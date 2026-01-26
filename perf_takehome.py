"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.
"""

from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class KernelBuilder:
    def __init__(self, enable_debug: bool = False):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}
        self.vconst_map = {}
        self.enable_debug = enable_debug

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def _range_set(self, base: int, length: int):
        return set(range(base, base + length))

    def _slot_reads_writes(self, engine: Engine, slot: tuple[int, ...]):
        reads = set()
        writes = set()
        match engine:
            case "alu":
                _, dest, a1, a2 = slot
                writes.add(dest)
                reads.update([a1, a2])
            case "valu":
                match slot:
                    case ("vbroadcast", dest, src):
                        writes.update(self._range_set(dest, VLEN))
                        reads.add(src)
                    case ("multiply_add", dest, a, b, c):
                        writes.update(self._range_set(dest, VLEN))
                        reads.update(self._range_set(a, VLEN))
                        reads.update(self._range_set(b, VLEN))
                        reads.update(self._range_set(c, VLEN))
                    case (op, dest, a1, a2):
                        writes.update(self._range_set(dest, VLEN))
                        reads.update(self._range_set(a1, VLEN))
                        reads.update(self._range_set(a2, VLEN))
            case "load":
                match slot:
                    case ("load", dest, addr):
                        writes.add(dest)
                        reads.add(addr)
                    case ("load_offset", dest, addr, offset):
                        writes.add(dest + offset)
                        reads.add(addr + offset)
                    case ("vload", dest, addr):
                        writes.update(self._range_set(dest, VLEN))
                        reads.add(addr)
                    case ("const", dest, _val):
                        writes.add(dest)
            case "store":
                match slot:
                    case ("store", addr, src):
                        reads.update([addr, src])
                    case ("vstore", addr, src):
                        reads.add(addr)
                        reads.update(self._range_set(src, VLEN))
            case "flow":
                match slot:
                    case ("select", dest, cond, a, b):
                        writes.add(dest)
                        reads.update([cond, a, b])
                    case ("add_imm", dest, a, _imm):
                        writes.add(dest)
                        reads.add(a)
                    case ("vselect", dest, cond, a, b):
                        writes.update(self._range_set(dest, VLEN))
                        reads.update(self._range_set(cond, VLEN))
                        reads.update(self._range_set(a, VLEN))
                        reads.update(self._range_set(b, VLEN))
                    case ("trace_write", val):
                        reads.add(val)
                    case ("cond_jump", cond, addr):
                        reads.update([cond, addr])
                    case ("cond_jump_rel", cond, _offset):
                        reads.add(cond)
                    case ("jump", addr):
                        reads.add(addr)
                    case ("jump_indirect", addr):
                        reads.add(addr)
                    case ("coreid", dest):
                        writes.add(dest)
            case "debug":
                match slot:
                    case ("compare", loc, _key):
                        reads.add(loc)
                    case ("vcompare", loc, _keys):
                        reads.update(self._range_set(loc, VLEN))
        return reads, writes

    def _schedule_segment(self, slots: list[tuple[Engine, tuple]]):
        if not slots:
            return []

        reads = []
        writes = []
        succ = [[] for _ in range(len(slots))]
        hard_indegree = [0] * len(slots)
        war_preds = [set() for _ in range(len(slots))]
        last_write = {}
        last_read = {}

        for idx, (engine, slot) in enumerate(slots):
            rset, wset = self._slot_reads_writes(engine, slot)
            reads.append(rset)
            writes.append(wset)
            deps = set()
            for addr in rset:
                if addr in last_write:
                    deps.add(last_write[addr])
            for addr in wset:
                if addr in last_write:
                    deps.add(last_write[addr])
                if addr in last_read:
                    war_preds[idx].add(last_read[addr])
            for dep in deps:
                succ[dep].append(idx)
                hard_indegree[idx] += 1
            for addr in rset:
                last_read[addr] = idx
            for addr in wset:
                last_write[addr] = idx

        slot_engines = [engine for engine, _slot in slots]
        engine_priority = {
            "load": 0,
            "valu": 1,
            "alu": 2,
            "store": 3,
            "flow": 4,
            "debug": 5,
        }
        rank = [1] * len(slots)
        for idx in range(len(slots) - 1, -1, -1):
            if succ[idx]:
                rank[idx] = 1 + max(rank[succ_idx] for succ_idx in succ[idx])
        ready = [i for i, deg in enumerate(hard_indegree) if deg == 0]
        ready.sort()
        instrs = []
        scheduled = 0
        scheduled_set = set()

        while scheduled < len(slots):
            if not ready:
                raise RuntimeError("Scheduler stalled: no ready slots.")

            current = {}
            engine_counts = defaultdict(int)
            current_reads = set()
            current_writes = set()
            scheduled_in_cycle = []
            available_war = set(scheduled_set)

            ready.sort(key=lambda i: (engine_priority[slot_engines[i]], -rank[i], i))
            for idx in ready:
                engine, slot = slots[idx]
                rset = reads[idx]
                wset = writes[idx]
                has_conflict = bool(wset & current_writes)
                if engine_counts[engine] >= SLOT_LIMITS[engine] or has_conflict:
                    continue
                if not war_preds[idx].issubset(available_war):
                    continue
                if engine not in current:
                    current[engine] = []
                current[engine].append(slot)
                engine_counts[engine] += 1
                current_reads.update(rset)
                current_writes.update(wset)
                scheduled_in_cycle.append(idx)
                available_war.add(idx)

            if not scheduled_in_cycle:
                idx = ready[0]
                engine, slot = slots[idx]
                current = {engine: [slot]}
                scheduled_in_cycle = [idx]
                available_war.add(idx)

            instrs.append(current)
            scheduled += len(scheduled_in_cycle)
            scheduled_set.update(scheduled_in_cycle)

            for idx in scheduled_in_cycle:
                ready.remove(idx)
                for succ_idx in succ[idx]:
                    hard_indegree[succ_idx] -= 1
                    if hard_indegree[succ_idx] == 0:
                        ready.append(succ_idx)
            ready.sort()

        return instrs

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        # VLIW slot packing with conservative dependency avoidance.
        if not vliw:
            instrs = []
            for engine, slot in slots:
                instrs.append({engine: [slot]})
            return instrs

        instrs = []
        segment = []
        for engine, slot in slots:
            if engine == "debug":
                instrs.extend(self._schedule_segment(segment))
                segment = []
                instrs.append({engine: [slot]})
                continue
            segment.append((engine, slot))
        instrs.extend(self._schedule_segment(segment))
        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def scratch_const_vector(self, val, name=None):
        if val not in self.vconst_map:
            addr = self.alloc_scratch(name, VLEN)
            scalar_addr = self.scratch_const(val)
            self.add("valu", ("vbroadcast", addr, scalar_addr))
            self.vconst_map[val] = addr
        return self.vconst_map[val]

    def vbroadcast_from(self, src_addr, name=None):
        addr = self.alloc_scratch(name, VLEN)
        self.add("valu", ("vbroadcast", addr, src_addr))
        return addr

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            if op1 == "+" and op2 == "+" and op3 == "<<":
                factor = 1 + (1 << val3)
                slots.append(("alu", ("*", tmp1, val_hash_addr, self.scratch_const(factor))))
                slots.append(("alu", ("+", val_hash_addr, tmp1, self.scratch_const(val1))))
            else:
                slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
                slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
                slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            if self.enable_debug:
                slots.append(
                    ("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi)))
                )

        return slots

    def build_hash_vector(self, val_hash_addr, tmp1, tmp2, round, i_base):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            v1 = self.scratch_const_vector(val1)
            v3 = self.scratch_const_vector(val3)
            slots.append(("valu", (op1, tmp1, val_hash_addr, v1)))
            slots.append(("valu", (op3, tmp2, val_hash_addr, v3)))
            slots.append(("valu", (op2, val_hash_addr, tmp1, tmp2)))
            if self.enable_debug:
                keys = [(round, i_base + lane, "hash_stage", hi) for lane in range(VLEN)]
                slots.append(("debug", ("vcompare", val_hash_addr, keys)))

        return slots

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Vector-first kernel with scratch-resident idx/val, hashed across rounds,
        then written back once at the end. Scalar tail handled separately.
        """
        # Scratch space addresses
        init_vars = [
            ("n_nodes", 1),
            ("forest_values_p", 4),
            ("inp_values_p", 6),
        ]
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")
        tmp3 = self.alloc_scratch("tmp3")
        for name, _idx in init_vars:
            self.alloc_scratch(name, 1)
        for name, idx in init_vars:
            self.add("load", ("const", tmp1, idx))
            self.add("load", ("load", self.scratch[name], tmp1))

        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)
        three_const = self.scratch_const(3)

        # Pause instructions are matched up with yield statements in the reference
        # kernel to let you debug at intermediate steps. The testing harness in this
        # file requires these match up to the reference kernel's yields, but the
        # submission harness ignores them.
        self.add("flow", ("pause",))

        body = []  # array of slots

        # Shared scalar temps
        tmp_idx_addr = self.alloc_scratch("tmp_idx_addr")
        tmp_val_addr = self.alloc_scratch("tmp_val_addr")
        tmp_addr = self.alloc_scratch("tmp_addr")

        # Vector scratch buffers (full batch resident)
        vec_end = batch_size - (batch_size % VLEN)
        idx_buf = self.alloc_scratch("idx_buf", vec_end)
        val_buf = self.alloc_scratch("val_buf", vec_end)

        # Scalar tail buffers (<= VLEN-1)
        tail_len = batch_size - vec_end
        tail_idx = self.alloc_scratch("tail_idx", tail_len if tail_len > 0 else 1)
        tail_val = self.alloc_scratch("tail_val", tail_len if tail_len > 0 else 1)

        # Vector temps
        pipe = 13
        addr_v_base = self.alloc_scratch("addr_v", VLEN * pipe)
        node_v0_base = self.alloc_scratch("node_v0", VLEN * pipe)
        node_v1_base = self.alloc_scratch("node_v1", VLEN * pipe)
        tmp1_v_base = self.alloc_scratch("tmp1_v", VLEN * pipe)
        tmp2_v_base = self.alloc_scratch("tmp2_v", VLEN * pipe)

        zero_v = self.scratch_const_vector(0)
        one_v = self.scratch_const_vector(1)
        two_v = self.scratch_const_vector(2)
        three_v = self.scratch_const_vector(3)
        n_nodes_v = self.vbroadcast_from(self.scratch["n_nodes"], "n_nodes_v")
        forest_base_v = self.vbroadcast_from(
            self.scratch["forest_values_p"], "forest_values_v"
        )

        # Preload root and first-level child node values for rounds with tiny depth.
        root_val = self.alloc_scratch("root_val")
        left_val = self.alloc_scratch("left_val")
        right_val = self.alloc_scratch("right_val")
        root_v = self.alloc_scratch("root_v", VLEN)
        left_v = self.alloc_scratch("left_v", VLEN)
        right_v = self.alloc_scratch("right_v", VLEN)
        left_minus_right_v = self.alloc_scratch("left_minus_right_v", VLEN)

        self.add("load", ("load", root_val, self.scratch["forest_values_p"]))
        self.add("alu", ("+", tmp1, self.scratch["forest_values_p"], one_const))
        self.add("load", ("load", left_val, tmp1))
        self.add("alu", ("+", tmp1, self.scratch["forest_values_p"], two_const))
        self.add("load", ("load", right_val, tmp1))
        self.add("valu", ("vbroadcast", root_v, root_val))
        self.add("valu", ("vbroadcast", left_v, left_val))
        self.add("valu", ("vbroadcast", right_v, right_val))
        self.add("valu", ("-", left_minus_right_v, left_v, right_v))

        # Preload depth-2 node values (indices 3..6) and deltas for depth==2 selection.
        d2_node3 = self.alloc_scratch("d2_node3")
        d2_node4 = self.alloc_scratch("d2_node4")
        d2_node5 = self.alloc_scratch("d2_node5")
        d2_node6 = self.alloc_scratch("d2_node6")
        d2_node3_v = self.alloc_scratch("d2_node3_v", VLEN)
        d2_node4_v = self.alloc_scratch("d2_node4_v", VLEN)
        d2_node5_v = self.alloc_scratch("d2_node5_v", VLEN)
        d2_node6_v = self.alloc_scratch("d2_node6_v", VLEN)
        d2_delta34_v = self.alloc_scratch("d2_delta34_v", VLEN)
        d2_delta56_v = self.alloc_scratch("d2_delta56_v", VLEN)

        self.add("alu", ("+", tmp1, self.scratch["forest_values_p"], three_const))
        self.add("load", ("load", d2_node3, tmp1))
        self.add("alu", ("+", tmp1, tmp1, one_const))
        self.add("load", ("load", d2_node4, tmp1))
        self.add("alu", ("+", tmp1, tmp1, one_const))
        self.add("load", ("load", d2_node5, tmp1))
        self.add("alu", ("+", tmp1, tmp1, one_const))
        self.add("load", ("load", d2_node6, tmp1))
        self.add("valu", ("vbroadcast", d2_node3_v, d2_node3))
        self.add("valu", ("vbroadcast", d2_node4_v, d2_node4))
        self.add("valu", ("vbroadcast", d2_node5_v, d2_node5))
        self.add("valu", ("vbroadcast", d2_node6_v, d2_node6))
        self.add("valu", ("-", d2_delta34_v, d2_node4_v, d2_node3_v))
        self.add("valu", ("-", d2_delta56_v, d2_node6_v, d2_node5_v))


        hash_plan = []
        for op1, val1, op2, op3, val3 in HASH_STAGES:
            if op1 == "+" and op2 == "+" and op3 == "<<":
                factor = 1 + (1 << val3)
                hash_plan.append(
                    (
                        "fused_add",
                        self.scratch_const_vector(factor),
                        self.scratch_const_vector(val1),
                    )
                )
            else:
                hash_plan.append(
                    (
                        "generic",
                        self.scratch_const_vector(val1),
                        self.scratch_const_vector(val3),
                        op1,
                        op2,
                        op3,
                    )
                )

        # Initial load of idx/val into scratch buffers
        for i in range(0, vec_end, VLEN):
            i_const = self.scratch_const(i)
            body.append(
                ("alu", ("+", tmp_val_addr, self.scratch["inp_values_p"], i_const))
            )
            body.append(("load", ("vload", val_buf + i, tmp_val_addr)))

        # Scalar tail load
        for i in range(tail_len):
            i_const = self.scratch_const(vec_end + i)
            body.append(
                ("alu", ("+", tmp_idx_addr, self.scratch["inp_indices_p"], i_const))
            )
            body.append(("load", ("load", tail_idx + i, tmp_idx_addr)))
            body.append(
                ("alu", ("+", tmp_val_addr, self.scratch["inp_values_p"], i_const))
            )
            body.append(("load", ("load", tail_val + i, tmp_val_addr)))

        # Rounds over scratch-resident buffers
        blocks = list(range(0, vec_end, VLEN))
        groups = [blocks[i : i + pipe] for i in range(0, len(blocks), pipe)]
        period = forest_height + 1

        for round_i in range(rounds):
            depth = round_i % period
            use_root = depth == 0
            use_children = depth == 1
            use_depth2 = depth == 2
            use_gather = not (use_root or use_children or use_depth2)
            use_wrap = depth == forest_height

            for gi_group, group in enumerate(groups):
                cur_node_base = node_v0_base if (gi_group % 2 == 0) else node_v1_base
                next_node_base = node_v1_base if (gi_group % 2 == 0) else node_v0_base
                next_group = (
                    groups[gi_group + 1]
                    if use_gather and gi_group + 1 < len(groups)
                    else None
                )

                if self.enable_debug:
                    for i in group:
                        keys = [(round_i, i + lane, "idx") for lane in range(VLEN)]
                        body.append(("debug", ("vcompare", idx_buf + i, keys)))
                        keys = [(round_i, i + lane, "val") for lane in range(VLEN)]
                        body.append(("debug", ("vcompare", val_buf + i, keys)))

                load_slots_next = []
                if use_gather:
                    if gi_group == 0:
                        for gi, i in enumerate(group):
                            lane_base = gi * VLEN
                            addr_v = addr_v_base + lane_base
                            body.append(("valu", ("+", addr_v, idx_buf + i, forest_base_v)))
                        for gi, i in enumerate(group):
                            lane_base = gi * VLEN
                            addr_v = addr_v_base + lane_base
                            node_v = cur_node_base + lane_base
                            for lane in range(VLEN):
                                body.append(("load", ("load_offset", node_v, addr_v, lane)))
                    if next_group:
                        for gi, i in enumerate(next_group):
                            lane_base = gi * VLEN
                            addr_v = addr_v_base + lane_base
                            body.append(("valu", ("+", addr_v, idx_buf + i, forest_base_v)))
                        for gi, i in enumerate(next_group):
                            lane_base = gi * VLEN
                            addr_v = addr_v_base + lane_base
                            node_v = next_node_base + lane_base
                            for lane in range(VLEN):
                                load_slots_next.append(
                                    ("load", ("load_offset", node_v, addr_v, lane))
                                )

                load_idx = [0]
                load_per_point = 2 if load_slots_next else 0

                def emit_loads():
                    if load_per_point == 0:
                        return
                    for _ in range(load_per_point):
                        if load_idx[0] >= len(load_slots_next):
                            break
                        body.append(load_slots_next[load_idx[0]])
                        load_idx[0] += 1

                if use_children:
                    for gi, i in enumerate(group):
                        lane_base = gi * VLEN
                        tmp1_v = tmp1_v_base + lane_base
                        body.append(("valu", ("&", tmp1_v, idx_buf + i, one_v)))
                    emit_loads()
                    for gi, i in enumerate(group):
                        lane_base = gi * VLEN
                        tmp1_v = tmp1_v_base + lane_base
                        node_v = cur_node_base + lane_base
                        body.append(
                            (
                                "valu",
                                ("multiply_add", node_v, tmp1_v, left_minus_right_v, right_v),
                            )
                        )
                    emit_loads()

                if use_depth2:
                    for gi, i in enumerate(group):
                        lane_base = gi * VLEN
                        tmp1_v = tmp1_v_base + lane_base
                        body.append(("valu", ("-", tmp1_v, idx_buf + i, three_v)))
                    emit_loads()
                    for gi, i in enumerate(group):
                        lane_base = gi * VLEN
                        tmp1_v = tmp1_v_base + lane_base
                        tmp2_v = tmp2_v_base + lane_base
                        body.append(("valu", ("&", tmp2_v, tmp1_v, one_v)))
                    emit_loads()
                    for gi, i in enumerate(group):
                        lane_base = gi * VLEN
                        tmp1_v = tmp1_v_base + lane_base
                        body.append(("valu", (">>", tmp1_v, tmp1_v, one_v)))
                    emit_loads()
                    for gi, i in enumerate(group):
                        lane_base = gi * VLEN
                        tmp2_v = tmp2_v_base + lane_base
                        node_v = cur_node_base + lane_base
                        body.append(
                            (
                                "valu",
                                (
                                    "multiply_add",
                                    node_v,
                                    tmp2_v,
                                    d2_delta34_v,
                                    d2_node3_v,
                                ),
                            )
                        )
                    emit_loads()
                    for gi, i in enumerate(group):
                        lane_base = gi * VLEN
                        tmp2_v = tmp2_v_base + lane_base
                        body.append(
                            (
                                "valu",
                                (
                                    "multiply_add",
                                    tmp2_v,
                                    tmp2_v,
                                    d2_delta56_v,
                                    d2_node5_v,
                                ),
                            )
                        )
                    emit_loads()
                    for gi, i in enumerate(group):
                        lane_base = gi * VLEN
                        tmp2_v = tmp2_v_base + lane_base
                        node_v = cur_node_base + lane_base
                        body.append(("valu", ("-", tmp2_v, tmp2_v, node_v)))
                    emit_loads()
                    for gi, i in enumerate(group):
                        lane_base = gi * VLEN
                        tmp1_v = tmp1_v_base + lane_base
                        tmp2_v = tmp2_v_base + lane_base
                        node_v = cur_node_base + lane_base
                        body.append(
                            ("valu", ("multiply_add", node_v, tmp1_v, tmp2_v, node_v))
                        )
                    emit_loads()

                for gi, i in enumerate(group):
                    lane_base = gi * VLEN
                    node_src = root_v if use_root else cur_node_base + lane_base
                    body.append(("valu", ("^", val_buf + i, val_buf + i, node_src)))
                emit_loads()

                for hi, plan in enumerate(hash_plan):
                    match plan:
                        case ("fused_add", factor_v, v1):
                            for gi, i in enumerate(group):
                                body.append(
                                    (
                                        "valu",
                                        ("multiply_add", val_buf + i, val_buf + i, factor_v, v1),
                                    )
                                )
                            emit_loads()
                        case ("generic", v1, v3, op1, op2, op3):
                            for gi, i in enumerate(group):
                                lane_base = gi * VLEN
                                tmp1_v = tmp1_v_base + lane_base
                                body.append(("valu", (op1, tmp1_v, val_buf + i, v1)))
                            emit_loads()
                            for gi, i in enumerate(group):
                                lane_base = gi * VLEN
                                tmp2_v = tmp2_v_base + lane_base
                                body.append(("valu", (op3, tmp2_v, val_buf + i, v3)))
                            emit_loads()
                            for gi, i in enumerate(group):
                                lane_base = gi * VLEN
                                tmp1_v = tmp1_v_base + lane_base
                                tmp2_v = tmp2_v_base + lane_base
                                body.append(("valu", (op2, val_buf + i, tmp1_v, tmp2_v)))
                            emit_loads()

                if self.enable_debug:
                    for i in group:
                        keys = [(round_i, i + lane, "hashed_val") for lane in range(VLEN)]
                        body.append(("debug", ("vcompare", val_buf + i, keys)))

                if use_root:
                    for gi, i in enumerate(group):
                        lane_base = gi * VLEN
                        tmp1_v = tmp1_v_base + lane_base
                        body.append(("valu", ("&", tmp1_v, val_buf + i, one_v)))
                    emit_loads()
                    for gi, i in enumerate(group):
                        lane_base = gi * VLEN
                        tmp1_v = tmp1_v_base + lane_base
                        body.append(("valu", ("+", idx_buf + i, tmp1_v, one_v)))
                    emit_loads()
                else:
                    for gi, i in enumerate(group):
                        lane_base = gi * VLEN
                        tmp1_v = tmp1_v_base + lane_base
                        body.append(("valu", ("&", tmp1_v, val_buf + i, one_v)))
                    emit_loads()
                    for gi, i in enumerate(group):
                        lane_base = gi * VLEN
                        tmp1_v = tmp1_v_base + lane_base
                        body.append(("valu", ("+", tmp1_v, tmp1_v, one_v)))
                    emit_loads()
                    for gi, i in enumerate(group):
                        lane_base = gi * VLEN
                        tmp1_v = tmp1_v_base + lane_base
                        body.append(
                            ("valu", ("multiply_add", idx_buf + i, idx_buf + i, two_v, tmp1_v))
                        )
                    emit_loads()
                if self.enable_debug:
                    for i in group:
                        keys = [(round_i, i + lane, "next_idx") for lane in range(VLEN)]
                        body.append(("debug", ("vcompare", idx_buf + i, keys)))
                if use_wrap:
                    for gi, i in enumerate(group):
                        lane_base = gi * VLEN
                        tmp2_v = tmp2_v_base + lane_base
                        body.append(("valu", ("<", tmp2_v, idx_buf + i, n_nodes_v)))
                    emit_loads()
                    for gi, i in enumerate(group):
                        lane_base = gi * VLEN
                        tmp2_v = tmp2_v_base + lane_base
                        body.append(("valu", ("*", idx_buf + i, idx_buf + i, tmp2_v)))
                    emit_loads()
                if self.enable_debug:
                    for i in group:
                        keys = [(round_i, i + lane, "wrapped_idx") for lane in range(VLEN)]
                        body.append(("debug", ("vcompare", idx_buf + i, keys)))

                if load_slots_next:
                    while load_idx[0] < len(load_slots_next):
                        body.append(load_slots_next[load_idx[0]])
                        load_idx[0] += 1
 

            # Scalar tail per round
            for ti in range(tail_len):
                if self.enable_debug:
                    body.append(
                        ("debug", ("compare", tail_idx + ti, (round_i, vec_end + ti, "idx")))
                    )
                    body.append(
                        ("debug", ("compare", tail_val + ti, (round_i, vec_end + ti, "val")))
                    )
                body.append(
                    ("alu", ("+", tmp_addr, self.scratch["forest_values_p"], tail_idx + ti))
                )
                body.append(("load", ("load", tmp3, tmp_addr)))
                if self.enable_debug:
                    body.append(
                        ("debug", ("compare", tmp3, (round_i, vec_end + ti, "node_val")))
                    )
                body.append(("alu", ("^", tail_val + ti, tail_val + ti, tmp3)))
                body.extend(self.build_hash(tail_val + ti, tmp1, tmp2, round_i, vec_end + ti))
                if self.enable_debug:
                    body.append(
                        ("debug", ("compare", tail_val + ti, (round_i, vec_end + ti, "hashed_val")))
                    )
                body.append(("alu", ("&", tmp1, tail_val + ti, one_const)))
                if use_root:
                    body.append(("alu", ("+", tail_idx + ti, tmp1, one_const)))
                else:
                    body.append(("alu", ("+", tmp3, tmp1, one_const)))
                    body.append(("alu", ("*", tail_idx + ti, tail_idx + ti, two_const)))
                    body.append(("alu", ("+", tail_idx + ti, tail_idx + ti, tmp3)))
                if self.enable_debug:
                    body.append(
                        ("debug", ("compare", tail_idx + ti, (round_i, vec_end + ti, "next_idx")))
                    )
                if use_wrap:
                    body.append(("alu", ("<", tmp1, tail_idx + ti, self.scratch["n_nodes"])))
                    body.append(
                        ("flow", ("select", tail_idx + ti, tmp1, tail_idx + ti, zero_const))
                    )
                if self.enable_debug:
                    body.append(
                        ("debug", ("compare", tail_idx + ti, (round_i, vec_end + ti, "wrapped_idx")))
                    )

        # Store back final vectors
        for i in range(0, vec_end, VLEN):
            i_const = self.scratch_const(i)
            body.append(
                ("alu", ("+", tmp_val_addr, self.scratch["inp_values_p"], i_const))
            )
            body.append(("store", ("vstore", tmp_val_addr, val_buf + i)))

        # Store scalar tail
        for i in range(tail_len):
            i_const = self.scratch_const(vec_end + i)
            body.append(
                ("alu", ("+", tmp_val_addr, self.scratch["inp_values_p"], i_const))
            )
            body.append(("store", ("store", tmp_val_addr, tail_val + i)))

        body_instrs = self.build(body, vliw=True)
        self.instrs.extend(body_instrs)
        # Required to match with the yield in reference_kernel2
        self.instrs.append({"flow": [("pause",)]})

BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
        trace_path="trace/trace.json" if trace else None,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
