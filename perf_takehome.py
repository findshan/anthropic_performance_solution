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

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        # Greedy VLIW slot packing with conservative dependency avoidance.
        if not vliw:
            instrs = []
            for engine, slot in slots:
                instrs.append({engine: [slot]})
            return instrs

        instrs = []
        current = {}
        engine_counts = defaultdict(int)
        current_reads = set()
        current_writes = set()

        def flush():
            nonlocal current, engine_counts, current_reads, current_writes
            if current:
                instrs.append(current)
            current = {}
            engine_counts = defaultdict(int)
            current_reads = set()
            current_writes = set()

        for engine, slot in slots:
            if engine == "debug":
                flush()
                instrs.append({engine: [slot]})
                continue

            reads, writes = self._slot_reads_writes(engine, slot)
            has_conflict = (
                bool(reads & current_writes)
                or bool(writes & current_reads)
                or bool(writes & current_writes)
            )
            if engine_counts[engine] >= SLOT_LIMITS[engine] or has_conflict:
                flush()

            if engine not in current:
                current[engine] = []
            current[engine].append(slot)
            engine_counts[engine] += 1
            current_reads.update(reads)
            current_writes.update(writes)

        flush()
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
        Like reference_kernel2 but building actual instructions.
        Scalar implementation using only scalar ALU and load/store.
        """
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")
        tmp3 = self.alloc_scratch("tmp3")
        # Scratch space addresses
        init_vars = [
            "rounds",
            "n_nodes",
            "batch_size",
            "forest_height",
            "forest_values_p",
            "inp_indices_p",
            "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)

        # Pause instructions are matched up with yield statements in the reference
        # kernel to let you debug at intermediate steps. The testing harness in this
        # file requires these match up to the reference kernel's yields, but the
        # submission harness ignores them.
        self.add("flow", ("pause",))

        body = []  # array of slots

        # Scalar scratch registers
        tmp_idx = self.alloc_scratch("tmp_idx")
        tmp_val = self.alloc_scratch("tmp_val")
        tmp_node_val = self.alloc_scratch("tmp_node_val")
        tmp_addr = self.alloc_scratch("tmp_addr")
        tmp_idx_addr = self.alloc_scratch("tmp_idx_addr")
        tmp_val_addr = self.alloc_scratch("tmp_val_addr")
        tmp_idx_addr0 = self.alloc_scratch("tmp_idx_addr0")
        tmp_val_addr0 = self.alloc_scratch("tmp_val_addr0")
        tmp_idx_addr1 = self.alloc_scratch("tmp_idx_addr1")
        tmp_val_addr1 = self.alloc_scratch("tmp_val_addr1")

        # Vector scratch registers
        idx_v0 = self.alloc_scratch("idx_v0", VLEN)
        val_v0 = self.alloc_scratch("val_v0", VLEN)
        node_v0 = self.alloc_scratch("node_v0", VLEN)
        addr_v0 = self.alloc_scratch("addr_v0", VLEN)
        idx_v1 = self.alloc_scratch("idx_v1", VLEN)
        val_v1 = self.alloc_scratch("val_v1", VLEN)
        node_v1 = self.alloc_scratch("node_v1", VLEN)
        addr_v1 = self.alloc_scratch("addr_v1", VLEN)
        tmp1_v = self.alloc_scratch("tmp1_v", VLEN)
        tmp2_v = self.alloc_scratch("tmp2_v", VLEN)

        one_v = self.scratch_const_vector(1)
        two_v = self.scratch_const_vector(2)
        n_nodes_v = self.vbroadcast_from(self.scratch["n_nodes"], "n_nodes_v")
        forest_base_v = self.vbroadcast_from(
            self.scratch["forest_values_p"], "forest_values_v"
        )

        vec_end = batch_size - (batch_size % VLEN)
        for round in range(rounds):
            buffers = [
                {
                    "idx": idx_v0,
                    "val": val_v0,
                    "node": node_v0,
                    "addr": addr_v0,
                    "idx_addr": tmp_idx_addr0,
                    "val_addr": tmp_val_addr0,
                },
                {
                    "idx": idx_v1,
                    "val": val_v1,
                    "node": node_v1,
                    "addr": addr_v1,
                    "idx_addr": tmp_idx_addr1,
                    "val_addr": tmp_val_addr1,
                },
            ]

            def emit_vec_load(i, buf):
                i_const = self.scratch_const(i)
                body.append(
                    ("alu", ("+", buf["idx_addr"], self.scratch["inp_indices_p"], i_const))
                )
                body.append(("load", ("vload", buf["idx"], buf["idx_addr"])))
                if self.enable_debug:
                    keys = [(round, i + lane, "idx") for lane in range(VLEN)]
                    body.append(("debug", ("vcompare", buf["idx"], keys)))
                body.append(
                    ("alu", ("+", buf["val_addr"], self.scratch["inp_values_p"], i_const))
                )
                body.append(("load", ("vload", buf["val"], buf["val_addr"])))
                if self.enable_debug:
                    keys = [(round, i + lane, "val") for lane in range(VLEN)]
                    body.append(("debug", ("vcompare", buf["val"], keys)))
                body.append(("valu", ("+", buf["addr"], buf["idx"], forest_base_v)))
                for lane in range(VLEN):
                    body.append(("load", ("load_offset", buf["node"], buf["addr"], lane)))
                if self.enable_debug:
                    keys = [(round, i + lane, "node_val") for lane in range(VLEN)]
                    body.append(("debug", ("vcompare", buf["node"], keys)))

            def emit_vec_compute(i, buf, next_i=None, next_buf=None):
                load_lane = 0
                if next_i is not None and next_buf is not None:
                    i_const = self.scratch_const(next_i)
                    body.append(
                        (
                            "alu",
                            ("+", next_buf["idx_addr"], self.scratch["inp_indices_p"], i_const),
                        )
                    )
                    body.append(("load", ("vload", next_buf["idx"], next_buf["idx_addr"])))
                    if self.enable_debug:
                        keys = [(round, next_i + lane, "idx") for lane in range(VLEN)]
                        body.append(("debug", ("vcompare", next_buf["idx"], keys)))
                    body.append(
                        (
                            "alu",
                            ("+", next_buf["val_addr"], self.scratch["inp_values_p"], i_const),
                        )
                    )
                    body.append(("load", ("vload", next_buf["val"], next_buf["val_addr"])))
                    if self.enable_debug:
                        keys = [(round, next_i + lane, "val") for lane in range(VLEN)]
                        body.append(("debug", ("vcompare", next_buf["val"], keys)))
                    body.append(
                        ("valu", ("+", next_buf["addr"], next_buf["idx"], forest_base_v))
                    )
                body.append(("valu", ("^", buf["val"], buf["val"], buf["node"])))
                for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                    v1 = self.scratch_const_vector(val1)
                    v3 = self.scratch_const_vector(val3)
                    body.append(("valu", (op1, tmp1_v, buf["val"], v1)))
                    body.append(("valu", (op3, tmp2_v, buf["val"], v3)))
                    if next_buf is not None and load_lane < VLEN:
                        for _ in range(2):
                            if load_lane >= VLEN:
                                break
                            body.append(
                                ("load", ("load_offset", next_buf["node"], next_buf["addr"], load_lane))
                            )
                            load_lane += 1
                    body.append(("valu", (op2, buf["val"], tmp1_v, tmp2_v)))
                    if self.enable_debug:
                        keys = [
                            (round, i + lane, "hash_stage", hi) for lane in range(VLEN)
                        ]
                        body.append(("debug", ("vcompare", buf["val"], keys)))
                if self.enable_debug:
                    keys = [(round, i + lane, "hashed_val") for lane in range(VLEN)]
                    body.append(("debug", ("vcompare", buf["val"], keys)))
                body.append(("valu", ("%", tmp1_v, buf["val"], two_v)))
                body.append(("valu", ("+", tmp1_v, tmp1_v, one_v)))
                body.append(("valu", ("*", buf["idx"], buf["idx"], two_v)))
                body.append(("valu", ("+", buf["idx"], buf["idx"], tmp1_v)))
                if self.enable_debug:
                    keys = [(round, i + lane, "next_idx") for lane in range(VLEN)]
                    body.append(("debug", ("vcompare", buf["idx"], keys)))
                body.append(("valu", ("<", tmp1_v, buf["idx"], n_nodes_v)))
                body.append(("valu", ("*", buf["idx"], buf["idx"], tmp1_v)))
                if self.enable_debug:
                    keys = [(round, i + lane, "wrapped_idx") for lane in range(VLEN)]
                    body.append(("debug", ("vcompare", buf["idx"], keys)))
                body.append(("store", ("vstore", buf["idx_addr"], buf["idx"])))
                body.append(("store", ("vstore", buf["val_addr"], buf["val"])))

            if vec_end > 0:
                emit_vec_load(0, buffers[0])
                for i in range(0, vec_end, VLEN):
                    buf = buffers[(i // VLEN) % 2]
                    next_i = i + VLEN
                    if next_i < vec_end:
                        next_buf = buffers[((i // VLEN) + 1) % 2]
                        emit_vec_compute(i, buf, next_i, next_buf)
                    else:
                        emit_vec_compute(i, buf)

            for i in range(vec_end, batch_size):
                i_const = self.scratch_const(i)
                # idx = mem[inp_indices_p + i]
                body.append(
                    ("alu", ("+", tmp_idx_addr, self.scratch["inp_indices_p"], i_const))
                )
                body.append(("load", ("load", tmp_idx, tmp_idx_addr)))
                if self.enable_debug:
                    body.append(("debug", ("compare", tmp_idx, (round, i, "idx"))))
                # val = mem[inp_values_p + i]
                body.append(
                    ("alu", ("+", tmp_val_addr, self.scratch["inp_values_p"], i_const))
                )
                body.append(("load", ("load", tmp_val, tmp_val_addr)))
                if self.enable_debug:
                    body.append(("debug", ("compare", tmp_val, (round, i, "val"))))
                # node_val = mem[forest_values_p + idx]
                body.append(("alu", ("+", tmp_addr, self.scratch["forest_values_p"], tmp_idx)))
                body.append(("load", ("load", tmp_node_val, tmp_addr)))
                if self.enable_debug:
                    body.append(("debug", ("compare", tmp_node_val, (round, i, "node_val"))))
                # val = myhash(val ^ node_val)
                body.append(("alu", ("^", tmp_val, tmp_val, tmp_node_val)))
                body.extend(self.build_hash(tmp_val, tmp1, tmp2, round, i))
                if self.enable_debug:
                    body.append(("debug", ("compare", tmp_val, (round, i, "hashed_val"))))
                # idx = 2*idx + (1 if val % 2 == 0 else 2)
                body.append(("alu", ("%", tmp1, tmp_val, two_const)))
                body.append(("alu", ("==", tmp1, tmp1, zero_const)))
                body.append(("flow", ("select", tmp3, tmp1, one_const, two_const)))
                body.append(("alu", ("*", tmp_idx, tmp_idx, two_const)))
                body.append(("alu", ("+", tmp_idx, tmp_idx, tmp3)))
                if self.enable_debug:
                    body.append(("debug", ("compare", tmp_idx, (round, i, "next_idx"))))
                # idx = 0 if idx >= n_nodes else idx
                body.append(("alu", ("<", tmp1, tmp_idx, self.scratch["n_nodes"])))
                body.append(("flow", ("select", tmp_idx, tmp1, tmp_idx, zero_const)))
                if self.enable_debug:
                    body.append(("debug", ("compare", tmp_idx, (round, i, "wrapped_idx"))))
                # mem[inp_indices_p + i] = idx
                body.append(("store", ("store", tmp_idx_addr, tmp_idx)))
                # mem[inp_values_p + i] = val
                body.append(("store", ("store", tmp_val_addr, tmp_val)))

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
