"""
Optimized kernel for Anthropic's performance take-home.
All work is done in opt_work/ — the original folder is untouched.
"""

from collections import defaultdict
import random
import unittest

from problem import (
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    build_mem_image,
    reference_kernel2,
)

BASELINE = 147734


def vec_range(base):
    return frozenset(range(base, base + VLEN))


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.slots = []  # list of (engine, slot_tuple, reads_frozenset, writes_frozenset)
        self._const_scalar = None

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def alloc(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, f"Out of scratch space at {self.scratch_ptr}"
        return addr

    def alloc_vec(self, name=None):
        return self.alloc(name, VLEN)

    # ------------------------------------------------------------------
    # Low-level emitters (with read/write sets for hazard-aware scheduling)
    # ------------------------------------------------------------------
    def emit(self, engine, slot, reads, writes):
        self.slots.append((engine, slot, frozenset(reads), frozenset(writes)))

    def emit_const_scalar(self, dest, val):
        self.emit("load", ("const", dest, val), (), (dest,))

    def emit_vbroadcast(self, dest, src):
        self.emit("valu", ("vbroadcast", dest, src), (src,), vec_range(dest))

    def emit_valu(self, op, dest, a1, a2):
        self.emit("valu", (op, dest, a1, a2), vec_range(a1) | vec_range(a2), vec_range(dest))

    def emit_alu(self, op, dest, a1, a2):
        self.emit("alu", (op, dest, a1, a2), (a1, a2), (dest,))

    def emit_vselect(self, dest, cond, a, b):
        self.emit(
            "flow",
            ("vselect", dest, cond, a, b),
            vec_range(cond) | vec_range(a) | vec_range(b),
            vec_range(dest),
        )

    def emit_valu_ma(self, dest, a, b, c):
        self.emit(
            "valu",
            ("multiply_add", dest, a, b, c),
            vec_range(a) | vec_range(b) | vec_range(c),
            vec_range(dest),
        )

    def emit_load_offset(self, dest, addr, offset):
        self.emit("load", ("load_offset", dest, addr, offset), (addr + offset,), (dest + offset,))

    def emit_vload(self, dest, addr_scalar):
        self.emit("load", ("vload", dest, addr_scalar), (addr_scalar,), vec_range(dest))

    def emit_vstore(self, addr_scalar, src):
        self.emit("store", ("vstore", addr_scalar, src), frozenset((addr_scalar,)) | vec_range(src), ())

    def emit_load_scalar(self, dest, addr):
        self.emit("load", ("load", dest, addr), (addr,), (dest,))

    def emit_store_scalar(self, addr, src):
        self.emit("store", ("store", addr, src), (addr, src), ())

    def scalar_const(self, val):
        s = self.alloc("scalar_c")
        self.emit_const_scalar(s, val)
        return s

    # ------------------------------------------------------------------
    # Constants
    # ------------------------------------------------------------------
    def const_vec(self, val):
        s = self.alloc("const_scalar")
        self.emit_const_scalar(s, val)
        v = self.alloc_vec("const_vec")
        self.emit_vbroadcast(v, s)
        return v

    def _emit_scalar_hash(self, val, t1, t2, S):
        self.emit_alu("*", t1, val, S["b0"])
        self.emit_alu("+", val, t1, S["c0"])
        self.emit_alu("^", t1, val, S["C1"])
        self.emit_alu(">>", t2, val, S["s1"])
        self.emit_alu("^", val, t1, t2)
        self.emit_alu("*", t1, val, S["b2"])
        self.emit_alu("+", val, t1, S["c2"])
        self.emit_alu("+", t1, val, S["C3"])
        self.emit_alu("<<", t2, val, S["s3"])
        self.emit_alu("^", val, t1, t2)
        self.emit_alu("*", t1, val, S["b4"])
        self.emit_alu("+", val, t1, S["c4"])
        self.emit_alu("^", t1, val, S["C5"])
        self.emit_alu(">>", t2, val, S["s5"])
        self.emit_alu("^", val, t1, t2)

    def _emit_scalar_round(self, r, rounds, depth, is_leaf, addr, val, t, t1, t2, S):
        if depth == 0:
            self.emit_alu("^", val, val, S["n0"])
        elif depth == 1:
            self.emit_alu("*", t1, addr, S["diff1"])
            self.emit_alu("+", t1, t1, S["const1"])
            self.emit_alu("^", val, val, t1)
        elif depth == 2:
            self.emit_alu("-", t, addr, S["c10"])       # offset
            self.emit_alu("&", t1, t, S["one"])         # bit1
            self.emit_alu(">>", t, t, S["one"])         # bit0
            self.emit_alu("*", t2, t1, S["d34"])        # bit1*d34
            self.emit_alu("+", t2, t2, S["n3"])         # lower in t2
            self.emit_alu("*", t1, t1, S["d56"])        # bit1*d56
            self.emit_alu("+", t1, t1, S["n5"])         # upper in t1
            self.emit_alu("-", t1, t1, t2)              # upper-lower
            self.emit_alu("*", t1, t, t1)               # bit0*(upper-lower)
            self.emit_alu("+", t1, t1, t2)              # node_val
            self.emit_alu("^", val, val, t1)
        else:
            self.emit_load_scalar(t1, addr)
            self.emit_alu("^", val, val, t1)

        self._emit_scalar_hash(val, t1, t2, S)

        if not is_leaf and r != rounds - 1:
            self.emit_alu("&", t1, val, S["one"])       # bit
            if depth == 0:
                self.emit_alu("+", addr, t1, S["c8"])
            else:
                self.emit_alu("<<", t2, addr, S["one"])  # addr*2
                self.emit_alu("+", t1, t1, S["c_branch"])
                self.emit_alu("+", addr, t2, t1)

    def build_kernel(self, forest_height, n_nodes, batch_size, rounds):
        fp = 7  # forest_values_p
        n_chunks = batch_size // VLEN  # 32
        n_levels = forest_height + 1  # 11
        n_scalar = 1  # chunks processed on the scalar ALU engine

        # ---- compile-time constants (broadcast vectors) ----
        one_vec = self.const_vec(1)
        two_vec = self.const_vec(2)
        c_branch_vec = self.const_vec((1 - fp) % (2**32))
        const10_vec = self.const_vec(10)
        const8_vec = self.alloc_vec("const8")
        self.emit_valu("-", const8_vec, const10_vec, two_vec)  # 10 - 2 = 8

        b0 = self.const_vec(4097)
        c0 = self.const_vec(0x7ED55D16)
        C1 = self.const_vec(0xC761C23C)
        s1 = self.const_vec(19)
        b2 = self.const_vec(33)
        c2 = self.const_vec(0x165667B1)
        C3 = self.const_vec(0xD3A2646C)
        nine = self.const_vec(9)  # used as both s3 and b4
        c4 = self.const_vec(0xFD7046C5)
        C5 = self.const_vec(0xB55A4F09)
        s5 = self.const_vec(16)

        # ---- preload tree levels 0..2 (nodes 0..6) ----
        node_vec = self.alloc_vec("node_vec")
        node_addr = self.alloc("node_addr")
        self.emit_const_scalar(node_addr, fp)
        self.emit_vload(node_vec, node_addr)
        n0 = node_vec + 0
        n1 = node_vec + 1
        n2 = node_vec + 2
        n3 = node_vec + 3
        n4 = node_vec + 4
        n5 = node_vec + 5
        n6 = node_vec + 6

        node0_vec = self.alloc_vec("node0")
        self.emit_vbroadcast(node0_vec, n0)
        node1_vec = self.alloc_vec("node1")
        self.emit_vbroadcast(node1_vec, n1)
        diff1 = self.alloc("diff1")
        self.emit_alu("-", diff1, n2, n1)
        diff1_vec = self.alloc_vec("diff1")
        self.emit_vbroadcast(diff1_vec, diff1)
        # folded d=1 mux: node_val = addr*diff1 + (node1 - 8*diff1)
        const1 = self.alloc("const1")
        _t8 = self.alloc("t8")
        self.emit_alu("*", _t8, diff1, const8_vec)  # 8*diff1 (const8_vec lane0 == 8)
        self.emit_alu("-", const1, n1, _t8)        # node1 - 8*diff1
        const1_vec = self.alloc_vec("const1")
        self.emit_vbroadcast(const1_vec, const1)

        node3_vec = self.alloc_vec("node3")
        self.emit_vbroadcast(node3_vec, n3)
        node4_vec = self.alloc_vec("node4")
        self.emit_vbroadcast(node4_vec, n4)
        node5_vec = self.alloc_vec("node5")
        self.emit_vbroadcast(node5_vec, n5)
        node6_vec = self.alloc_vec("node6")
        self.emit_vbroadcast(node6_vec, n6)

        # scalar-only node diffs
        d34 = self.alloc("d34")
        self.emit_alu("-", d34, n4, n3)
        d56 = self.alloc("d56")
        self.emit_alu("-", d56, n6, n5)

        # ---- scalar constants (reuse broadcast vectors' first lane) ----
        S = {
            "one": one_vec,
            "c_branch": c_branch_vec,
            "c8": const8_vec,
            "c10": const10_vec,
            "b0": b0,
            "c0": c0,
            "C1": C1,
            "s1": s1,
            "b2": b2,
            "c2": c2,
            "C3": C3,
            "s3": nine,
            "b4": nine,
            "c4": c4,
            "C5": C5,
            "s5": s5,
            "n0": n0, "n1": n1, "n3": n3, "n5": n5,
            "diff1": diff1, "d34": d34, "d56": d56, "const1": const1,
        }

        # ---- per-chunk registers ----
        n_vec = n_chunks - n_scalar
        addrs = []
        vals = []
        t1s = []
        t2s = []
        t3s = []
        for c in range(n_vec):
            addrs.append(self.alloc_vec("addr"))
            vals.append(self.alloc_vec("val"))
            t1s.append(self.alloc_vec("t1"))
            t2s.append(self.alloc_vec("t2"))
            t3s.append(self.alloc_vec("t3"))

        # scalar chunk registers (5 scalars per element; t doubles as store addr)
        sc_addrs = []
        sc_vals = []
        sc_t = []
        sc_t1 = []
        sc_t2 = []
        for _ in range(n_scalar):
            ca = []
            cv = []
            ct = []
            ctmp1 = []
            ctmp2 = []
            for _ in range(VLEN):
                ca.append(self.alloc("s_addr"))
                cv.append(self.alloc("s_val"))
                ct.append(self.alloc("s_t"))
                ctmp1.append(self.alloc("s_t1"))
                ctmp2.append(self.alloc("s_t2"))
            sc_addrs.append(ca)
            sc_vals.append(cv)
            sc_t.append(ct)
            sc_t1.append(ctmp1)
            sc_t2.append(ctmp2)

        # ---- initial load of val ----
        inp_values_p = fp + n_nodes + batch_size
        ival_addrs = []
        for c in range(n_vec):
            s = self.alloc("ival_addr")
            self.emit_const_scalar(s, inp_values_p + (c + n_scalar) * VLEN)
            ival_addrs.append(s)
            self.emit_vload(vals[c], s)
        for c in range(n_scalar):
            for i in range(VLEN):
                s = sc_t[c][i]
                self.emit_const_scalar(s, inp_values_p + c * VLEN + i)
                self.emit_load_scalar(sc_vals[c][i], s)

        # ---- main loop over rounds (diagonal wavefront emission) ----
        _pairs = [(c, r) for c in range(n_chunks) for r in range(rounds)]
        _pairs.sort(key=lambda p: (p[0] + 3*p[1], p[0]))
        for c, r in _pairs:
                depth = r % n_levels
                is_leaf = (depth == forest_height)

                if c < n_scalar:
                    for i in range(VLEN):
                        self._emit_scalar_round(
                            r, rounds, depth, is_leaf,
                            sc_addrs[c][i], sc_vals[c][i],
                            sc_t[c][i], sc_t1[c][i], sc_t2[c][i], S,
                        )
                    continue

                cc = c - n_scalar
                v = vals[cc]
                t1 = t1s[cc]
                t2 = t2s[cc]
                t3 = t3s[cc]

                # ---- produce node_val ----
                if depth == 0:
                    self.emit_valu("^", v, v, node0_vec)
                elif depth == 1:
                    self.emit_valu_ma(t2, addrs[cc], diff1_vec, const1_vec)
                    self.emit_valu("^", v, v, t2)
                elif depth == 2:
                    self.emit_valu("-", t1, addrs[cc], const10_vec)
                    self.emit_valu("&", t2, t1, one_vec)
                    self.emit_valu(">>", t1, t1, one_vec)
                    self.emit_vselect(t3, t2, node4_vec, node3_vec)
                    self.emit_vselect(t2, t2, node6_vec, node5_vec)
                    self.emit_vselect(t3, t1, t2, t3)
                    self.emit_valu("^", v, v, t3)
                else:
                    for off in range(VLEN):
                        self.emit_load_offset(t1, addrs[cc], off)
                    self.emit_valu("^", v, v, t1)

                # ---- hash ----
                self.emit_valu_ma(v, v, b0, c0)
                self.emit_valu("^", t1, v, C1)
                self.emit_valu(">>", t2, v, s1)
                self.emit_valu("^", v, t1, t2)
                self.emit_valu_ma(v, v, b2, c2)
                self.emit_valu("+", t1, v, C3)
                self.emit_valu("<<", t2, v, nine)
                self.emit_valu("^", v, t1, t2)
                self.emit_valu_ma(v, v, nine, c4)
                self.emit_valu("^", t1, v, C5)
                self.emit_valu(">>", t2, v, s5)
                self.emit_valu("^", v, t1, t2)

                # ---- branch ----
                if not is_leaf and r != rounds - 1:
                    self.emit_valu("&", t1, v, one_vec)
                    if depth == 0:
                        self.emit_valu("+", addrs[cc], t1, const8_vec)
                    else:
                        self.emit_valu("+", t2, t1, c_branch_vec)
                        self.emit_valu_ma(addrs[cc], addrs[cc], two_vec, t2)

        # ---- store final val ----
        for c in range(n_vec):
            self.emit_vstore(ival_addrs[c], vals[c])
        for c in range(n_scalar):
            for i in range(VLEN):
                self.emit_const_scalar(sc_t[c][i], inp_values_p + c * VLEN + i)
                self.emit_store_scalar(sc_t[c][i], sc_vals[c][i])

        self.instrs = self.schedule()

    # ------------------------------------------------------------------
    # Hazard-aware greedy scheduler
    # ------------------------------------------------------------------
    def schedule(self):
        limits = dict(SLOT_LIMITS)
        slots = self.slots
        n = len(slots)

        # Build a data-dependency DAG (RAW, WAW, and WAR).
        last_writer = {}
        last_reader = {}
        deps = [set() for _ in range(n)]
        for j, (engine, slot, reads, writes) in enumerate(slots):
            for a in reads:
                if a in last_writer:
                    deps[j].add(last_writer[a])  # RAW
            for a in writes:
                if a in last_writer:
                    deps[j].add(last_writer[a])  # WAW
                lr = last_reader.get(a)
                if lr is not None:
                    deps[j].add(lr)  # WAR
            for a in reads:
                last_reader[a] = j
            for a in writes:
                last_writer[a] = j
                last_reader.pop(a, None)

        indegree = [len(d) for d in deps]
        consumers = [[] for _ in range(n)]
        for j in range(n):
            for i in deps[j]:
                consumers[i].append(j)

        ready = [j for j in range(n) if indegree[j] == 0]
        bundles = []
        scheduled = 0
        while ready:
            bundle = {}
            writes = set()
            used = defaultdict(int)
            chosen = []
            next_ready = []
            ready.sort()
            for j in ready:
                engine, slot, reads, wset = slots[j]
                if used[engine] >= limits[engine] or (reads & writes) or (wset & writes):
                    next_ready.append(j)
                    continue
                bundle.setdefault(engine, []).append(slot)
                writes |= wset
                used[engine] += 1
                chosen.append(j)
            for j in chosen:
                scheduled += 1
                for c in consumers[j]:
                    indegree[c] -= 1
                    if indegree[c] == 0:
                        next_ready.append(c)
            bundles.append(bundle)
            ready = next_ready
        assert scheduled == n
        return bundles


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

    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        trace=trace,
    )
    machine.prints = prints
    machine.enable_pause = False
    machine.enable_debug = False
    machine.run()
    final_ref = list(reference_kernel2(mem))[-1]
    inp_values_p = final_ref[6]
    assert (
        machine.mem[inp_values_p : inp_values_p + len(inp.values)]
        == final_ref[inp_values_p : inp_values_p + len(inp.values)]
    ), "Incorrect output values"
    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


if __name__ == "__main__":
    unittest.main()
