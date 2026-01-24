# 架构图（中文）

## 核心与引擎
```mermaid
flowchart TB
  subgraph Core[核心]
    SCR[Scratch 内存]\\n大小=SCRATCH_SIZE
    PC[程序计数器]
    ST[状态]
  end

  subgraph Engines[每核引擎]
    ALU[alu × SLOT_LIMITS.alu]
    VALU[valu × SLOT_LIMITS.valu]
    LD[load × SLOT_LIMITS.load]
    STOR[store × SLOT_LIMITS.store]
    FLOW[flow × SLOT_LIMITS.flow]
  end

  Program[程序: list[指令包]]
  Mem[内存镜像]

  Program -->|每周期一个指令包| Engines
  Engines -->|读写| SCR
  Engines -->|读写| Mem
  Core --> Engines
```

## 内核控制流（基线）
```mermaid
flowchart LR
  S[开始] --> P0[加载常量与参数]
  P0 --> L0[外层: 轮次]
  L0 --> L1[内层: 批元素]
  L1 --> G1[取 idx]
  G1 --> G2[取 val]
  G2 --> G3[取 node_val]
  G3 --> H[哈希混合 (HASH_STAGES)]
  H --> N1[计算下一索引]
  N1 --> N2[越界回绕]
  N2 --> W1[写回 idx]
  W1 --> W2[写回 val]
  W2 --> L1
  L1 -->|结束| L0
  L0 --> E[结束]
```

