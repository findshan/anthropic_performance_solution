# Diagrams

## Machine Components

```mermaid
flowchart TB
  subgraph Core[Core]
    direction TB
    SCR[Scratch RAM]\nsize=SCRATCH_SIZE
    PC[Program Counter]
    ST[State]
  end

  subgraph Engines[Engines per Core]
    ALU[alu x SLOT_LIMITS.alu]
    VALU[valu x SLOT_LIMITS.valu]
    LD[load x SLOT_LIMITS.load]
    STOR[store x SLOT_LIMITS.store]
    FLOW[flow x SLOT_LIMITS.flow]
  end

  Program[Program: list[Instruction]]
  Mem[Memory Image]

  Program -->|bundle per cycle| Engines
  Engines -->|read/write| SCR
  Engines -->|read/write| Mem
  Core --> Engines
```

## Kernel Control Flow (Baseline)

```mermaid
flowchart LR
  S[Start] --> P0[Load constants & params]
  P0 --> L0[For each round]
  L0 --> L1[For each batch item]
  L1 --> G1[Load idx]
  G1 --> G2[Load val]
  G2 --> G3[Load node_val]
  G3 --> H[Hash mix over HASH_STAGES]
  H --> N1[Compute next idx]
  N1 --> N2[Bounds wrap]
  N2 --> W1[Store idx]
  W1 --> W2[Store val]
  W2 --> L1
  L1 -->|done| L0
  L0 --> E[End]
```

