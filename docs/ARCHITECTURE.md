# QPU Simulator Architecture

## Overview

This document describes the architecture of the Cross-Platform Quantum Processing Unit (QPU) Simulator, a multi-scale simulation framework for comparing quantum hardware architectures including neutral atom arrays, trapped ions, and cavity QED systems.

## Design Philosophy

### Key Principles

1. **Multi-scale Simulation**: Physics simulations (micro-layer) are separated from architectural simulation (macro-layer)
2. **Hardware Abstraction**: A common primitive API allows the same circuits to run on different hardware models
3. **Offline Calibration**: Micro-physics simulations run offline to generate parameters, not during circuit execution
4. **Cross-Platform Comparison**: Same QEC protocols can be benchmarked across different hardware

### What This Simulator Is

- An **architectural simulator** for studying QEC, compilation, and system-level tradeoffs
- A framework for **comparing hardware platforms** using calibrated physical parameters
- A tool for understanding **how hardware constraints affect logical error rates**

### What This Simulator Is NOT

- A full physics simulator that solves Maxwell + Schrödinger equations
- A replacement for QuTiP or other micro-physics tools (it uses them)
- A real-time control system simulator

---

## Three-Layer Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                    LAYER 0: MICRO-PHYSICS                     │
│                    (Hardware-Specific)                        │
│                                                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐           │
│  │   Neutral   │  │   Trapped   │  │   Cavity    │           │
│  │    Atoms    │  │    Ions     │  │    QED      │           │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘           │
│         │                │                │                   │
│    QuTiP/ODE        QuTiP/ODE        QuTiP                   │
│    Simulations      Simulations      Simulations              │
└─────────┼────────────────┼────────────────┼───────────────────┘
          │                │                │
          ▼                ▼                ▼
     ┌─────────────────────────────────────────┐
     │        CALIBRATED PARAMETERS            │
     │  (CPTP maps, timing, loss rates, etc.)  │
     └─────────────────────┬───────────────────┘
                           │
                           ▼
┌───────────────────────────────────────────────────────────────┐
│                    LAYER 1: PRIMITIVES                        │
│                  (Hardware Abstraction)                       │
│                                                               │
│   ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐     │
│   │  Move  │ │ Single │ │  Two   │ │Measure │ │  Cool  │     │
│   │        │ │ Qubit  │ │ Qubit  │ │        │ │        │     │
│   │        │ │  Gate  │ │  Gate  │ │        │ │        │     │
│   └────────┘ └────────┘ └────────┘ └────────┘ └────────┘     │
│                                                               │
│              SAME API ACROSS ALL PLATFORMS                    │
└───────────────────────────┬───────────────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────────┐
│                   LAYER 2: ARCHITECTURE                       │
│                    (System-Level)                             │
│                                                               │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐  │
│  │ Scheduler │  │  Topology │  │    QEC    │  │  Compiler │  │
│  └───────────┘  └───────────┘  └───────────┘  └───────────┘  │
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │                    SIMULATOR ENGINE                      │  │
│  │              (Stim + NetworkX + Python)                  │  │
│  └─────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────┘
```

---

## Layer 0: Micro-Physics

### Purpose

Simulate low-level physics for 1-3 qubits to extract calibrated parameters.

### Key Rule

> **Level-0/1 tools inform the parameters. They do not execute the computation.**

### Inputs

| Category | Examples |
|----------|----------|
| System State | Atom/ion state, positions, motional excitation |
| Hardware Parameters | Rabi frequencies, trap depths, decay rates, distances |
| Architectural | Gate type, target qubits, timing constraints |

### Outputs

| Output | Description |
|--------|-------------|
| CPTP Map | Error channel for the operation |
| Duration | Gate/operation time |
| Fidelity | 1 - error probability |
| Loss Probability | Chance of losing atom/ion |
| Heating | Motional energy added |
| Leakage | Population outside computational basis |

### Hardware-Specific Models

#### Neutral Atoms (Optical Tweezers)
- **rydberg_gates.py**: QuTiP simulation of Rydberg CZ gates
- **single_qubit_gates.py**: Raman/microwave rotations
- **optical_tweezers.py**: Harmonic trap model
- **aod_slm_motion.py**: Atom reconfiguration dynamics
- **measurement.py**: Fluorescence detection
- **cooling.py**: Optical molasses, sideband cooling

#### Trapped Ions
- **QCCD**: Segmented traps with shuttling
- **Penning**: 2D crystals with collective modes
- **RF Paul**: Standard RF traps with micromotion

#### Cavity QED
- **cavity_gates.py**: Photon-mediated entanglement
- **atom_cavity_coupling.py**: Jaynes-Cummings dynamics

### Tools Used

| Tool | Purpose |
|------|---------|
| QuTiP | Lindblad master equation for gates |
| SciPy ODE | Classical motion in traps |
| NumPy | Waveform generation, linear algebra |
| Monte Carlo | Stochastic loss and measurement |

---

## Layer 1: Primitives

### Purpose

Provide a **unified API** across all hardware platforms.

### Core Primitives

```
Move(qubit_id, start, end, duration)
    → MoveResult(duration, heating, loss_prob, trajectory)

SingleQubitGate(qubit_id, gate_type, angle, axis)
    → GateResult(duration, fidelity, error_map, leakage)

TwoQubitGate(q1, q2, gate_type, distance)
    → GateResult(duration, fidelity, error_map, crosstalk)

Measure(qubit_id, basis)
    → MeasureResult(fidelity, confusion_matrix, duration, loss_prob)

Cool(qubit_id, target_temp, method)
    → CoolResult(duration, final_temp, success_prob)

Idle(qubit_id, duration)
    → IdleResult(error_map, loss_prob, heating)
```

### Platform Mapping

| Primitive | Neutral Atoms | Trapped Ions | Cavity QED |
|-----------|---------------|--------------|------------|
| TwoQubitGate | Rydberg CZ | Mølmer-Sørensen | Cavity-mediated |
| Move | AOD/SLM transport | Shuttling | Atom transport |
| Measure | Fluorescence | Fluorescence | Cavity transmission |
| Cool | Optical molasses | Sideband cooling | Cavity cooling |

---

## Layer 2: Architecture

### Scheduler

Schedules operations respecting hardware constraints:
- Physical adjacency for two-qubit gates
- Blockade radius conflicts (atoms)
- Zone occupancy limits (ions)
- Parallelism exploitation

### Topology

Manages physical qubit layout:
- 2D grids, arbitrary graphs, zone-based
- Distance-based interaction strengths
- Reconfigurable layouts (neutral atoms)

### QEC Module

Quantum error correction implementations:
- **Surface Code**: Rotated, lattice surgery
- **Color Codes**: Transversal Cliffords
- **LDPC Codes**: High-rate codes

### Decoders

- Minimum Weight Perfect Matching (PyMatching)
- Union-Find
- Belief Propagation (for LDPC)

### Compiler

Circuit compilation pipeline:
1. Gate decomposition to native gates
2. Qubit mapping (logical → physical)
3. Routing (SWAP/MOVE insertion)
4. Scheduling
5. Optimization

### Simulator Engine

Main simulation loop:
1. Load scheduled circuit
2. Apply operations with error injection from primitives
3. Track syndrome measurements
4. Decode and compute logical outcome
5. Report metrics

---

## Data Flow

```
┌──────────────────┐
│  QuTiP/Physics   │  ← Run OFFLINE, once per parameter regime
│    Simulations   │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   Calibration    │  ← JSON files with error rates, timings
│      Data        │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   Primitives     │  ← Load calibration, expose API
│    Library       │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Architecture    │  ← Schedule, apply errors, decode
│    Simulator     │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   Performance    │
│    Metrics       │
│  (Logical error  │
│   rate, timing,  │
│    resources)    │
└──────────────────┘
```

---

## Directory Structure

```
src/qpu_simulator/
├── __init__.py
├── micro_physics/           # Layer 0
│   ├── base.py
│   ├── neutral_atoms/
│   │   ├── rydberg_gates.py
│   │   ├── single_qubit_gates.py
│   │   ├── optical_tweezers.py
│   │   ├── aod_slm_motion.py
│   │   ├── measurement.py
│   │   └── cooling.py
│   ├── trapped_ions/
│   │   ├── qccd/
│   │   ├── penning/
│   │   └── rf_paul/
│   └── cavity_qed/
├── primitives/              # Layer 1
│   ├── base.py
│   ├── move.py
│   ├── gates.py
│   ├── measurement.py
│   ├── cooling.py
│   └── idle.py
├── architecture/            # Layer 2
│   ├── scheduler.py
│   ├── topology.py
│   ├── qec/
│   │   ├── surface_code.py
│   │   ├── color_code.py
│   │   ├── ldpc_codes.py
│   │   └── decoders.py
│   ├── compiler/
│   │   ├── decomposition.py
│   │   └── routing.py
│   └── simulator.py
├── hardware_configs/        # Parameter definitions
├── noise_models/            # Error channel definitions
└── utils/                   # Common utilities

calibration_data/            # Pre-computed physics results
tests/                       # Test suite
examples/                    # Example notebooks/scripts
```

---

## Development Roadmap

### Phase 0: Foundation
- [x] Define module structure
- [x] Document architecture
- [ ] Set up build system (pyproject.toml)
- [ ] Establish testing framework

### Phase 1: MVP (Neutral Atoms)
- [ ] Implement QuTiP-based Rydberg gate simulation
- [ ] Implement harmonic trap + AOD motion model
- [ ] Create neutral atom primitives
- [ ] Basic scheduler and topology
- [ ] Surface code + MWPM decoder
- [ ] End-to-end Clifford QEC simulation

### Phase 2: Ion Traps
- [ ] QCCD shuttling model
- [ ] MS gate simulation
- [ ] Ion trap primitives
- [ ] Cross-platform comparison (atoms vs ions)

### Phase 3: Advanced Features
- [ ] High-rate LDPC codes
- [ ] Non-Clifford simulation (T gates)
- [ ] Compiler optimizations
- [ ] Cavity QED models

### Phase 4: Validation & Publication
- [ ] Benchmark against experimental data
- [ ] Cross-platform QEC comparisons
- [ ] Performance scaling studies

---

## Key Design Decisions

### Why Separate Micro-Physics from Architecture?

1. **Scalability**: Can't simulate Schrödinger equation for 1000+ qubits
2. **Reproducibility**: Cached calibration ensures consistent results
3. **Speed**: Architecture simulator runs fast with pre-computed errors
4. **Clarity**: Clean separation of concerns

### Why a Common Primitive API?

1. **Fair Comparison**: Same circuit, different hardware
2. **Code Reuse**: QEC and compiler code shared across platforms
3. **Extensibility**: Easy to add new hardware types

### Why Stim for Clifford Simulation?

1. **Performance**: Handles millions of operations efficiently
2. **QEC Support**: Built-in detector error models
3. **Maturity**: Well-tested, widely used

---

## References

### Tools and Libraries
- [QuTiP](https://qutip.org/): Quantum Toolbox in Python
- [Stim](https://github.com/quantumlib/Stim): Fast Clifford simulator
- [PyMatching](https://github.com/oscarhiggott/PyMatching): MWPM decoder
- [NetworkX](https://networkx.org/): Graph algorithms

### Key Papers
- Bluvstein et al., Nature 2022: Rydberg atom QEC
- Quantinuum QCCD architecture papers
- Surface code threshold calculations

---

## Contributing

See individual module docstrings for implementation details and TODOs.
