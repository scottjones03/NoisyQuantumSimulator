# QPU Simulator

**Cross-Platform Quantum Processing Unit Architecture Simulator**

A multi-scale simulation framework for comparing quantum hardware architectures, including neutral atom arrays, trapped ions, and cavity QED systems.

---

## 🎯 What This Project Does

This simulator enables **hardware-aware quantum error correction (QEC) studies** by:

1. **Modeling real physics** (Rydberg gates, ion shuttling, cavity interactions) via offline micro-simulations
2. **Abstracting hardware** into a common primitive API (Move, Gate, Measure, Cool)
3. **Running QEC protocols** (surface codes, LDPC, color codes) on realistic noise models
4. **Comparing platforms** fairly using calibrated physical parameters

```
┌─────────────────────┐
│   Micro-Physics     │  QuTiP simulations → CPTP maps, timing, loss
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│    Primitives       │  Move(), TwoQubitGate(), Measure(), Cool()
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│   Architecture      │  Scheduler, QEC, Compiler, Stim integration
└─────────────────────┘
```

---

## 🚀 Quick Start

```bash
# Clone and install
git clone <repo-url>
cd NoisyQuantumSimulator
pip install -e .

# Run tests
pytest tests/
```

---

## 📦 Supported Hardware Platforms

| Platform | Status | Key Features |
|----------|--------|--------------|
| **Neutral Atoms** (Rydberg) | 🔨 In Progress | Blockade gates, AOD reconfiguration, optical tweezers |
| **QCCD Ions** | 📋 Planned | MS gates, shuttling, segmented traps |
| **Penning Ions** | 📋 Planned | 2D crystals, collective modes |
| **Cavity QED** | 📋 Planned | Photon-mediated gates |

---

## 🏗️ Architecture Overview

### Layer 0: Micro-Physics (Hardware-Specific)

Small-scale QuTiP/ODE simulations that run **offline** to extract:
- Gate error channels (CPTP maps)
- Gate durations
- Loss/leakage probabilities
- Heating rates

### Layer 1: Primitives (Hardware Abstraction)

Common API across all platforms:
```python
Move(qubit_id, start, end)        # Atom/ion transport
TwoQubitGate(q1, q2, "CZ")        # Entangling gate
Measure(qubit_id, basis="Z")      # State readout
Cool(qubit_id)                     # Reset motional state
```

### Layer 2: Architecture (System-Level)

- **Scheduler**: Respects parallelism, adjacency, timing
- **Topology**: Qubit layout and connectivity
- **QEC**: Surface codes, LDPC, color codes
- **Compiler**: Gate decomposition, routing, optimization

---

## 📂 Project Structure

```
src/qpu_simulator/
├── micro_physics/       # Layer 0: Physics simulations
│   ├── neutral_atoms/   # Rydberg, tweezers, AOD
│   ├── trapped_ions/    # QCCD, Penning, RF Paul
│   └── cavity_qed/      # Cavity-mediated gates
├── primitives/          # Layer 1: Hardware abstraction
├── architecture/        # Layer 2: Scheduling, QEC, compilation
├── hardware_configs/    # Parameter definitions
├── noise_models/        # Error channel definitions
└── utils/               # Common utilities

calibration_data/        # Pre-computed physics results
tests/                   # Test suite
examples/                # Example scripts
docs/                    # Documentation
```

---

## 📖 Documentation

- [**ARCHITECTURE.md**](docs/ARCHITECTURE.md): Detailed design documentation
- Module docstrings: Each file contains comprehensive documentation

---

## 🔬 Key Design Principle

> **"Level-0/1 tools inform the parameters. They do not execute the computation."**

Micro-physics simulations (QuTiP) run **once** to calibrate error models.  
The architectural simulator uses those cached parameters for **fast QEC simulation**.

This separation enables:
- Scalability to 1000+ qubit simulations
- Fair cross-platform comparisons
- Reproducible results

---

## 🗺️ Roadmap

- [x] **Phase 0**: Architecture design and module structure
- [ ] **Phase 1**: MVP with neutral atoms (Rydberg gates, surface code)
- [ ] **Phase 2**: Trapped ion support (QCCD)
- [ ] **Phase 3**: Advanced QEC (LDPC codes, compiler optimization)
- [ ] **Phase 4**: Validation against experimental data

---

## 📚 Dependencies

- **QuTiP**: Lindblad master equation solving
- **Stim**: Fast Clifford circuit simulation
- **NumPy/SciPy**: Numerical computation
- **NetworkX**: Graph-based topology management
- **PyMatching**: MWPM decoder for surface codes

---

## 🤝 Contributing

This project follows a modular design. See [ARCHITECTURE.md](docs/ARCHITECTURE.md) for implementation guidelines.

---

## 📄 License

[To be determined]

---

## 📬 Contact

[Project maintainer information]
