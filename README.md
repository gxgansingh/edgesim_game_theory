# Priority-Aware Hierarchical Mean-Field–Stackelberg Edge Computing Simulator

## Version 0.2.0

This version extends the simulation foundation with the current baseline utility-driven probabilistic allocation policy.

### Implemented

- IoT task generation
- Edge ingress gateway
- Heterogeneous edge servers
- Heterogeneous edge nodes
- Resource-aware feasibility filtering
- Current node utility: `U_i = 100 - load_i`
- Utility-normalized selection probabilities
- Weighted random node selection
- Immediate task allocation and release for deterministic foundation experiments
- Baseline metric collection
- Automated tests

### Research-layer status

The research specification defines a priority-aware multi-population Mean-Field Game at the edge-node layer and a cooperative multi-leader Stackelberg Game at the edge-server layer. Module-2 additionally introduces Weighted Nash Bargaining. Those layers are not yet claimed as implemented in this version.

The exact HJB-FPK equations, state dynamics, control set, and research utility equation from the completed mathematical formulation must be mapped into dedicated solver modules before final research experiments are reported.

## Run

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the baseline simulation:

```bash
python -m src.edge_game.main
```

Run tests:

```bash
pytest
```
