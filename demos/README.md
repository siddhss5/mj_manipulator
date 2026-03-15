# Demos

Integration demos using real MuJoCo robot models (UR5e, Franka Panda).

Unlike `tests/` (which are automated, mock-based, CI-friendly), these are
standalone scripts that load real models and show the framework working
end-to-end.

## Running

```bash
cd mj_manipulator
uv run python demos/<script>.py
```

## Available Demos

| Script | What it shows |
|---|---|
| `ik_solver.py` | EAIK analytical IK: kinematic extraction from MuJoCo, multi-config IK with solution analysis, FK round-trip verification |
| `arm_planning.py` | Motion planning with CBiRRT: plan to configuration, plan to pose (via TSRs), trajectory retiming with TOPP-RA |
| `collision_check.py` | Collision checking: simple mode, grasp-aware mode, batch configuration validation |
| `cartesian_control.py` | Cartesian velocity control: Jacobian analysis, QP-based twist-to-joint-velocity, multi-step trajectory following |
