#!/usr/bin/env python3
"""OpenMM CUDA production MD from GROMACS solvated coordinates."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def run_openmm_md(workdir: Path, outdir: Path, gpu_id: int = 0, prod_ns: float = 2.0) -> dict:
    import os

    import openmm as mm
    import openmm.app as app
    import openmm.unit as unit

    gro = workdir / "solv_ions.gro"
    top = workdir / "topol.top"
    if not gro.exists() or not top.exists():
        raise FileNotFoundError(f"Missing solv_ions.gro/topol.top in {workdir}")

    ff_link = workdir / "amber99sb-ildn.ff"
    ff_src = Path(os.environ.get("CONDA_PREFIX", "")) / "share/gromacs/top/amber99sb-ildn.ff"
    if not ff_link.exists() and ff_src.exists():
        ff_link.symlink_to(ff_src)

    outdir.mkdir(parents=True, exist_ok=True)
    gro_p = app.GromacsGroFile(str(gro))
    box = gro_p.getPeriodicBoxVectors()
    top_p = app.GromacsTopFile(str(top), periodicBoxVectors=box, includeDir=str(workdir))

    platform = mm.Platform.getPlatformByName("CUDA")
    props = {"CudaDeviceIndex": str(gpu_id), "Precision": "mixed"}

    def make_sim(barostat: bool = False) -> app.Simulation:
        system = top_p.createSystem(
            nonbondedMethod=app.PME,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=app.HBonds,
            ewaldErrorTolerance=0.0005,
        )
        if barostat:
            system.addForce(mm.MonteCarloBarostat(1.0 * unit.bar, 300 * unit.kelvin, 25))
        integrator = mm.LangevinMiddleIntegrator(
            300 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picoseconds
        )
        return app.Simulation(top_p.topology, system, integrator, platform, props)

    def log(msg: str) -> None:
        print(msg, flush=True)

    log(f"Atoms: {top_p.topology.getNumAtoms()}, GPU: {gpu_id}, prod_ns: {prod_ns}")
    sim = make_sim(barostat=False)
    sim.context.setPositions(gro_p.positions)
    sim.context.setPeriodicBoxVectors(*box)

    em_pdb = workdir / "em.gro"
    if em_pdb.exists():
        em_gro = app.GromacsGroFile(str(em_pdb))
        sim.context.setPositions(em_gro.positions)

    log("Energy minimization...")
    sim.minimizeEnergy(maxIterations=2000)

    log("NVT 100 ps...")
    sim.step(50000)

    log("NPT 100 ps...")
    state = sim.context.getState(getPositions=True, getVelocities=True, enforcePeriodicBox=True)
    sim_npt = make_sim(barostat=True)
    sim_npt.context.setState(state)
    sim_npt.step(50000)

    steps = int(prod_ns * 500000)
    traj = outdir / "prod.dcd"
    logf = outdir / "prod.log"
    sim_npt.reporters.append(app.DCDReporter(str(traj), 5000))
    sim_npt.reporters.append(
        app.StateDataReporter(str(logf), 5000, step=True, temperature=True, potentialEnergy=True, speed=True)
    )
    log(f"Production {prod_ns} ns ({steps} steps)...")
    sim_npt.step(steps)

    pdb_out = outdir / "final.pdb"
    with open(pdb_out, "w", encoding="utf-8") as fh:
        st = sim_npt.context.getState(getPositions=True, enforcePeriodicBox=True)
        app.PDBFile.writeFile(sim_npt.topology, st.getPositions(), fh)

    sim_npt.saveCheckpoint(str(outdir / "prod.chk"))

    result = {
        "platform": "OpenMM-CUDA",
        "gpu_id": gpu_id,
        "production_ns": prod_ns,
        "n_atoms": top_p.topology.getNumAtoms(),
        "trajectory": str(traj),
        "final_pdb": str(pdb_out),
        "log": str(logf),
        "status": "done",
    }
    (outdir / "md_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    workdir = Path(sys.argv[1])
    outdir = Path(sys.argv[2]) if len(sys.argv) > 2 else workdir.parent / "outputs"
    gpu_id = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    prod_ns = float(sys.argv[4]) if len(sys.argv) > 4 else 2.0
    result = run_openmm_md(workdir, outdir, gpu_id=gpu_id, prod_ns=prod_ns)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
