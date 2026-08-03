#!/usr/bin/env python3
"""GROMACS system preparation for one enzyme-substrate complex."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import gemmi

from config import MDP_DIR


def run(cmd: list[str], cwd: Path, env: dict | None = None, input_text: str | None = None) -> None:
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        input=input_text,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )


def split_structure(cif_path: Path, protein_pdb: Path, ligand_pdb: Path) -> None:
    st = gemmi.read_structure(str(cif_path))
    model = st[0]

    prot = st.clone()
    prot[0].remove_chain("B")
    prot.write_pdb(str(protein_pdb))

    lig = gemmi.read_structure(str(cif_path))
    lig[0].remove_chain("A")
    lig.write_pdb(str(ligand_pdb))


def read_gro_atoms(gro_path: Path) -> tuple[str, list[str], str]:
    lines = gro_path.read_text(encoding="utf-8").splitlines()
    title = lines[0]
    natoms = int(lines[1].strip())
    atoms = lines[2 : 2 + natoms]
    box = lines[2 + natoms]
    return title, atoms, box


def write_gro(gro_path: Path, title: str, atoms: list[str], box: str) -> None:
    with open(gro_path, "w", encoding="utf-8") as fh:
        fh.write(f"{title}\n")
        fh.write(f"{len(atoms)}\n")
        fh.write("\n".join(atoms) + "\n")
        fh.write(box + "\n")


def merge_gro(protein_gro: Path, ligand_gro: Path, out_gro: Path) -> None:
    title, patoms, pbox = read_gro_atoms(protein_gro)
    _, latoms, _ = read_gro_atoms(ligand_gro)
    write_gro(out_gro, title, patoms + latoms, pbox)


def merge_topology(protein_top: Path, ligand_itp: Path, out_top: Path) -> None:
    lines = protein_top.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    inserted = False
    for line in lines:
        if not inserted and line.strip().startswith("[ moleculetype ]"):
            out.append(f'#include "{ligand_itp.name}"')
            out.append("")
            inserted = True
        out.append(line)
        if line.strip() == "Protein_chain_A     1":
            out.append("LIG                 1")
    out_top.write_text("\n".join(out) + "\n", encoding="utf-8")


def link_forcefield(workdir: Path) -> None:
    ff_link = workdir / "amber99sb-ildn.ff"
    if ff_link.exists():
        return
    prefix = os.environ.get("CONDA_PREFIX", "")
    ff_src = Path(prefix) / "share/gromacs/top/amber99sb-ildn.ff"
    if ff_src.exists():
        ff_link.symlink_to(ff_src)


def setup_job(
    job_dir: Path,
    cif_path: Path,
    smiles: str = "",
    gmx: str = "gmx",
) -> Path:
    work = job_dir / "work"
    work.mkdir(parents=True, exist_ok=True)
    done_flag = work / "setup.done"
    if done_flag.exists():
        return work

    for mdp in MDP_DIR.glob("*.mdp"):
        shutil.copy2(mdp, work / mdp.name)

    protein_pdb = work / "protein.pdb"
    ligand_pdb = work / "ligand.pdb"
    split_structure(cif_path, protein_pdb, ligand_pdb)

    run(
        [gmx, "pdb2gmx", "-f", "protein.pdb", "-o", "protein.gro", "-p", "topol.top",
         "-water", "tip3p", "-ignh", "-ff", "amber99sb-ildn", "-merge", "all"],
        work,
        input_text="1\n",
    )

    acpype_dir = work / "LIG.acpype"
    if not (acpype_dir / "LIG_GMX.itp").exists():
        run(["acpype", "-i", "ligand.pdb", "-b", "LIG", "-c", "gas", "-a", "gaff2"], work)

    shutil.copy2(acpype_dir / "LIG_GMX.itp", work / "LIG_GMX.itp")
    merge_gro(work / "protein.gro", acpype_dir / "LIG_GMX.gro", work / "complex.gro")
    merge_topology(work / "topol.top", work / "LIG_GMX.itp", work / "topol.top")

    link_forcefield(work)

    run([gmx, "editconf", "-f", "complex.gro", "-o", "box.gro", "-c", "-d", "1.2", "-bt", "dodecahedron"], work)
    run([gmx, "solvate", "-cp", "box.gro", "-cs", "spc216.gro", "-o", "solv.gro", "-p", "topol.top"], work)
    run([gmx, "grompp", "-f", "ions.mdp", "-c", "solv.gro", "-p", "topol.top", "-o", "ions.tpr", "-maxwarn", "10"], work)
    run(
        [gmx, "genion", "-s", "ions.tpr", "-o", "solv_ions.gro", "-p", "topol.top",
         "-pname", "NA", "-nname", "CL", "-neutral", "-conc", "0.15"],
        work,
        input_text="SOL\n",
    )

    run([gmx, "grompp", "-f", "em.mdp", "-c", "solv_ions.gro", "-p", "topol.top", "-o", "em.tpr", "-maxwarn", "10"], work)
    run([gmx, "mdrun", "-deffnm", "em", "-v", "-ntomp", "8"], work)

    done_flag.write_text("ok\n", encoding="utf-8")
    return work


def main() -> None:
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <job_dir> <cif_path> [smiles]", file=sys.stderr)
        sys.exit(1)
    job_dir = Path(sys.argv[1])
    cif_path = Path(sys.argv[2])
    smiles = sys.argv[3] if len(sys.argv) > 3 else ""
    work = setup_job(job_dir, cif_path, smiles=smiles)
    print(f"Setup complete: {work}")


if __name__ == "__main__":
    main()
