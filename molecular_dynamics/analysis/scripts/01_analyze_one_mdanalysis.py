#!/usr/bin/env python3
import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import MDAnalysis as mda
from MDAnalysis.analysis import align
from MDAnalysis.analysis.rms import RMSF
from MDAnalysis.analysis.rms import rmsd
from MDAnalysis.analysis.distances import distance_array


def radius_of_gyration(atomgroup):
    coords = atomgroup.positions
    center = coords.mean(axis=0)
    diff = coords - center
    value = (diff * diff).sum(axis=1).mean()
    return float(np.sqrt(value))


def simple_rmsd(atomgroup, ref_positions):
    diff = atomgroup.positions - ref_positions
    value = (diff * diff).sum(axis=1).mean()
    return float(np.sqrt(value))


def save_plot(df, xcol, ycol, ylabel, title, outfile):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(df[xcol], df[ycol], lw=1.4)
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


def save_rmsf_plot(rmsf_df, title, outfile):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(rmsf_df["resid"], rmsf_df["ca_rmsf_A"], lw=1.2)
    ax.set_xlabel("Residue ID")
    ax.set_ylabel("C-alpha RMSF (A)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


def get_last_fraction(df, sim_ns, fraction):
    cutoff = sim_ns * (1.0 - fraction)
    return df[df["time_ns"] >= cutoff].copy()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze OpenMM DCD trajectories for enzyme-ligand MD."
    )
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--topology", required=True)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--sim-ns", type=float, default=2.0)
    parser.add_argument("--ligand-resname", default="LIG")
    parser.add_argument("--contact-cutoff", type=float, default=4.0)
    parser.add_argument("--summary-fraction", type=float, default=0.20)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    universe = mda.Universe(args.topology, args.trajectory)
    reference = mda.Universe(args.topology, args.trajectory)
    reference.trajectory[0]

    protein = universe.select_atoms("protein")
    ca_atoms = universe.select_atoms("protein and name CA")
    ligand = universe.select_atoms(f"resname {args.ligand_resname}")

    if protein.n_atoms == 0:
        raise ValueError("Protein selection is empty.")
    if ca_atoms.n_atoms == 0:
        raise ValueError("C-alpha selection is empty.")
    if ligand.n_atoms == 0:
        msg = f"Ligand selection is empty: resname {args.ligand_resname}"
        raise ValueError(msg)

    n_frames = len(universe.trajectory)
    time_ns = np.linspace(0.0, args.sim_ns, n_frames)

    align.AlignTraj(
        universe,
        reference,
        select="protein and name CA",
        in_memory=True,
    ).run()

    universe.trajectory[0]
    ref_ca_positions = ca_atoms.positions.copy()
    ref_ligand_positions = ligand.positions.copy()

    rows = []

    for frame_index, timestep in enumerate(universe.trajectory):
        ca_rmsd = float(rmsd(ca_atoms.positions, ref_ca_positions))
        ligand_rmsd = simple_rmsd(ligand, ref_ligand_positions)
        rg_value = radius_of_gyration(protein)

        ligand_center = ligand.center_of_geometry()
        protein_center = protein.center_of_geometry()
        com_distance = float(np.linalg.norm(ligand_center - protein_center))

        dist_matrix = distance_array(
            ligand.positions,
            protein.positions,
            box=timestep.dimensions,
        )

        min_distance = float(dist_matrix.min())
        contacts = int((dist_matrix < args.contact_cutoff).sum())

        rows.append(
            {
                "frame": frame_index,
                "time_ns": time_ns[frame_index],
                "ca_rmsd_A": ca_rmsd,
                "ligand_rmsd_A": ligand_rmsd,
                "protein_rg_A": rg_value,
                "protein_ligand_com_distance_A": com_distance,
                "protein_ligand_min_distance_A": min_distance,
                "protein_ligand_contacts_4A": contacts,
            }
        )

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(outdir / "timeseries_metrics.csv", index=False)

    rmsf_result = RMSF(ca_atoms).run()
    rmsf_df = pd.DataFrame(
        {
            "resid": ca_atoms.resids,
            "resname": ca_atoms.resnames,
            "ca_rmsf_A": rmsf_result.results.rmsf,
        }
    )
    rmsf_df.to_csv(outdir / "ca_rmsf.csv", index=False)

    last_df = get_last_fraction(
        metrics_df,
        args.sim_ns,
        args.summary_fraction,
    )

    summary = {
        "candidate": args.candidate,
        "topology": args.topology,
        "trajectory": args.trajectory,
        "ligand_resname": args.ligand_resname,
        "n_atoms": int(universe.atoms.n_atoms),
        "n_frames": int(n_frames),
        "sim_ns": float(args.sim_ns),
        "protein_atoms": int(protein.n_atoms),
        "ca_atoms": int(ca_atoms.n_atoms),
        "ligand_atoms": int(ligand.n_atoms),
        "ca_rmsd_final_A": metrics_df["ca_rmsd_A"].iloc[-1],
        "ca_rmsd_last20pct_mean_A": last_df["ca_rmsd_A"].mean(),
        "ca_rmsd_last20pct_std_A": last_df["ca_rmsd_A"].std(),
        "ligand_rmsd_final_A": metrics_df["ligand_rmsd_A"].iloc[-1],
        "ligand_rmsd_last20pct_mean_A": last_df["ligand_rmsd_A"].mean(),
        "ligand_rmsd_last20pct_std_A": last_df["ligand_rmsd_A"].std(),
        "protein_rg_last20pct_mean_A": last_df["protein_rg_A"].mean(),
        "protein_rg_last20pct_std_A": last_df["protein_rg_A"].std(),
        "pl_com_distance_last20pct_mean_A": (
            last_df["protein_ligand_com_distance_A"].mean()
        ),
        "pl_min_distance_last20pct_mean_A": (
            last_df["protein_ligand_min_distance_A"].mean()
        ),
        "pl_contacts_4A_last20pct_mean": (
            last_df["protein_ligand_contacts_4A"].mean()
        ),
        "ca_rmsf_mean_A": rmsf_df["ca_rmsf_A"].mean(),
        "ca_rmsf_max_A": rmsf_df["ca_rmsf_A"].max(),
    }

    pd.DataFrame([summary]).to_csv(outdir / "summary.csv", index=False)

    title = args.candidate

    save_plot(
        metrics_df,
        "time_ns",
        "ca_rmsd_A",
        "Protein C-alpha RMSD (A)",
        title,
        outdir / "ca_rmsd.png",
    )

    save_plot(
        metrics_df,
        "time_ns",
        "ligand_rmsd_A",
        "Ligand RMSD after protein alignment (A)",
        title,
        outdir / "ligand_rmsd.png",
    )

    save_plot(
        metrics_df,
        "time_ns",
        "protein_rg_A",
        "Protein radius of gyration (A)",
        title,
        outdir / "protein_rg.png",
    )

    save_plot(
        metrics_df,
        "time_ns",
        "protein_ligand_com_distance_A",
        "Protein-ligand COM distance (A)",
        title,
        outdir / "protein_ligand_com_distance.png",
    )

    save_plot(
        metrics_df,
        "time_ns",
        "protein_ligand_min_distance_A",
        "Protein-ligand minimum distance (A)",
        title,
        outdir / "protein_ligand_min_distance.png",
    )

    save_plot(
        metrics_df,
        "time_ns",
        "protein_ligand_contacts_4A",
        "Protein-ligand contacts within 4 A",
        title,
        outdir / "protein_ligand_contacts_4A.png",
    )

    save_rmsf_plot(
        rmsf_df,
        title,
        outdir / "ca_rmsf.png",
    )

    universe.trajectory[-1]
    final_selection = universe.select_atoms(
        f"protein or resname {args.ligand_resname}"
    )
    final_selection.write(str(outdir / "final_protein_ligand.pdb"))

    print("DONE", args.candidate)
    print("Output:", outdir)
    print(pd.DataFrame([summary]).T)


if __name__ == "__main__":
    main()