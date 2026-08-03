#!/usr/bin/env python3
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import MDAnalysis as mda
from MDAnalysis.analysis import align
from MDAnalysis.lib.distances import capped_distance


def save_plot(df, ycol, ylabel, outfile, title):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(df["time_ns"], df[ycol], lw=1.4)
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


def get_pairs(pos_a, pos_b, cutoff, box):
    if len(pos_a) == 0 or len(pos_b) == 0:
        return np.empty((0, 2), dtype=int)

    pairs = capped_distance(
        pos_a,
        pos_b,
        max_cutoff=cutoff,
        box=box,
        return_distances=False,
    )

    if pairs is None:
        return np.empty((0, 2), dtype=int)

    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--topology", required=True)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--sim-ns", type=float, default=2.0)
    parser.add_argument("--ligand-resname", default="LIG")
    parser.add_argument("--contact-cutoff", type=float, default=4.0)
    parser.add_argument("--polar-cutoff", type=float, default=3.5)
    parser.add_argument("--water-cutoff", type=float, default=4.0)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    u = mda.Universe(args.topology, args.trajectory)
    ref = mda.Universe(args.topology, args.trajectory)
    ref.trajectory[0]

    align.AlignTraj(
        u,
        ref,
        select="protein and name CA",
        in_memory=True,
    ).run()

    protein = u.select_atoms("protein")
    ligand = u.select_atoms(f"resname {args.ligand_resname}")

    polar_protein = u.select_atoms(
        "protein and (name N* or name O* or name S*)"
    )
    polar_ligand = u.select_atoms(
        f"resname {args.ligand_resname} "
        "and (name N* or name O* or name S*)"
    )

    water_oxygen = u.select_atoms(
        "(resname SOL or resname WAT or resname HOH "
        "or resname TIP3P) and (name O* or name OW or name OH2)"
    )

    if protein.n_atoms == 0:
        raise ValueError("No protein atoms selected.")
    if ligand.n_atoms == 0:
        raise ValueError("No ligand atoms selected.")

    n_frames = len(u.trajectory)
    times = np.linspace(0.0, args.sim_ns, n_frames)

    rows = []
    residue_counter = Counter()

    for i, ts in enumerate(u.trajectory):
        box = ts.dimensions

        lp_pairs = get_pairs(
            ligand.positions,
            protein.positions,
            args.contact_cutoff,
            box,
        )

        protein_contacts = int(len(lp_pairs))

        if len(lp_pairs) > 0:
            protein_atom_ids = np.unique(lp_pairs[:, 1])
            close_atoms = protein[protein_atom_ids]
            close_residues = close_atoms.residues

            for res in close_residues:
                key = f"{res.resname}{res.resid}"
                residue_counter[key] += 1

            pocket_residue_count = len(close_residues)
        else:
            pocket_residue_count = 0

        polar_pairs = get_pairs(
            polar_ligand.positions,
            polar_protein.positions,
            args.polar_cutoff,
            box,
        )

        water_pairs = get_pairs(
            ligand.positions,
            water_oxygen.positions,
            args.water_cutoff,
            box,
        )

        rows.append(
            {
                "frame": i,
                "time_ns": times[i],
                "polar_contacts_3p5A": int(len(polar_pairs)),
                "protein_contacts_4A": protein_contacts,
                "water_oxygen_contacts_4A": int(len(water_pairs)),
                "pocket_residue_count_4A": pocket_residue_count,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "extra_timeseries.csv", index=False)

    occ_rows = []
    for residue, count in residue_counter.items():
        occ_rows.append(
            {
                "residue": residue,
                "frames_present": count,
                "occupancy": count / n_frames,
            }
        )

    occ_df = pd.DataFrame(occ_rows)
    if not occ_df.empty:
        occ_df = occ_df.sort_values(
            "occupancy",
            ascending=False,
        )

    occ_df.to_csv(
        outdir / "contact_residue_occupancy.csv",
        index=False,
    )

    last = df[df["time_ns"] >= args.sim_ns * 0.8]

    if not occ_df.empty:
        top_residues = ";".join(
            occ_df.head(15)["residue"].tolist()
        )
    else:
        top_residues = ""

    summary = {
        "candidate": args.candidate,
        "sim_ns": args.sim_ns,
        "n_frames": n_frames,
        "ligand_resname": args.ligand_resname,
        "polar_contacts_last20pct_mean": (
            last["polar_contacts_3p5A"].mean()
        ),
        "protein_contacts_last20pct_mean": (
            last["protein_contacts_4A"].mean()
        ),
        "water_oxygen_contacts_last20pct_mean": (
            last["water_oxygen_contacts_4A"].mean()
        ),
        "pocket_residue_count_last20pct_mean": (
            last["pocket_residue_count_4A"].mean()
        ),
        "top_contact_residues": top_residues,
    }

    pd.DataFrame([summary]).to_csv(
        outdir / "extra_summary.csv",
        index=False,
    )

    title = args.candidate

    save_plot(
        df,
        "polar_contacts_3p5A",
        "Polar contacts < 3.5 A",
        outdir / "polar_contacts.png",
        title,
    )

    save_plot(
        df,
        "protein_contacts_4A",
        "Protein-LIG contacts < 4 A",
        outdir / "protein_contacts.png",
        title,
    )

    save_plot(
        df,
        "water_oxygen_contacts_4A",
        "Water oxygen-LIG contacts < 4 A",
        outdir / "water_contacts.png",
        title,
    )

    save_plot(
        df,
        "pocket_residue_count_4A",
        "Pocket residues within 4 A",
        outdir / "pocket_residue_count.png",
        title,
    )

    print("DONE", args.candidate)
    print("Output:", outdir)


if __name__ == "__main__":
    main()