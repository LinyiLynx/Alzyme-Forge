from __future__ import annotations

import csv
import math
import re
from pathlib import Path

import pandas as pd


HITS_FILENAME = "gt_hits.tsv"
ANNOTATION_FILENAME = "uniprot_reviewed_glycosyltransferase_KW-0328.tsv"

EXCLUDE_ANNOTATION_TERMS = (
    "xyloglucan endotransglucosylase",
    "endotransglycosylase",
    "exostosin",
    "hyaluronan",
    "chitin",
    "peptidoglycan",
    "dolichyl",
    "glycogen",
    "glycosidase",
    "hydrolase",
)

UGT_LIKE_TERMS = (
    "udp-glycosyltransferase",
    "udp-glucosyltransferase",
    "udp-glucuronosyltransferase",
    "udp-arabinosyltransferase",
    "udp-rhamnosyltransferase",
    "udp-xylosyltransferase",
    "flavonoid",
    "anthocyanidin",
    "anthocyanin",
    "glucosyltransferase",
    "rhamnosyltransferase",
    "xylosyltransferase",
    "glycosyltransferase",
    "ugt",
)

GT_OUTPUT_COLUMNS = [
    "gt_candidate",
    "gt_tier",
    "gt_reason",
    "gt_tail_hcgw",
    "gt_tail_hcg",
    "gt_tail_pspg_like",
    "gt_hit_target",
    "gt_hit_accession",
    "gt_hit_identity",
    "gt_hit_align_len",
    "gt_hit_evalue",
    "gt_hit_bitscore",
    "gt_hit_query_coverage",
    "gt_target_entry_name",
    "gt_target_protein_names",
    "gt_target_gene_names",
    "gt_target_organism",
    "gt_target_keywords",
]


def as_float(value: str | float | int | None) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def contains_any(text: str, terms: tuple[str, ...]) -> bool:
    lower = text.lower()
    return any(term in lower for term in terms)


def parse_uniprot_accession(target: str) -> str:
    pieces = target.split("|")
    if len(pieces) >= 2:
        return pieces[1]
    return target


def resolve_gt_reference_dir(reference_dir: str | Path | None = None) -> Path:
    if reference_dir:
        candidates = [Path(reference_dir)]
    else:
        cwd = Path.cwd()
        candidates = [
            cwd / "data" / "gt_reference",
            cwd / "uniprot_gt_reference",
            cwd.parent / "uniprot_gt_reference",
        ]

    missing_messages = []
    for candidate in candidates:
        hits_path = candidate / HITS_FILENAME
        annotation_path = candidate / ANNOTATION_FILENAME
        if hits_path.exists() and annotation_path.exists():
            return candidate
        missing_messages.append(f"{candidate} needs {HITS_FILENAME} and {ANNOTATION_FILENAME}")

    raise FileNotFoundError("Unable to resolve strict GT reference directory: " + "; ".join(missing_messages))


def read_hits(path: Path) -> dict[str, dict[str, str]]:
    hits: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 7:
                continue
            query, target, pident, align_len, evalue, bitscore, qcov = parts[:7]
            current = {
                "gt_hit_target": target,
                "gt_hit_accession": parse_uniprot_accession(target),
                "gt_hit_identity": pident,
                "gt_hit_align_len": align_len,
                "gt_hit_evalue": evalue,
                "gt_hit_bitscore": bitscore,
                "gt_hit_query_coverage": qcov,
            }
            previous = hits.get(query)
            if previous is None or as_float(current["gt_hit_bitscore"]) > as_float(previous["gt_hit_bitscore"]):
                hits[query] = current
    return hits


def read_uniprot_annotations(path: Path) -> dict[str, dict[str, str]]:
    annotations: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            entry = str(row.get("Entry", "")).strip()
            if not entry:
                continue
            annotations[entry] = {
                "gt_target_entry_name": row.get("Entry Name", ""),
                "gt_target_protein_names": row.get("Protein names", ""),
                "gt_target_gene_names": row.get("Gene Names", ""),
                "gt_target_organism": row.get("Organism", ""),
                "gt_target_keywords": row.get("Keywords", ""),
            }
    return annotations


def motif_flags(sequence: str) -> dict[str, str]:
    tail = str(sequence or "")[-180:]
    return {
        "gt_tail_hcgw": "1" if "HCGW" in tail else "0",
        "gt_tail_hcg": "1" if "HCG" in tail else "0",
        "gt_tail_pspg_like": "1" if re.search(r"H[ACGSTY][A-Z]{0,4}[GS][A-Z]{0,8}W", tail) else "0",
    }


def choose_tier(row: dict[str, str]) -> tuple[str, str]:
    length = int(row.get("length") or 0)
    identity = as_float(row.get("gt_hit_identity"))
    evalue = as_float(row.get("gt_hit_evalue"))
    bitscore = as_float(row.get("gt_hit_bitscore"))
    qcov = as_float(row.get("gt_hit_query_coverage"))
    annotation_text = " ".join(
        [
            row.get("gt_target_entry_name", ""),
            row.get("gt_target_protein_names", ""),
            row.get("gt_target_keywords", ""),
        ]
    )
    annotation_ugt_like = contains_any(annotation_text, UGT_LIKE_TERMS)
    annotation_excluded = contains_any(annotation_text, EXCLUDE_ANNOTATION_TERMS)
    strong_hit = identity >= 30 and evalue <= 1e-20 and bitscore >= 100 and qcov >= 50
    very_strong_hit = identity >= 45 and evalue <= 1e-50 and bitscore >= 200 and qcov >= 70
    length_family1 = 350 <= length <= 650
    length_broad = 250 <= length <= 750
    tail_hcgw = row.get("gt_tail_hcgw") == "1"
    tail_hcg = row.get("gt_tail_hcg") == "1"

    if not row.get("gt_hit_target"):
        return "", "no DIAMOND reviewed GT hit"
    if annotation_excluded:
        return "", "excluded by broad non-UGT annotation term"
    if strong_hit and length_family1 and tail_hcgw and annotation_ugt_like:
        return "A_family1_ugt_high_confidence", "strong DIAMOND + 350-650 aa + C-terminal HCGW + UGT-like annotation"
    if strong_hit and length_family1 and tail_hcg:
        return "B_pspg_like_gt_candidate", "strong DIAMOND + 350-650 aa + C-terminal HCG motif"
    if very_strong_hit and length_broad and annotation_ugt_like:
        return "C_reviewed_gt_high_similarity", "very strong DIAMOND + broad GT annotation"
    return "", "did not meet strict GT tier rules"


def apply_strict_gt_gate(proteins: pd.DataFrame, reference_dir: str | Path | None = None) -> pd.DataFrame:
    resolved_reference_dir = resolve_gt_reference_dir(reference_dir)
    hits = read_hits(resolved_reference_dir / HITS_FILENAME)
    annotations = read_uniprot_annotations(resolved_reference_dir / ANNOTATION_FILENAME)

    screened_rows = []
    for protein in proteins.to_dict("records"):
        row = dict(protein)
        hit = hits.get(str(row.get("protein_id", "")), {})
        annotation = annotations.get(hit.get("gt_hit_accession", ""), {})
        evidence = {
            "gt_candidate": "0",
            "gt_tier": "",
            "gt_reason": "",
            "gt_hit_target": "",
            "gt_hit_accession": "",
            "gt_hit_identity": "",
            "gt_hit_align_len": "",
            "gt_hit_evalue": "",
            "gt_hit_bitscore": "",
            "gt_hit_query_coverage": "",
            "gt_target_entry_name": "",
            "gt_target_protein_names": "",
            "gt_target_gene_names": "",
            "gt_target_organism": "",
            "gt_target_keywords": "",
            **motif_flags(str(row.get("sequence", ""))),
            **hit,
            **annotation,
        }
        tier, reason = choose_tier({**row, **evidence})
        evidence["gt_candidate"] = "1" if tier else "0"
        evidence["gt_tier"] = tier
        evidence["gt_reason"] = reason
        screened_rows.append({**row, **evidence})

    return pd.DataFrame(screened_rows)


def summarize_gt_gate(proteins: pd.DataFrame) -> dict[str, int]:
    if proteins.empty or "gt_candidate" not in proteins.columns:
        return {
            "gt_candidates_total": 0,
            "gt_tier_A": 0,
            "gt_tier_B": 0,
            "gt_tier_C": 0,
            "gt_zeroed_total": int(len(proteins)),
        }
    candidates = proteins["gt_candidate"].astype(str) == "1"
    tiers = proteins["gt_tier"].fillna("").astype(str)
    return {
        "gt_candidates_total": int(candidates.sum()),
        "gt_tier_A": int((tiers == "A_family1_ugt_high_confidence").sum()),
        "gt_tier_B": int((tiers == "B_pspg_like_gt_candidate").sum()),
        "gt_tier_C": int((tiers == "C_reviewed_gt_high_similarity").sum()),
        "gt_zeroed_total": int((~candidates).sum()),
    }
