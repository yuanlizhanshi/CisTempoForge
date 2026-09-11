from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


STRUCTURE_COLUMNS = (
    "utr5_length_log10", "orf_length_log10", "intron_length_log10",
    "utr3_length_log10", "utr5_gc", "orf_gc", "utr3_gc",
    "orf_exon_junction_density",
)
ATTRIBUTE = re.compile(r'(\w+) "([^"]*)"')


def parse_attributes(text: str) -> dict[str, str]:
    return dict(ATTRIBUTE.findall(text))


def merge_intervals(intervals):
    merged = []
    for start, end in sorted(set(intervals)):
        if not merged or start > merged[-1][1] + 1:
            merged.append([int(start), int(end)])
        else:
            merged[-1][1] = max(merged[-1][1], int(end))
    return [tuple(value) for value in merged]


def interval_length(intervals) -> int:
    return sum(end - start + 1 for start, end in merge_intervals(intervals))


def gc_fraction(fasta: Any, chrom: str, intervals) -> float:
    gc = valid = 0
    for start, end in merge_intervals(intervals):
        sequence = str(fasta[chrom][start - 1:end]).upper()
        gc += sequence.count("G") + sequence.count("C")
        valid += sum(sequence.count(base) for base in "ACGT")
    return float(gc / valid) if valid else float("nan")


def classify_utrs(utrs, orf, strand):
    if not orf:
        return [], []
    left, right = min(value[0] for value in orf), max(value[1] for value in orf)
    five, three = [], []
    for start, end in utrs:
        if end < left:
            (five if strand == "+" else three).append((start, end))
        elif start > right:
            (three if strand == "+" else five).append((start, end))
        else:
            if start < left:
                (five if strand == "+" else three).append((start, left - 1))
            if end > right:
                (three if strand == "+" else five).append((right + 1, end))
    return merge_intervals(five), merge_intervals(three)


def transcript_features(record, fasta):
    orf = merge_intervals(record["cds"] + record["stop_codon"])
    if not orf:
        return None
    utr5, utr3 = classify_utrs(record["utr"], orf, record["strand"])
    exons = merge_intervals(record["exon"])
    intron_length = sum(
        max(0, exons[index + 1][0] - exons[index][1] - 1)
        for index in range(len(exons) - 1)
    )
    lengths = {
        "utr5_length_bp": interval_length(utr5), "orf_length_bp": interval_length(orf),
        "intron_length_bp": int(intron_length), "utr3_length_bp": interval_length(utr3),
    }
    junctions = max(0, len(orf) - 1)
    row = {
        **lengths, "utr5_gc": gc_fraction(fasta, record["chrom"], utr5),
        "orf_gc": gc_fraction(fasta, record["chrom"], orf),
        "utr3_gc": gc_fraction(fasta, record["chrom"], utr3),
        "orf_exon_junction_count": int(junctions),
        "orf_exon_junction_density": float(1000 * junctions / lengths["orf_length_bp"]),
        "n_exons": len(exons), "n_orf_blocks": len(orf),
    }
    for region in ("utr5", "orf", "intron", "utr3"):
        row[f"{region}_length_log10"] = float(np.log10(row[f"{region}_length_bp"] + 0.1))
    return row


def build_structure_table(gtf_path: str | Path, fasta_path: str | Path,
                          metadata: pd.DataFrame) -> pd.DataFrame:
    try:
        from pyfaidx import Fasta
    except ImportError as exc:
        raise ImportError("install CisTempoForge[preprocessing] to build structure features") from exc
    wanted = set(metadata["gene"].astype(str))
    transcripts: dict[str, dict] = {}
    with Path(gtf_path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip().split("\t")
            if len(fields) != 9 or fields[2] not in {
                "gene", "transcript", "exon", "CDS", "stop_codon", "UTR"
            }:
                continue
            attrs = parse_attributes(fields[8])
            name = attrs.get("gene_name")
            if name not in wanted or attrs.get("gene_type") != "protein_coding" or fields[2] == "gene":
                continue
            transcript_id = attrs.get("transcript_id")
            if transcript_id is None:
                continue
            record = transcripts.setdefault(transcript_id, {
                "transcript_id": transcript_id, "gene": name,
                "gene_id": attrs.get("gene_id"), "chrom": fields[0], "strand": fields[6],
                "exon": [], "cds": [], "stop_codon": [], "utr": [],
            })
            key = {"exon": "exon", "CDS": "cds", "stop_codon": "stop_codon", "UTR": "utr"}.get(fields[2])
            if key is not None:
                record[key].append((int(fields[3]), int(fields[4])))
    metadata_chrom = metadata.set_index("gene")["chrom"].to_dict()
    by_gene = defaultdict(list)
    for record in transcripts.values():
        if record["chrom"] == metadata_chrom.get(record["gene"]):
            by_gene[record["gene"]].append(record)
    fasta = Fasta(str(fasta_path), as_raw=True, sequence_always_upper=True, rebuild=False)
    rows = []
    for gene, chrom, split in metadata[["gene", "chrom", "split"]].itertuples(index=False):
        candidates = []
        for record in by_gene.get(gene, []):
            values = transcript_features(record, fasta)
            if values is not None:
                candidates.append((record, values))
        candidates.sort(key=lambda item: (
            -item[1]["orf_length_bp"], -item[1]["utr5_length_bp"],
            -item[1]["utr3_length_bp"], item[0]["transcript_id"],
        ))
        if candidates:
            record, values = candidates[0]
            row = {"gene": gene, "chrom": chrom, "split": split,
                   "gene_id": record["gene_id"], "transcript_id": record["transcript_id"],
                   "strand": record["strand"], "structure_transcript_available": True, **values}
        else:
            row = {column: np.nan for column in STRUCTURE_COLUMNS}
            row.update({"gene": gene, "chrom": chrom, "split": split,
                        "gene_id": None, "transcript_id": None,
                        "structure_transcript_available": False})
        rows.append(row)
    result = pd.DataFrame(rows)
    if len(result) != len(metadata) or result["gene"].duplicated().any():
        raise RuntimeError("gene structure alignment is not one-to-one")
    return result


def fit_transform_structure(frame: pd.DataFrame):
    values = frame.loc[:, STRUCTURE_COLUMNS].to_numpy(np.float64)
    if np.isinf(values).any():
        raise ValueError("infinite gene-structure values are not allowed")
    masks = np.isfinite(values)
    train = frame["split"].to_numpy() == "train"
    if not train.any():
        raise ValueError("structure table has no training rows")
    medians = np.nanmedian(np.where(masks[train], values[train], np.nan), axis=0)
    if not np.isfinite(medians).all():
        raise ValueError("a structure feature has no finite training values")
    imputed = np.where(masks, values, medians)
    means, stds = imputed[train].mean(0), imputed[train].std(0)
    stds = np.where(stds < 1e-8, 1.0, stds)
    normalized = ((imputed - means) / stds).astype(np.float32)
    scaler = {"columns": list(STRUCTURE_COLUMNS), "medians": medians.tolist(),
              "means": means.tolist(), "stds": stds.tolist(),
              "fit_split": "train", "n_train": int(train.sum())}
    return normalized, masks.astype(np.float32), scaler
