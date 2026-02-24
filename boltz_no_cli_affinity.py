#!/usr/bin/env python3
"""
No-CLI, no-YAML Boltz2 pipeline on Narval.

Input CSV: protein_sequence, peptide_sequence, kd_value (uM)
- Peptide -> RDKit SMILES
- Build boltz schema dict in-memory (with local MSA paths)
- parse_boltz_schema -> dump processed assets
- Step 1: structure prediction (conf ckpt + BoltzWriter)
- Step 2: affinity prediction (aff ckpt + BoltzAffinityWriter)
- Collect affinity json -> merge w/ CSV -> save results + mapping audit
"""

from __future__ import annotations

import json
import math
import os
import pickle
import re
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from pytorch_lightning import Trainer, seed_everything
from rdkit import Chem

from boltz.data.mol import load_canonicals
from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
from boltz.data.parse.a3m import parse_a3m
from boltz.data.parse.csv import parse_csv as parse_msa_csv
from boltz.data.parse.schema import parse_boltz_schema
from boltz.data.types import Manifest, Record
from boltz.data.write.writer import BoltzAffinityWriter, BoltzWriter
from boltz.model.models.boltz2 import Boltz2


# -------------------------
# PATHS (Narval)
# -------------------------
SCRATCH = Path(os.environ.get("SCRATCH", "/scratch/asarrafz")).resolve()

PROJECT_DIR = SCRATCH / "boltz_project"
IN_CSV = PROJECT_DIR / "binding_data_kd.csv"

MSA_DIR = PROJECT_DIR / "msas"
MSA_MASTER_FASTA = PROJECT_DIR / "all_proteins_for_msa.fasta"

CACHE_DIR = Path(os.environ.get("BOLTZ_CACHE", str(SCRATCH / "boltz_cache"))).expanduser().resolve()
CONF_CKPT = CACHE_DIR / "boltz2_conf.ckpt"
AFF_CKPT  = CACHE_DIR / "boltz2_aff.ckpt"
MOL_DIR   = CACHE_DIR / "mols"

# if running under Slurm, we put outputs in /scratch/asarrafz/run_setB/affinity_run_<jobid>
JOBID = os.environ.get("SLURM_JOB_ID", "interactive")
WORKDIR = SCRATCH / "run_setB" / f"affinity_run_{JOBID}"

# -------------------------
# RUNTIME SETTINGS
# -------------------------
ACCELERATOR = "gpu" if torch.cuda.is_available() else "cpu"
DEVICES = 1
NUM_WORKERS = 2
SEED = 42

# structure step
RECYCLING_STEPS = 3
SAMPLING_STEPS = 200
DIFFUSION_SAMPLES = 1

# affinity step
RECYCLING_STEPS_AFF = 5
SAMPLING_STEPS_AFF = 200
DIFFUSION_SAMPLES_AFF = 3


# -------------------------
# dataclass mirrors
# -------------------------
@dataclass
class PairformerArgsV2:
    num_blocks: int = 64
    num_heads: int = 16
    dropout: float = 0.0
    activation_checkpointing: bool = False
    offload_to_cpu: bool = False
    v2: bool = True

@dataclass
class MSAModuleArgs:
    msa_s: int = 64
    msa_blocks: int = 4
    msa_dropout: float = 0.0
    z_dropout: float = 0.0
    use_paired_feature: bool = True
    pairwise_head_width: int = 32
    pairwise_num_heads: int = 4
    activation_checkpointing: bool = False
    offload_to_cpu: bool = False
    subsample_msa: bool = False
    num_subsampled_msa: int = 1024

@dataclass
class Boltz2DiffusionParams:
    gamma_0: float = 0.8
    gamma_min: float = 1.0
    noise_scale: float = 1.003
    rho: float = 7
    step_scale: float = 1.5
    sigma_min: float = 0.0001
    sigma_max: float = 160.0
    sigma_data: float = 16.0
    P_mean: float = -1.2
    P_std: float = 1.5
    coordinate_augmentation: bool = True
    alignment_reverse_diff: bool = True
    synchronize_sigmas: bool = True

@dataclass
class BoltzSteeringParams:
    fk_steering: bool = False
    num_particles: int = 3
    fk_lambda: float = 4.0
    fk_resampling_interval: int = 3
    physical_guidance_update: bool = False
    contact_guidance_update: bool = True
    num_gd_steps: int = 20


# -------------------------
# debug / sanity helpers
# -------------------------
def dbg(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def sanity(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(f"SANITY FAIL: {msg}")

def file_info(p: Path) -> str:
    try:
        if not p.exists():
            return f"{p} (MISSING)"
        if p.is_dir():
            # count entries (non-recursive)
            n = len(list(p.iterdir()))
            return f"{p} (dir, {n} entries)"
        return f"{p} (file, {p.stat().st_size} bytes)"
    except Exception as e:
        return f"{p} (exists? error: {e})"


# -------------------------
# helpers
# -------------------------
def ensure_col(df: pd.DataFrame, col: str):
    if col not in df.columns:
        raise ValueError(f"Missing column '{col}'. Found: {list(df.columns)}")

def pkd_from_uM(kd_uM: float) -> float:
    kd_M = float(kd_uM) * 1e-6
    return float(-math.log10(kd_M)) if kd_M > 0 else float("nan")

def peptide_to_smiles(seq: str) -> str:
    mol = Chem.MolFromSequence(seq.strip().upper())
    if mol is None:
        raise ValueError(f"RDKit could not parse peptide: {seq[:40]}...")
    return Chem.MolToSmiles(mol)

def load_fasta_seq_to_index(fasta_path: Path) -> dict[str, int]:
    seq_to_idx: dict[str, int] = {}
    idx = -1
    cur: list[str] = []
    with fasta_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur:
                    idx += 1
                    seq_to_idx["".join(cur).replace(" ", "").replace("\r", "").upper()] = idx
                    cur = []
                continue
            cur.append(line)
    if cur:
        idx += 1
        seq_to_idx["".join(cur).replace(" ", "").replace("\r", "").upper()] = idx
    return seq_to_idx

def a3m_has_content(p: Path) -> bool:
    try:
        with p.open() as f:
            for line in f:
                s = line.strip()
                if s and not s.startswith(">"):
                    return True
    except Exception:
        pass
    return False

def _if_exists(p: Path) -> Optional[Path]:
    return p if p.exists() else None

def norm_seq(s: str) -> str:
    return str(s).strip().replace(" ", "").replace("\r", "").upper()


# -------------------------
# Step 1: process CSV -> processed/ + manifest
# -------------------------
def process_csv_to_processed(
    csv_path: Path,
    out_dir: Path,
    mol_dir: Path,
    msa_master_fasta: Path,
    msa_dir: Path,
    max_msa_seqs: int = 8192,
) -> tuple[pd.DataFrame, Manifest]:

    dbg(f"Reading CSV: {file_info(csv_path)}")
    df = pd.read_csv(csv_path)

    dbg(f"CSV loaded. shape={df.shape}. columns={list(df.columns)}")
    ensure_col(df, "protein_sequence")
    ensure_col(df, "peptide_sequence")
    ensure_col(df, "kd_value")

    if "row_id" not in df.columns:
        df["row_id"] = range(len(df))
        dbg("Added row_id column (0..N-1).")
    else:
        dbg("row_id column already present.")

    # quick NA / basic stats
    dbg(f"NA counts: protein={int(df['protein_sequence'].isna().sum())}, peptide={int(df['peptide_sequence'].isna().sum())}, kd={int(df['kd_value'].isna().sum())}")
    dbg(f"Unique proteins={df['protein_sequence'].nunique(dropna=True)} | Unique peptides={df['peptide_sequence'].nunique(dropna=True)}")

    # peptide -> smiles
    dbg("Converting peptides -> SMILES (RDKit)...")
    smiles_list = []
    n_fail = 0
    fail_examples = []
    for seq in df["peptide_sequence"].astype(str):
        s = seq.strip()
        if (not s) or (s.lower() == "nan"):
            smiles_list.append(None)
            continue
        try:
            smiles_list.append(peptide_to_smiles(s))
        except Exception as e:
            n_fail += 1
            if len(fail_examples) < 5:
                fail_examples.append((s[:60], str(e)))
            smiles_list.append(None)

    df["ligand_smiles"] = smiles_list
    dbg(f"Converted peptides -> SMILES: {int(df['ligand_smiles'].notna().sum())}/{len(df)} (failures={n_fail})")
    if fail_examples:
        dbg("Example peptide->SMILES failures (up to 5):")
        for ex_seq, ex_err in fail_examples:
            dbg(f"  peptide='{ex_seq}' err='{ex_err}'")

    # msa mapping
    dbg(f"Checking MSA master FASTA: {file_info(msa_master_fasta)}")
    dbg(f"Checking MSA dir: {file_info(msa_dir)}")
    if not msa_master_fasta.exists():
        raise FileNotFoundError(f"Missing MSA master FASTA: {msa_master_fasta}")
    if not msa_dir.exists():
        raise FileNotFoundError(f"Missing MSA dir: {msa_dir}")

    dbg("Loading master FASTA index (seq -> idx)...")
    seq_to_idx = load_fasta_seq_to_index(msa_master_fasta)
    dbg(f"Loaded master FASTA seqs: {len(seq_to_idx)}")
    sanity(len(seq_to_idx) > 0, "MSA master FASTA index is empty.")

    # output dirs
    processed_dir = out_dir / "processed"
    structure_dir = processed_dir / "structures"
    processed_msa_dir = processed_dir / "msa"
    processed_mols_dir = processed_dir / "mols"
    processed_constraints_dir = processed_dir / "constraints"
    processed_templates_dir = processed_dir / "templates"
    records_dir = processed_dir / "records"
    predictions_dir = out_dir / "predictions"

    for d in [structure_dir, processed_msa_dir, processed_mols_dir,
              processed_constraints_dir, processed_templates_dir,
              records_dir, predictions_dir]:
        d.mkdir(parents=True, exist_ok=True)

    dbg(f"Output dirs ready under: {processed_dir}")

    dbg(f"Loading canonicals from: {file_info(mol_dir)}")
    ccd = load_canonicals(mol_dir)
    Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

    records: list[Record] = []
    n_msa_found = 0

    # counters for visibility
    n_total = 0
    n_skip_empty = 0
    n_skip_no_msa = 0
    n_parse_fail = 0

    dbg("Beginning schema parse / asset dump loop...")
    for row in df.itertuples(index=False):
        n_total += 1
        rid = int(row.row_id)
        prot = norm_seq(row.protein_sequence)
        smi = row.ligand_smiles

        if not prot or prot == "NAN" or smi is None or str(smi).lower() == "nan":
            n_skip_empty += 1
            continue

        # resolve MSA path using master fasta index
        msa_path_str: Optional[str] = None
        msa_idx = seq_to_idx.get(prot)
        if msa_idx is not None:
            a3m_path = msa_dir / f"{msa_idx}.a3m"
            if a3m_path.exists() and a3m_has_content(a3m_path):
                msa_path_str = str(a3m_path)
                n_msa_found += 1
            else:
                n_skip_no_msa += 1
        else:
            n_skip_no_msa += 1

        # build schema dict (no YAML)
        protein_entry = {"id": "A", "sequence": prot}
        if msa_path_str is not None:
            protein_entry["msa"] = msa_path_str  # real file path

        schema = {
            "version": 1,
            "sequences": [
                {"protein": protein_entry},
                {"ligand": {"id": "B", "smiles": smi}},
            ],
            "properties": [{"affinity": {"binder": "B"}}],
        }
        name = f"row_{rid}"

        try:
            target = parse_boltz_schema(name, schema, ccd, mol_dir, boltz_2=True)
        except Exception as e:
            n_parse_fail += 1
            if n_parse_fail <= 5:
                dbg(f"SKIP {name}: parse error — {e}")
            continue

        # normalize msa_id: never allow "empty"/None strings to survive
        for chain in target.record.chains:
            if chain.msa_id in ("empty", "", None):
                chain.msa_id = -1

        target_id = target.record.id

        # dump MSAs -> processed/msa/*.npz and rewrite chain.msa_id to processed name
        msa_id_map: dict[str, str] = {}
        msas_unique = sorted({
            c.msa_id for c in target.record.chains
            if isinstance(c.msa_id, str) and c.msa_id.endswith((".a3m", ".csv"))
        })

        for msa_i, raw_msa_id in enumerate(msas_unique):
            raw_msa_path = Path(raw_msa_id)
            processed_name = f"{target_id}_{msa_i}"
            processed_path = processed_msa_dir / f"{processed_name}.npz"
            msa_id_map[raw_msa_id] = processed_name

            if not processed_path.exists() and raw_msa_path.exists():
                if raw_msa_path.suffix == ".a3m":
                    msa_obj = parse_a3m(raw_msa_path, taxonomy=None, max_seqs=max_msa_seqs)
                elif raw_msa_path.suffix == ".csv":
                    msa_obj = parse_msa_csv(raw_msa_path, max_seqs=max_msa_seqs)
                else:
                    continue
                msa_obj.dump(processed_path)

        for chain in target.record.chains:
            if isinstance(chain.msa_id, str) and chain.msa_id in msa_id_map:
                chain.msa_id = msa_id_map[chain.msa_id]
            elif chain.msa_id in ("empty", "", None, 0):
                chain.msa_id = -1

        # dump templates/constraints/extra_mols/structure/record
        for template_id, template_struct in target.templates.items():
            tpl_path = processed_templates_dir / f"{target_id}_{template_id}.npz"
            if not tpl_path.exists():
                template_struct.dump(tpl_path)

        con_path = processed_constraints_dir / f"{target_id}.npz"
        if not con_path.exists():
            target.residue_constraints.dump(con_path)

        mol_path = processed_mols_dir / f"{target_id}.pkl"
        if not mol_path.exists():
            with mol_path.open("wb") as f:
                pickle.dump(target.extra_mols, f)

        struct_path = structure_dir / f"{target_id}.npz"
        if not struct_path.exists():
            target.structure.dump(struct_path)

        rec_path = records_dir / f"{target_id}.json"
        target.record.dump(rec_path)

        records.append(target.record)

        # occasional progress
        if len(records) % 100 == 0:
            dbg(f"Progress: records={len(records)} / rows_seen={n_total}")

    dbg(f"Loop done. rows_seen={n_total} | records={len(records)} | skip_empty={n_skip_empty} | no_msa={n_skip_no_msa} | parse_fail={n_parse_fail}")
    manifest = Manifest(records)
    manifest_path = processed_dir / "manifest.json"
    manifest.dump(manifest_path)

    sanity(manifest_path.exists(), f"manifest.json not written at {manifest_path}")
    sanity(manifest_path.stat().st_size > 10, f"manifest.json too small ({manifest_path.stat().st_size} bytes)")

    dbg(f"Processed inputs: {len(records)}")
    dbg(f"MSA coverage (local .a3m found): {n_msa_found}/{len(records)} had a local .a3m path")
    dbg(f"Manifest written: {file_info(manifest_path)}")
    dbg(f"Processed dir snapshot: structures={len(list(structure_dir.glob('*.npz')))} records={len(list(records_dir.glob('*.json')))} msa_npz={len(list(processed_msa_dir.glob('*.npz')))}")

    return df, manifest


# -------------------------
# Step 2: structure prediction
# -------------------------
def run_structure_prediction(manifest: Manifest, out_dir: Path, mol_dir: Path, conf_ckpt: Path) -> None:
    processed_dir = out_dir / "processed"
    predictions_dir = out_dir / "predictions"

    dbg("Sanity before structure prediction:")
    dbg(f"  conf_ckpt: {file_info(conf_ckpt)}")
    dbg(f"  processed_dir: {file_info(processed_dir)}")
    dbg(f"  structures_in: {file_info(processed_dir / 'structures')}")
    dbg(f"  msa_dir: {file_info(processed_dir / 'msa')}")
    dbg(f"  mol_dir: {file_info(mol_dir)}")
    sanity(conf_ckpt.exists(), "conf_ckpt missing")
    sanity((processed_dir / "manifest.json").exists(), "processed/manifest.json missing")

    predict_args = {
        "recycling_steps": RECYCLING_STEPS,
        "sampling_steps": SAMPLING_STEPS,
        "diffusion_samples": DIFFUSION_SAMPLES,
        "max_parallel_samples": 1,
        "write_confidence_summary": True,
        "write_full_pae": False,
        "write_full_pde": False,
    }

    dbg("Loading structure checkpoint...")
    model = Boltz2.load_from_checkpoint(
        str(conf_ckpt),
        strict=True,
        predict_args=predict_args,
        map_location="cpu",
        diffusion_process_args=asdict(Boltz2DiffusionParams()),
        ema=False,
        pairformer_args=asdict(PairformerArgsV2()),
        msa_args=asdict(MSAModuleArgs()),
        steering_args=asdict(BoltzSteeringParams()),
    )
    model.eval()
    dbg("Model loaded. Building datamodule...")

    dm = Boltz2InferenceDataModule(
        manifest=manifest,
        target_dir=processed_dir / "structures",
        msa_dir=processed_dir / "msa",
        mol_dir=mol_dir,
        num_workers=NUM_WORKERS,
        constraints_dir=_if_exists(processed_dir / "constraints"),
        template_dir=_if_exists(processed_dir / "templates"),
        extra_mols_dir=_if_exists(processed_dir / "mols"),
    )

    writer = BoltzWriter(
        data_dir=processed_dir / "structures",
        output_dir=predictions_dir,
        output_format="mmcif",
        boltz2=True,
    )

    dbg(f"Starting trainer.predict (structure). accelerator={ACCELERATOR} devices={DEVICES}")
    trainer = Trainer(
        default_root_dir=str(out_dir),
        strategy="auto",
        callbacks=[writer],
        accelerator=ACCELERATOR,
        devices=DEVICES,
        precision="bf16-mixed",
        enable_progress_bar=True,
        log_every_n_steps=1,
    )
    trainer.predict(model, datamodule=dm, return_predictions=False)

    # post-check: did we write anything?
    sanity(predictions_dir.exists(), "predictions_dir missing after structure prediction")
    n_any = len(list(predictions_dir.rglob("*.cif"))) + len(list(predictions_dir.rglob("*.mmcif"))) + len(list(predictions_dir.rglob("*.npz")))
    dbg(f"Structure prediction done. predictions_dir={predictions_dir} | written_files≈{n_any}")


# -------------------------
# Step 3: affinity prediction
# -------------------------
def run_affinity_prediction(manifest: Manifest, out_dir: Path, mol_dir: Path, aff_ckpt: Path) -> None:
    affinity_records = [r for r in manifest.records if r.affinity is not None]
    dbg(f"Affinity records in manifest: {len(affinity_records)}/{len(manifest.records)}")
    if not affinity_records:
        dbg("No affinity records found in manifest. Skipping affinity step.")
        return

    affinity_manifest = Manifest(affinity_records)
    processed_dir = out_dir / "processed"
    predictions_dir = out_dir / "predictions"

    dbg("Sanity before affinity prediction:")
    dbg(f"  aff_ckpt: {file_info(aff_ckpt)}")
    dbg(f"  predictions_dir: {file_info(predictions_dir)}")
    dbg(f"  msa_dir: {file_info(processed_dir / 'msa')}")
    dbg(f"  structures_dir: {file_info(processed_dir / 'structures')}")
    sanity(aff_ckpt.exists(), "aff_ckpt missing")
    sanity(predictions_dir.exists(), "predictions_dir missing before affinity prediction (structure step probably failed)")

    predict_args = {
        "recycling_steps": RECYCLING_STEPS_AFF,
        "sampling_steps": SAMPLING_STEPS_AFF,
        "diffusion_samples": DIFFUSION_SAMPLES_AFF,
        "max_parallel_samples": 1,
        "write_confidence_summary": False,
        "write_full_pae": False,
        "write_full_pde": False,
    }

    steering = BoltzSteeringParams()
    steering.fk_steering = False
    steering.physical_guidance_update = False
    steering.contact_guidance_update = False

    dbg("Loading affinity checkpoint...")
    model = Boltz2.load_from_checkpoint(
        str(aff_ckpt),
        strict=True,
        predict_args=predict_args,
        map_location="cpu",
        diffusion_process_args=asdict(Boltz2DiffusionParams()),
        ema=False,
        pairformer_args=asdict(PairformerArgsV2()),
        msa_args=asdict(MSAModuleArgs()),
        steering_args=asdict(steering),
    )
    model.eval()
    dbg("Model loaded. Building datamodule...")

    dm = Boltz2InferenceDataModule(
        manifest=affinity_manifest,
        target_dir=predictions_dir,   # uses predicted structures outputs
        msa_dir=processed_dir / "msa",
        mol_dir=mol_dir,
        num_workers=NUM_WORKERS,
        constraints_dir=_if_exists(processed_dir / "constraints"),
        template_dir=_if_exists(processed_dir / "templates"),
        extra_mols_dir=_if_exists(processed_dir / "mols"),
        override_method="other",
        affinity=True,
    )

    writer = BoltzAffinityWriter(
        data_dir=processed_dir / "structures",
        output_dir=predictions_dir,
    )

    dbg(f"Starting trainer.predict (affinity). accelerator={ACCELERATOR} devices={DEVICES}")
    trainer = Trainer(
        default_root_dir=str(out_dir),
        strategy="auto",
        callbacks=[writer],
        accelerator=ACCELERATOR,
        devices=DEVICES,
        precision="bf16-mixed",
        enable_progress_bar=True,
        log_every_n_steps=1,
    )
    trainer.predict(model, datamodule=dm, return_predictions=False)

    # post-check: did we write any affinity json?
    n_aff = len(list(predictions_dir.rglob("*affinity*.json")))
    dbg(f"Affinity prediction done. affinity_json_count={n_aff}")
    sanity(n_aff > 0, "No *affinity*.json written under predictions_dir; writer/pathing likely wrong.")


# -------------------------
# Step 4: collect results + audit mapping
# -------------------------
def collect_results(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    predictions_dir = out_dir / "predictions"
    dbg(f"Collecting results from: {file_info(predictions_dir)}")
    rows = []

    for fp in sorted(predictions_dir.rglob("*affinity*.json")):
        try:
            data = json.loads(fp.read_text())
        except Exception:
            continue
        m = re.search(r"(row_\d+)", str(fp))
        if not m:
            continue
        record_id = m.group(1)
        data["record_id"] = record_id
        rows.append(data)

    if not rows:
        dbg("No affinity predictions found under predictions/.")
        df["affinity_pred_value"] = None
        df["affinity_probability_binary"] = None
        return df

    aff = pd.DataFrame(rows)
    aff["row_id"] = aff["record_id"].str.extract(r"row_(\d+)").astype(int)

    out = df.merge(
        aff[["row_id", "affinity_pred_value", "affinity_probability_binary"]],
        on="row_id",
        how="left",
    )

    out["true_log10_kd_uM"] = out["kd_value"].apply(lambda x: math.log10(x) if pd.notna(x) and float(x) > 0 else None)
    out["true_pKd"] = out["kd_value"].apply(pkd_from_uM)

    dbg(f"Collected predictions: {int(out['affinity_pred_value'].notna().sum())}/{len(out)}")
    return out


def write_mapping_audit(df: pd.DataFrame, out_dir: Path) -> None:
    rec_dir = out_dir / "processed" / "records"
    dbg(f"Writing mapping audit from records dir: {file_info(rec_dir)}")
    rows = []
    for rid in df["row_id"].tolist():
        rec_path = rec_dir / f"row_{rid}.json"
        if not rec_path.exists():
            continue
        try:
            rec = json.loads(rec_path.read_text())
        except Exception:
            continue

        prot_seq = None
        try:
            for ch in rec.get("chains", []):
                if ch.get("chain_id") == "A":
                    prot_seq = ch.get("sequence")
                    break
        except Exception:
            prot_seq = None

        rows.append({
            "row_id": rid,
            "record_file": str(rec_path),
            "csv_protein_sequence": df.loc[df["row_id"] == rid, "protein_sequence"].values[0],
            "record_protein_sequence_chainA": prot_seq,
            "csv_peptide_sequence": df.loc[df["row_id"] == rid, "peptide_sequence"].values[0],
        })

    audit = pd.DataFrame(rows)
    audit_path = out_dir / "mapping_audit.csv"
    audit.to_csv(audit_path, index=False)
    dbg(f"Wrote mapping audit -> {audit_path} (rows={len(audit)})")


# -------------------------
# main
# -------------------------
def main():
    seed_everything(SEED)
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("highest")

    dbg("=== BOOT ===")
    dbg(f"Python: {sys.version.split()[0]}")
    dbg(f"Torch: {torch.__version__} | cuda_available={torch.cuda.is_available()} | accelerator={ACCELERATOR}")
    dbg(f"SCRATCH: {SCRATCH}")
    dbg(f"JOBID: {JOBID}")
    dbg(f"WORKDIR (planned): {WORKDIR}")

    # force unbuffered-ish behavior for some libs
    os.environ["PYTHONUNBUFFERED"] = os.environ.get("PYTHONUNBUFFERED", "1")

    for key in ["CUEQ_DEFAULT_CONFIG", "CUEQ_DISABLE_AOT_TUNING"]:
        os.environ[key] = os.environ.get(key, "1")

    dbg("=== PATH SANITY ===")
    dbg(file_info(PROJECT_DIR))
    dbg(file_info(IN_CSV))
    dbg(file_info(MSA_DIR))
    dbg(file_info(MSA_MASTER_FASTA))
    dbg(file_info(CACHE_DIR))
    dbg(file_info(CONF_CKPT))
    dbg(file_info(AFF_CKPT))
    dbg(file_info(MOL_DIR))

    # sanity required paths
    for p in [IN_CSV, MSA_DIR, MSA_MASTER_FASTA]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required path: {p}")

    for name, p in [("CONF_CKPT", CONF_CKPT), ("AFF_CKPT", AFF_CKPT), ("MOL_DIR", MOL_DIR)]:
        if not p.exists():
            raise FileNotFoundError(f"{name} missing: {p} (your BOLTZ_CACHE likely wrong or cache not downloaded)")

    dbg("=== WORKDIR CREATE + WRITE TEST ===")
    WORKDIR.mkdir(parents=True, exist_ok=True)
    sanity(WORKDIR.exists(), "WORKDIR not created")
    # write test file to confirm permissions
    test_path = WORKDIR / "_write_test.txt"
    test_path.write_text("ok\n")
    sanity(test_path.exists(), "Cannot write in WORKDIR")
    dbg(f"WORKDIR ready + writable: {WORKDIR}")

    print("\n=== STEP 1: CSV -> processed/manifest ===", flush=True)
    df, manifest = process_csv_to_processed(
        csv_path=IN_CSV,
        out_dir=WORKDIR,
        mol_dir=MOL_DIR,
        msa_master_fasta=MSA_MASTER_FASTA,
        msa_dir=MSA_DIR,
    )

    if not manifest.records:
        raise RuntimeError("No records created. Check CSV columns and sequences.")
    dbg(f"Manifest records: {len(manifest.records)}")

    print("\n=== STEP 2: Structure prediction ===", flush=True)
    run_structure_prediction(manifest, WORKDIR, MOL_DIR, CONF_CKPT)

    print("\n=== STEP 3: Affinity prediction ===", flush=True)
    run_affinity_prediction(manifest, WORKDIR, MOL_DIR, AFF_CKPT)

    print("\n=== STEP 4: Collect + save ===", flush=True)
    results = collect_results(df, WORKDIR)
    out_csv = WORKDIR / "affinity_results.csv"
    results.to_csv(out_csv, index=False)
    dbg(f"Saved results -> {out_csv} ({file_info(out_csv)})")

    write_mapping_audit(df, WORKDIR)

    n = len(results)
    n_pred = int(results["affinity_pred_value"].notna().sum()) if "affinity_pred_value" in results.columns else 0
    dbg(f"DONE. Rows={n} | with_predictions={n_pred}")


if __name__ == "__main__":
    main()
