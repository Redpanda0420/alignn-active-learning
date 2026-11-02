"""
任务 3.1 核心 · 模型集成不确定性采样
==========================================
功能：用 4 个 ALIGNN 集成模型（不同种子）对全部候选结构预测，
      用预测标准差作为不确定性，输出不确定性排序清单。
输入：
  - data/candidates/*.cif          全部候选（掺杂+缺陷）
  - models/alignn_energy_best.pt + seed1/2/3 集成模型
输出：
  - data/active_learning/round1_query.csv  候选清单（含均值+不确定性）
依赖：alignn conda 环境
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# ====== 项目根目录 ======
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent))
DATA_DIR = PROJECT_ROOT / "data"
CAND_DIR = DATA_DIR / "candidates"
AL_DIR = DATA_DIR / "active_learning"
MODEL_DIR = PROJECT_ROOT / "models"
RUNS_DIR = PROJECT_ROOT / "runs"
AL_DIR.mkdir(parents=True, exist_ok=True)

# 集成模型清单（主模型 + 3 个种子）
ENSEMBLE_MODELS = [
    ("alignn_energy_best.pt", "alignn_energy"),
    ("alignn_energy_seed1_best.pt", "alignn_energy_seed1"),
    ("alignn_energy_seed2_best.pt", "alignn_energy_seed2"),
    ("alignn_energy_seed3_best.pt", "alignn_energy_seed3"),
]


def load_model(ckpt_name, run_name):
    """加载单个 ALIGNN 模型"""
    from alignn.models.alignn_atomwise_pure import (
        ALIGNNAtomWisePure, ALIGNNAtomWisePureConfig,
    )
    ckpt_path = MODEL_DIR / ckpt_name
    cfg_path = RUNS_DIR / run_name / "config.json"
    if not ckpt_path.exists():
        print(f"[WARN] 未找到 {ckpt_path}，跳过")
        return None
    config = json.load(open(cfg_path))
    model_cfg = ALIGNNAtomWisePureConfig(**config["model"])
    model = ALIGNNAtomWisePure(model_cfg)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model, config


def structure_to_atoms_dict(structure):
    """pymatgen Structure → JARVIS atoms dict"""
    from jarvis.core.atoms import Atoms
    return Atoms(
        lattice_mat=structure.lattice.matrix.tolist(),
        coords=structure.cart_coords.tolist(),
        elements=[sp.symbol for sp in structure.species],
        props=[sp.Z for sp in structure.species],
        cartesian=True,
    )


def predict_model(model, structure, model_cfg, device):
    """单个模型预测"""
    from alignn.torch_graph_builder import build_pure_torch_graph
    atoms = structure_to_atoms_dict(structure)
    g, lg = build_pure_torch_graph(
        atoms=atoms,
        two_body_cutoff=model_cfg.get("cutoff", 5.0),
        three_body_cutoff=model_cfg.get("three_body_cutoff"),
        max_neighbors=model_cfg.get("max_neighbors", 12),
        atom_features=model_cfg.get("atom_features", "cgcnn"),
        use_lattice_prop=False,
        compute_line_graph=True,
        device=device,
    )
    lattice = torch.tensor(atoms.lattice_mat, dtype=torch.float32, device=device)
    with torch.no_grad():
        result = model([g, lg, lattice])
    return result["out"].cpu().item()


def main():
    print("=== 模型集成不确定性采样 ===")

    parser = argparse.ArgumentParser()
    parser.add_argument("--cif_dir", type=str, default=None)
    parser.add_argument("--top_k", type=int, default=50,
                        help="输出不确定性最高的 Top K")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---------- 1. 加载集成模型 ----------
    print("\n[STEP] 加载集成模型...")
    models = []
    for ckpt, run in ENSEMBLE_MODELS:
        m = load_model(ckpt, run)
        if m is not None:
            models.append(m)
            print(f"  [OK] {ckpt}")
    if len(models) < 2:
        print("[ERROR] 集成模型不足 2 个")
        sys.exit(1)
    print(f"[OK] 集成共 {len(models)} 个模型")

    # ---------- 2. 读取候选 ----------
    cif_dir = Path(args.cif_dir) if args.cif_dir else CAND_DIR
    cif_files = sorted(cif_dir.glob("*.cif")) if cif_dir.exists() else []
    if len(cif_files) == 0:
        print(f"[ERROR] 无候选 CIF: {cif_dir}")
        sys.exit(1)
    print(f"[OK] 候选结构: {len(cif_files)} 个")

    # ---------- 3. 集成预测 ----------
    from pymatgen.core import Structure

    print("\n[STEP] 集成预测（4 模型 × 87 候选）...")
    results = []
    for cif_path in tqdm(cif_files, desc="集成预测"):
        try:
            struct = Structure.from_file(str(cif_path))
            preds = [predict_model(m, struct, cfg, device)
                     for m, cfg in models]
            results.append({
                "candidate_id": cif_path.stem,
                "n_atoms": len(struct),
                "pred_energy_mean": float(np.mean(preds)),
                "pred_energy_std": float(np.std(preds)),
                "ensemble_preds": [float(p) for p in preds],
            })
        except Exception as e:
            print(f"  [WARN] {cif_path.name} 失败: {e}")
            continue

    if len(results) == 0:
        print("[ERROR] 无成功预测")
        sys.exit(1)

    df = pd.DataFrame(results)

    # ---------- 4. 排序输出 ----------
    # 不确定性最高在前（标准差最大 = 模型最没把握）
    df_unc = df.sort_values("pred_energy_std", ascending=False)
    out_csv = AL_DIR / "round1_query.csv"
    df_unc.to_csv(out_csv, index=False)
    print(f"\n[OK] 清单已保存: {out_csv} ({len(df_unc)} 条)")

    # 展示 Top K
    top = df_unc.head(args.top_k)
    print(f"\n{'='*70}")
    print(f"  不确定性最高的 Top {min(args.top_k, len(df_unc))}（推荐 VASP 计算）")
    print(f"{'='*70}")
    print(top[["candidate_id", "pred_energy_mean", "pred_energy_std"]]
          .to_string(index=False, float_format="%.4f"))
    print(f"{'='*70}")

    # 统计
    print(f"\n  形成能均值范围: {df['pred_energy_mean'].min():.3f} ~ "
          f"{df['pred_energy_mean'].max():.3f} eV/atom")
    print(f"  不确定性范围:   {df['pred_energy_std'].min():.4f} ~ "
          f"{df['pred_energy_std'].max():.4f} eV/atom")

    print("\n=== DONE ===")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[FATAL] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
