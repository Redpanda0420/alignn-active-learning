"""
任务 2.5 · 用训练好的 ALIGNN 做批量推理
==========================================
功能：加载形成能 + 带隙两个模型，对候选 CIF 结构批量预测，
      输出形成能和带隙，支持 --top_k 筛选最稳定候选。
输入：
  - models/alignn_energy_best.pt + runs/alignn_energy/config.json
  - models/alignn_gap_best.pt + runs/alignn_gap/config.json
  - data/candidates/*.cif       候选结构目录
输出：
  - data/candidates_predictions.csv
参数：
  --top_k N   只输出预测形成能最低（最稳定）的 Top N
依赖：alignn conda 环境
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# ====== 项目根目录 ======
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent))
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
RUNS_DIR = PROJECT_ROOT / "runs"

CANDIDATE_DIR = DATA_DIR / "candidates"
OUT_CSV = DATA_DIR / "candidates_predictions.csv"

# 与训练一致的图参数（重要！必须和训练时完全相同）
CUTOFF = 5.0
MAX_NEIGHBORS = 12
ATOM_FEATURES = "cgcnn"


def load_model(ckpt_name, run_name):
    """从 checkpoint + config 重建 ALIGNN 模型"""
    from alignn.models.alignn_atomwise_pure import (
        ALIGNNAtomWisePure,
        ALIGNNAtomWisePureConfig,
    )

    ckpt_path = MODEL_DIR / ckpt_name
    cfg_path = RUNS_DIR / run_name / "config.json"
    if not ckpt_path.exists():
        print(f"[ERROR] 未找到模型 {ckpt_path}")
        sys.exit(1)
    if not cfg_path.exists():
        print(f"[ERROR] 未找到配置 {cfg_path}")
        sys.exit(1)

    config = json.load(open(cfg_path))
    model_cfg = ALIGNNAtomWisePureConfig(**config["model"])
    model = ALIGNNAtomWisePure(model_cfg)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    print(f"[OK] 模型加载: {ckpt_path} (hidden={model_cfg.hidden_features})")
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


def predict_one(model, structure, model_config, device):
    """对单个结构预测，返回标量"""
    from alignn.torch_graph_builder import build_pure_torch_graph

    atoms = structure_to_atoms_dict(structure)
    g, lg = build_pure_torch_graph(
        atoms=atoms,
        two_body_cutoff=model_config.get("cutoff", CUTOFF),
        three_body_cutoff=model_config.get("three_body_cutoff"),
        max_neighbors=model_config.get("max_neighbors", MAX_NEIGHBORS),
        atom_features=model_config.get("atom_features", ATOM_FEATURES),
        use_lattice_prop=False,
        compute_line_graph=True,
        device=device,
    )
    lattice = torch.tensor(atoms.lattice_mat, dtype=torch.float32, device=device)
    with torch.no_grad():
        result = model([g, lg, lattice])
    return result["out"].cpu().item()


def main():
    print("=== TASK 2.5: ALIGNN 批量推理 ===")

    parser = argparse.ArgumentParser()
    parser.add_argument("--top_k", type=int, default=None,
                        help="只输出预测形成能最低的 Top K")
    parser.add_argument("--cif_dir", type=str, default=None,
                        help="候选 CIF 目录（默认 data/candidates）")
    args = parser.parse_args()

    # ---------- 1. GPU ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---------- 2. 加载两个模型 ----------
    print("\n[STEP] 加载模型...")
    model_e, cfg_e = load_model("alignn_energy_best.pt", "alignn_energy")
    model_g, cfg_g = load_model("alignn_gap_best.pt", "alignn_gap")

    # ---------- 3. 读取候选结构 ----------
    cif_dir = Path(args.cif_dir) if args.cif_dir else CANDIDATE_DIR
    cif_files = []
    if cif_dir.exists():
        cif_files = sorted(cif_dir.glob("*.cif"))
    if len(cif_files) == 0:
        print(f"[WARN] {cif_dir} 下没有 .cif 文件")
        # 回退：用训练结构目录做演示
        fallback = DATA_DIR / "structures"
        cif_files = sorted(fallback.glob("*.cif"))[:20]
        print(f"  回退用 data/structures 前 {len(cif_files)} 个结构做演示")
    print(f"[OK] 找到 {len(cif_files)} 个候选结构")

    # ---------- 4. 批量预测 ----------
    from pymatgen.core import Structure

    results = []
    for cif_path in tqdm(cif_files, desc="预测"):
        try:
            struct = Structure.from_file(str(cif_path))
            e_pred = predict_one(model_e, struct, cfg_e, device)
            g_pred = predict_one(model_g, struct, cfg_g, device)
            results.append({
                "material_id": cif_path.stem,
                "predicted_energy": e_pred,
                "predicted_gap": g_pred,
            })
        except Exception as e:
            print(f"  [WARN] {cif_path.name} 预测失败: {e}")
            continue

    if len(results) == 0:
        print("[ERROR] 无成功预测")
        sys.exit(1)

    # ---------- 5. 排序 + Top K ----------
    df = __import__("pandas").DataFrame(results)
    df = df.sort_values("predicted_energy")  # 形成能最低 = 最稳定

    if args.top_k and args.top_k < len(df):
        print(f"\n[INFO] 只保留预测最稳定的 Top {args.top_k}")
        df = df.head(args.top_k)

    df.to_csv(OUT_CSV, index=False)
    print(f"\n[OK] 预测结果已保存: {OUT_CSV}")
    print(f"  共 {len(df)} 条")

    # ---------- 6. 展示 ----------
    print(f"\n{'='*60}")
    print(f"  预测结果（按形成能升序 = 最稳定在前）")
    print(f"{'='*60}")
    print(df.to_string(index=False, float_format="%.4f"))
    print(f"{'='*60}")

    # 统计
    print(f"\n  形成能范围: {df['predicted_energy'].min():.3f} ~ "
          f"{df['predicted_energy'].max():.3f} eV/atom")
    print(f"  带隙范围:   {df['predicted_gap'].min():.3f} ~ "
          f"{df['predicted_gap'].max():.3f} eV")

    print("\n=== DONE ===")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[FATAL] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
