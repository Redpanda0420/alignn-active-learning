"""
任务 2.2 · 把 CIF 结构转成 ALIGNN 训练数据
============================================
功能：将 data/structures/*.cif 转成 ALIGNN 标准 JSON 数据集
      并按 80/10/10 切分训练/验证/测试。
输入：
  - data/mp_lifepo4_raw.csv   材料属性表（标签来源）
  - data/structures/*.cif     结构文件
输出：
  - data/alignn/formation_energy/{train,val,test}.json
  - data/alignn/band_gap/{train,val,test}.json
依赖：alignn conda 环境（pymatgen + jarvis-tools）
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ====== 项目根目录 ======
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent))
DATA_DIR = PROJECT_ROOT / "data"
STRUCT_DIR = DATA_DIR / "structures"
ALIGNN_DIR = DATA_DIR / "alignn"

# ====== 防坑规则 ======
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def structure_to_atoms_dict(structure):
    """把 pymatgen Structure 转成 JARVIS atoms dict（ALIGNN 标准格式）"""
    return {
        "lattice_mat": structure.lattice.matrix.tolist(),
        "coords": structure.cart_coords.tolist(),
        "elements": [sp.symbol for sp in structure.species],
        "props": [sp.Z for sp in structure.species],
        "cartesian": True,
    }


def build_dataset(records):
    """把记录列表转成 ALIGNN 数据集（list of dict）"""
    dataset = []
    for r in records:
        dataset.append({
            "jid": r["material_id"],
            "atoms": r["atoms"],
            r["target_key"]: r["target_value"],
        })
    return dataset


def split_and_save(dataset, target_name, out_dir):
    """按 80/10/10 切分并保存 JSON"""
    n = len(dataset)
    idx = np.arange(n)
    rng = np.random.RandomState(RANDOM_STATE)
    rng.shuffle(idx)

    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    n_test = n - n_train - n_val

    splits = {
        "train": [dataset[i] for i in idx[:n_train]],
        "val": [dataset[i] for i in idx[n_train:n_train + n_val]],
        "test": [dataset[i] for i in idx[n_train + n_val:]],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, data in splits.items():
        path = out_dir / f"{name}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  [OK] {path.name}: {len(data)} 条")

    return n_train, n_val, n_test


def main():
    print("=== TASK 2.2: CIF → ALIGNN 训练数据 ===")

    # ---------- 1. 读取标签 ----------
    csv_path = DATA_DIR / "mp_lifepo4_raw.csv"
    if not csv_path.exists():
        print(f"[ERROR] 未找到 {csv_path}")
        sys.exit(1)
    df = pd.read_csv(csv_path)
    print(f"[OK] 标签表: {len(df)} 条")

    # ---------- 2. 逐结构转换 ----------
    from pymatgen.core import Structure

    records_energy = []   # 形成能数据集
    records_gap = []      # 带隙数据集（过滤 band_gap < 0）
    n_fail = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="转换结构"):
        mid = row["material_id"]
        cif_path = STRUCT_DIR / f"{mid}.cif"
        if not cif_path.exists():
            n_fail += 1
            continue
        try:
            struct = Structure.from_file(str(cif_path))
            atoms = structure_to_atoms_dict(struct)

            # 形成能数据集（所有样本）
            records_energy.append({
                "material_id": mid,
                "atoms": atoms,
                "target_key": "formation_energy_per_atom",
                "target_value": float(row["formation_energy_per_atom"]),
            })

            # 带隙数据集（过滤缺失值）
            gap = float(row["band_gap"])
            if gap >= 0:
                records_gap.append({
                    "material_id": mid,
                    "atoms": atoms,
                    "target_key": "band_gap",
                    "target_value": gap,
                })
        except Exception as e:
            n_fail += 1
            print(f"  [WARN] {mid}: {e}")

    print(f"\n[OK] 形成能样本: {len(records_energy)}, 带隙样本: {len(records_gap)}, 失败: {n_fail}")

    # ---------- 3. 切分并保存 ----------
    print("\n[STEP] 切分形成能数据集 (80/10/10)...")
    e_dir = ALIGNN_DIR / "formation_energy"
    n1, n2, n3 = split_and_save(build_dataset(records_energy), "formation_energy_per_atom", e_dir)
    print(f"  形成能: train={n1}, val={n2}, test={n3}")

    print("\n[STEP] 切分带隙数据集 (80/10/10)...")
    g_dir = ALIGNN_DIR / "band_gap"
    n1, n2, n3 = split_and_save(build_dataset(records_gap), "band_gap", g_dir)
    print(f"  带隙: train={n1}, val={n2}, test={n3}")

    # ---------- 4. 统计 ----------
    e_vals = [r["target_value"] for r in records_energy]
    g_vals = [r["target_value"] for r in records_gap]

    print(f"\n{'='*50}")
    print(f"  数据集统计")
    print(f"{'='*50}")
    print(f"  形成能: n={len(e_vals)}, mean={np.mean(e_vals):.4f}, std={np.std(e_vals):.4f}")
    print(f"           min={np.min(e_vals):.4f}, max={np.max(e_vals):.4f} eV/atom")
    print(f"  带隙:   n={len(g_vals)}, mean={np.mean(g_vals):.4f}, std={np.std(g_vals):.4f}")
    print(f"           min={np.min(g_vals):.4f}, max={np.max(g_vals):.4f} eV")
    print(f"{'='*50}")

    # 展示一个样本的结构
    sample = records_energy[0]
    print(f"\n  样本预览: {sample['material_id']}")
    print(f"    atoms: {len(sample['atoms']['elements'])} atoms, "
          f"elements={sample['atoms']['elements'][:6]}...")
    print(f"    {sample['target_key']}: {sample['target_value']:.4f}")

    print("\n=== DONE ===")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[FATAL] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
