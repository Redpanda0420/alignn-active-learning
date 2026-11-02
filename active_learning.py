"""
任务 3.1 · 主动学习：候选结构生成 + 不确定性采样
====================================================
功能：基于优化后的 LiFePO4 1x2x2 超胞生成掺杂/缺陷候选结构，
      用 ALIGNN 模型预测形成能，输出待 VASP 计算的候选清单。
输入：
  - data/candidates/LiFePO4_122_opt.cif   优化后超胞
  - models/alignn_energy_best.pt          形成能模型
输出：
  - data/active_learning/round1_query.csv  候选清单（含预测值）
  - data/candidates/*.cif                  生成的候选结构
依赖：alignn conda 环境（pymatgen + torch）
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
AL_DIR.mkdir(parents=True, exist_ok=True)
CAND_DIR.mkdir(parents=True, exist_ok=True)

# 母体结构（优化后的 1x2x2 超胞）
PARENT_CIF = CAND_DIR / "LiFePO4_122_opt.cif"

# 掺杂元素（替换 Fe 位）与替换 Li 位
DOPANTS_FE = ["Mg", "Mn", "Co", "Ni", "Na", "K", "Al", "V", "Ti",
              "Zn", "Ca", "Cu", "Cr", "Nb"]
DOPANTS_LI = ["Na", "K", "Mg", "Ca"]

np.random.seed(42)
torch.manual_seed(42)


def load_parent():
    """加载优化后的母体超胞"""
    from pymatgen.core import Structure
    if not PARENT_CIF.exists():
        print(f"[ERROR] 未找到母体结构: {PARENT_CIF}")
        sys.exit(1)
    struct = Structure.from_file(str(PARENT_CIF))
    print(f"[OK] 母体: {struct.composition.formula} "
          f"({len(struct)} 原子)")
    return struct


def generate_candidates(parent):
    """生成大规模掺杂/缺陷候选结构（目标 100+）"""
    from pymatgen.core import Structure

    fe_indices = [i for i, s in enumerate(parent) if s.specie.symbol == "Fe"]
    li_indices = [i for i, s in enumerate(parent) if s.specie.symbol == "Li"]
    p_indices = [i for i, s in enumerate(parent) if s.specie.symbol == "P"]
    candidates = []

    # ---- 1. Fe 位单掺 (1/16 = 6.25%) ----
    for dop in DOPANTS_FE:
        c = parent.copy()
        c.replace(fe_indices[0], dop)
        candidates.append({
            "id": f"Fe_{dop}x1",
            "desc": f"Fe 位掺 {dop} (1/16)",
            "structure": c,
        })

    # ---- 2. Fe 位双掺 (2/16 = 12.5%)，多种构型 ----
    for dop in DOPANTS_FE:
        for conf, idxs in enumerate([[0, 1], [0, 4], [0, 8]]):
            c = parent.copy()
            for idx in idxs:
                c.replace(fe_indices[idx], dop)
            candidates.append({
                "id": f"Fe_{dop}x2_c{conf}",
                "desc": f"Fe 位掺 {dop} x2 (2/16, 构型{conf})",
                "structure": c,
            })

    # ---- 3. Fe 位三掺 (3/16 = 18.75%)，常见元素 ----
    for dop in ["Mg", "Mn", "Co", "Ni", "Na"]:
        for conf, idxs in enumerate([[0, 1, 2], [0, 4, 8]]):
            c = parent.copy()
            for idx in idxs:
                c.replace(fe_indices[idx], dop)
            candidates.append({
                "id": f"Fe_{dop}x3_c{conf}",
                "desc": f"Fe 位掺 {dop} x3 (3/16, 构型{conf})",
                "structure": c,
            })

    # ---- 4. Li 位掺杂 (1/16) ----
    for dop in DOPANTS_LI:
        c = parent.copy()
        c.replace(li_indices[0], dop)
        candidates.append({
            "id": f"Li_{dop}x1",
            "desc": f"Li 位掺 {dop} (1/16)",
            "structure": c,
        })

    # ---- 5. 缺陷：空位 ----
    for el, idx_list, name in [("Li", li_indices, "V_Li"),
                               ("Fe", fe_indices, "V_Fe"),
                               ("P", p_indices, "V_P")]:
        c = parent.copy()
        c.remove_sites([idx_list[0]])
        candidates.append({
            "id": name,
            "desc": f"{el} 空位",
            "structure": c,
        })

    # ---- 5b. Li-Fe 反位缺陷（LiFePO4 最著名缺陷！）----
    # Li 位换成 Fe + 附近 Fe 位换成 Li（antisite）
    for conf, (li_i, fe_i) in enumerate([(0, 0), (0, 4), (4, 0)]):
        c = parent.copy()
        c.replace(li_indices[li_i], "Fe")
        c.replace(fe_indices[fe_i], "Li")
        candidates.append({
            "id": f"AS_li{li_i}fe{fe_i}",
            "desc": f"Li-Fe 反位 (构型{conf})",
            "structure": c,
        })

    # ---- 5c. 双 Li 空位（近邻/远邻构型）----
    for conf, idxs in enumerate([[0, 1], [0, 8]]):
        c = parent.copy()
        c.remove_sites([li_indices[i] for i in idxs])
        candidates.append({
            "id": f"V2_Li_c{conf}",
            "desc": f"双 Li 空位 (构型{conf})",
            "structure": c,
        })

    # ---- 5d. Li 间隙缺陷（在晶胞空隙插入 Li）----
    # 用 pymatgen 找最大空隙位置插入 Li
    try:
        from pymatgen.analysis.structure_matcher import StructureMatcher
        # 简化：在晶胞中心附近找离所有原子最远的点
        c = parent.copy()
        import numpy as np
        # 在分数坐标网格上找最大空隙
        frac_grid = np.array([[x, y, z] for x in np.linspace(0, 1, 8)
                              for y in np.linspace(0, 1, 8)
                              for z in np.linspace(0, 1, 8)])
        cart_grid = c.lattice.get_cartesian_coords(frac_grid)
        best_d = -1
        best_p = None
        for p in cart_grid:
            d = min(c.lattice.get_distance_and_image(p, site.coords)[0]
                    for site in c)
            if d > best_d:
                best_d, best_p = d, p
        if best_p is not None and best_d > 1.5:  # 距原子至少 1.5 Å
            c.append("Li", best_p, coords_are_cartesian=True)
            candidates.append({
                "id": "Li_interstitial",
                "desc": f"Li 间隙 (最近距 {best_d:.2f} Å)",
                "structure": c,
            })
    except Exception as e:
        print(f"  [WARN] Li 间隙生成失败: {e}")

    # ---- 5e. 空位 + 反位组合 ----
    c = parent.copy()
    c.remove_sites([li_indices[0]])
    c.replace(li_indices[1], "Fe")
    c.replace(fe_indices[0], "Li")
    candidates.append({
        "id": "V_Li_AS",
        "desc": "Li 空位 + Li-Fe 反位",
        "structure": c,
    })

    # ---- 6. 组合掺杂（共掺两种元素）----
    for d1, d2 in [("Mg", "Mn"), ("Co", "Ni"), ("Mg", "Co"),
                   ("Mn", "Ni"), ("Al", "V"), ("Na", "Mg")]:
        c = parent.copy()
        c.replace(fe_indices[0], d1)
        c.replace(fe_indices[4], d2)
        candidates.append({
            "id": f"Fe_{d1}+{d2}",
            "desc": f"Fe 位共掺 {d1}+{d2}",
            "structure": c,
        })

    # ---- 7. Li 空位 + Fe 位掺杂组合（电荷补偿机制）----
    # 关键物理：高价掺杂（Al3+/V3+/Cr3+/Ti4+/Nb5+）替换 Fe2+ 时多出的
    # 正电荷，靠 Li 空位（V_Li'）补偿——这是这些掺杂在 LiFePO4 中的
    # 真实存在形态。同价 2+ 掺杂（Mg/Mn/Co/Ni/Zn/Ca/Cu）实验上也可能
    # 伴随缺陷，一并生成以便对比。
    for dop in DOPANTS_FE:
        c = parent.copy()
        c.remove_sites([li_indices[0]])
        c.replace(fe_indices[0], dop)
        candidates.append({
            "id": f"V_Li_Fe{dop}",
            "desc": f"Li 空位 + Fe 位掺 {dop}（电荷补偿）",
            "structure": c,
        })

    print(f"[OK] 生成 {len(candidates)} 个候选结构")
    return candidates


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


def load_energy_model():
    """加载形成能模型"""
    from alignn.models.alignn_atomwise_pure import (
        ALIGNNAtomWisePure, ALIGNNAtomWisePureConfig,
    )
    ckpt = MODEL_DIR / "alignn_energy_best.pt"
    cfg_path = PROJECT_ROOT / "runs" / "alignn_energy" / "config.json"
    cfg = json.load(open(cfg_path))
    model_cfg = ALIGNNAtomWisePureConfig(**cfg["model"])
    model = ALIGNNAtomWisePure(model_cfg)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model, cfg


def predict_energy(model, structure, model_cfg, device):
    """预测单个结构的形成能"""
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
    print("=== TASK 3.1: 候选结构生成 + 不确定性采样 ===")

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_cif", action="store_true",
                        help="把候选结构保存为 CIF（默认保存）")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---------- 1. 加载母体 + 生成候选 ----------
    parent = load_parent()
    candidates = generate_candidates(parent)

    # ---------- 2. 保存候选 CIF ----------
    for c in candidates:
        cif_path = CAND_DIR / f"{c['id']}.cif"
        c["structure"].to(filename=str(cif_path), fmt="cif")
    print(f"[OK] 候选 CIF 已保存到 {CAND_DIR}/")

    # ---------- 3. 用形成能模型预测 ----------
    print("\n[STEP] 用 ALIGNN 形成能模型预测...")
    model_e, cfg_e = load_energy_model()

    results = []
    for c in tqdm(candidates, desc="预测"):
        try:
            e_pred = predict_energy(model_e, c["structure"], cfg_e, device)
            results.append({
                "candidate_id": c["id"],
                "description": c["desc"],
                "predicted_energy": e_pred,
                "n_atoms": len(c["structure"]),
            })
        except Exception as e:
            print(f"  [WARN] {c['id']} 预测失败: {e}")
            continue

    if len(results) == 0:
        print("[ERROR] 无成功预测")
        sys.exit(1)

    df = pd.DataFrame(results)

    # ---------- 4. 排序输出 ----------
    # 最稳定（形成能最低）的候选
    df = df.sort_values("predicted_energy")

    out_csv = AL_DIR / "round1_query.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[OK] 候选清单已保存: {out_csv}")

    print(f"\n{'='*65}")
    print(f"  候选结构预测（按形成能升序 = 稳定性）")
    print(f"{'='*65}")
    print(df.to_string(index=False, float_format="%.4f"))
    print(f"{'='*65}")

    print(f"\n  预测形成能范围: {df['predicted_energy'].min():.3f} ~ "
          f"{df['predicted_energy'].max():.3f} eV/atom")

    # ---------- 5. 关于不确定性 ----------
    print(f"\n{'='*65}")
    print("  关于不确定性采样")
    print(f"{'='*65}")
    print("  当前为单模型预测（不确定性未包含）。")
    print("  严格的不确定性需要模型集成（多个随机种子训练）。")
    print("  如需，可训练 3 个种子模型后用标准差排序（真正的不确定性采样）。")

    print("\n=== DONE ===")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[FATAL] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
