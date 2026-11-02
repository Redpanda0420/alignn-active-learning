"""
主动学习 · DFT 结果解析器（V100/P40 数据接收端）
================================================
功能：扫描回传的 VASP 结果目录，解析 OUTCAR 总能量，按 MP 定义
      计算 formation_energy_per_atom，输出可直接合并训练集的数据表。
输入（回传格式，见 VASP_DOPING_GUIDE.md / VASP_DEFECT_GUIDE.md）：
  - data/dft/doping/<candidate_id>/OUTCAR     （V100 掺杂）
  - data/dft/defects/<candidate_id>/OUTCAR    （P40 缺陷）
  - data/dft/references.json                  （可选：元素参考态化学势）
输出：
  - data/dft/dft_results.csv    统一结果表（含 formation_energy_per_atom）
依赖：ai4sci conda 环境（pymatgen）
"""

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
DFT_DIR = PROJECT_ROOT / "data" / "dft"
CAND_DIR = PROJECT_ROOT / "data" / "candidates"
REFERENCES_FILE = DFT_DIR / "references.json"

# 各任务类型对应的结构来源目录
TASK_TYPES = {"doping": "doping", "defects": "defects"}


def parse_outcar(outcar_path):
    """从 OUTCAR 提取最终总能（eV）。返回 None 表示失败。"""
    from pymatgen.io.vasp import Outcar
    try:
        outcar = Outcar(str(outcar_path))
        if outcar.final_energy is not None:
            return outcar.final_energy
        # 兼容：取最后一个离子步能量
        if outcar.ionic_steps:
            return outcar.ionic_steps[-1]["e_0_energy_cont"]
    except Exception as e:
        print(f"  [WARN] 解析失败 {outcar_path}: {e}")
    return None


def load_references():
    """加载元素参考态化学势 {element: eV/atom}。缺失时返回 {}。"""
    if not REFERENCES_FILE.exists():
        print(f"\n[WARN] 未找到参考态文件 {REFERENCES_FILE}")
        print("       将只输出 E_total，不计算 formation_energy_per_atom。")
        print("       参考态模板：")
        print("       { \"Li\": -1.90, \"Fe\": -8.32, \"P\": -5.39, \"O\": -4.94, \"Mg\": -1.53, ... }")
        print("       （用 V100 算 Li_bcc/Fe_bcc/P_ortho/O2 分子的每原子总能填入）")
        return {}
    with open(REFERENCES_FILE) as f:
        refs = json.load(f)
    print(f"[OK] 参考态化学势: {refs}")
    return refs


def formation_energy_per_atom(total_e, comp_dict, refs):
    """按 MP 定义计算 formation energy per atom：
    E_f = (E_total - Σ n_i·μ_i) / N
    comp_dict: {元素符号: 原子数}
    """
    if not refs:
        return None
    missing = [el for el in comp_dict if el not in refs]
    if missing:
        print(f"  [WARN] 缺少参考态: {missing}，跳过形成能")
        return None
    n_total = sum(comp_dict.values())
    e_ref = sum(n * refs[el] for el, n in comp_dict.items())
    return (total_e - e_ref) / n_total


def main():
    print("=== 解析 DFT 回传数据 ===")
    if not DFT_DIR.exists():
        print(f"[ERROR] 未找到 {DFT_DIR}，请先按指导书回传数据")
        sys.exit(1)

    refs = load_references()

    rows = []
    for task in TASK_TYPES:
        task_dir = DFT_DIR / task
        if not task_dir.exists():
            continue
        for job_dir in sorted(task_dir.iterdir()):
            if not job_dir.is_dir():
                continue
            cid = job_dir.name
            outcar_path = job_dir / "OUTCAR"
            # 优先用 CONTCAR（若有），否则用候选 CIF 作为结构
            contcar_path = job_dir / "CONTCAR"
            struct = None
            from pymatgen.core import Structure
            if contcar_path.exists():
                try:
                    struct = Structure.from_file(str(contcar_path))
                except Exception as e:
                    print(f"  [WARN] CONTCAR 解析失败 {cid}: {e}")
            if struct is None:
                cif_path = CAND_DIR / f"{cid}.cif"
                if cif_path.exists():
                    struct = Structure.from_file(str(cif_path))
                else:
                    print(f"  [WARN] 无结构文件: {cid}")
                    continue

            e_total = parse_outcar(outcar_path) if outcar_path.exists() else None
            if e_total is None:
                print(f"  [WARN] {cid}: 无有效 OUTCAR 能量，跳过")
                continue

            comp_dict = {el.symbol: int(n) for el, n in
                         struct.composition.items()}
            e_f = formation_energy_per_atom(e_total, comp_dict, refs)

            rows.append({
                "task": task,
                "candidate_id": cid,
                "e_total": e_total,
                "n_atoms": len(struct),
                "composition": struct.composition.formula,
                "formation_energy_per_atom": e_f,
            })
            print(f"  [OK] {task}/{cid}: E={e_total:.3f} eV, "
                  f"{'Ef=' + format(e_f, '.4f') if e_f is not None else '无参考态'}")

    if not rows:
        print("[ERROR] 未解析到任何 DFT 结果")
        sys.exit(1)

    df = pd.DataFrame(rows)
    out_csv = DFT_DIR / "dft_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[OK] 结果表已保存: {out_csv} ({len(df)} 条)")

    # 摘要
    n_with_ef = df["formation_energy_per_atom"].notna().sum()
    print(f"[OK] 有形成能标签: {n_with_ef}/{len(df)} 条")
    if n_with_ef > 0:
        valid = df.dropna(subset=["formation_energy_per_atom"])
        print(f"  形成能范围: {valid['formation_energy_per_atom'].min():.4f} ~ "
              f"{valid['formation_energy_per_atom'].max():.4f} eV/atom")
    print("\n=== DONE ===")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[FATAL] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
