"""
任务 3.3 · 主动学习循环编排器
================================
功能：把「DFT 数据回传 → 合并训练集 → 重训 ALIGNN → 生成新候选 →
      集成采样 → 输出下一批 VASP 输入」串成一条可重复的流水线。
用法（每轮数据回来后）：
  conda activate alignn
  python active_loop.py --round 2 --parse --merge --train --sample --vasp
参数：
  --round N      当前轮次（默认 2，第 1 轮已手工完成）
  --parse        解析 DFT 回传数据（需 data/dft/<task>/<id>/OUTCAR）
  --merge        把新 DFT 标签合并进训练集 train.json
  --train        重训形成能模型（--n_models 个种子做集成）
  --sample       集成不确定性采样（覆盖 round1_query.csv）
  --vasp         生成下一批掺杂 VASP 输入（vasp_jobs/doping_roundN）
  --n_models N   训练模型数（默认 1；>=3 才适合集成）
依赖：alignn conda 环境；需先按 VASP_DOPING_GUIDE 回传数据
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
ALIGNN_DIR = DATA_DIR / "alignn"
CAND_DIR = DATA_DIR / "candidates"
AL_DIR = DATA_DIR / "active_learning"
DFT_DIR = DATA_DIR / "dft"
MODEL_DIR = PROJECT_ROOT / "models"


def run(script, *args):
    """用当前 Python 解释器运行子脚本"""
    print(f"\n>>> python {Path(script).name} {' '.join(args)}")
    r = subprocess.run([sys.executable, str(PROJECT_ROOT / script), *args])
    if r.returncode != 0:
        print(f"[FATAL] {script} 失败 (exit {r.returncode})")
        sys.exit(1)


# ---------- 1. 解析 DFT 数据 ----------
def step_parse():
    run("parse_dft_results.py")


# ---------- 2. 合并到训练集 ----------
def step_merge():
    print("\n=== 合并 DFT 数据到训练集 ===")
    csv_path = DFT_DIR / "dft_results.csv"
    if not csv_path.exists():
        print(f"[ERROR] 未找到 {csv_path}，先跑 --parse")
        sys.exit(1)
    import pandas as pd
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["formation_energy_per_atom"])
    if len(df) == 0:
        print("[ERROR] 无带形成能标签的数据（缺参考态？）")
        sys.exit(1)
    print(f"[OK] 可合并: {len(df)} 条")

    # 读现有训练集
    train_json = ALIGNN_DIR / "formation_energy" / "train.json"
    train_data = json.load(open(train_json))
    existing_jids = {d["jid"] for d in train_data}
    print(f"[OK] 现有训练集: {len(train_data)} 条")

    # 结构 → atoms dict
    from pymatgen.core import Structure
    new_records = []
    added = 0
    for _, row in df.iterrows():
        cid = row["candidate_id"]
        cif_path = CAND_DIR / f"{cid}.cif"
        if not cif_path.exists():
            print(f"  [WARN] 无 CIF: {cid}，跳过")
            continue
        jid = f"dft_{cid}"
        if jid in existing_jids:
            print(f"  [SKIP] {cid} 已存在")
            continue
        struct = Structure.from_file(str(cif_path))
        atoms = {
            "lattice_mat": struct.lattice.matrix.tolist(),
            "coords": struct.cart_coords.tolist(),
            "elements": [sp.symbol for sp in struct.species],
            "props": [sp.Z for sp in struct.species],
            "cartesian": True,
        }
        new_records.append({
            "jid": jid,
            "atoms": atoms,
            "formation_energy_per_atom": row["formation_energy_per_atom"],
        })
        added += 1

    train_data.extend(new_records)
    json.dump(train_data, open(train_json, "w"), indent=2)
    print(f"[OK] 新增 {added} 条 → 训练集共 {len(train_data)} 条")
    # 备份原始（防误操作）
    shutil.copy(train_json, ALIGNN_DIR / "formation_energy" /
                "train_before_merge.json")


# ---------- 3. 重训模型 ----------
def step_train(round_no, n_models):
    print(f"\n=== 第 {round_no} 轮重训（{n_models} 个模型）===")
    seeds = [42, 1, 2, 3]
    for i in range(n_models):
        seed = seeds[i % len(seeds)]
        tag = f"r{round_no}_s{seed}"
        run("train_alignn_energy.py", "--seed", str(seed),
            "--tag", tag, "--epochs", "300")
    # 把当前轮主模型拷为通用名（后续采样脚本用）
    # 采样脚本读 models/alignn_energy_best.pt + seed1/2/3，这里提示
    print(f"[INFO] 集成采样将使用 models/alignn_energy_best.pt + seed1/2/3")
    print(f"       本轮模型为 alignn_energy_{'r'}{round_no}_s42_best.pt 等")
    print(f"       如需用本轮模型做集成，请手动更新 ENSEMBLE_MODELS 列表")


# ---------- 4. 集成采样 ----------
def step_sample(round_no):
    run("ensemble_uncertainty.py", "--top_k", "60")
    # 备份本轮清单
    src = AL_DIR / "round1_query.csv"
    dst = AL_DIR / f"round{round_no}_query.csv"
    shutil.copy(src, dst)
    print(f"[OK] 本轮清单备份: {dst}")


# ---------- 5. 生成下一批 VASP 输入 ----------
def step_vasp(round_no):
    out_dir = PROJECT_ROOT / "vasp_jobs" / f"doping_round{round_no}"
    run("generate_doping_inputs.py", "--top_k", "60",
        "--unique_composition", "--out_dir", str(out_dir))
    # 打包
    zip_path = PROJECT_ROOT / f"vasp_jobs_doping_round{round_no}.zip"
    if zip_path.exists():
        zip_path.unlink()
    print(f"\n[INFO] 请手工打包: Compress-Archive {out_dir}\\* {zip_path}")


def main():
    parser = argparse.ArgumentParser(description="主动学习循环")
    parser.add_argument("--round", type=int, default=2)
    parser.add_argument("--parse", action="store_true")
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--vasp", action="store_true")
    parser.add_argument("--n_models", type=int, default=1)
    parser.add_argument("--all", action="store_true", help="执行全部步骤")
    args = parser.parse_args()

    print(f"=== 主动学习 第 {args.round} 轮 ===")
    if args.all:
        args.parse = args.merge = args.train = args.sample = args.vasp = True

    if args.parse:
        step_parse()
    if args.merge:
        step_merge()
    if args.train:
        step_train(args.round, args.n_models)
    if args.sample:
        step_sample(args.round)
    if args.vasp:
        step_vasp(args.round)

    print(f"\n=== 第 {args.round} 轮流水线完成 ===")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[FATAL] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
