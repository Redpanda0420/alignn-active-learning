"""
任务 2.3 · 训练 ALIGNN 预测形成能
==================================
功能：在 RTX 4090 上训练 ALIGNN 模型预测 energy_per_atom，
      支持自动 batch_size 降级，保存最佳模型与评估图表。
输入：data/alignn/formation_energy/{train,val,test}.json
输出：
  - models/alignn_energy_best.pt
  - runs/alignn_energy/           TensorBoard 日志
  - figures/alignn_energy_pred_vs_true.png
  - figures/alignn_energy_loss_curve.png
依赖：alignn conda 环境（torch + dgl + alignn）
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data.sampler import Sampler


class SizeBucketBatchSampler(Sampler):
    """按结构大小（原子数）排序分批的 BatchSampler。

    解决：随机 batch 中若混入多个超大结构（如 168 原子），
    line graph 计算量会爆炸导致训练卡死/显存 OOM。
    思路：同一 batch 内的图大小接近，避免大结构叠加。
    """

    def __init__(self, sizes, batch_size, drop_last=True, shuffle_seed=42):
        self.sizes = np.asarray(sizes)
        self.batch_size = batch_size
        order = np.argsort(self.sizes)  # 小 → 大
        self.batches = []
        for i in range(0, len(order), batch_size):
            batch = order[i:i + batch_size]
            if drop_last and len(batch) < batch_size:
                continue
            self.batches.append(batch.tolist())
        # 打乱 batch 顺序（保持随机性，但 batch 内部大小接近）
        rng = np.random.RandomState(shuffle_seed)
        rng.shuffle(self.batches)

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)

# ====== 项目根目录 ======
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent))
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
FIG_DIR = PROJECT_ROOT / "figures"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ====== 防坑规则 ======
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)


def load_json_data(task_dir, task):
    """加载 ALIGNN 训练/验证/测试 JSON"""
    train = json.load(open(task_dir / "train.json"))
    val = json.load(open(task_dir / "val.json"))
    test = json.load(open(task_dir / "test.json"))
    print(f"[OK] {task}: train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test


def build_loaders(train_data, val_data, test_data, target, batch_size,
                  line_graph=True, cutoff=5.0, cutoff_extra=3.0,
                  max_neighbors=12):
    """自建纯 PyTorch DataLoader（不依赖 DGL/CUDA），
    避免 drop_last=True 导致小验证集为空。
    注意：cutoff=5.0 是 ALIGNN 论文默认值，过大的 cutoff（如 8.0）
    会产生过多边和超大 line graph，导致训练卡死/显存爆炸。"""
    from alignn.pure_lmdb_dataset import get_torch_dataset
    from torch.utils.data import DataLoader

    # 确保临时输出目录存在
    tmp_out = PROJECT_ROOT / "runs" / "_tmp"
    tmp_out.mkdir(parents=True, exist_ok=True)

    common = dict(
        id_tag="jid",
        target=target,
        neighbor_strategy="pure_torch",
        atom_features="cgcnn",
        use_canonize=True,
        line_graph=line_graph,
        cutoff=cutoff,
        cutoff_extra=cutoff_extra,
        max_neighbors=max_neighbors,
        classification=False,
        output_dir=str(tmp_out),
    )

    # 训练集（按结构大小排序分批，避免大结构叠加爆炸）
    # 注意：必须先算 sizes 再调 get_torch_dataset！
    # get_torch_dataset 写 LMDB 后会 dataset[idx]=None 释放内存。
    train_sizes = [len(d["atoms"]["elements"]) for d in train_data]
    train_ds = get_torch_dataset(dataset=train_data, name="train",
                                 tmp_name="train_data", **common)
    collate = train_ds.collate_line_graph if line_graph else train_ds.collate
    train_batch_sampler = SizeBucketBatchSampler(
        train_sizes, batch_size=batch_size, drop_last=True,
        shuffle_seed=RANDOM_STATE,
    )
    train_loader = DataLoader(
        train_ds, batch_sampler=train_batch_sampler,
        collate_fn=collate, num_workers=0,
    )

    # 验证集（小 batch + drop_last=False，避免巨型 batch 计算爆炸）
    val_ds = get_torch_dataset(dataset=val_data, name="val",
                               tmp_name="val_data", **common)
    val_batch = min(16, len(val_data)) if len(val_data) > 0 else 1
    val_loader = DataLoader(
        val_ds, batch_size=val_batch, shuffle=False,
        collate_fn=collate, drop_last=False, num_workers=0,
    )

    # 测试集（batch_size=1 全量）
    test_ds = get_torch_dataset(dataset=test_data, name="test",
                                tmp_name="test_data", **common)
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False,
        collate_fn=collate, drop_last=False, num_workers=0,
    )

    prepare_batch = train_ds.prepare_batch
    return train_loader, val_loader, test_loader, prepare_batch


def main():
    print("=== TASK 2.3: 训练 ALIGNN 预测形成能 ===")

    # ---------- 0. 命令行参数 ----------
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=300, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="批大小")
    parser.add_argument("--hidden", type=int, default=128, help="隐藏层维度")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--tag", type=str, default="",
                        help="模型标签（集成训练用，如 seed1/seed2/seed3）")
    args = parser.parse_args()

    # 集成训练：覆盖随机种子（RANDOM_STATE 是模块级常量，这里用 args.seed）
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ---------- 0. GPU 检查 ----------
    if not torch.cuda.is_available():
        print("[ERROR] CUDA 不可用，无法训练")
        sys.exit(1)
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    if torch.cuda.get_device_name(0) not in ["NVIDIA GeForce RTX 4090"]:
        print("[WARN] 检测到非 4090 GPU，训练可能较慢")

    # ---------- 1. 加载数据 ----------
    task_dir = DATA_DIR / "alignn" / "formation_energy"
    train_data, val_data, test_data = load_json_data(task_dir, "formation_energy")

    # 拼接数据（保持顺序：train+val+test）
    all_data = train_data + val_data + test_data

    # ---------- 2. 训练配置 ----------
    target = "formation_energy_per_atom"
    batch_size = args.batch_size
    epochs = args.epochs
    run_tag = args.tag if args.tag else ""
    out_dir_name = f"alignn_energy_{run_tag}" if run_tag else "alignn_energy"
    model_out_name = f"alignn_energy_{run_tag}_best.pt" if run_tag else "alignn_energy_best.pt"

    config = {
        "dataset": "user_data",
        "target": target,
        "id_tag": "jid",
        "atom_features": "cgcnn",
        "neighbor_strategy": "k-nearest",
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": 0.001,
        "optimizer": "adamw",
        "scheduler": "onecycle",
        "criterion": "mse",
        "weight_decay": 1e-5,
        "num_workers": 0,
        "pin_memory": False,
        "use_lmdb": False,
        "random_seed": seed,
        "n_early_stopping": 50,
        "keep_data_order": True,
        "log_tensorboard": True,
        "output_dir": str(PROJECT_ROOT / "runs" / out_dir_name),
        "model": {
            "name": "alignn_atomwise_pure",
            "alignn_layers": 4,
            "gcn_layers": 4,
            "embedding_features": 64,
            "hidden_features": args.hidden,
            "output_features": 1,
            "atom_input_features": 92,  # CGCNN 特征（纯 torch 图默认）
            "calculate_gradient": False,
            "graphwise_weight": 1.0,
            "atomwise_weight": 0.0,
            "gradwise_weight": 0.0,
            "stresswise_weight": 0.0,
            "energy_mult_natoms": False,
            "lg_on_fly": False,  # 关键：用预计算键角特征，避免每次前向重算
        },
    }

    # ---------- 3. 构建 loader 并训练（含 OOM 自动降级） ----------
    from alignn.train import train_dgl

    # 训练循环（OOM 降级）
    for bs in [batch_size, 32, 16]:
        try:
            print(f"\n[STEP] batch_size={bs} 开始训练...")
            train_loader, val_loader, test_loader, prepare_batch = build_loaders(
                train_data, val_data, test_data, target, batch_size=bs,
            )
            config["batch_size"] = bs

            result = train_dgl(
                config,
                train_val_test_loaders=[
                    train_loader, val_loader, test_loader, prepare_batch,
                ],
            )
            break
        except torch.cuda.OutOfMemoryError:
            print(f"[WARN] OOM at batch_size={bs}，降到下一档重试...")
            torch.cuda.empty_cache()
            continue
        except Exception as e:
            if "out of memory" in str(e).lower():
                print(f"[WARN] OOM at batch_size={bs}，降到下一档重试...")
                torch.cuda.empty_cache()
                continue
            raise

    # ---------- 4. 结果处理 ----------
    print("\n[STEP] 读取训练结果...")
    result_dir = Path(config["output_dir"])

    # 训练历史（loss 曲线）：list of [total, graph, atom, grad, stress, addn]
    history_train = json.load(open(result_dir / "history_train.json"))
    history_val = json.load(open(result_dir / "history_val.json"))
    train_losses = [h[0] for h in history_train]
    val_losses = [h[0] for h in history_val]

    # 测试集预测结果：list of {id, target_out, pred_out}
    test_results = json.load(open(result_dir / "Test_results.json"))
    preds = np.array([r["pred_out"][0] for r in test_results])
    targets = np.array([r["target_out"][0] for r in test_results])
    ids = [r["id"] for r in test_results]

    # MAD
    mad_val = np.mean(np.abs(targets - np.mean(targets)))

    # 评估指标
    mae = np.mean(np.abs(preds - targets))
    rmse = np.sqrt(np.mean((preds - targets) ** 2))
    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - ss_res / ss_tot

    print(f"\n{'='*55}")
    print(f"  ALIGNN 形成能模型评估（测试集 {len(targets)} 条）")
    print(f"{'='*55}")
    print(f"  MAE:  {mae:.4f} eV/atom")
    print(f"  RMSE: {rmse:.4f} eV/atom")
    print(f"  R2:   {r2:.4f}")
    print(f"  MAD:  {mad_val:.4f} eV/atom")
    print(f"  MAE/MAD: {mae/mad_val:.3f}")
    print(f"{'='*55}")

    if mae < mad_val:
        print("[OK] MAE < MAD，模型具有预测价值")
    else:
        print("[WARN] MAE >= MAD，模型无预测价值！")

    # ---------- 5. 保存最佳模型 ----------
    import shutil
    ckpt_path = result_dir / "best_model.pt"
    best_path = MODEL_DIR / model_out_name
    if ckpt_path.exists():
        shutil.copy(ckpt_path, best_path)
        print(f"\n[OK] 最佳模型已保存至 {best_path}")
    else:
        print(f"[WARN] 未找到 {ckpt_path}，模型未复制")

    # ---------- 6. 画图 ----------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 150, "font.size": 11})

    # 6.1 预测 vs 真实
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(targets, preds, alpha=0.6, edgecolors="k", linewidth=0.3, s=40)
    lims = [min(targets.min(), preds.min()) - 0.2,
            max(targets.max(), preds.max()) + 0.2]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="y = x")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("True energy (eV/atom)")
    ax.set_ylabel("Predicted energy (eV/atom)")
    ax.set_title(f"ALIGNN: Predicted vs True (MAE={mae:.4f}, $R^2$={r2:.4f})")
    ax.legend(loc="upper left")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "alignn_energy_pred_vs_true.png")
    plt.close(fig)
    print("  [OK] figures/alignn_energy_pred_vs_true.png")

    # 6.2 Loss 曲线
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_losses, label="train loss", color="steelblue", linewidth=1.5)
    ax.plot(val_losses, label="val loss", color="crimson", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title("ALIGNN Training Loss Curve")
    ax.legend()
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "alignn_energy_loss_curve.png")
    plt.close(fig)
    print("  [OK] figures/alignn_energy_loss_curve.png")

    print("\n=== DONE ===")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[FATAL] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
