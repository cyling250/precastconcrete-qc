"""用保存的最佳模型评估测试集"""
import sys, os, json, torch, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.pointnet2 import PointNetPlusPlus, PerSampleWeightedCE
from lib.pointnet2.dataset import create_dataloaders, _clean_shm
from lib.pointnet2.metrics import compute_iou, compute_precision_recall_f1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_POINTS = 8192
BATCH_SIZE = 8
NUM_CLASSES = 3
MAX_WEIGHT = 15.0
NOISE_WEIGHT = 0.5
CLASS_NAMES = {0: "噪声", 1: "混凝土", 2: "钢筋"}

def validate(model, make_loader, criterion):
    """与 experiment 中 validate() 相同的逻辑, 返回 (loss, ious, acc, prf, clean_miou)"""
    loader = make_loader()
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for batch in loader:
            pts, geo, labels, scale = batch  # tuple unpack
            pts = pts.to(DEVICE).float()       # (B, N, 3)
            geo = geo.to(DEVICE).float()       # (B, N, 3)
            labels = labels.to(DEVICE).long()  # (B, N)
            scale = scale.to(DEVICE).float()    # (B,)

            logits = model(pts, geo_features=geo, scale=scale)  # (B, N, C)
            loss = criterion(logits.permute(0, 2, 1), labels)
            total_loss += loss.item()

            preds = logits.argmax(dim=-1)  # (B, N)
            all_preds.append(preds.cpu())
            all_targets.append(labels.cpu())

    preds = torch.cat(all_preds, dim=0).flatten()
    targets = torch.cat(all_targets, dim=0).flatten()

    ious = compute_iou(preds, targets, CLASS_NAMES)
    prf = compute_precision_recall_f1(preds, targets, CLASS_NAMES)
    acc = (preds == targets).float().mean().item()

    # 干净 mIoU（仅混凝土+钢筋）
    clean_ious = [v for k, v in ious.items() if k not in ("mIoU", "噪声")]
    clean_miou = float(np.mean(clean_ious)) if clean_ious else 0.0

    return total_loss / len(loader), ious, float(acc), prf, clean_miou


def main():
    print(f"Device: {DEVICE}")
    _clean_shm()

    # ── 创建 dataloaders (只需要 test loader) ──
    print("正在初始化数据管道...")
    _, _, make_test_loader = create_dataloaders(
        num_points=NUM_POINTS,
        batch_size=BATCH_SIZE,
        train_ratio=0.6,
        val_ratio=0.2,
        keep_ratio=0.4,
        use_double_buffer=True,
        only_test=True,
    )

    # ── 加载模型 ──
    model = PointNetPlusPlus(num_classes=NUM_CLASSES).to(DEVICE)
    ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "..", "checkpoints", "best_pointnet2.pth")
    ckpt = torch.load(ckpt_path, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    print(f"加载模型: epoch={ckpt['epoch']}, val_mIoU={ckpt.get('val_miou', '?')}")

    criterion = PerSampleWeightedCE(max_weight=MAX_WEIGHT,
                                     noise_weight=NOISE_WEIGHT)

    # ── 测试集评估 ──
    print("\n" + "=" * 55)
    print("  测试集评估")
    print("=" * 55)
    test_loss, test_ious, test_acc, test_prf, test_clean_miou = validate(
        model, make_test_loader, criterion
    )

    print(f"  test_loss={test_loss:.4f}")
    print(f"  test_acc={test_acc:.4f}")
    print(f"  clean_mIoU={test_clean_miou:.4f}")
    print(f"  mIoU={test_ious['mIoU']:.4f}")
    for name in ["噪声", "混凝土", "钢筋"]:
        print(f"    {name} IoU: {test_ious[name]:.4f}")
    print(f"  PRF:")
    for name, m in test_prf.items():
        print(f"    {name}: P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f}")

    # ── 保存结果 ──
    result = {
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_clean_miou": test_clean_miou,
        "test_ious": test_ious,
        "test_prf": test_prf,
    }
    out_dir = os.path.dirname(ckpt_path)
    with open(os.path.join(out_dir, "test_metrics.json"), "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n结果保存至: {out_dir}/test_metrics.json")


if __name__ == "__main__":
    main()
