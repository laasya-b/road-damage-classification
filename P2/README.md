# P2: Baseline (frozen ResNet-50 + MLP head)

## Run

    python P2/train_baseline.py --csv P1/splits.csv --data-root <patches_dir> --out <out_dir>

Defaults: ImageNet ResNet-50 (fully frozen), MLP head 2048→256→4 (dropout 0.3), class-weighted cross-entropy,
AdamW (head lr 1e-3, wd 1e-4), batch 32, up to 20 epochs, early stopping on dev macro-F1 (patience 5), seed 0.
Augmentation: horizontal/vertical flips, ±10° rotation, brightness ±20%. No 90° rotations, since they would turn
a longitudinal crack (D00) into a transverse one (D10).

**P3 (layer-4 comparison):** same command plus `--unfreeze-layer4` (layer-4 lr 1e-4); keep everything else the same.

## Dev results (best epoch 16)

| Metric | Value |
|---|---|
| Macro-F1 | 0.742 |
| Accuracy | 0.744 |
| Recall D00 / D10 / D20 / D40 | 0.72 / 0.77 / 0.70 / 0.83 |
| Macro-F1 India / Japan / Norway | 0.58 / 0.70 / 0.61 |

Main error: over-predicting D10 (17% of D00 and 16% of D20 dev patches → D10; D10 precision ≈ 0.53).

## Files

- `train_baseline.py`: training script
- `dev_metrics.json`: dev metrics, confusion matrix, full config
- `history.csv`: per-epoch losses and dev F1
- `dev_logits.csv`: raw dev logits per patch (for P4 temperature scaling and review threshold)
- `best_model.pt`: in Google Drive (too large for GitHub): https://drive.google.com/drive/folders/1COXnTPZ_V__AVdtL9gAnlkdG6uQseZ_e?usp=sharing
