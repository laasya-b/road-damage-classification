# P3: Layer-4 fine-tuning comparison

P3 compares the frozen ResNet-50 + MLP baseline with training ResNet layer4 + the same MLP, using paired seeds 0, 1, and 2. All runs use `P2/train_baseline.py` unchanged. Training starts from the same ImageNet weights for both variants; the fine-tuned run does not resume from the trained P2 checkpoint.

## Completed results

Three paired GPU runs give mean dev macro-F1 **0.7414 +/- 0.0017 frozen** versus **0.8190 +/- 0.0038 layer4** (sample standard deviation). Mean D40 recall is 0.8640 versus 0.8782, with a decline in one seed. Layer4 seed 0 is the prespecified P4 handoff. The final test split remains unevaluated.

See the [full report](results/report.md), [visual error review](results/visual_review.md), and [verification record](results/verification.json). All six runs' metrics were recomputed from saved logits, and checkpoint comparisons confirmed that earlier backbone tensors stayed unchanged.

## Run locally or on a Colab GPU

Run from the repository root in a Python environment with a CUDA-enabled PyTorch installation:

```bash
python -m pip install -r P3/requirements.txt
python P3/run_comparison.py --data-root /path/to/patches --check-only
python P3/run_comparison.py --data-root /path/to/patches
```

The patch root must directly contain `India/`, `Japan/`, and `Norway/`. The archive already used by P2 is [rdd_patches.zip](https://drive.google.com/file/d/17hruEEXaUr4MFu_HD6nwN9BnqPYTB3gk/view). The runner checks all train/dev files and source-photo separation before training. Test membership is checked for leakage, but test images are never loaded.

For this workspace, the environment is `../.venv-p3` and patches are in `../.p3-data`:

```bash
../.venv-p3/bin/python P3/run_comparison.py --data-root ../.p3-data
```

To resume, repeat the same command with `--resume`. Completed runs are skipped only if the manifest matches the split hash, training code hash, hyperparameters, seeds, environment versions, and data root. An interrupted individual run restarts from its initialization. Use a new `--out` directory if changing the experiment settings.

The executable notebook is `P3/layer4_comparison.ipynb`; it also supports uploading the three local P3 scripts before they are pushed to GitHub.

## Controls

| Setting | Both variants |
|---|---|
| Backbone | torchvision ImageNet ResNet-50, IMAGENET1K_V2 |
| Head | 2048 -> 256 -> 4, ReLU, dropout 0.3 |
| Loss | Training-class-weighted cross-entropy |
| Optimizer | AdamW; head lr 1e-3; weight decay 1e-4 |
| Training | Batch 32, max 20 epochs, early stopping patience 5 on dev macro-F1 |
| Split | P1/splits.csv, fixed across all seeds |
| Augmentation | Horizontal/vertical flips, +/-10 degree rotation, brightness +/-20% |
| Seeds | 0, 1, 2, paired across variants |

The only within-pair change is unfreezing layer4, including its BatchNorm statistics, with learning rate 1e-4. Earlier layers stay frozen and in eval mode. No 90-degree rotations are used because these can swap longitudinal and transverse crack semantics. The stopping rule is shared; the actual best and stopping epochs can differ.

## Results and handoff

The runner generates `results/report.md`, `comparison.csv`, `paired_deltas.csv`, `summary.json`, learning curves, confusion matrices, and dev errors. It reports mean and sample standard deviation of macro-F1, accuracy, per-class recall, D40 recall, and per-country F1. Paired deltas are layer4 minus frozen for the same seed.

Architecture selection uses higher mean dev macro-F1 (ties favor frozen). The checkpoint handed to P4 uses the first prespecified seed, 0, rather than choosing the luckiest seed. D40 recall is reported separately so the team can assess the pothole tradeoff. Three seeds on one split describe training variability and do not establish significance or cross-country generalization.

`results/selected_dev_logits.csv` and `selected_dev_metrics.json` are the P4 inputs. `summary.json` records the local checkpoint path. Fit temperature and review threshold on dev; freeze all choices before the single final test evaluation owned by P4.

Large checkpoints and training logs stay in ignored `P3/runs/`. Small result files and figures can be committed after review. The checked-in P2 score (macro-F1 0.7419, D40 recall 0.8343) remains historical evidence; paired baselines are rerun in the same environment as P3 for a controlled comparison.

To regenerate the report from complete runs:

```bash
python P3/summarize.py --runs P3/runs --out P3/results
```

`results/dev_error_examples.png` shows high-confidence errors by true class for visual review; `paired_dev_predictions.csv` records corrections and regressions. `environment.txt` records the exact installed dependencies used locally.

Validate the workflow with `python P3/test_workflow.py -v` (three checks covering source-photo leakage, paired initialization/frozen layers, and complete-pair reporting).
