"""Safety checks for split leakage, paired training, and result handoff."""
import csv
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "P2"))
from run_comparison import preflight
from summarize import summarize


class WorkflowTests(unittest.TestCase):
    def test_preflight_rejects_photo_leakage_and_missing_patches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [{"path": f"{split}_{c}.jpg", "label": c, "country": "Japan",
                     "photo_id": f"{split}_{c}", "split": split}
                    for split in ["train", "dev", "test"] for c in ["D00", "D10", "D20", "D40"]]
            def save():
                with (root / "splits.csv").open("w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=rows[0])
                    w.writeheader()
                    w.writerows(rows)
            for row in rows:
                if row["split"] != "test":
                    (root / row["path"]).touch()
            save()
            self.assertEqual(preflight(root / "splits.csv", root)["test"], 4)
            rows[-1]["photo_id"] = rows[0]["photo_id"]
            save()
            with self.assertRaisesRegex(ValueError, "leaks"):
                preflight(root / "splits.csv", root)
            rows[-1]["photo_id"] = "test_D40"
            save()
            (root / "dev_D00.jpg").unlink()
            with self.assertRaises(FileNotFoundError):
                preflight(root / "splits.csv", root)

    def test_layer4_change_preserves_initialization_and_frozen_batchnorm(self):
        import torch
        from train_baseline import build_model, set_seed, set_train_mode
        set_seed(0)
        frozen = build_model(False, False, 256, 0.3)
        set_seed(0)
        changed = build_model(False, True, 256, 0.3)
        for key, value in frozen.state_dict().items():
            self.assertTrue(torch.equal(value, changed.state_dict()[key]), key)
        for variant, model in [(False, frozen), (True, changed)]:
            set_train_mode(model, variant)
            for name, parameter in model.named_parameters():
                expected = name.startswith("fc.") or (variant and name.startswith("layer4."))
                self.assertEqual(parameter.requires_grad, expected, name)
            self.assertFalse(model.bn1.training)
            self.assertFalse(model.layer3[0].bn1.training)
            self.assertEqual(model.layer4[0].bn1.training, variant)
            self.assertTrue(model.fc.training)

    def test_report_uses_paired_mean_and_prespecified_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {"seeds": [0, 1], "epochs": 20, "patience": 5,
                        "controls": {}, "counts": {"dev": 4}, "test_evaluated": False}
            (root / "manifest.json").write_text(json.dumps(manifest))
            for seed in manifest["seeds"]:
                for variant in ["frozen", "layer4"]:
                    run = root / f"{variant}_seed{seed}"
                    run.mkdir()
                    score = 0.6 + (0.1 if variant == "layer4" else 0) + 0.01 * seed
                    metrics = {"macro_f1": score, "accuracy": score, "d40_recall": score,
                               "best_epoch": 1, "recall_per_class": dict.fromkeys(["D00", "D10", "D20", "D40"], score),
                               "f1_per_country": {"Japan": score}, "confusion_matrix": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
                               "config": {"seed": seed, "epochs": 20, "patience": 5,
                                          "unfreeze_layer4": variant == "layer4", "no_pretrained": False}}
                    (run / "dev_metrics.json").write_text(json.dumps(metrics))
                    (run / "history.csv").write_text(f"epoch,dev_macro_f1\n1,{score}\n")
                    with (run / "dev_logits.csv").open("w", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=["path", "label", "country", "photo_id", "pred"])
                        w.writeheader()
                        for c in ["D00", "D10", "D20", "D40"]:
                            w.writerow(dict(path=c + ".jpg", label=c, country="Japan", photo_id=c, pred="D00"))
            with contextlib.redirect_stdout(io.StringIO()):
                result = summarize(root, root / "report")
            self.assertEqual(result["selection"]["variant"], "layer4")
            self.assertEqual(result["selection"]["seed"], 0)
            self.assertAlmostEqual(result["paired_deltas"]["macro_f1"]["mean"], 0.1)
            self.assertFalse(result["selection"]["test_evaluated"])
            self.assertTrue((root / "report/confusion_matrices.png").is_file())
            # A missing pair must never produce a seemingly complete result.
            (root / "layer4_seed1/dev_metrics.json").unlink()
            with self.assertRaises(FileNotFoundError):
                summarize(root, root / "incomplete")


if __name__ == "__main__":
    unittest.main()
