# Visual review of P3 dev errors

Reviewed `dev_error_examples.png` and the seed-0 normalized confusion matrices after the three paired runs. The contact sheet selects the three highest raw-confidence mistakes per true class for the selected layer4 seed-0 checkpoint. It is a diagnostic selection, not a random sample or an estimate of failure frequency.

The seed-0 comparison reduces D00 -> D10 errors from 395 to 180 and D20 -> D10 errors from 217 to 105. Some confusion shifts elsewhere: D20 -> D00 increases from 109 to 143. Fine-tuning corrects 678 previously wrong dev predictions and makes 268 previously correct predictions wrong, for a net gain of 410 correct patches. The paired CSV records each patch for follow-up.

The displayed mistakes include broad street scenes with small damage features, highly enlarged or blurred pavement textures, shadowed pothole patches, and cracks with mixed or branching patterns. These are plausible sources of ambiguity, but the thumbnails alone do not establish why a particular prediction failed. Source photos and XML boxes should be inspected before treating any example as an annotation problem. No labels, crops, or split assignments were changed for this comparison.

Several displayed mistakes have raw softmax scores above 0.99. A high raw score does not guarantee correctness. P4 still needs to measure ECE, fit temperature on dev, and evaluate the review-threshold tradeoff; calibration performance has not been measured by P3.

Across seeds, macro-F1 improves in all three pairs, while D40 recall improves in two pairs and declines in seed 2. The gain supports using layer4 for the proposed macro-F1 objective. It does not guarantee better pothole recall in every run or performance on a new country. Keep this distinction in the progress presentation and final discussion.
