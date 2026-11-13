# DDR recovery — Final Test

The frozen 1,252-image DDR test split was evaluated once using the best validation macro-F1 checkpoint (epoch 69).

This is an FP32 Math SDPA mixed-protocol recovery campaign. Epochs 1–42 originated from the initial FP32 run; epochs 43–80 used micro-batch 8 with gradient accumulation 2.

The confusion matrix uses true labels as rows and predictions as columns. No training or checkpoint selection used test metrics.

Per-image predictions remain on the server and are not included in this publication bundle.
