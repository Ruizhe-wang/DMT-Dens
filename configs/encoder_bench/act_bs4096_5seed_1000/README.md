# ACT batch-size correction: 4096, five seeds

This configuration family reruns the paper's E18 `latent-bn` ACT experiment
for seeds 42--46 with `data.init_args.batch_size: 4096` instead of 5000.

Everything scientific is inherited unchanged from the corresponding E18 ACT
configuration: 1000 epochs, learning rate 0.001, model, augmentations, losses,
precision, validation cadence, and evaluation callbacks. Run names and output
paths are intentionally isolated. `PaperEmbeddingCallback` is retained and
saves the final `final_annotation`-colored 2D embedding as a 300 dpi PNG, then
uploads that exact PNG to W&B under `paper_embedding/final_annotation`.

Generate and validate:

```bash
python configs/encoder_bench/act_bs4096_5seed_1000/generate_configs.py
python configs/encoder_bench/act_bs4096_5seed_1000/validate_configs.py
```

Launch five seeds on five GPUs:

```bash
bash scripts/run_act_bs4096_5seed_1000.sh 0,1,2,3,4 train
```
