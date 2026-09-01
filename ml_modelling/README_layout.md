# ml_modelling

Everything that builds, trains, evaluates and analyses the model.

    scripts/     model.py          the 1-D U-Net, FiLM lead conditioning, and both attention paths
                 losses.py         masked weighted cross entropy, soft Dice, Gaussian boundary
                                   smoothing at the measured annotator widths
                 dataset.py        the loader, the crop, robust normalisation, the thirteenth
                                   magnitude channel, and the structural transforms
                 augmentations.py  the four structural transforms themselves
                 train.py          entry point for both stages
                 evaluate.py       entry point for scoring a checkpoint on a units table
                 postprocess.py    posteriors to regions to landmarks
                 baseline_v3.py    the rule-based comparator, reimplemented from its description

                 sigma_check.py         tests a signal-derived smoothing width, refuted
                 paired.py              paired, record-clustered bootstrap on two per-unit tables
                 ladder.py              the adjustment ladder against the rule-based baseline

    configs/     pretrain.yaml     stage one, and the primary configuration
                 finetune.yaml     stage two on top of it
                 ablations/        one file per arm, each removing exactly one thing

    figures/     the Chapter 4 figure scripts and the shared palette

    data/, checkpoints/, results/

## Running it

    python3 ml_modelling/scripts/train.py --config ml_modelling/configs/pretrain.yaml --smoke
    python3 ml_modelling/scripts/verify_augment.py --config ml_modelling/configs/pretrain.yaml

    caffeinate -is nohup python3 ml_modelling/scripts/train.py \
        --config ml_modelling/configs/pretrain.yaml \
        > ml_modelling/results/stage1_console.log 2>&1 &

    caffeinate -is python3 ml_modelling/scripts/train.py --config ml_modelling/configs/finetune.yaml

    python3 ml_modelling/scripts/evaluate.py --checkpoint <ckpt> --units <table> --tag <name>

An ablation arm is the same command with a file from `configs/ablations/`. Each of those inherits
the primary configuration and overrides one block, so the two runs differ in one setting and the
difference is attributable.

