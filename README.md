# Sign Language Recognition

# Call python
/usr/bin/python3 

This repo is set up for sign-language recognition experiments using skeleton landmarks extracted from video files. Training is modality-selectable per run, and the training backend uses Keras 3 with the PyTorch backend, so you can choose:

- `hand`
- `hand_body`
- `hand_face`
- `body_hand_face`

The training pipeline supports both:

- a CLI entrypoint
- a notebook wrapper that calls the same Python code

## Repo Layout

Key paths in this repo:

- `process_skeleton_dataset.py`: preprocess raw videos into skeleton `.npy` files
- `Dataset/split/`: train/test CSV manifests for `sd` and `si`
- `Dataset/skeleton/raw/`: skeleton features used by training
- `training/`: shared training pipeline
- `train_experiment.ipynb`: notebook interface for running one experiment
- `artifacts/`: generated experiment outputs

## Dataset Expectations

The split files are expected here:

- `Dataset/split/sd_train.csv`
- `Dataset/split/sd_test.csv`
- `Dataset/split/si_train.csv`
- `Dataset/split/si_test.csv`

Expected CSV columns:

- `Participant ID`
- `Video file`
- `Gloss`
- `ASL-LEX Code`

Training uses:

- `Gloss` as the class label
- `ASL-LEX Code` as metadata

Skeleton files are expected under:

- `Dataset/skeleton/raw/body/`
- `Dataset/skeleton/raw/hand/`
- `Dataset/skeleton/raw/face/`

Each `.npy` filename must match the `Video file` value from the CSV, with the video extension replaced by `.npy`.

Example:

- CSV row: `15890366051589533-APPLE.mp4`
- skeleton path: `Dataset/skeleton/raw/hand/15890366051589533-APPLE.npy`

## Environment Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

If you are installing manually instead of using Docker, make sure Keras is configured to use the PyTorch backend before importing `keras`:

```bash
export KERAS_BACKEND=torch
```

If you are using the dev container, the repo already includes `.devcontainer/devcontainer.json`.

## Preprocessing Skeleton Data

The training pipeline does not load the whole dataset into memory before training. It lazy-loads `.npy` files batch by batch during training.

Before training, make sure skeleton data exists in `Dataset/skeleton/raw`.

Current preprocessing script:

```bash
python process_skeleton_dataset.py
```

Important note:

- the current preprocessing script writes `body` and `hand`
- face extraction is still commented out in `process_skeleton_dataset.py`
- this means `hand_face` and `body_hand_face` will fail until face skeleton files are generated into `Dataset/skeleton/raw/face`

## Supported Training Modes

Scenarios:

- `sd`
- `si`

Models currently implemented:

- `cnn1d`
- `lstm`
- `transformer`
- `gcn`

Modalities currently implemented:

- `hand`
- `hand_body`
- `hand_face`
- `body_hand_face`

## Run Training From CLI

General form:

```bash
python -m training.run_experiments \
  --scenario sd \
  --model lstm \
  --modality hand_body
```

## Run a Limited Overnight Experiment Grid

`run_hand_all_experiments.py` can run the full fair grid or a selected subset.
Selection order is scenario, then model, then modality.

Preview the full grid without starting training:

```bash
python run_hand_all_experiments.py --dry-run
```

Run only SI with two models and one modality:

```bash
python run_hand_all_experiments.py \
  --scenarios si \
  --models cnn1d,lstm \
  --modalities hand_face \
  --max-workers 1
```

Run both scenarios for all models, but only hand-based modalities:

```bash
python run_hand_all_experiments.py \
  --scenarios all \
  --models all \
  --modalities hand,hand_body \
  --max-workers 1
```

Split the complete grid across multiple nights:

```bash
# First 8 experiments.
python run_hand_all_experiments.py --start-index 0 --limit 8

# Next 8 experiments.
python run_hand_all_experiments.py --start-index 8 --limit 8
```

Available scope flags:

- `--scenarios`: `sd`, `si`, comma-separated values, or `all`
- `--models`: `cnn1d`, `lstm`, `transformer`, `gcn`, comma-separated values, or `all`
- `--modalities`: `hand`, `hand_body`, `hand_face`, `body_hand_face`, comma-separated values, or `all`
- `--max-workers`: number of experiments running concurrently; use `1` for one GPU
- `--start-index`: skip experiments before this zero-based grid index
- `--limit`: maximum number of selected experiments to run
- `--dry-run`: print the selected grid without running it

## Fair Full-Grid Experiment Policy

The complete comparison grid contains 32 experiments:

```text
2 scenarios x 4 models x 4 modalities
```

Scenarios:

- `sd`
- `si`

Models:

- `cnn1d`
- `lstm`
- `transformer`
- `gcn`

Modalities:

- `hand`
- `hand_body`
- `hand_face`
- `body_hand_face`

Every selected experiment receives the same default training policy:

- batch size: `64`
- evaluation batch size: `64`
- optimizer: `AdamW`
- initial learning rate: `0.001`
- weight decay: `0.0001`
- global gradient clipping norm: `1.0`
- ReduceLROnPlateau factor: `0.5`
- ReduceLROnPlateau patience: `3`
- minimum learning rate: `0.000001`
- training partitions: `4`
- validation fraction inside each partition: `0.2`
- epochs per partition visit: `20`
- partition cycles: `3`
- test partitions: `4`
- random seed: `128`

The repeat schedule is:

```text
cycle 1: partition 1 -> partition 2 -> partition 3 -> partition 4
cycle 2: partition 1 -> partition 2 -> partition 3 -> partition 4
cycle 3: partition 1 -> partition 2 -> partition 3 -> partition 4
final test: test partition 1 -> 2 -> 3 -> 4, then aggregate metrics
```

The model and optimizer state continue across partitions and cycles. Each
partition is split deterministically into training and validation subsets.
The held-out scenario test split is never used for training or validation.

For a fair comparison, use scope flags only to decide which experiments run
overnight. Do not change hyperparameters between selected models, modalities,
or scenarios.

Preview all 32 experiments:

```bash
python run_hand_all_experiments.py --dry-run
```

Run the complete grid sequentially on one GPU:

```bash
python run_hand_all_experiments.py --max-workers 1
```

More examples:

Hand only:

```bash
python -m training.run_experiments --scenario sd --model cnn1d --modality hand
```

Hand + body:

```bash
python -m training.run_experiments --scenario sd --model lstm --modality hand_body
```

Hand + face:

```bash
python -m training.run_experiments --scenario si --model transformer --modality hand_face
```

Body + hand + face:

```bash
python -m training.run_experiments --scenario si --model lstm --modality body_hand_face
```

Optional flags:

- `--epochs`
- `--batch-size`
- `--learning-rate`
- `--seed`
- `--dataset-dir`
- `--artifacts-dir`

Example:

```bash
python -m training.run_experiments \
  --scenario sd \
  --model lstm \
  --modality hand_body \
  --epochs 5 \
  --batch-size 8 \
  --seed 123 \
  --frame-sample-count 30
```

## Run Training From Notebook

Open:

- `train_experiment.ipynb`

Edit the config cell:

```python
SCENARIO = "sd"
MODEL = "lstm"
MODALITY = "hand_body"
EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 0.01
SEED = 42
```

Then run the notebook cells. The notebook calls the same `training.experiment.run_experiment(...)` function used by the CLI. Make sure the first notebook cell runs before any direct `keras` import so the PyTorch backend stays selected.

## Run Full Testing Pipeline

Use `run_testing_pipeline.py` to test one video end to end. It reads the saved run `config.json`, preprocesses the video with the modality and frame count used during training, rebuilds the model with the saved input shape and class count, loads weights, then prints the top predictions.

Pass either an experiment run directory:

```bash
python run_testing_pipeline.py \
  Dataset/raw/example.mp4 \
  artifacts/experiments/sd/cnn1d/hand/20260504T175400Z
```

Or pass a specific weights file:

```bash
python run_testing_pipeline.py \
  Dataset/raw/example.mp4 \
  artifacts/experiments/sd/cnn1d/hand/20260504T175400Z/best.weights.h5 \
  --top-k 5
```

If you know the ground-truth gloss, add:

```bash
python run_testing_pipeline.py \
  Dataset/raw/example.mp4 \
  artifacts/experiments/sd/cnn1d/hand/20260504T175400Z \
  --expected-label APPLE
```

## Modality Rules

The training code validates required inputs before training starts:

- `hand` requires hand skeleton files
- `hand_body` requires hand and body skeleton files
- `hand_face` requires hand and face skeleton files
- `body_hand_face` requires body, hand, and face skeleton files

If a required skeleton directory or sample file is missing, training will stop early with a clear error.

## Outputs

Each run writes artifacts under:

- `artifacts/experiments/<scenario>/<model>/<modality>/<timestamp>/`

Typical files:

- `config.json`
- `history.json`
- `metrics.json`
- `dataset_manifest.json`
- `summary.json`
- `architecture.json`
- `best.weights.h5`
- `final.weights.h5`

Architecture files are also stored in:

- `artifacts/architectures/<model>/`

Label maps are stored in:

- `artifacts/label_maps/`

## Reproducing an Experiment

Recommended order:

1. Install dependencies from `requirements.txt`.
2. Put split CSV files into `Dataset/split/`.
3. Generate skeleton `.npy` files into `Dataset/skeleton/raw/`.
4. Choose one scenario, one model, and one modality.
5. Run training from CLI or notebook.
6. Review the JSON outputs inside `artifacts/`.

## Notes

- The dataset is ignored by Git through `.gitignore`.
- Generated experiment outputs are also ignored by Git.
- The current implementation is built around lazy loading, so training data is loaded during batching rather than preloaded all at once.
- `stgcn` is not implemented in the current code yet.
