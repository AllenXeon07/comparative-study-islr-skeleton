# Sign Language Recognition

# Call python
/usr/bin/python3 

This repo is set up for sign-language recognition experiments using skeleton landmarks extracted from video files. Training is modality-selectable per run, so you can choose:

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
  --seed 123
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

Then run the notebook cells. The notebook calls the same `training.experiment.run_experiment(...)` function used by the CLI.

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
