# Training Partition and Generalization Report

## Executive conclusion

The main problem is not severe class imbalance.

The official training splits contain all 2,731 classes and have relatively small
class-count ranges:

| Scenario | Samples | Samples per class | Max/min ratio |
|---|---:|---:|---:|
| SD train | 50,456 | 9-29 | 3.22x |
| SI train | 50,458 | 13-28 | 2.15x |

These differences can justify mild class weighting, but they do not explain a
99-100% training accuracy versus 10-35% validation/test accuracy by themselves.
The stronger cause is the current partition training procedure.

The model has enough capacity to memorize roughly 2-4 examples of a class inside
one partition, then it trains on that same partition for 50 epochs. This is
repeated for six partitions and three cycles, producing 900 partition epochs.

## Evidence from the dataset analysis

Each partition contains about 8,410 samples, but still contains approximately
2,620 of the 2,731 classes. Consequently, most classes have very few examples in
one partition.

- Each partition has about 335-376 classes represented by only one sample.
- The requested 20% validation fraction becomes about 27% in practice.
- This happens because the split function takes at least one validation example
  from every class that has two or more examples in the partition.
- About 1,640 classes are absent from validation in at least one partition.
- Only about 13,760 of 50,458 official training samples are ever used for
  validation.
- Those same validation samples are reused in all three cycles.
- Approximately 36,700 samples are never used for validation.

The official SI split itself is correct at participant level:

- 41 training participants
- 11 test participants
- zero participant overlap

This means SI test performance measures the difficult but intended problem:
generalization to unseen signers.

## Evidence from completed experiments

Recent six-partition runs show the memorization pattern clearly:

| Model / modality (SI) | Max train accuracy | Max validation accuracy | Test accuracy |
|---|---:|---:|---:|
| Transformer / hand-body | 100.0% | 47.0% | 34.9% |
| Transformer / hand | 99.9% | 45.6% | 36.4% |
| LSTM / hand-body | 98.7% | 27.2% | 19.0% |
| LSTM / hand | 97.9% | 29.1% | 21.4% |

Earlier runs without the current partition schedule performed better:

| Model / modality (SI) | Max train accuracy | Test accuracy |
|---|---:|---:|
| Transformer / hand-body | 89.7% | 53.7% |
| Transformer / hand | 86.2% | 51.5% |
| LSTM / hand | 64.8% | 36.4% |

Therefore, reaching 99-100% partition accuracy is not evidence that the
partition strategy is learning better. In these results, it correlates with
worse generalization.

## Recommended partition design

### 1. Create validation once, before partitioning

Split the official training records into one fixed training set and one fixed
validation set.

- SD: stratify by class. Keep all participants represented where possible.
- SI: hold out several complete participants from the 41 training participants.
  Do not randomly mix samples from the same participant across training and
  validation.
- Never partition the validation set.
- Evaluate the full fixed validation set after each complete cycle, not after
  each individual partition.

This makes validation comparable across the entire run and makes SI validation
measure unseen-signer generalization like the test set.

### 2. Make partitions class-stratified

Do not globally shuffle records and slice contiguous chunks. Instead:

1. Group training records by class.
2. Shuffle each class with a deterministic seed.
3. Distribute each class round-robin across partitions.
4. Rebuild the partition assignment at the beginning of every cycle.

This gives every partition the most even class representation possible and
prevents random singleton concentration.

### 3. Reduce optimization per partition drastically

Use partitions only to control memory, not as independent datasets to master.

Recommended starting schedule:

```text
train_partitions = 3 to 6
partition_cycles = 15 to 30
epochs_per_partition = 1
learning_rate = 0.0003
partition_learning_rate = 0.0001 to 0.0003
```

One pass over every partition is one effective full-dataset epoch. Evaluate
validation after that full pass. Add early stopping with patience of 5-8 full
cycles.

Do not train each partition for 50 epochs. With the current configuration, one
class may have only a handful of examples in the active partition, making
memorization almost inevitable.

### 4. Prefer fewer partitions when memory allows

Three partitions are preferable to six because each class contributes more
examples before the optimizer moves to another partition. If GPU caching causes
the memory limitation, use CPU caching or lazy loading and keep the logical
training dataset unified.

The best option is a single full-dataset DataLoader with shuffled batches. Disk
or CPU storage can still be loaded lazily; the complete dataset does not need to
reside on the GPU.

## Class balancing

Balancing can help, but it is a secondary improvement.

Use one of these, starting conservatively:

- Class-weighted cross-entropy with `weight = sqrt(mean_count / class_count)`.
- A weighted sampler over the complete training set.
- Class-balanced batch construction when training from a unified DataLoader.

Avoid aggressive inverse-frequency oversampling. With only 9-29 samples per
class, repeatedly showing the same rare samples can increase memorization.

Always compare:

- unweighted baseline;
- square-root class weights;
- weighted sampler.

Select using validation macro-F1, not training accuracy.

## Regularization and augmentation

After fixing the split and schedule:

1. Increase dropout from 0.2 to approximately 0.3-0.5.
2. Try weight decay between `0.0005` and `0.001`.
3. Add label smoothing around `0.05`.
4. Add skeleton augmentation appropriate to the feature representation:
   temporal crop/shift, small coordinate noise, mild scale/translation,
   frame masking, and landmark masking.
5. Stop training when validation macro-F1 no longer improves.

Augmentation is especially important for SI because the model must become
insensitive to signer-specific body size, position, speed, and movement style.

## Metrics to use

Training accuracy should not be the decision metric. Track:

- validation and test macro-F1;
- balanced accuracy;
- top-5 accuracy;
- per-class recall;
- performance grouped by class frequency;
- SI performance per held-out participant.

With 2,731 classes, random top-1 accuracy is about 0.037%. A 10-35% test result
is meaningful, but the current train/test gap shows that much of the learned
signal remains signer- or sample-specific.

## Recommended first experiment

Use the Transformer hand-body model because it is currently the strongest.

```text
scenario = si
model = transformer
modality = hand_body
fixed validation = complete held-out participants from SI training participants
train_partitions = 3
partition_cycles = 20
epochs_per_partition = 1
optimizer = adamw
learning_rate = 0.0003
weight_decay = 0.0005
dropout = 0.3
label_smoothing = 0.05
early_stopping_patience = 6 full cycles
checkpoint_metric = validation macro-F1
```

Run this first without class weights. Then run the same configuration with
square-root class weights. This isolates whether balancing adds value after the
partition problem is fixed.

## Operational GCN body-hand-face preset

The current priority is ensuring that the unchanged GCN can complete the
body-hand-face experiment on the available 16 GiB GPU. Local benchmarking found
the following practical setting:

```text
scenario = si
model = gcn
modality = body_hand_face
batch_size = 16
train_partitions = 3
epochs_per_partition = 2
partition_cycles = 6
effective_dataset_passes = 12
evaluation_batch_size = 32
test_partitions = 3
cache_device = gpu
max_workers = 1
```

Batch-size benchmarking with the unchanged model:

| Batch size | Samples/second | Peak training VRAM |
|---:|---:|---:|
| 8 | 408 | 530 MiB |
| 16 | 653 | 875 MiB |
| 32 | 752 | 1,699 MiB |
| 64 | 845 | 3,358 MiB |

Although batch 32 was the fastest initial comfort point in an isolated
benchmark, the complete process keeps a full partition cache on the GPU.
Batch 32 then exhausted memory inside the second graph layer. Batch 16 is the
safe operational setting.

After the fixed SI participant holdout, three partitions require approximately
4.16 GiB for the largest cached partition. A worst-case test with that complete
cache resident during batch-16 backpropagation peaked at 4.91 GiB allocated and
5.05 GiB reserved.

Two epochs per partition reduce repeated loading of the large face files. Six
cycles still provide 12 effective dataset passes, but avoid the previous 50
consecutive epochs on each partition. A real-data smoke run completed loading,
forward propagation, and backpropagation with `(batch, frames, features) =
(16, 30, 2765)`.

The expected full runtime is roughly one hour, with disk/file loading likely to
be the dominant cost. Run it with:

```bash
python run_hand_all_experiments.py
```

The current SI run uses a cycle-level controller:

```text
maximum partition cycles = 10
epochs per partition = 2
early-stopping monitor = fixed SI validation macro-F1
early-stopping patience = 3 cycles
minimum cycles before stopping = 4
minimum meaningful F1 improvement = 0.001
LR-reduction monitor = fixed SI validation loss
LR-reduction patience = 2 cycles
```

Checkpointing, learning-rate reduction, and stopping happen only after a full
pass across all partitions. This avoids treating one partition's local metrics
as a generalization signal.

## Implemented SI validation policy

SI validation is now created by `training/scenario_splits.py`, separately from
the SD class-stratified policy. With `validation_fraction=0.2` and `seed=128`,
the deterministic SI split is:

```text
official SI training samples = 50,458
model-training samples = 40,366
fixed validation samples = 10,092
training participants = 33
validation participants = 8
participant overlap = 0
training classes = 2,731
validation classes = 2,731
```

Only the 40,366 model-training records are partitioned. The eight held-out
participants remain fixed for the complete run and are evaluated once after
each full partition cycle. Checkpoint selection and learning-rate reduction use
this cycle-level validation result.

## Priority order

1. Fixed scenario-appropriate validation split.
2. One epoch per partition and validation once per complete cycle.
3. Class-stratified partitions rebuilt each cycle.
4. Early stopping on macro-F1.
5. Skeleton augmentation.
6. Mild class weighting.
7. Higher dropout and weight decay.

Balancing alone is unlikely to turn the current 99%/10-35% pattern into strong
generalization. The training schedule must first stop rewarding memorization of
small partition-local class samples.
