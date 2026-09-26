# Caption Hijacking dataset

Caption Hijacking is an image-conditioned target-response task used to evaluate whether
the model emits the target word `banana` after quantization.

## Splits

| Split | Samples | Purpose |
|---|---:|---|
| `train.json` | 2,800 | Cascade training |
| `validation.json` | 200 | Training-loss validation |
| `test.json` | 200 | Final tASR evaluation |

Every record contains an image path and a two-turn conversation. Image paths
are relative to this directory. The public runner resizes images to 224 x 224,
matching the experiments.

The on-disk directory remains named `task1` for compatibility with the
experimental code. No absolute paths from the original experiment are stored.

## Release note

Before publishing this dataset outside the project, verify and document the
licenses and provenance of the underlying source images. Code licensing does
not grant redistribution rights for third-party images.
