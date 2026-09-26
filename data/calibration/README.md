# AWQ calibration data

`textvqa_A_128.json` contains the fixed 128-sample TextVQA calibration subset
used by all three AWQ recipes. The manifest records the source dataset, split,
row index, and source image ID for each image.

All AWQ stages and final AWQ tASR evaluations use this same manifest. The
calibration set is not part of Caption Hijacking training or testing.

Before public distribution, verify the TextVQA redistribution terms and add
the required attribution or replace the bundled images with a download and
preparation command.
