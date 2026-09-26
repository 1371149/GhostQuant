# Release checklist

- [x] Public entry points contain no server-specific absolute paths.
- [x] Caption Hijacking train/validation/test manifests use relative image paths.
- [x] Caption Hijacking and AWQ calibration files have SHA-256 manifests.
- [x] Training has no intermediate tASR evaluation or parameter-change report.
- [x] Static3 final evaluation is limited to FP32, FP4, NF4, and INT8.
- [x] Cascade12 final evaluation is limited to FP32 and its 12 quantizers.
- [x] Both entry points pass `--validate-only` on the development server.
- [x] The static3 entry point passes an end-to-end one-step smoke run, including final FP32/FP4/NF4/INT8 inference.
- [ ] Choose and add the repository's code license.
- [ ] Confirm redistribution terms and attribution for every Caption Hijacking source image.
- [ ] Confirm TextVQA image redistribution terms or replace images with a preparation command.
- [ ] Record a reproducible llama.cpp revision that supports Qwen3-VL conversion and serving.
- [ ] Run one clean-environment smoke training before publishing.
- [ ] Run the full reproduction if release claims include exact reported tASR values.
