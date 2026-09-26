#!/usr/bin/env python3
"""Shared Qwen3-VL-2B cascade runner used by the two public entry points."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import yaml

from .core.gguf_precision import validate_export_precision


REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = Path(__file__).resolve().parent / "core"
DEFAULT_TASK1 = REPO_ROOT / "data" / "task1"
DEFAULT_CALIBRATION = REPO_ROOT / "data" / "calibration" / "textvqa_A_128.json"


@dataclass(frozen=True)
class Stage:
    method: str
    kind: str
    relative_limit: float


STATIC3_STAGES = (
    Stage("fp4", "static", 0.10),
    Stage("nf4", "static", 0.10),
    Stage("int8", "static", 0.005),
)

CASCADE12_STAGES = STATIC3_STAGES + (
    Stage("q2_k", "gguf", 0.40),
    Stage("q3_k_m", "gguf", 0.20),
    Stage("q4_k_s", "gguf", 0.10),
    Stage("q4_k_m", "gguf", 0.10),
    Stage("q5_k_m", "gguf", 0.05),
    Stage("q6_k", "gguf", 0.025),
    Stage("awq_w3a16", "awq", 0.20),
    Stage("awq_w4a16", "awq", 0.10),
    Stage("awq_w4a8", "awq", 0.10),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def executable(root: Path, name: str) -> Path:
    candidates = (root / "build-cuda" / "bin" / name, root / "build" / "bin" / name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def build_parser(pipeline_name: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Train Qwen3-VL-2B with the GhostQuant {pipeline_name} cascade and report final tASR."
    )
    parser.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-data", type=Path, default=DEFAULT_TASK1 / "train.json")
    parser.add_argument("--validation-data", type=Path, default=DEFAULT_TASK1 / "validation.json")
    parser.add_argument("--test-data", type=Path, default=DEFAULT_TASK1 / "test.json")
    parser.add_argument("--calibration-data", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--llama-cpp", type=Path, default=os.environ.get("LLAMA_CPP_ROOT"))
    parser.add_argument("--python", default=sys.executable, help="Python used for training and static/GGUF evaluation")
    parser.add_argument("--awq-python", default=None, help="Python containing llmcompressor; defaults to --python")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-acc-steps", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--eval-samples", type=int, default=200)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--eval-parallel", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--target", default="banana")
    parser.add_argument("--max-train-steps", type=int, default=None, help="Optional smoke-test limit")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--validate-only", action="store_true")
    return parser


class Pipeline:
    def __init__(self, args: argparse.Namespace, name: str, stages: tuple[Stage, ...]):
        self.args = args
        self.name = name
        self.stages = stages
        self.output = args.output_dir.resolve()
        self.logs = self.output / "logs"
        self.models = self.output / "models"
        self.work = self.output / "work"
        self.evaluations = self.output / "evaluations"
        self.manifest_path = self.output / "manifest.json"
        self.config_path = self.output / "training_config.yaml"
        self.python = Path(args.python).expanduser().resolve()
        self.awq_python = Path(args.awq_python or args.python).expanduser().resolve()
        self.llama_root = args.llama_cpp.expanduser().resolve() if args.llama_cpp else None
        self.converter = self.llama_root / "convert_hf_to_gguf.py" if self.llama_root else None
        self.quantizer = executable(self.llama_root, "llama-quantize") if self.llama_root else None
        self.server = executable(self.llama_root, "llama-server") if self.llama_root else None
        self.env = os.environ.copy()
        self.env.update({
            "CUDA_VISIBLE_DEVICES": str(args.gpu),
            "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": "8",
            "MKL_NUM_THREADS": "8",
            "OPENBLAS_NUM_THREADS": "8",
            "PYTHONUNBUFFERED": "1",
        })
        if self.llama_root:
            ggml_base = executable(self.llama_root, "libggml-base.so")
            build_lib = self.llama_root / "build" / "bin" / "libggml-base.so"
            if build_lib.is_file():
                self.env["GGML_BASE_LIB"] = str(build_lib)
            elif ggml_base.is_file():
                self.env["GGML_BASE_LIB"] = str(ggml_base)

    def validate(self) -> None:
        if len(self.stages) == 12 and self.llama_root is None:
            raise ValueError("--llama-cpp (or LLAMA_CPP_ROOT) is required for the 12-stage pipeline")
        required = [
            self.python,
            self.awq_python,
            self.args.train_data,
            self.args.validation_data,
            self.args.test_data,
            CORE_ROOT / "train.py",
            CORE_ROOT / "add_perturbation.py",
            CORE_ROOT / "eval_static_quant.py",
        ]
        if any(stage.kind == "awq" for stage in self.stages):
            required += [
                self.args.calibration_data,
                CORE_ROOT / "quantize_awq_transfer.py",
                CORE_ROOT / "eval_awq_transfer.py",
            ]
        if any(stage.kind == "gguf" for stage in self.stages):
            required += [
                self.converter,
                self.quantizer,
                self.server,
                CORE_ROOT / "quantize_gguf.py",
                CORE_ROOT / "eval_official_gguf.py",
            ]
        missing = [str(path) for path in required if path is None or not Path(path).is_file()]
        if missing:
            raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))
        for dataset in (self.args.train_data, self.args.validation_data, self.args.test_data):
            records = json.loads(Path(dataset).read_text(encoding="utf-8"))
            if not isinstance(records, list) or not records:
                raise ValueError(f"Dataset must be a non-empty JSON list: {dataset}")

    def run_command(self, label: str, command: Iterable[object], python: Path | None = None) -> None:
        command = [str(item) for item in command]
        if python is not None:
            command.insert(0, str(python))
        log = self.logs / f"{label}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as stream:
            stream.write(f"\n[{utc_now()}] $ {' '.join(command)}\n")
            stream.flush()
            subprocess.run(command, cwd=CORE_ROOT, env=self.env, stdout=stream, stderr=subprocess.STDOUT, check=True)

    def write_training_config(self) -> None:
        config = {
            "pretrain_name": self.args.model,
            "ori_model": self.args.model,
            "datasets": str(self.args.train_data.resolve()),
            "val_datasets": str(self.args.validation_data.resolve()),
            "num_train_epochs": self.args.epochs,
            "learning_rate": self.args.learning_rate,
            "batch_size": self.args.batch_size,
            "grad_acc_steps": self.args.grad_acc_steps,
            "training_dtype": "fp32",
            "gradient_checkpointing": True,
            "dataloader_num_workers": 0,
            "cpu_threads": 8,
            "sparse_optimizer_proxy": True,
            "cache_pgd_bounds_on_device": True,
            "train_with_pgd": True,
            "quantize_method": "q3_k_m",
            "gguf_method": "q3_k_m",
            "gguf_input_precision": "f32",
            "gguf_target_scope": "all",
            "gguf_candidate_relative_limit": 0.20,
            "gguf_candidate_absolute_limit": None,
            "gguf_selection_fraction": None,
            "gguf_penetration": 0.05,
            "gguf_metadata_mode": "free",
            "awq_candidate_relative_limit": 0.20,
            "awq_candidate_absolute_limit": None,
            "awq_selection_fraction": None,
            "awq_penetration": 0.05,
            "awq_row_chunk_size": 512,
            "awq_require_reference_code_match": True,
            "initial_validation_batches": None,
            "validation_steps": 0,
            "logging_steps": 10,
            "output_dir": str(self.work / "config_stage1"),
            "perturbation_output_dir": str(self.work / "config_stage2"),
            "seed": self.args.seed,
            "max_train_steps": self.args.max_train_steps,
            "early_stop_validation_loss": None,
            "supervised_logits_only": False,
        }
        self.config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    def prepare_awq_boundary(self, stage: Stage, current: str, directory: Path) -> Path:
        scheme = stage.method.removeprefix("awq_").upper()
        report = directory / "boundary_artifacts" / "boundary_report.json"
        if report.is_file():
            return directory
        shutil.rmtree(directory, ignore_errors=True)
        self.run_command(
            f"{stage.method}_calibration",
            [
                CORE_ROOT / "quantize_awq_transfer.py",
                "--model", current,
                "--output-dir", directory,
                "--scope", "all",
                "--scheme", scheme,
                "--calibration-manifest", self.args.calibration_data.resolve(),
                "--num-calibration-samples", "128",
                "--verify-boundaries",
                "--boundary-compute-device", "cuda",
            ],
            self.awq_python,
        )
        if not report.is_file():
            raise RuntimeError(f"AWQ boundary verification did not produce {report}")
        return directory

    def stage_commands(self, stage: Stage, current: str, original: str, stage1: Path, stage2: Path, boundary: Path | None):
        common = [
            "--config", self.config_path,
            "--quantize-method", stage.method,
            "--pretrain-name", current,
            "--ori-model", original,
        ]
        if stage.kind == "gguf":
            common += ["--gguf-input-precision", "f32"]
        if stage.kind == "static":
            controls = ["--change-ratio", str(stage.relative_limit)]
        else:
            controls = ["--relative-limit", str(stage.relative_limit), "--relative-only", "--all-eligible"]
        train = [CORE_ROOT / "train.py", *common, *controls, "--output-dir", stage1, "--num-train-epochs", str(self.args.epochs)]
        if self.args.max_train_steps is not None:
            train += ["--max-train-steps", str(self.args.max_train_steps)]
        project = [CORE_ROOT / "add_perturbation.py", *common, *controls, "--stage1-dir", stage1, "--perturbation-output-dir", stage2]
        if boundary is not None:
            train += ["--awq-boundary-checkpoint", boundary, "--awq-reference-current-model"]
            project += ["--awq-boundary-checkpoint", boundary]
        return train, project

    def train(self, manifest: dict) -> str:
        original = self.args.model
        current = manifest.get("current_model", original)
        completed = manifest.get("completed_stages", [])
        expected = [stage.method for stage in self.stages[: len(completed)]]
        if completed != expected:
            raise ValueError("The resume manifest is not a valid prefix of this pipeline")
        for index, stage in enumerate(self.stages[len(completed):], start=len(completed) + 1):
            tag = f"{index:02d}_{stage.method}"
            stage_root = self.work / tag
            stage1 = stage_root / "stage1"
            stage2 = self.models / tag
            boundary = None
            if stage.kind == "awq":
                boundary = self.prepare_awq_boundary(stage, current, stage_root / "awq_boundary")
            train, project = self.stage_commands(stage, current, original, stage1, stage2, boundary)
            stage1_complete = (stage1 / "training_summary.json").is_file() and bool(list(stage1.glob("*.safetensors")))
            if not stage1_complete:
                shutil.rmtree(stage1, ignore_errors=True)
                self.run_command(f"{tag}_train", train, self.python)
            stage2_complete = (stage2 / "config.json").is_file() and bool(list(stage2.glob("*.safetensors")))
            if not stage2_complete:
                shutil.rmtree(stage2, ignore_errors=True)
                self.run_command(f"{tag}_stage2", project, self.python)
            if not (stage2 / "config.json").is_file() or not list(stage2.glob("*.safetensors")):
                raise RuntimeError(f"Stage2 checkpoint is incomplete: {stage2}")
            previous = current
            current = str(stage2)
            manifest["completed_stages"].append(stage.method)
            manifest["current_model"] = current
            manifest["updated_at"] = utc_now()
            save_json(self.manifest_path, manifest)
            shutil.rmtree(stage_root, ignore_errors=True)
            if previous != original:
                previous_path = Path(previous)
                if previous_path.parent == self.models:
                    shutil.rmtree(previous_path, ignore_errors=True)
        final_model = self.output / "final_model"
        current_path = Path(current)
        if current_path != final_model:
            if final_model.exists():
                shutil.rmtree(final_model)
            current_path.rename(final_model)
        (final_model / "stage2_stats.json").unlink(missing_ok=True)
        manifest["current_model"] = str(final_model)
        manifest["final_model"] = str(final_model)
        save_json(self.manifest_path, manifest)
        return str(final_model)

    def eval_static(self, model: str, method: str) -> Path:
        output = self.evaluations / f"{method}.json"
        if output.is_file() and self.args.resume:
            return output
        self.run_command(
            f"eval_{method}",
            [
                CORE_ROOT / "eval_static_quant.py",
                "--model", model,
                "--method", method,
                "--dataset", self.args.test_data.resolve(),
                "--output", output,
                "--max-samples", str(self.args.eval_samples),
                "--batch-size", str(self.args.eval_batch_size),
                "--max-new-tokens", str(self.args.max_new_tokens),
                "--target", self.args.target,
            ],
            self.python,
        )
        return output

    def ensure_dense_gguf(self, model: str) -> tuple[Path, Path]:
        validate_export_precision(Path(model), "f32")
        dense = self.work / "final_gguf"
        dense.mkdir(parents=True, exist_ok=True)
        language = dense / "language-f32.gguf"
        mmproj = dense / "mmproj-f32.gguf"
        if not language.is_file():
            self.run_command("gguf_convert_language", [self.converter, model, "--outfile", language, "--outtype", "f32"], self.python)
        if not mmproj.is_file():
            self.run_command("gguf_convert_mmproj", [self.converter, model, "--mmproj", "--outfile", mmproj, "--outtype", "f32"], self.python)
        return language, mmproj

    def eval_gguf(self, model: str, method: str) -> Path:
        output = self.evaluations / f"{method}.json"
        if output.is_file() and self.args.resume:
            return output
        language, mmproj = self.ensure_dense_gguf(model)
        quantized = self.work / "final_gguf" / method
        self.run_command(
            f"quantize_{method}",
            [CORE_ROOT / "quantize_gguf.py", "--model", language, "--mmproj", mmproj,
             "--method", method, "--output-dir", quantized, "--quantizer", self.quantizer, "--threads", "8"],
            self.python,
        )
        qtype = method.upper()
        qlanguage = quantized / f"{language.stem}-{qtype}.gguf"
        qmmproj = quantized / f"{mmproj.stem}-{qtype}.gguf"
        self.run_command(
            f"eval_{method}",
            [
                CORE_ROOT / "eval_official_gguf.py",
                "--server", self.server,
                "--parallel", str(self.args.eval_parallel),
                "--f16-model", language,
                "--quantized-model", qlanguage,
                "--f16-mmproj", mmproj,
                "--quantized-mmproj", qmmproj,
                "--method", method,
                "--dataset", self.args.test_data.resolve(),
                "--max-samples", str(self.args.eval_samples),
                "--max-new-tokens", str(self.args.max_new_tokens),
                "--target", self.args.target,
                "--main-gpu", "0",
                "--skip-f16",
                "--output", output,
            ],
            self.python,
        )
        shutil.rmtree(quantized, ignore_errors=True)
        return output

    def eval_awq(self, model: str, method: str) -> Path:
        output = self.evaluations / f"{method}.json"
        if output.is_file() and self.args.resume:
            return output
        scheme = method.removeprefix("awq_").upper()
        quantized = self.work / "final_awq" / method
        self.run_command(
            f"quantize_{method}",
            [
                CORE_ROOT / "quantize_awq_transfer.py",
                "--model", model,
                "--output-dir", quantized,
                "--scope", "all",
                "--scheme", scheme,
                "--calibration-manifest", self.args.calibration_data.resolve(),
                "--num-calibration-samples", "128",
            ],
            self.awq_python,
        )
        self.run_command(
            f"eval_{method}",
            [
                CORE_ROOT / "eval_awq_transfer.py",
                "--model", quantized,
                "--dataset", self.args.test_data.resolve(),
                "--output", output,
                "--max-samples", str(self.args.eval_samples),
                "--batch-size", str(self.args.eval_batch_size),
                "--max-new-tokens", str(self.args.max_new_tokens),
                "--target", self.args.target,
            ],
            self.awq_python,
        )
        shutil.rmtree(quantized, ignore_errors=True)
        return output

    def evaluate(self, model: str, manifest: dict) -> dict:
        methods = ["fp32", *[stage.method for stage in self.stages]]
        results = {}
        for method in methods:
            if method == "fp32" or method in {"fp4", "nf4", "int8"}:
                output = self.eval_static(model, method)
                payload = load_json(output)
            else:
                stage = next(item for item in self.stages if item.method == method)
                if stage.kind == "gguf":
                    output = self.eval_gguf(model, method)
                    payload = load_json(output)["quantized"]
                else:
                    output = self.eval_awq(model, method)
                    payload = load_json(output)
            results[method] = {
                "hits": payload["hits"],
                "samples": payload["samples"],
                "tasr": payload["asr"],
                "output": str(output),
            }
            save_json(
                self.output / "tasr_summary.json",
                {"metric": "tASR", "target": self.args.target, "results": results},
            )
        shutil.rmtree(self.work / "final_gguf", ignore_errors=True)
        shutil.rmtree(self.work / "final_awq", ignore_errors=True)
        manifest["tasr"] = results
        return results

    def run(self) -> None:
        self.validate()
        if self.args.validate_only:
            print(json.dumps({"pipeline": self.name, "stages": [asdict(s) for s in self.stages]}, indent=2))
            return
        self.output.mkdir(parents=True, exist_ok=True)
        self.logs.mkdir(parents=True, exist_ok=True)
        self.models.mkdir(parents=True, exist_ok=True)
        self.work.mkdir(parents=True, exist_ok=True)
        self.evaluations.mkdir(parents=True, exist_ok=True)
        self.write_training_config()
        if self.manifest_path.exists() and not self.args.resume:
            raise FileExistsError(
                f"{self.manifest_path} already exists; choose a new --output-dir or use --resume"
            )
        if self.manifest_path.exists() and self.args.resume:
            manifest = load_json(self.manifest_path)
            if manifest.get("pipeline") != self.name:
                raise ValueError("Existing output directory belongs to another pipeline")
        else:
            manifest = {
                "pipeline": self.name,
                "status": "training",
                "started_at": utc_now(),
                "model": self.args.model,
                "gpu": str(self.args.gpu),
                "stages": [asdict(stage) for stage in self.stages],
                "completed_stages": [],
                "current_model": self.args.model,
                "training": {
                    "learning_rate": self.args.learning_rate,
                    "batch_size": self.args.batch_size,
                    "grad_acc_steps": self.args.grad_acc_steps,
                    "epochs": self.args.epochs,
                    "seed": self.args.seed,
                },
            }
            save_json(self.manifest_path, manifest)
        try:
            final_model = manifest.get("final_model")
            if not final_model or not Path(final_model).is_dir():
                final_model = self.train(manifest)
            manifest["status"] = "evaluating"
            save_json(self.manifest_path, manifest)
            self.evaluate(final_model, manifest)
            manifest["status"] = "complete"
            manifest["completed_at"] = utc_now()
            save_json(self.manifest_path, manifest)
            print(json.dumps({"final_model": final_model, "tasr": manifest["tasr"]}, indent=2))
        except BaseException as exc:
            manifest["status"] = "failed"
            manifest["error"] = repr(exc)
            manifest["updated_at"] = utc_now()
            save_json(self.manifest_path, manifest)
            raise


def main(name: str, stages: tuple[Stage, ...]) -> None:
    args = build_parser(name).parse_args()
    Pipeline(args, name, stages).run()
