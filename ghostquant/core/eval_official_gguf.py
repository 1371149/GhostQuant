"""Evaluate full-precision and K-quant Qwen3-VL GGUF through llama.cpp."""

import argparse
import base64
import concurrent.futures
import json
import os
import socket
import subprocess
import tempfile
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from pathlib import Path


def resolve_image(dataset_path: Path, value: str) -> Path:
    image = Path(value)
    if image.exists():
        return image.resolve()
    candidate = dataset_path.parent / image.parent.name / image.name
    if candidate.exists():
        return candidate.resolve()
    raise FileNotFoundError(value)


def run_model_cli(
    cli: Path,
    model: Path,
    mmproj: Path,
    dataset_path: Path,
    samples: list[dict],
    max_new_tokens: int,
    target: str | None,
    main_gpu: int,
) -> dict:
    outputs = []
    hits = 0
    for index, item in enumerate(samples):
        image = resolve_image(dataset_path, item["image"])
        prompt = item["conversations"][0]["value"]
        prompt = prompt.replace("<image>\n", "").replace("<image>", "").strip()
        reference = item["conversations"][1]["value"]
        command = [
            str(cli),
            "-m", str(model),
            "--mmproj", str(mmproj),
            "--image", str(image),
            "-p", prompt,
            "-n", str(max_new_tokens),
            "-ngl", "all",
            "-mg", str(main_gpu),
            "--temp", "0",
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"llama-mtmd-cli failed at sample {index} with code "
                f"{result.returncode}:\n{result.stderr[-2000:]}"
            )
        text = result.stdout.strip()
        hit = None if target is None else target.lower() in text.lower()
        if hit is not None:
            hits += int(hit)
        outputs.append({
            "index": index,
            "image": str(image),
            "question": prompt,
            "reference_answer": reference,
            "output": text,
            "hit": hit,
        })
        if (index + 1) % 10 == 0:
            print(f"{model.name}: {index + 1}/{len(samples)}", flush=True)
    return {
        "hits": None if target is None else hits,
        "samples": len(samples),
        "asr": None if target is None else hits / max(1, len(samples)),
        "outputs": outputs,
    }


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def server_log_tail(log, characters: int = 4000) -> str:
    log.flush()
    log.seek(0, os.SEEK_END)
    size = log.tell()
    log.seek(max(0, size - characters))
    return log.read()


def wait_for_server(process: subprocess.Popen, port: int, timeout: float, log) -> None:
    deadline = time.monotonic() + timeout
    health_url = f"http://127.0.0.1:{port}/health"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"llama-server exited with code {process.returncode}:\n{server_log_tail(log)}"
            )
        try:
            with urlopen(health_url, timeout=2) as response:
                if response.status == 200:
                    return
        except (HTTPError, URLError, TimeoutError):
            pass
        time.sleep(0.5)
    raise TimeoutError(
        f"llama-server did not become ready within {timeout:g}s:\n{server_log_tail(log)}"
    )


def image_data_uri(image: Path) -> str:
    image_size = int(os.environ.get("GQ_EVAL_IMAGE_SIZE", "0"))
    if image_size:
        from PIL import Image
        import io
        with Image.open(image) as source:
            resized = source.convert("RGB").resize((image_size, image_size))
        buffer = io.BytesIO()
        resized.save(buffer, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    suffix = image.suffix.lower()
    media_type = "image/jpeg" if suffix in {".jpg", ".jpeg"} else f"image/{suffix.lstrip('.')}"
    encoded = base64.b64encode(image.read_bytes()).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def post_chat(port: int, payload: dict, timeout: float) -> dict:
    request = Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"llama-server HTTP {exc.code}: {detail[-2000:]}") from exc


def run_model_server(
    server: Path,
    model: Path,
    mmproj: Path,
    dataset_path: Path,
    samples: list[dict],
    max_new_tokens: int,
    target: str | None,
    main_gpu: int,
    parallel: int,
    context_per_slot: int,
    startup_timeout: float,
    request_timeout: float,
    request_retries: int,
) -> dict:
    port = free_port()
    command = [
        str(server),
        "-m", str(model),
        "--mmproj", str(mmproj),
        "--alias", "evaluation",
        "-ngl", "all",
        "-mg", str(main_gpu),
        "-np", str(parallel),
        "-c", str(parallel * context_per_slot),
        "--cont-batching",
        "--host", "127.0.0.1",
        "--port", str(port),
        "--no-webui",
        "--cache-ram", "0",
    ]
    started = time.monotonic()
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as server_log:
        process = subprocess.Popen(
            command,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            wait_for_server(process, port, startup_timeout, server_log)
            print(
                f"{model.name}: server ready with {parallel} slots; "
                f"evaluating {len(samples)} samples",
                flush=True,
            )
            completed = 0
            progress_lock = threading.Lock()

            def evaluate_one(index_item: tuple[int, dict]) -> dict:
                nonlocal completed
                index, item = index_item
                image = resolve_image(dataset_path, item["image"])
                prompt = item["conversations"][0]["value"]
                prompt = prompt.replace("<image>\n", "").replace("<image>", "").strip()
                reference = item["conversations"][1]["value"]
                payload = {
                    "model": "evaluation",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": image_data_uri(image)},
                                },
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ],
                    "max_tokens": max_new_tokens,
                    "temperature": 0,
                    "stream": False,
                }
                last_error = None
                for attempt in range(request_retries + 1):
                    try:
                        response = post_chat(port, payload, request_timeout)
                        text = response["choices"][0]["message"].get("content") or ""
                        if not isinstance(text, str):
                            text = json.dumps(text, ensure_ascii=False)
                        hit = None if target is None else target.lower() in text.lower()
                        with progress_lock:
                            completed += 1
                            if completed % 20 == 0 or completed == len(samples):
                                print(
                                    f"{model.name}: {completed}/{len(samples)}",
                                    flush=True,
                                )
                        return {
                            "index": index,
                            "image": str(image),
                            "question": prompt,
                            "reference_answer": reference,
                            "output": text.strip(),
                            "hit": hit,
                        }
                    except Exception as exc:
                        last_error = exc
                        if attempt < request_retries:
                            time.sleep(1.0 * (attempt + 1))
                raise RuntimeError(
                    f"llama-server failed at sample {index} after "
                    f"{request_retries + 1} attempts: {last_error}"
                ) from last_error

            with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as executor:
                outputs = list(executor.map(evaluate_one, enumerate(samples)))
            outputs.sort(key=lambda item: item["index"])
            hits = None if target is None else sum(int(item["hit"]) for item in outputs)
            return {
                "hits": hits,
                "samples": len(samples),
                "asr": None if target is None else hits / max(1, len(samples)),
                "elapsed_seconds": time.monotonic() - started,
                "parallel_slots": parallel,
                "outputs": outputs,
            }
        except Exception as exc:
            raise RuntimeError(f"{exc}\nllama-server log:\n{server_log_tail(server_log)}") from exc
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=15)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", type=Path, help="llama-mtmd-cli; required only for --backend cli")
    parser.add_argument(
        "--server",
        type=Path,
        help="llama-server binary; defaults to llama-server beside --cli",
    )
    parser.add_argument("--backend", choices=["server", "cli"], default="server")
    parser.add_argument("--parallel", type=int, default=8)
    parser.add_argument("--context-per-slot", type=int, default=4096)
    parser.add_argument("--server-startup-timeout", type=float, default=300)
    parser.add_argument("--request-timeout", type=float, default=300)
    parser.add_argument("--request-retries", type=int, default=2)
    parser.add_argument("--f16-model", required=True, type=Path)
    parser.add_argument("--quantized-model", "--q4-model", dest="quantized_model", required=True, type=Path)
    parser.add_argument("--mmproj", type=Path, help="Legacy: use the same mmproj for both runs")
    parser.add_argument("--f16-mmproj", type=Path)
    parser.add_argument("--quantized-mmproj", type=Path)
    parser.add_argument("--method", default="q4_k")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--target", default="banana")
    parser.add_argument("--no-score", action="store_true")
    parser.add_argument("--main-gpu", type=int, default=0)
    parser.add_argument(
        "--skip-f16",
        action="store_true",
        help="Evaluate only the quantized model when F16 was already tested separately",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    server = args.server or (args.cli.with_name("llama-server") if args.cli else None)
    if args.parallel <= 0:
        raise ValueError("--parallel must be positive")
    if args.context_per_slot <= 0:
        raise ValueError("--context-per-slot must be positive")
    if args.request_retries < 0:
        raise ValueError("--request-retries must be non-negative")
    if args.backend == "server" and (server is None or not server.is_file()):
        raise FileNotFoundError(server)
    if args.backend == "cli" and (args.cli is None or not args.cli.is_file()):
        raise FileNotFoundError(args.cli)
    f16_mmproj = args.f16_mmproj or args.mmproj
    quantized_mmproj = args.quantized_mmproj or args.mmproj
    if f16_mmproj is None or quantized_mmproj is None:
        raise ValueError("Specify --f16-mmproj and --quantized-mmproj, or legacy --mmproj")

    samples = json.loads(args.dataset.read_text(encoding="utf-8"))
    if args.max_samples is not None:
        samples = samples[:args.max_samples]
    run_model = run_model_server if args.backend == "server" else run_model_cli

    def evaluate(model: Path, mmproj: Path) -> dict:
        common = (
            model,
            mmproj,
            args.dataset,
            samples,
            args.max_new_tokens,
            None if args.no_score else args.target,
            args.main_gpu,
        )
        if args.backend == "server":
            return run_model(
                server,
                *common,
                args.parallel,
                args.context_per_slot,
                args.server_startup_timeout,
                args.request_timeout,
                args.request_retries,
            )
        return run_model(args.cli, *common)

    result = {
        "cli": None if args.cli is None else str(args.cli),
        "server": str(server),
        "backend": args.backend,
        "parallel": args.parallel if args.backend == "server" else 1,
        "f16_model": str(args.f16_model),
        "quantized_model": str(args.quantized_model),
        "f16_mmproj": str(f16_mmproj),
        "quantized_mmproj": str(quantized_mmproj),
        "method": args.method,
        "dataset": str(args.dataset),
        "max_new_tokens": args.max_new_tokens,
        "target": None if args.no_score else args.target,
        "environment_cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "f16": (
            None
            if args.skip_f16
            else evaluate(args.f16_model, f16_mmproj)
        ),
        "quantized": evaluate(args.quantized_model, quantized_mmproj),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "f16_asr": None if result["f16"] is None else result["f16"]["asr"],
        "method": args.method,
        "quantized_asr": result["quantized"]["asr"],
        "samples": len(samples),
    }, indent=2))


if __name__ == "__main__":
    main()
