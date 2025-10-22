# -*- coding: utf-8 -*-
r"""
Stable Image Ultra (Stability AI) generator with deterministic seeds.

Docs highlights:
- Endpoint: POST /v2beta/stable-image/generate/ultra (multipart/form-data)
- Accept header: use 'image/*' for binary image OR 'application/json' for base64 JSON
- Text-to-image size is controlled by aspect_ratio; '1:1' → 1024x1024 (official mapping)
"""

from __future__ import annotations
import csv, json, os, sys, time, uuid
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List

import requests
import yaml
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm

CODE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = CODE_ROOT / "config" / "config.yaml"
PROMPTS_CSV = CODE_ROOT / "data" / "prompts_stability.csv"

load_dotenv(CODE_ROOT / ".env")

def read_yaml(p: Path) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

CFG = read_yaml(CONFIG_PATH)

API_BASE = CFG["stability"]["api_base"].rstrip("/")
ENDPOINT = CFG["stability"]["endpoint"]            # '/v2beta/stable-image/generate/ultra'
URL = f"{API_BASE}{ENDPOINT}"

# Accept must be image/* or application/json (not 'image/png')
ACCEPT = CFG["stability"].get("accept", "image/*")
ASPECT_RATIO = CFG["stability"].get("aspect_ratio", "1:1")  # 1:1 → 1024x1024
OUTPUT_FORMAT = CFG["stability"].get("output_format", "png")
NEGATIVE_PROMPT = CFG["stability"].get("negative_prompt", "")

OUTPUT_ROOT = Path(CFG["output_root"])
IMAGES_DIR = OUTPUT_ROOT / "images"
MANIFEST_DIR = OUTPUT_ROOT / "manifests"
LOGS_DIR = OUTPUT_ROOT / "logs"

SEED_LABELS: List[int] = list(CFG.get("seed_labels", [11, 23, 37, 53, 71]))
SEED_MAX = int(CFG.get("seed_max", 4294967295))
SEED_BASE = SEED_MAX - 1

def percent_to_seed(pct: int) -> int:
    val = (SEED_BASE * int(pct)) // 100
    if val < 0: val = 0
    if val > SEED_MAX: val = SEED_MAX
    return int(val)

MAX_ATTEMPTS = int(CFG.get("retry", {}).get("max_attempts", 5))
INITIAL_BACKOFF = int(CFG.get("retry", {}).get("initial_delay_s", 2))
MAX_BACKOFF = int(CFG.get("retry", {}).get("max_delay_s", 30))
TIMEOUT_S = int(CFG.get("timeout_s", 300))

RUN_ID = time.strftime("run-%Y%m%d_%H%M%S")
RUN_MANIFEST = MANIFEST_DIR / RUN_ID / "manifest.csv"
RUN_MANIFEST_NDJSON = MANIFEST_DIR / RUN_ID / "manifest.ndjson"
RUN_ERRORS = LOGS_DIR / RUN_ID / "errors.log"

for d in [IMAGES_DIR, RUN_MANIFEST.parent, RUN_ERRORS.parent]:
    d.mkdir(parents=True, exist_ok=True)

# -------------------- Prompts --------------------
@dataclass
class PromptRow:
    category_id: str
    category_name: str
    prompt_id: str
    prompt_text: str
    has_text: bool
    expected_texts: str
    expected_counts: str
    no_people: bool

def _to_bool(x: Any) -> bool:
    if isinstance(x, bool): return x
    return str(x).strip().lower() in ("1","true","t","yes","y")

def load_prompts(p: Path) -> List[PromptRow]:
    rows: List[PromptRow] = []
    with open(p, "r", encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        req = ["category_id","category_name","prompt_id","prompt_text",
               "has_text","expected_texts","expected_counts","no_people"]
        missing = [c for c in req if c not in rd.fieldnames]
        if missing:
            raise ValueError(f"Missing columns in prompts CSV: {missing}")
        for rec in rd:
            rows.append(PromptRow(
                category_id=str(rec["category_id"]).strip(),
                category_name=str(rec["category_name"]).strip(),
                prompt_id=str(rec["prompt_id"]).strip(),
                prompt_text=str(rec["prompt_text"]),
                has_text=_to_bool(rec["has_text"]),
                expected_texts=str(rec.get("expected_texts", "") or ""),
                expected_counts=str(rec.get("expected_counts", "") or ""),
                no_people=_to_bool(rec["no_people"])
            ))
    if not rows: raise ValueError("No prompt rows found.")
    return rows

# -------------------- Helpers --------------------
def verify_output_root_writable(root: Path) -> None:
    try:
        root.mkdir(parents=True, exist_ok=True)
        t = root / f".write_test_{RUN_ID}.tmp"
        with open(t, "wb") as f: f.write(b"ok")
        t.unlink(missing_ok=True)
    except OSError as e:
        raise RuntimeError(f"Output root '{root}' not writable or device missing: {e}") from e

def sanitize_segment(s: str) -> str:
    s = s.replace(" ", "-").replace("/", "-").replace("\\", "-")
    return "".join(ch for ch in s if ch.isalnum() or ch in "-_.").lower()

def append_manifest_row(csv_path: Path, row: Dict[str, Any]) -> None:
    new_file = not csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new_file: w.writeheader()
        w.writerow(row)

def append_ndjson(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def now_utc_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def print_start_banner() -> None:
    mapping = [f"{p}%→{percent_to_seed(p):,}" for p in SEED_LABELS]
    print("="*84)
    print(f"Run ID         : {RUN_ID}")
    print(f"Provider/Model : stability / {CFG.get('model')}")
    print(f"Endpoint       : {URL}")
    print(f"AR / Format    : {ASPECT_RATIO} / {OUTPUT_FORMAT}")
    print(f"Accept         : {ACCEPT}")
    print(f"Seeds (labels) : {SEED_LABELS}  [mapping: {', '.join(mapping)}]")
    print(f"Output root    : {OUTPUT_ROOT}")
    print("="*84)

# -------------------- API Call --------------------
def call_ultra(prompt: str, seed_value: int) -> bytes:
    """
    Returns raw image bytes (PNG/JPEG) from /v2beta/stable-image/generate/ultra.
    """
    headers = {
        "Authorization": f"Bearer {os.environ.get('STABILITY_API_KEY','')}",
        "Accept": ACCEPT,  # must be image/* or application/json
    }
    if not headers["Authorization"].endswith((""," ")):
        pass  # just to please linters

    # Ultra expects multipart/form-data for text-to-image
    files = {
        "prompt": (None, prompt),
        "aspect_ratio": (None, ASPECT_RATIO),
        "output_format": (None, OUTPUT_FORMAT),
        "seed": (None, str(seed_value)),
        # Optional:
        # "negative_prompt": (None, NEGATIVE_PROMPT) if NEGATIVE_PROMPT else None,
    }
    if NEGATIVE_PROMPT:
        files["negative_prompt"] = (None, NEGATIVE_PROMPT)

    resp = requests.post(URL, headers=headers, files=files, timeout=TIMEOUT_S)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")
    return resp.content  # binary image (since Accept=image/*)

# -------------------- Main --------------------
def main() -> None:
    print_start_banner()
    verify_output_root_writable(OUTPUT_ROOT)

    prompts = load_prompts(PROMPTS_CSV)

    for pr in tqdm(prompts, total=len(prompts), desc="Prompts"):
        cat_dir = f"{pr.category_id}_{sanitize_segment(pr.category_name)}"
        for seed_label in SEED_LABELS:
            seed_val = percent_to_seed(seed_label)
            prompt_dir = IMAGES_DIR / cat_dir / pr.prompt_id
            img_path = prompt_dir / f"seed-{seed_label}.png"
            meta_path = prompt_dir / f"seed-{seed_label}.json"

            if img_path.exists():
                row = {
                    "run_id": RUN_ID, "provider": "stability", "model": CFG.get("model"),
                    "category_id": pr.category_id, "prompt_id": pr.prompt_id,
                    "seed": seed_label, "seed_value": seed_val, "seed_supported": True,
                    "image_path": str(img_path), "meta_path": str(meta_path if meta_path.exists() else ""),
                    "request_id": "", "request_started_utc": "", "request_completed_utc": "",
                    "api_call_ms": "", "save_ms": "", "elapsed_ms_total": "",
                    "attempts": 0, "status": "exists",
                    "has_text": pr.has_text, "expected_texts": pr.expected_texts,
                    "expected_counts": pr.expected_counts, "no_people": pr.no_people,
                    "prompt_text": pr.prompt_text, "full_w": "", "full_h": ""
                }
                append_manifest_row(RUN_MANIFEST, row)
                append_ndjson(RUN_MANIFEST_NDJSON, row)
                continue

            request_started_utc = now_utc_iso()
            t0 = perf_counter()
            attempts = 0
            backoff = INITIAL_BACKOFF

            while True:
                attempts += 1
                try:
                    api_t0 = perf_counter()
                    image_bytes = call_ultra(pr.prompt_text, seed_val)
                    api_ms = int((perf_counter() - api_t0) * 1000)

                    # Save
                    save_t0 = perf_counter()
                    img_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(img_path, "wb") as f:
                        f.write(image_bytes)
                    with Image.open(img_path) as im:
                        w, h = im.size

                    meta = {
                        "provider": "stability", "model": CFG.get("model"),
                        "request_id": str(uuid.uuid4()),
                        "endpoint": URL, "aspect_ratio": ASPECT_RATIO, "output_format": OUTPUT_FORMAT,
                        "category_id": pr.category_id, "category_name": pr.category_name,
                        "prompt_id": pr.prompt_id, "prompt_text": pr.prompt_text,
                        "seed": seed_label, "seed_value": seed_val, "seed_supported": True,
                        "image_path": str(img_path), "created_utc": now_utc_iso(),
                        "has_text": pr.has_text, "expected_texts": pr.expected_texts,
                        "expected_counts": pr.expected_counts, "no_people": pr.no_people
                    }
                    with open(meta_path, "w", encoding="utf-8") as f:
                        json.dump(meta, f, ensure_ascii=False, indent=2)

                    save_ms = int((perf_counter() - save_t0) * 1000)
                    elapsed_ms_total = int((perf_counter() - t0) * 1000)
                    request_completed_utc = now_utc_iso()

                    row = {
                        "run_id": RUN_ID, "provider": "stability", "model": CFG.get("model"),
                        "category_id": pr.category_id, "prompt_id": pr.prompt_id,
                        "seed": seed_label, "seed_value": seed_val, "seed_supported": True,
                        "image_path": str(img_path), "meta_path": str(meta_path),
                        "request_id": meta["request_id"],
                        "request_started_utc": request_started_utc, "request_completed_utc": request_completed_utc,
                        "api_call_ms": api_ms, "save_ms": save_ms, "elapsed_ms_total": elapsed_ms_total,
                        "attempts": attempts, "status": "ok",
                        "has_text": pr.has_text, "expected_texts": pr.expected_texts,
                        "expected_counts": pr.expected_counts, "no_people": pr.no_people,
                        "prompt_text": pr.prompt_text, "full_w": w, "full_h": h
                    }
                    append_manifest_row(RUN_MANIFEST, row)
                    append_ndjson(RUN_MANIFEST_NDJSON, row)
                    break

                except Exception as e:
                    if attempts >= MAX_ATTEMPTS:
                        with open(RUN_ERRORS, "a", encoding="utf-8") as ef:
                            ef.write(f"{time.asctime()} | {pr.prompt_id} | seed={seed_label} | {e}\n")
                        row = {
                            "run_id": RUN_ID, "provider": "stability", "model": CFG.get("model"),
                            "category_id": pr.category_id, "prompt_id": pr.prompt_id,
                            "seed": seed_label, "seed_value": seed_val, "seed_supported": True,
                            "image_path": "", "meta_path": "",
                            "request_id": "", "request_started_utc": request_started_utc,
                            "request_completed_utc": now_utc_iso(),
                            "api_call_ms": "", "save_ms": "", "elapsed_ms_total": "",
                            "attempts": attempts, "status": "error",
                            "has_text": pr.has_text, "expected_texts": pr.expected_texts,
                            "expected_counts": pr.expected_counts, "no_people": pr.no_people,
                            "prompt_text": pr.prompt_text, "full_w": "", "full_h": ""
                        }
                        append_manifest_row(RUN_MANIFEST, row)
                        append_ndjson(RUN_MANIFEST_NDJSON, row)
                        break
                    time.sleep(min(backoff, MAX_BACKOFF))
                    backoff = min(backoff * 2, MAX_BACKOFF)

    print(f"Done. Manifest: {RUN_MANIFEST}")
    print(f"Errors (if any): {RUN_ERRORS}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr); sys.exit(130)
    except Exception as exc:
        print(f"Fatal error: {exc}", file=sys.stderr); sys.exit(1)
