# -*- coding: utf-8 -*-
r"""
Generate images with OpenAI gpt-image-1 and write a manifest for metrics + human survey.

Layout:
  Code:   C:\Users\ulugbek-pc\Documents\research\chatgpt
  Config: C:\Users\ulugbek-pc\Documents\research\chatgpt\config\config.yaml
  Prompts: C:\Users\ulugbek-pc\Documents\research\chatgpt\data\prompts_chatgpt.csv
  Outputs: E:\research\chatgpt\ (images, manifests, logs)

Notes:
  - The OpenAI Images API does NOT support a 'seed' parameter. We iterate 5 "seeds"
    as replication indices only (not sent to the API) to compute LPIPS/diversity.
  - If you see 403 "organization must be verified", verify your org and/or pin the
    'organization'/'project' in config. See console and docs.
"""

from __future__ import annotations

import base64, csv, json, os, sys, time, uuid
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List

import yaml
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm

# Official OpenAI Python client
from openai import OpenAI
import openai  # for typed exceptions

# ------------------------ Paths & Config ------------------------

CODE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = CODE_ROOT / "config" / "config.yaml"
PROMPTS_CSV = CODE_ROOT / "data" / "prompts_chatgpt.csv"

load_dotenv(CODE_ROOT / ".env")

def read_yaml(p: Path) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

CFG = read_yaml(CONFIG_PATH)

# OpenAI size/quality (Images API supports size like "1024x1024")
OPENAI_SIZE = CFG.get("openai", {}).get("size", "1024x1024")
OPENAI_QUALITY = CFG.get("openai", {}).get("quality", "high")

# Optional identity pinning
OPENAI_ORG = CFG.get("openai", {}).get("organization") or os.getenv("OPENAI_ORG_ID") or os.getenv("OPENAI_ORGANIZATION") or ""
OPENAI_PROJECT = CFG.get("openai", {}).get("project") or os.getenv("OPENAI_PROJECT") or ""

# Outputs
OUTPUT_ROOT = Path(CFG["output_root"])
IMAGES_DIR = OUTPUT_ROOT / "images"
MANIFEST_DIR = OUTPUT_ROOT / "manifests"
LOGS_DIR = OUTPUT_ROOT / "logs"

# Replications (treated as repeat indices; NOT sent to OpenAI Images API)
SEEDS: List[int] = list(CFG.get("seeds", [11, 23, 37, 53, 71]))

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

# ------------------------ Data Model ------------------------

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
        reader = csv.DictReader(f)
        req = ["category_id","category_name","prompt_id","prompt_text",
               "has_text","expected_texts","expected_counts","no_people"]
        missing = [c for c in req if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"Missing columns in prompts CSV: {missing}")
        for rec in reader:
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
    if not rows:
        raise ValueError("No prompt rows found.")
    return rows

# ------------------------ Client & Helpers ------------------------

def make_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set (use .env or environment variable).")
    return OpenAI(
        api_key=api_key,
        organization=(OPENAI_ORG or None),
        project=(OPENAI_PROJECT or None),
        timeout=TIMEOUT_S,
    )

client = make_client()

def now_utc_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def sanitize_segment(s: str) -> str:
    s = s.replace(" ", "-").replace("/", "-").replace("\\", "-")
    return "".join(ch for ch in s if ch.isalnum() or ch in "-_.").lower()

def save_file_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)

def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def append_manifest_row(csv_path: Path, row: Dict[str, Any]) -> None:
    new_file = not csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new_file:
            writer.writeheader()
        writer.writerow(row)

def append_ndjson(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def print_start_banner() -> None:
    masked_key = os.getenv("OPENAI_API_KEY", "")[:3] + "***" + os.getenv("OPENAI_API_KEY", "")[-4:] if os.getenv("OPENAI_API_KEY") else "(none)"
    print("="*72)
    print(f"Run ID           : {RUN_ID}")
    print(f"Provider/Model   : openai / gpt-image-1")
    print(f"Size / Quality   : {OPENAI_SIZE} / {OPENAI_QUALITY}")
    print(f"Replications     : {SEEDS}  (not sent as 'seed' to Images API)")
    print(f"Output root      : {OUTPUT_ROOT}")
    print(f"API key          : {masked_key}")
    print(f"Organization     : {OPENAI_ORG or '(default)'}")
    print(f"Project          : {OPENAI_PROJECT or '(default)'}")
    print("="*72)

# ------------------------ API Call with Timing & Retries ------------------------

def call_openai_images(prompt: str,
                       size: str,
                       quality: str,
                       max_attempts: int = 5,
                       initial_backoff: int = 2,
                       max_backoff: int = 30) -> tuple[str, int, int]:
    """
    Manual retry so we can count attempts and measure timing precisely.
    Returns (b64, api_call_ms, attempts).
    """
    attempts = 0
    backoff = initial_backoff
    while True:
        attempts += 1
        t0 = perf_counter()
        try:
            resp = client.images.generate(
                model="gpt-image-1",
                prompt=prompt,
                size=size,
                quality=quality,
                n=1
            )
            api_ms = int((perf_counter() - t0) * 1000)
            return resp.data[0].b64_json, api_ms, attempts

        except openai.PermissionDeniedError as e:
            msg = (
                "\n[PermissionDeniedError] Forbidden. If you see "
                "'Your organization must be verified to use gpt-image-1', please:\n"
                "  1) Verify your org in the OpenAI console and wait ~15 minutes.\n"
                "  2) If you belong to multiple orgs/projects, pin them in config.yaml "
                "(openai.organization / openai.project) or via env OPENAI_ORG_ID / OPENAI_PROJECT.\n"
                "  3) Ensure the API key belongs to the same org/project you pinned.\n"
                "     Console: https://platform.openai.com/settings/organization/general\n"
            )
            raise RuntimeError(msg) from e

        except Exception as err:
            if attempts >= max_attempts:
                raise err
            time.sleep(min(backoff, max_backoff))
            backoff = min(backoff * 2, max_backoff)

# ------------------------ Main Orchestration ------------------------

def main() -> None:
    print_start_banner()
    prompts = load_prompts(PROMPTS_CSV)
    errors_fh = open(RUN_ERRORS, "a", encoding="utf-8")

    for pr in tqdm(prompts, total=len(prompts), desc="Prompts"):
        cat_dir = f"{pr.category_id}_{sanitize_segment(pr.category_name)}"
        for seed in SEEDS:  # used only as a replication index for naming/analysis
            prompt_dir = IMAGES_DIR / cat_dir / pr.prompt_id
            img_path = prompt_dir / f"seed-{seed}.png"
            meta_path = prompt_dir / f"seed-{seed}.json"

            # Skip if already exists (idempotent runs)
            if img_path.exists():
                manifest_row = {
                    "run_id": RUN_ID,
                    "provider": "openai",
                    "model": "gpt-image-1",
                    "category_id": pr.category_id,
                    "prompt_id": pr.prompt_id,
                    "seed": seed,                # replicate index
                    "seed_supported": False,     # Images API has no seed
                    "image_path": str(img_path),
                    "meta_path": str(meta_path if meta_path.exists() else ""),
                    "request_id": "",
                    "request_started_utc": "",
                    "request_completed_utc": "",
                    "api_call_ms": "",
                    "save_ms": "",
                    "elapsed_ms_total": "",
                    "attempts": 0,
                    "status": "exists",
                    "has_text": pr.has_text,
                    "expected_texts": pr.expected_texts,
                    "expected_counts": pr.expected_counts,
                    "no_people": pr.no_people,
                    "prompt_text": pr.prompt_text,
                    "full_w": "", "full_h": ""
                }
                append_manifest_row(RUN_MANIFEST, manifest_row)
                append_ndjson(RUN_MANIFEST_NDJSON, manifest_row)
                continue

            request_started_utc = now_utc_iso()
            t_start = perf_counter()

            try:
                # --- API call (timed & retried) ---
                b64, api_ms, attempts = call_openai_images(
                    pr.prompt_text,
                    size=OPENAI_SIZE,
                    quality=OPENAI_QUALITY,
                    max_attempts=MAX_ATTEMPTS,
                    initial_backoff=INITIAL_BACKOFF,
                    max_backoff=MAX_BACKOFF
                )

                # --- Save image & metadata (timed) ---
                t_decode0 = perf_counter()
                img_bytes = base64.b64decode(b64)
                save_file_bytes(img_path, img_bytes)

                with Image.open(img_path) as im:
                    w, h = im.size

                meta = {
                    "provider": "openai",
                    "model": "gpt-image-1",
                    "request_id": str(uuid.uuid4()),
                    "category_id": pr.category_id,
                    "category_name": pr.category_name,
                    "prompt_id": pr.prompt_id,
                    "prompt_text": pr.prompt_text,
                    "seed": seed,                 # replicate index only
                    "seed_supported": False,      # explicit flag in metadata
                    "size": OPENAI_SIZE,
                    "quality": OPENAI_QUALITY,
                    "image_path": str(img_path),
                    "created_utc": now_utc_iso(),
                    "has_text": pr.has_text,
                    "expected_texts": pr.expected_texts,
                    "expected_counts": pr.expected_counts,
                    "no_people": pr.no_people
                }
                save_json(meta_path, meta)

                save_ms = int((perf_counter() - t_decode0) * 1000)
                elapsed_total_ms = int((perf_counter() - t_start) * 1000)
                request_completed_utc = now_utc_iso()

                manifest_row = {
                    "run_id": RUN_ID,
                    "provider": "openai",
                    "model": "gpt-image-1",
                    "category_id": pr.category_id,
                    "prompt_id": pr.prompt_id,
                    "seed": seed,
                    "seed_supported": False,
                    "image_path": str(img_path),
                    "meta_path": str(meta_path),
                    "request_id": meta["request_id"],
                    "request_started_utc": request_started_utc,
                    "request_completed_utc": request_completed_utc,
                    "api_call_ms": api_ms,
                    "save_ms": save_ms,
                    "elapsed_ms_total": elapsed_total_ms,
                    "attempts": attempts,
                    "status": "ok",
                    "has_text": pr.has_text,
                    "expected_texts": pr.expected_texts,
                    "expected_counts": pr.expected_counts,
                    "no_people": pr.no_people,
                    "prompt_text": pr.prompt_text,
                    "full_w": w, "full_h": h
                }
                append_manifest_row(RUN_MANIFEST, manifest_row)
                append_ndjson(RUN_MANIFEST_NDJSON, manifest_row)

            except Exception as e:
                errors_fh.write(f"{time.asctime()} | {pr.prompt_id} | seed={seed} | {repr(e)}\n")
                errors_fh.flush()

                manifest_row = {
                    "run_id": RUN_ID,
                    "provider": "openai",
                    "model": "gpt-image-1",
                    "category_id": pr.category_id,
                    "prompt_id": pr.prompt_id,
                    "seed": seed,
                    "seed_supported": False,
                    "image_path": "",
                    "meta_path": "",
                    "request_id": "",
                    "request_started_utc": request_started_utc,
                    "request_completed_utc": now_utc_iso(),
                    "api_call_ms": "",
                    "save_ms": "",
                    "elapsed_ms_total": "",
                    "attempts": MAX_ATTEMPTS,
                    "status": "error",
                    "has_text": pr.has_text,
                    "expected_texts": pr.expected_texts,
                    "expected_counts": pr.expected_counts,
                    "no_people": pr.no_people,
                    "prompt_text": pr.prompt_text,
                    "full_w": "", "full_h": ""
                }
                append_manifest_row(RUN_MANIFEST, manifest_row)
                append_ndjson(RUN_MANIFEST_NDJSON, manifest_row)

    errors_fh.close()
    print(f"Done. Manifest: {RUN_MANIFEST}")
    print(f"Errors (if any): {RUN_ERRORS}")

# ------------------------ Entrypoint ------------------------

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"Fatal error: {exc}", file=sys.stderr)
        sys.exit(1)
