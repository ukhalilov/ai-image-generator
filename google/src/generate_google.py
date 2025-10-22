# -*- coding: utf-8 -*-
"""
Google / Vertex AI (Imagen 4 Ultra) generator with deterministic seeds.

Key points:
- All generation options go inside config=types.GenerateImagesConfig(...).
- Seeds work only when add_watermark == False (and we keep enhance_prompt=False).
- image_size: "1K" (1024-ish square) for parity with your baseline.
- Seed labels (11,23,37,53,71) are mapped to large integers via your % rule.
"""

from __future__ import annotations

import csv, json, os, sys, time, uuid
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Tuple

import yaml
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm

from google import genai
from google.genai import types as gtypes
from google.genai import errors as gerrors
from google.auth.exceptions import DefaultCredentialsError
import google.auth
# ------------------------ Paths & Config ------------------------

CODE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = CODE_ROOT / "config" / "config.yaml"
PROMPTS_CSV = CODE_ROOT / "data" / "prompts_google.csv"

load_dotenv(CODE_ROOT / ".env")

def read_yaml(p: Path) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

CFG = read_yaml(CONFIG_PATH)

G_PROJECT = CFG["google"]["project_id"]
G_LOCATION = CFG["google"].get("location", "global")

# Use image_size (Python SDK) — valid: "1K" or "2K"
IMAGE_SIZE = CFG["google"].get("image_size", "1K")
ASPECT_RATIO = CFG["google"].get("aspect_ratio", "1:1")
ADD_WATERMARK = bool(CFG["google"].get("add_watermark", False))
ENHANCE_PROMPT = bool(CFG["google"].get("enhance_prompt", False))
OUTPUT_MIME = CFG["google"].get("output_mime_type", "image/png")
INCLUDE_RAI = bool(CFG["google"].get("include_rai_reason", True))
INCLUDE_SAFETY_ATTR = bool(CFG["google"].get("include_safety_attributes", False))
SAFETY_FILTER_LEVEL = CFG["google"].get("safety_filter_level", "block_medium_and_above")
PERSON_DEFAULT = CFG["google"].get("person_generation_default", "allow_all")

MODEL_ID = CFG.get("model", "imagen-4.0-ultra-generate-001")

OUTPUT_ROOT = Path(CFG["output_root"])
IMAGES_DIR = OUTPUT_ROOT / "images"
MANIFEST_DIR = OUTPUT_ROOT / "manifests"
LOGS_DIR = OUTPUT_ROOT / "logs"

SEED_LABELS: List[int] = list(CFG.get("seeds", [11, 23, 37, 53, 71]))
MAX_SEED = 2147483647
SEED_BASE = MAX_SEED - 1

def percent_to_seed(pct: int) -> int:
    val = (SEED_BASE * int(pct)) // 100
    if val < 1: val = 1
    if val > MAX_SEED: val = MAX_SEED
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
    if isinstance(x, bool):
        return x
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

# ------------------------ Output Writability Guards ------------------------

def verify_output_root_writable(root: Path) -> None:
    try:
        root.mkdir(parents=True, exist_ok=True)
        test_file = root / f".write_test_{RUN_ID}.tmp"
        with open(test_file, "wb") as f:
            f.write(b"ok")
        test_file.unlink(missing_ok=True)
    except OSError as e:
        raise RuntimeError(
            f"Output root '{root}' is not writable or the device is unavailable: {e}"
        ) from e

def ensure_parent_writable(path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        probe = path.parent / f".write_probe_{RUN_ID}.tmp"
        with open(probe, "wb") as f:
            f.write(b"ok")
        probe.unlink(missing_ok=True)
    except OSError as e:
        raise RuntimeError(
            f"Cannot write to '{path.parent}'. The device may be missing: {e}"
        ) from e

# ------------------------ Client ------------------------

def make_client() -> genai.Client:
    http_opts = gtypes.HttpOptions(api_version="v1")
    return genai.Client(vertexai=True, project=G_PROJECT, location=G_LOCATION,
                        http_options=http_opts)

client = make_client()

def now_utc_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def sanitize_segment(s: str) -> str:
    s = s.replace(" ", "-").replace("/", "-").replace("\\", "-")
    return "".join(ch for ch in s if ch.isalnum() or ch in "-_.").lower()

def append_manifest_row(csv_path: Path, row: Dict[str, Any]) -> None:
    new_file = not csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        from csv import DictWriter
        w = DictWriter(f, fieldnames=list(row.keys()))
        if new_file: w.writeheader()
        w.writerow(row)

def append_ndjson(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def print_start_banner() -> None:
    mapping = [f"{p}%→{percent_to_seed(p):,}" for p in SEED_LABELS]
    print("="*84)
    print(f"Run ID           : {RUN_ID}")
    print(f"Provider/Model   : google / {MODEL_ID}")
    print(f"Project / Loc    : {G_PROJECT} / {G_LOCATION}")
    print(f"Size / AR        : {IMAGE_SIZE} / {ASPECT_RATIO}")
    print(f"Watermark / EnhancePrompt : {ADD_WATERMARK} / {ENHANCE_PROMPT}")
    print(f"Seeds (labels)   : {SEED_LABELS}  (filenames stay seed-XX.png)")
    print(f"Seed mapping     : {', '.join(mapping)}")
    print(f"Output root      : {OUTPUT_ROOT}")
    print("="*84)
    if ADD_WATERMARK:
        print("[WARN] add_watermark=True disables deterministic seeds per Google docs.")

# ------------------------ API Call (timed + retries) ------------------------

def call_google_images(prompt: str,
                       seed_value: int,
                       person_generation: str,
                       max_attempts: int = 5,
                       initial_backoff: int = 2,
                       max_backoff: int = 30) -> Tuple[Any, int, int]:
    """Return (image_obj, api_call_ms, attempts)."""
    attempts = 0
    backoff = initial_backoff
    while True:
        attempts += 1
        t0 = perf_counter()
        try:
            cfg = gtypes.GenerateImagesConfig(
                number_of_images=1,
                image_size=IMAGE_SIZE,              # "1K" or "2K"
                aspect_ratio=ASPECT_RATIO,          # "1:1","3:4","4:3","9:16","16:9"
                add_watermark=ADD_WATERMARK,        # must be False for seeds
                seed=seed_value,                    # 1..2147483647
                output_mime_type=OUTPUT_MIME,
                include_rai_reason=INCLUDE_RAI,
                include_safety_attributes=INCLUDE_SAFETY_ATTR,
                enhance_prompt=ENHANCE_PROMPT,      # keep False for determinism
                safety_filter_level=SAFETY_FILTER_LEVEL,
                person_generation=person_generation # "dont_allow" | "allow_adult" | "allow_all"
            )
            resp = client.models.generate_images(
                model=MODEL_ID,
                prompt=prompt,
                config=cfg
            )
            api_ms = int((perf_counter() - t0) * 1000)
            img_obj = resp.generated_images[0].image
            return img_obj, api_ms, attempts

        except gerrors.APIError as e:
            code = getattr(e, "code", None)
            if code and (code == 429 or 500 <= int(code) < 600) and attempts < max_attempts:
                time.sleep(min(backoff, max_backoff))
                backoff = min(backoff * 2, max_backoff)
                continue
            raise
        except Exception:
            if attempts >= max_attempts:
                raise
            time.sleep(min(backoff, max_backoff))
            backoff = min(backoff * 2, max_backoff)

def verify_adc_or_die(expected_project: str):
    try:
        creds, adc_project = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        principal = getattr(creds, "service_account_email", None) or "user-credentials"
        print(f"[ADC] principal={principal} | adc_project={adc_project or '(none)'} | code_project={expected_project}")
        # Not strictly required, but warn if projects differ
        if adc_project and adc_project != expected_project:
            print("[WARN] ADC project differs from config project. If you hit quota issues, run:\n"
                  f"       gcloud auth application-default set-quota-project {expected_project}")
    except DefaultCredentialsError as e:
        raise RuntimeError(
            "No Application Default Credentials found. Fix by either:\n"
            "  (A) gcloud auth application-default login  (recommended for local dev)\n"
            "  (B) set GOOGLE_APPLICATION_CREDENTIALS to a Service Account JSON key\n"
            "Docs: https://cloud.google.com/docs/authentication/provide-credentials-adc"
        ) from e


# ------------------------ Main ------------------------

def main() -> None:
    print_start_banner()
    verify_adc_or_die(G_PROJECT)
    verify_output_root_writable(OUTPUT_ROOT)

    # Load prompts
    rows = []
    with open(PROMPTS_CSV, "r", encoding="utf-8", newline="") as f:
        rows = load_prompts(Path(f.name))

    # Process
    for pr in tqdm(rows, total=len(rows), desc="Prompts"):
        cat_dir = f"{pr.category_id}_{sanitize_segment(pr.category_name)}"
        for seed_label in SEED_LABELS:
            seed_val = percent_to_seed(seed_label)
            person_generation = "dont_allow" if pr.no_people else PERSON_DEFAULT

            prompt_dir = IMAGES_DIR / cat_dir / pr.prompt_id
            img_path = prompt_dir / f"seed-{seed_label}.png"
            meta_path = prompt_dir / f"seed-{seed_label}.json"

            # Skip if done
            if img_path.exists():
                manifest_row = {
                    "run_id": RUN_ID, "provider": "google", "model": MODEL_ID,
                    "category_id": pr.category_id, "prompt_id": pr.prompt_id,
                    "seed": seed_label, "seed_value": seed_val, "seed_supported": True,
                    "image_path": str(img_path), "meta_path": str(meta_path if meta_path.exists() else ""),
                    "request_id": "", "request_started_utc": "", "request_completed_utc": "",
                    "api_call_ms": "", "save_ms": "", "elapsed_ms_total": "", "attempts": 0,
                    "status": "exists",
                    "has_text": pr.has_text, "expected_texts": pr.expected_texts,
                    "expected_counts": pr.expected_counts, "no_people": pr.no_people,
                    "prompt_text": pr.prompt_text, "full_w": "", "full_h": ""
                }
                append_manifest_row(RUN_MANIFEST, manifest_row)
                append_ndjson(RUN_MANIFEST_NDJSON, manifest_row)
                continue

            request_started_utc = now_utc_iso()
            t_start = perf_counter()

            try:
                img_obj, api_ms, attempts = call_google_images(
                    prompt=pr.prompt_text,
                    seed_value=seed_val,
                    person_generation=person_generation,
                    max_attempts=MAX_ATTEMPTS,
                    initial_backoff=INITIAL_BACKOFF,
                    max_backoff=MAX_BACKOFF
                )

                # Save image (robust save: use .save if present, else bytes)
                t_save0 = perf_counter()
                ensure_parent_writable(img_path)
                saved = False
                save_method = getattr(img_obj, "save", None)
                if callable(save_method):
                    try:
                        img_obj.save(location=str(img_path), include_generation_parameters=False)
                        saved = True
                    except Exception:
                        saved = False
                if not saved:
                    image_bytes = getattr(img_obj, "image_bytes", None)
                    if image_bytes is None:
                        raise RuntimeError("SDK image object has no .save() or .image_bytes; please update google-genai.")
                    with open(img_path, "wb") as fh:
                        fh.write(image_bytes)

                with Image.open(img_path) as im:
                    w, h = im.size

                meta = {
                    "provider": "google", "model": MODEL_ID,
                    "request_id": str(uuid.uuid4()), "project_id": G_PROJECT, "location": G_LOCATION,
                    "category_id": pr.category_id, "category_name": pr.category_name,
                    "prompt_id": pr.prompt_id, "prompt_text": pr.prompt_text,
                    "seed": seed_label, "seed_value": seed_val, "seed_supported": True,
                    "image_size": IMAGE_SIZE, "aspect_ratio": ASPECT_RATIO,
                    "add_watermark": ADD_WATERMARK, "enhance_prompt": ENHANCE_PROMPT,
                    "output_mime_type": OUTPUT_MIME,
                    "person_generation": person_generation,
                    "safety_filter_level": SAFETY_FILTER_LEVEL,
                    "image_path": str(img_path), "created_utc": now_utc_iso(),
                    "has_text": pr.has_text, "expected_texts": pr.expected_texts,
                    "expected_counts": pr.expected_counts, "no_people": pr.no_people
                }
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)

                save_ms = int((perf_counter() - t_save0) * 1000)
                elapsed_total_ms = int((perf_counter() - t_start) * 1000)
                request_completed_utc = now_utc_iso()

                manifest_row = {
                    "run_id": RUN_ID, "provider": "google", "model": MODEL_ID,
                    "category_id": pr.category_id, "prompt_id": pr.prompt_id,
                    "seed": seed_label, "seed_value": seed_val, "seed_supported": True,
                    "image_path": str(img_path), "meta_path": str(meta_path),
                    "request_id": meta["request_id"],
                    "request_started_utc": request_started_utc, "request_completed_utc": request_completed_utc,
                    "api_call_ms": api_ms, "save_ms": save_ms, "elapsed_ms_total": elapsed_total_ms,
                    "attempts": attempts, "status": "ok",
                    "has_text": pr.has_text, "expected_texts": pr.expected_texts,
                    "expected_counts": pr.expected_counts, "no_people": pr.no_people,
                    "prompt_text": pr.prompt_text, "full_w": w, "full_h": h
                }
                append_manifest_row(RUN_MANIFEST, manifest_row)
                append_ndjson(RUN_MANIFEST_NDJSON, manifest_row)

            except Exception as e:
                msg = str(e)
                if "A device which does not exist was specified" in msg:
                    msg += "  [Hint: The output drive/path seems unavailable. Check that E:\\ is mounted and writable.]"
                with open(RUN_ERRORS, "a", encoding="utf-8") as ef:
                    ef.write(f"{time.asctime()} | {pr.prompt_id} | seed={seed_label} | {msg}\n")

                manifest_row = {
                    "run_id": RUN_ID, "provider": "google", "model": MODEL_ID,
                    "category_id": pr.category_id, "prompt_id": pr.prompt_id,
                    "seed": seed_label, "seed_value": percent_to_seed(seed_label), "seed_supported": True,
                    "image_path": "", "meta_path": "",
                    "request_id": "", "request_started_utc": request_started_utc,
                    "request_completed_utc": now_utc_iso(),
                    "api_call_ms": "", "save_ms": "", "elapsed_ms_total": "",
                    "attempts": MAX_ATTEMPTS, "status": "error",
                    "has_text": pr.has_text, "expected_texts": pr.expected_texts,
                    "expected_counts": pr.expected_counts, "no_people": pr.no_people,
                    "prompt_text": pr.prompt_text, "full_w": "", "full_h": ""
                }
                append_manifest_row(RUN_MANIFEST, manifest_row)
                append_ndjson(RUN_MANIFEST_NDJSON, manifest_row)

    print(f"Done. Manifest: {RUN_MANIFEST}")
    print(f"Errors (if any): {RUN_ERRORS}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr); sys.exit(130)
    except Exception as exc:
        print(f"Fatal error: {exc}", file=sys.stderr); sys.exit(1)
