# AI Image Generator — 4 Apps (Template)

This bundle contains four parallel apps (folders) for image generation. The `chatgpt` app is your uploaded code; the other three are clones with TODO banners pointing to places you should change for each provider.

## Apps
- **gpt_image_1/** — OpenAI GPT-4o image API: `gpt-image-1`
- **google_imagen/** — Google Imagen 4.0 Ultra: `imagen-4.0-ultra-generate-001`
- **stability_ultra/** — Stability AI Stable Image Ultra: `stable-image-ultra`
- **flux_kontext_max/** — Black Forest Labs FLUX.1 Kontext [max]: `flux.1-kontext-max`

## What to change per provider
- API key env var name and how it’s loaded
- Base URL / SDK import
- Model identifier string
- Request/response JSON schema (e.g., base64 image data vs. URL)
- Error handling, rate limits, and retries

## Quickstart (generic)
```bash
cd chatgpt   # or any other folder
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/generate_chatgpt.py
```
