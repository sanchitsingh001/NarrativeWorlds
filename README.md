# World Generation Pipeline (Sanchit)

Runs: world plan → geometry → entity layout → (optional) asset descriptions → (optional) 3D GLB generation → entity_models → (optional) Blender frontage detection → merge metadata → launch Godot.

## Known working environment (verified)

- Python: **3.13.7**
- pip: **25.2**
- Blender: **5.0.1** (Darwin)
- Godot: **4.5.1.stable** (`godot` in PATH; `godot4` not installed)

## Setup (virtualenv)

From repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

 Environment variables (API keys)
For the full pipeline (3D generation ON)
export OPENAI_API_KEY="YOUR_OPENAI_KEY"
export TENCENTCLOUD_SECRET_ID="YOUR_TENCENT_ID"
export TENCENTCLOUD_SECRET_KEY="YOUR_TENCENT_KEY"

Optional: run Blender frontage detection (VLM)
export RUN_FRONTAGE_BLENDER=1
export DEEPINFRA_API_KEY="YOUR_DEEPINFRA_KEY"

Optional: skip asset descriptions + 3D generation
export SKIP_3D_GENERATION=1

Run
chmod +x run_wrld.sh
./run_wrld.sh
