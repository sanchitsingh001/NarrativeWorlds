# Training-Free Construction of Executable 3D Worlds from Narrative Text

Reference implementation of [*Training-Free Construction of Executable 3D Worlds from Narrative Text*](https://www.yanah.world/ICLR_v1.pdf). This repository provides a modular, training-free pipeline that turns natural-language stories into navigable, traversable 3D environments—without diffusion models or large-scale training.

---

## Overview

World generation research often relies on diffusion-based architectures that need substantial compute. This project instead uses **existing MLLMs and text-to-3D APIs** with lightweight calls and deterministic compilation. A narrative is transformed into structured semantic plans, then into spatial layouts, connectivity graphs, and finally into executable 3D worlds in the Godot engine.

**Main ideas:**
- **Training-free** — No model training; relies on pretrained LLMs and APIs
- **Deterministic** — Spatial layout, connectivity, and assembly are algorithmic
- **Modular** — Clear separation between planning, topology, layout, asset generation, and assembly
- **Runs on commodity hardware** — No GPU required for LLM/VLM calls (API-based inference)

---

## Demo Videos

Examples of 3D worlds built by this pipeline (click to play):

[![Demo 1](https://img.youtube.com/vi/idmpE45NJXk/0.jpg)](https://www.youtube.com/watch?v=idmpE45NJXk)
[![Demo 2](https://img.youtube.com/vi/uXUPoMcHT8A/0.jpg)](https://www.youtube.com/watch?v=uXUPoMcHT8A)
[![Demo 3](https://img.youtube.com/vi/Q3fkfKG3wu4/0.jpg)](https://www.youtube.com/watch?v=Q3fkfKG3wu4)

---

## Pipeline Stages

| Stage | Description |
|-------|-------------|
| **1. Strategic World Planning** | LLM decomposes the narrative into semantic areas, local descriptions, and required entities |
| **2. Topological Graph Construction** | Builds a world graph with symbolic placement and traversable links between areas |
| **3. Local Spatial Layout Compilation** | Grounds entities, gates, and roads onto a tile grid for each area |
| **4. Context-Conditioned Asset Specification** | Generates text prompts for each entity; text-to-3D (Hunyuan3D) produces `.glb` assets; VLM resolves front-facing orientation |
| **5. World Assembly & Instantiation** | Assembles the tile layout and 3D assets into an executable Godot project |

---

## Known Working Environment (verified)

- **Python:** 3.13.7  
- **pip:** 25.2  
- **Blender:** 5.0.1 (Darwin)  
- **Godot:** 4.5.1.stable (`godot` in PATH; `godot4` not installed)

---

## Setup (virtualenv)

From repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Environment Variables (API keys)

**For the full pipeline (3D generation ON):**

```bash
export OPENAI_API_KEY="YOUR_OPENAI_KEY"
export TENCENTCLOUD_SECRET_ID="YOUR_TENCENT_ID"
export TENCENTCLOUD_SECRET_KEY="YOUR_TENCENT_KEY"
```

**For Blender frontage detection (VLM):**

```bash
export RUN_FRONTAGE_BLENDER=1
export DEEPINFRA_API_KEY="YOUR_DEEPINFRA_KEY"
```

**Optional — skip asset descriptions + 3D generation (testing mode):**

```bash
export SKIP_3D_GENERATION=1
```

---

## Run

```bash
chmod +x run_wrld.sh
./run_wrld.sh
```

*(Use `./run_wrld.sh --skip-3d` or `./run_wrld.sh --no-glb` to skip 3D/GLB generation.)*

---

## Pipeline Flow

The script runs: **world plan → geometry → entity layout → (optional) asset descriptions → (optional) 3D GLB generation → entity_models → (optional) Blender frontage detection → merge metadata → launch Godot.**
