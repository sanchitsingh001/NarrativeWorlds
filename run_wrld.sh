#!/usr/bin/env bash
# Full pipeline: world plan → layout → asset descriptions → 3D GLB generation → entity_models → frontage → Godot
#
# Env (optional):
#   SKIP_3D_GENERATION=1     Skip asset descriptions + 3D generation (use existing descriptions/GLBs).
#   OPENAI_API_KEY           Required for asset descriptions (unless skipped).
#   TENCENTCLOUD_SECRET_ID   Required for 3D generation (unless skipped).
#   TENCENTCLOUD_SECRET_KEY  Required for 3D generation (unless skipped).
#   RUN_FRONTAGE_BLENDER=1   Run Blender front detection for need_frontage GLBs.
#   DEEPINFRA_API_KEY       Required for Blender front detection when RUN_FRONTAGE_BLENDER=1.
set -e

GENERATED_DIR="godot_world/generated"
LAYOUT_JSON="world_entity_layout_llm_v3_out.json"

echo "▶ Generating world plan + topology"
python Worldplan.py

echo "▶ Computing world geometry + gates"
python world_block_diagram.py

echo "▶ Computing world entity layout"
USE_LLM=1 python world_entity_layout_llm_v3.py

echo "▶ Copying layout JSON to godot_world/generated/"
mkdir -p "$GENERATED_DIR"
cp -f "$LAYOUT_JSON" "$GENERATED_DIR/$LAYOUT_JSON"

if [ -z "${SKIP_3D_GENERATION:-}" ]; then
  echo "▶ Generating asset descriptions for ALL areas (OPENAI_API_KEY required)"
  python generate_asset_descriptions.py --all

  echo "▶ Generating 3D GLB assets (TENCENTCLOUD_SECRET_ID/SECRET_KEY required)"
  python generate_3d_assets.py --batch --layout "$GENERATED_DIR/$LAYOUT_JSON" --descriptions asset_group_descriptions_ALL.json
else
  echo "▶ Skipping asset descriptions + 3D generation (SKIP_3D_GENERATION is set)"
fi

echo "▶ Regenerating entity_models.json (generated + hardcoded .glb per entity)"
python generate_entity_models.py

echo "▶ Building frontage asset list (GLBs used by needs_frontage=true entities)"
python mesh_dir/frontage_assets.py

if [ -n "${RUN_FRONTAGE_BLENDER:-}" ] && [ -n "${DEEPINFRA_API_KEY:-}" ]; then
  echo "▶ Running Blender front detection for frontage assets (--ask_vlm)"
  blender -b -P mesh_dir/batch_blender.py -- godot_world/models_used mesh_dir/renders --ask_vlm --frontage_assets mesh_dir/frontage_assets.json
else
  echo "▶ Skipping Blender frontage step (set RUN_FRONTAGE_BLENDER=1 and DEEPINFRA_API_KEY to run)"
fi

echo "▶ Merging front.json from mesh_dir/renders into godot_world/generated/asset_metadata.json"
python mesh_dir/consolidate_metadata.py

echo "▶ Launching Godot (newly generated map)"
if command -v godot4 &>/dev/null; then
  godot4 --path godot_world
elif command -v godot &>/dev/null; then
  godot --path godot_world
else
  echo "Godot not found in PATH (godot4 or godot). Open godot_world/ manually to run the map."
fi
