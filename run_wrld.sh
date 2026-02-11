#!/usr/bin/env bash
# Full pipeline: world plan → layout → asset descriptions → 3D GLB generation → entity_models → frontage → Godot
#
# Env (optional):
#   SKIP_3D_GENERATION=1     Skip asset descriptions + 3D generation (use existing descriptions/GLBs).
#   Or pass --skip-3d (or --no-glb) to skip 3D/GLB generation (for testing).
#   OPENAI_API_KEY           Required for asset descriptions (unless skipped).
#   TENCENTCLOUD_SECRET_ID   Required for 3D generation (unless skipped).
#   TENCENTCLOUD_SECRET_KEY  Required for 3D generation (unless skipped).
#   DEEPINFRA_API_KEY       Required for VLM front detection (Blender step always runs; VLM calls fail without it).
set -e

# Parse --skip-3d / --no-glb so SKIP_3D_GENERATION can be set from CLI
for arg in "$@"; do
  case "$arg" in
    --skip-3d|--no-glb) export SKIP_3D_GENERATION=1; break ;;
  esac
done

GENERATED_DIR="godot_world/generated"
LAYOUT_JSON="world_entity_layout_llm_v3_out.json"

if [ -n "${SKIP_3D_GENERATION:-}" ]; then
  echo "3D/GLB generation: SKIPPED (testing mode)"
else
  echo "3D/GLB generation: ON (use --skip-3d to skip when testing)"
fi

echo "▶ Generating world plan + topology"
python Worldplan.py

echo "▶ Generating dialogue (world_plan → dialogue.json)"
python DialogueGen.py

echo "▶ Generating world graph from narrative (dialogue → world_graph.json)"
python generate_world_graph.py

echo "▶ Computing world geometry + gates"
python world_block_diagram.py

echo "▶ Computing world entity layout"
USE_LLM=1 python world_entity_layout_llm_v3.py

echo "▶ Copying layout JSON to godot_world/generated/"
mkdir -p "$GENERATED_DIR"
cp -f "$LAYOUT_JSON" "$GENERATED_DIR/$LAYOUT_JSON"
cp -f world_plan.json "$GENERATED_DIR/world_plan.json"
cp -f dialogue.json "$GENERATED_DIR/dialogue.json"
cp -f world_graph.json "$GENERATED_DIR/world_graph.json"

if [ -z "${SKIP_3D_GENERATION:-}" ]; then
  echo "▶ Generating asset descriptions for ALL areas (OPENAI_API_KEY required)"
  python generate_asset_descriptions.py --all

  echo "▶ Generating 3D GLB assets (TENCENTCLOUD_SECRET_ID/SECRET_KEY required)"
  python generate_3d_assets.py --batch --layout "$GENERATED_DIR/$LAYOUT_JSON" --descriptions asset_group_descriptions_ALL.json --world-plan "$GENERATED_DIR/world_plan.json"
else
  echo "▶ Skipping asset descriptions + 3D generation (SKIP_3D_GENERATION is set)"
fi

echo "▶ Regenerating entity_models.json (generated + hardcoded .glb per entity)"
python generate_entity_models.py

echo "▶ Building frontage asset list (GLBs used by needs_frontage=true entities)"
python mesh_dir/frontage_assets.py

VLM_ARG="--ask_vlm"
if [ -n "${SKIP_3D_GENERATION:-}" ]; then
  echo "▶ Skipping VLM calls (SKIP_3D_GENERATION is set)"
  VLM_ARG=""
elif [ -z "${DEEPINFRA_API_KEY:-}" ]; then
  echo "⚠ DEEPINFRA_API_KEY not set; VLM front detection will fail (Blender will still run)."
fi

echo "▶ Running Blender front detection for frontage assets (${VLM_ARG})"
blender -b -P mesh_dir/batch_blender.py -- godot_world/models_used mesh_dir/renders $VLM_ARG --frontage_assets mesh_dir/frontage_assets.json

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
