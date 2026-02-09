#!/usr/bin/env python3
"""
Build a list of asset names (GLB stems) used by entities with needs_frontage=true.
Use this with batch_blender.py --frontage_assets so those GLBs get front_yaw_deg
and the game can orient them to face the path.

Usage:
  python mesh_dir/frontage_assets.py [--world path] [--models path] [--npc-models path] [--out path]

Then:
  blender -b -P batch_blender.py -- glb_dir out_dir --ask_vlm --frontage_assets mesh_dir/frontage_assets.json
"""

import argparse
import json
import os


def asset_name_from_path(path: str) -> str:
    """Res://models_used/home_above_shop.glb -> home_above_shop (matches Godot _get_asset_name_from_path)."""
    if not path:
        return ""
    name = path.replace("\\", "/").split("/")[-1]
    if name.lower().endswith(".glb"):
        name = name[:-4]
    return name


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    ap = argparse.ArgumentParser(description="Build frontage_assets.json for batch_blender.py --frontage_assets")
    ap.add_argument(
        "--world",
        default=os.path.join(project_root, "godot_world", "generated", "world_entity_layout_llm_v3_out.json"),
        help="Path to world layout v3 JSON (world_space.areas[].entities_world with needs_frontage)",
    )
    ap.add_argument(
        "--models",
        default=os.path.join(project_root, "godot_world", "generated", "entity_models.json"),
        help="Path to entity_models.json (entity_id -> res://path/to/model.glb)",
    )
    ap.add_argument(
        "--npc-models",
        default=os.path.join(project_root, "godot_world", "generated", "npc_models.json"),
        help="Path to npc_models.json (npc_id -> res://path/to/model.glb); optional, skipped if missing",
    )
    ap.add_argument(
        "--out",
        default=os.path.join(script_dir, "frontage_assets.json"),
        help="Output JSON path (list of asset names)",
    )
    args = ap.parse_args()

    if not os.path.isfile(args.world):
        raise SystemExit("World layout not found: %s" % args.world)
    if not os.path.isfile(args.models):
        raise SystemExit("Entity models not found: %s" % args.models)

    with open(args.world, "r") as f:
        world = json.load(f)
    with open(args.models, "r") as f:
        entity_models = json.load(f)

    world_space = world.get("world_space", {})
    areas = world_space.get("areas", {})

    frontage_entity_ids = set()
    for aid, a_ws in areas.items():
        entities_world = a_ws.get("entities_world", {})
        for eid, e in entities_world.items():
            if e.get("needs_frontage") is True:
                frontage_entity_ids.add(eid)

    asset_names = set()
    for eid in frontage_entity_ids:
        path = entity_models.get(eid)
        if path:
            name = asset_name_from_path(path)
            if name:
                asset_names.add(name)

    # Include NPC models so they get front_yaw_deg from VLM (optional; skip if file missing)
    if os.path.isfile(args.npc_models):
        with open(args.npc_models, "r") as f:
            npc_models = json.load(f)
        for path in npc_models.values():
            if path:
                name = asset_name_from_path(path)
                if name:
                    asset_names.add(name)
        print("NPC models: %d added from %s" % (len(npc_models), args.npc_models))
    else:
        print("NPC models: skipped (file not found: %s)" % args.npc_models)

    out_list = sorted(asset_names)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out_list, f, indent=2)

    print("Frontage entities: %d" % len(frontage_entity_ids))
    print("Unique frontage assets: %d -> %s" % (len(out_list), args.out))
    for a in out_list:
        print("  -", a)


if __name__ == "__main__":
    main()
