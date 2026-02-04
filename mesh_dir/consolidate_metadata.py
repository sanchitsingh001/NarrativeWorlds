#!/usr/bin/env python3
"""
Consolidate all front.json files from renders/ into a single asset_metadata.json
for use by Godot world_loader.gd
"""

import json
import os
from pathlib import Path

RENDERS_DIR = Path(__file__).parent / "renders"
OUTPUT_PATH = Path(__file__).parent.parent / "godot_world" / "generated" / "asset_metadata.json"


def consolidate_metadata():
    """
    Read all front.json files from renders/ and merge into asset_metadata.json.
    Preserves existing entries so running batch_blender only for frontage assets
    does not wipe other metadata.
    """
    # Start from existing asset_metadata so we don't lose entries when only frontage was re-run
    metadata = {}
    if OUTPUT_PATH.exists():
        try:
            with open(OUTPUT_PATH, "r") as f:
                metadata = json.load(f)
            print(f"Loaded existing {len(metadata)} assets from {OUTPUT_PATH}")
        except Exception as e:
            print(f"Could not load existing metadata: {e}")
    
    if not RENDERS_DIR.exists():
        print(f"Renders directory not found: {RENDERS_DIR}")
        if metadata:
            with open(OUTPUT_PATH, "w") as f:
                json.dump(metadata, f, indent=2)
            print(f"Wrote existing metadata to {OUTPUT_PATH}")
        return
    
    # Find all subdirectories with front.json and merge/overwrite
    for subdir in RENDERS_DIR.iterdir():
        if not subdir.is_dir():
            continue
        
        front_json_path = subdir / "front.json"
        if not front_json_path.exists():
            continue
        
        try:
            with open(front_json_path, "r") as f:
                data = json.load(f)
            
            # Use the asset stem (without .glb) as the key
            asset_name = data.get("asset", "")
            if asset_name.endswith(".glb"):
                asset_key = asset_name[:-4]  # Remove .glb
            else:
                asset_key = subdir.name
            
            # Extract relevant fields for Godot
            metadata[asset_key] = {
                "front_yaw_deg": data.get("front_yaw_deg"),
                "front_panel": data.get("front_panel"),
                # Bounding box dimensions (may be None if not yet generated)
                "bbox_width": data.get("bbox_width"),
                "bbox_depth": data.get("bbox_depth"),
                "bbox_height": data.get("bbox_height"),
            }
            
            print(f"  ✓ {asset_key}: front_yaw={data.get('front_yaw_deg')}")
            
        except Exception as e:
            print(f"  ✗ Error reading {front_json_path}: {e}")
    
    # Ensure output directory exists and write merged metadata
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nMerged metadata: {len(metadata)} assets -> {OUTPUT_PATH}")


if __name__ == "__main__":
    print("Consolidating asset metadata...")
    consolidate_metadata()
