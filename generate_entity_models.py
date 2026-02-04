import json
import os
import re
import shutil
from pathlib import Path

INPUT_FILE = "godot_world/generated/world_entity_layout_llm_v3_out.json"
OUTPUT_FILE = "godot_world/generated/entity_models.json"
GODOT_DIR = Path("godot_world")

# Paths for asset optimization
MODELS_LIBRARY_DIR = GODOT_DIR / "models_library" # Source (all assets)
MODELS_USED_DIR = GODOT_DIR / "models_used"       # Destination (only exported assets)

# Hardcoded / Explicit Mappings (fallback for entities without generated GLBs)
# NOTE: These paths are relative to models_library/ but will be copied to models_used/
HARDCODED_MAPPINGS = {
    "market_stalls": "market.glb",
    "returning_shops": "market.glb",
    "tailor_shops": "tailor.glb",
    "crumbling_haveli": "Haveli.glb",
    "mural_puzzles": "f7dbdb715e70eb7c9cd869e2a8f3d816.glb",
    "library_building": "689786d474be9850db03f1d22af5ec6c.glb",
    "ritual_shrine": "Meshy_AI_An_ancient_Indian_god_0114020801_texture.glb",
    # Default fallback
    "__default__": "home_above_shop.glb"
}


def get_base_entity_id(entity_id: str) -> str:
    """
    Strip numeric suffix from entity ID.
    e.g., 'spell_shop_1' -> 'spell_shop'
          'spell_shop' -> 'spell_shop'
    """
    return re.sub(r'_\d+$', '', entity_id)


def ensure_used_model_exists(filename: str) -> str | None:
    """
    Ensure a model exists in models_used/ directory.
    If it exists in models_library/, copy it over.
    
    Args:
        filename: Name of the GLB file (e.g. "tree.glb")
        
    Returns:
        Godot resource path (res://models_used/{filename}) if successful, None if source not found
    """
    # 1. Check if already in models_used (optimization)
    dest_path = MODELS_USED_DIR / filename
    if dest_path.exists():
        return f"res://models_used/{filename}"
    
    # 2. Look for source in models_library
    source_path = MODELS_LIBRARY_DIR / filename
    
    if not source_path.exists():
        # Try case-insensitive search in library
        found = False
        if MODELS_LIBRARY_DIR.exists():
            for f in MODELS_LIBRARY_DIR.iterdir():
                if f.name.lower() == filename.lower():
                    source_path = f
                    filename = f.name # Update filename to match actual case
                    dest_path = MODELS_USED_DIR / filename
                    found = True
                    break
        
        if not found:
            return None

    # 3. Copy from library to used
    try:
        MODELS_USED_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, dest_path)
        print(f"  --> Copied {filename} to models_used/")
        return f"res://models_used/{filename}"
    except Exception as e:
        print(f"  Error copying {filename}: {e}")
        return None


def find_generated_glb(entity_id: str) -> str | None:
    """
    Check if a generated GLB file exists for the given entity ID in the library.
    If found, copy to used folder and return path.
    
    Priority:
    1. Exact match: models_library/{entity_id}.glb
    2. Base name match: models_library/{base_id}.glb (for spell_shop_1 -> spell_shop.glb)
    3. Case-insensitive match
    """
    
    # Check for exact match
    filename = f"{entity_id}.glb"
    path = ensure_used_model_exists(filename)
    if path: 
        return path
    
    # Check for base name match
    base_id = get_base_entity_id(entity_id)
    if base_id != entity_id:
        filename = f"{base_id}.glb"
        path = ensure_used_model_exists(filename)
        if path:
            return path
            
    # Case-insensitive handled inside ensure_used_model_exists via direct filename check
    # But checking specific ID variants loosely against directory listing:
    if MODELS_LIBRARY_DIR.exists():
        for f in MODELS_LIBRARY_DIR.iterdir():
            if f.suffix.lower() == ".glb":
                if f.stem.lower() == entity_id.lower() or f.stem.lower() == base_id.lower():
                    return ensure_used_model_exists(f.name)
    
    return None


def get_model_path_for_entity(entity_id: str) -> str:
    """
    Get the model path for an entity with the following priority:
    1. Generated GLB file (from Hunyuan or other 3D generation)
    2. Hardcoded mapping (exact or base name)
    3. Default fallback
    
    Args:
        entity_id: The entity identifier
        
    Returns:
        Godot resource path (res://models_used/...)
    """
    # Priority 1: Check for generated GLB (also checks base name)
    generated_path = find_generated_glb(entity_id)
    if generated_path:
        return generated_path
    
    # Priority 2: Hardcoded mapping (exact match)
    filename = None
    if entity_id in HARDCODED_MAPPINGS:
        filename = HARDCODED_MAPPINGS[entity_id]
    
    # Priority 2b: Hardcoded mapping (base name match)
    if not filename:
        base_id = get_base_entity_id(entity_id)
        if base_id in HARDCODED_MAPPINGS:
            filename = HARDCODED_MAPPINGS[base_id]
            
    if filename:
        # Resolve 'res://models/...' legacy paths in mapping if present
        if filename.startswith("res://"):
            filename = filename.split("/")[-1]
            
        path = ensure_used_model_exists(filename)
        if path:
            return path
    
    # Priority 3: Default fallback
    fallback_file = HARDCODED_MAPPINGS.get("__default__", "tree.glb")
    if fallback_file.startswith("res://"):
        fallback_file = fallback_file.split("/")[-1]
        
    path = ensure_used_model_exists(fallback_file)
    if path:
        return path
        
    return "" # Fallback failed completely


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    print(f"Reading {INPUT_FILE}...")
    with open(INPUT_FILE, 'r') as f:
        v3_layout = json.load(f)

    # Clean the used directory to ensure no stale assets?
    # No, let's keep it incremental for now to avoid re-copying everything constantly.
    # But ideally, we should clean it for a "clean build".
    # For now, we rely on the fact that we only map what exists.
    
    print(f"Preparing models...")
    print(f"  Library: {MODELS_LIBRARY_DIR}")
    print(f"  Export:  {MODELS_USED_DIR}")
    MODELS_USED_DIR.mkdir(parents=True, exist_ok=True)

    entity_models = {}
    generated_count = 0
    hardcoded_count = 0
    fallback_count = 0

    # v3 structure: {"areas": {area_id: {"placements": {entity_id: {...}}}}}
    areas = v3_layout.get("areas", {}) or {}
    if not isinstance(areas, dict):
        print("Error: v3 JSON missing top-level 'areas' object.")
        return

    entity_ids: set[str] = set()
    for area_id, area_blob in areas.items():
        if not isinstance(area_blob, dict):
            continue
        placements = area_blob.get("placements", {}) or {}
        if isinstance(placements, dict):
            for eid in placements.keys():
                if eid:
                    entity_ids.add(str(eid))
        elif isinstance(placements, list):
            for p in placements:
                if isinstance(p, dict) and p.get("id"):
                    entity_ids.add(str(p["id"]))

    for eid in sorted(entity_ids):
        model_path = get_model_path_for_entity(eid)
        if model_path:
            entity_models[eid] = model_path

            # Track statistics logic (simplified)
            if "tree.glb" in model_path and eid != "tree":  # Rough fallback check
                fallback_count += 1
            else:
                generated_count += 1  # Treating all found/hardcoded as 'success'
        else:
            print(f"WARNING: No model found for {eid}")

    print(f"Found {len(entity_models)} entities.")
    
    print(f"Writing to {OUTPUT_FILE}...")
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(entity_models, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()

