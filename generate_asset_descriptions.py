import argparse
import os
import json
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Optional

from openai import OpenAI

PLAN_PATH = "world_plan.json"
LAYOUT_PATH = "world_entity_layout_llm_v3_out.json"
OUT_DIR = "."

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.4"))

# Match godot world_loader: 1 tile = 1.8 m (player height)
TILE_SIZE_M = float(os.environ.get("TILE_SIZE_M", "1.8"))

SYSTEM_INSTRUCTIONS = """You generate visual, material-focused 3D asset descriptions for text-to-3D generators.

Hard rules:
- Describe ONLY the object (no people, no story, no actions, no camera language).
- 2–4 sentences max.
- Mention: primary materials, key shapes/forms, surface condition, and any distinctive architectural details.
- If needs_frontage_any is true, include a clear front with an entrance/door orientation detail.
- If context includes placement_dimensions, incorporate approximate footprint and height in the prompt so the 3D generator produces correctly scaled geometry (e.g. "building footprint about X×Y m, height about Z m").
- Keep it grounded in the provided context and tags.

Return ONLY valid JSON in this schema:
{
  "group": "...",
  "prompt": "..."
}
"""


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def index_plan_templates(plan: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    Returns:
      - template_by_entity_id: entity_id -> template (kind/type/tags/count)
      - area_by_id: area_id -> area dict (narrative/scale_hint/entities)
    """
    template_by_entity_id: Dict[str, Dict[str, Any]] = {}
    area_by_id: Dict[str, Dict[str, Any]] = {}

    for area in plan.get("areas", []):
        area_id = area.get("id")
        if area_id:
            area_by_id[area_id] = area

        for ent in area.get("entities", []):
            ent_id = ent.get("id")
            # First seen wins; fine for most pipelines
            if ent_id and ent_id not in template_by_entity_id:
                template_by_entity_id[ent_id] = ent

    return template_by_entity_id, area_by_id


def list_layout_areas(layout: Dict[str, Any]) -> List[str]:
    areas = layout.get("areas", {})
    # stable ordering
    return sorted(list(areas.keys()))


def prompt_area_selection(area_ids: List[str]) -> List[str]:
    """
    Returns the selected area_ids. If user chooses ALL, returns all area_ids.
    """
    if not area_ids:
        raise RuntimeError("No areas found in layout JSON.")

    print("\nSelect which area to generate prompts for:")
    for i, aid in enumerate(area_ids, start=1):
        print(f"  {i}. {aid}")
    all_idx = len(area_ids) + 1
    print(f"  {all_idx}. ALL areas")

    while True:
        raw = input("\nType a number and press Enter: ").strip()
        try:
            choice = int(raw)
        except ValueError:
            print("Please enter a valid number.")
            continue

        if 1 <= choice <= len(area_ids):
            return [area_ids[choice - 1]]
        if choice == all_idx:
            return area_ids

        print("Choice out of range. Try again.")


def collect_groups_from_layout(layout: Dict[str, Any], selected_area_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Collect unique groups only within selected areas.

    Returns:
      groups[group_name] = {
        "sample_instances": [...],
        "kinds": set(),
        "needs_frontage_any": bool,
        "areas": set(),
      }
    """
    groups: Dict[str, Dict[str, Any]] = {}
    areas = layout.get("areas", {})

    for area_id in selected_area_ids:
        area_blob = areas.get(area_id, {})
        placements = area_blob.get("placements", {})

        for instance_id, p in placements.items():
            group = p.get("group") or p.get("id") or instance_id
            if not group:
                continue

            if group not in groups:
                groups[group] = {
                    "sample_instances": [],
                    "kinds": set(),
                    "needs_frontage_any": False,
                    "areas": set(),
                }

            groups[group]["areas"].add(area_id)
            if p.get("kind"):
                groups[group]["kinds"].add(p["kind"])
            if bool(p.get("needs_frontage")):
                groups[group]["needs_frontage_any"] = True

            if len(groups[group]["sample_instances"]) < 6:
                groups[group]["sample_instances"].append(
                    {
                        "instance_id": p.get("id", instance_id),
                        "area_id": area_id,
                        "w": p.get("w"),
                        "h": p.get("h"),
                        "needs_frontage": bool(p.get("needs_frontage")),
                    }
                )

    return groups


def _placement_dimensions_from_instances(sample_instances: List[Dict[str, Any]]) -> Optional[str]:
    """
    Derive placement dimensions from layout sample_instances (w, h in tiles).
    Matches world_loader logic: footprint ~ w*h tiles, height ~ sqrt(w*h)*0.8 tiles, 1 tile = TILE_SIZE_M.
    """
    if not sample_instances:
        return None
    ws = [float(p.get("w", 0)) for p in sample_instances if p.get("w") is not None]
    hs = [float(p.get("h", 0)) for p in sample_instances if p.get("h") is not None]
    if not ws or not hs:
        return None
    w_tiles = sum(ws) / len(ws)
    h_tiles = sum(hs) / len(hs)
    w_m = w_tiles * TILE_SIZE_M
    h_m = h_tiles * TILE_SIZE_M
    # Height in world_loader: clamp(sqrt(ew*eh)*0.8, 2, 8) tiles
    height_tiles = max(2.0, min(8.0, (w_tiles * h_tiles) ** 0.5 * 0.8))
    height_m = height_tiles * TILE_SIZE_M
    return (
        f"Placement: footprint about {w_tiles:.1f}×{h_tiles:.1f} tiles ({w_m:.1f}×{h_m:.1f} m), "
        f"height in-world ~{height_m:.1f} m. Incorporate approximate size so the 3D generator produces correctly scaled geometry."
    )


def find_representative_context_for_group(
    group: str,
    groups_meta: Dict[str, Any],
    area_by_id: Dict[str, Any],
    template_by_entity_id: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build context for LLM:
    - representative area narrative (first area where group appears)
    - template from plan using entity_id==group if available
    - placement_dimensions from layout sample_instances (so prompts can include size)
    """
    area_ids = sorted(list(groups_meta.get("areas", [])))
    rep_area_id = area_ids[0] if area_ids else None
    rep_area = area_by_id.get(rep_area_id, {}) if rep_area_id else {}

    template = template_by_entity_id.get(group, {})
    sample_instances = groups_meta.get("sample_instances", [])
    placement_dimensions = _placement_dimensions_from_instances(sample_instances)

    ctx: Dict[str, Any] = {
        "group": group,
        "rep_area_id": rep_area_id,
        "rep_area_scale_hint": rep_area.get("scale_hint"),
        "rep_area_narrative": rep_area.get("narrative"),
        "template": {
            "kind": template.get("kind"),
            "type": template.get("type"),
            "tags": template.get("tags", []),
        },
        "layout_hints": {
            "kinds_seen": sorted(list(groups_meta.get("kinds", []))),
            "needs_frontage_any": bool(groups_meta.get("needs_frontage_any")),
            "sample_instances": sample_instances,
        },
    }
    if placement_dimensions:
        ctx["placement_dimensions"] = placement_dimensions
    return ctx


def call_openai_for_group(client: OpenAI, model: str, context: Dict[str, Any]) -> Dict[str, Any]:
    payload = {
        "task": "Generate a single 3D asset prompt for this group, suitable for text-to-3D.",
        "context": context,
    }

    resp = client.responses.create(
        model=model,
        temperature=TEMPERATURE,
        instructions=SYSTEM_INSTRUCTIONS,
        input=json.dumps(payload, ensure_ascii=False),
    )

    text = resp.output_text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"group": context["group"], "prompt": text, "warning": "Model did not return valid JSON."}


def safe_filename(s: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in s)


def main():
    parser = argparse.ArgumentParser(description="Generate 3D asset prompts per group from world plan + v3 layout.")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Non-interactive: generate descriptions for ALL areas (no area selection prompt).",
    )
    args = parser.parse_args()

    use_all_areas = args.all or os.getenv("GENERATE_ALL_AREAS", "").strip().lower() in ("1", "true", "yes")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in environment.")

    plan = load_json(PLAN_PATH)
    layout = load_json(LAYOUT_PATH)

    template_by_entity_id, area_by_id = index_plan_templates(plan)

    area_ids = list_layout_areas(layout)
    if use_all_areas:
        selected_area_ids = area_ids
    else:
        selected_area_ids = prompt_area_selection(area_ids)

    # Use model from layout meta if present, else env/default
    model = layout.get("meta", {}).get("openai_model") or DEFAULT_MODEL

    groups = collect_groups_from_layout(layout, selected_area_ids)
    if not groups:
        raise RuntimeError(f"No placements/groups found for selected area(s): {selected_area_ids}")

    client = OpenAI(api_key=api_key)

    # Output naming
    if len(selected_area_ids) == len(area_ids):
        out_name = "asset_group_descriptions_ALL.json"
        selection_label = "ALL"
    else:
        out_name = f"asset_group_descriptions_{safe_filename(selected_area_ids[0])}.json"
        selection_label = selected_area_ids[0]

    out = {
        "meta": {
            "plan_path": PLAN_PATH,
            "layout_path": LAYOUT_PATH,
            "selected": selection_label,
            "selected_area_ids": selected_area_ids,
            "model": model,
            "temperature": TEMPERATURE,
        },
        "groups": {},
    }

    for group_name, gmeta in sorted(groups.items(), key=lambda x: x[0]):
        context = find_representative_context_for_group(
            group=group_name,
            groups_meta=gmeta,
            area_by_id=area_by_id,
            template_by_entity_id=template_by_entity_id,
        )

        result = call_openai_for_group(client, model, context)

        out["groups"][group_name] = {
            "prompt": result.get("prompt"),
            "rep_area_id": context.get("rep_area_id"),
            "template": context.get("template"),
            "layout_hints": context.get("layout_hints"),
        }

    out_path = os.path.join(OUT_DIR, out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Wrote {out_path}")
    print(f"   Groups generated: {len(out['groups'])}")
    print(f"   Selection: {selection_label}")


if __name__ == "__main__":
    main()