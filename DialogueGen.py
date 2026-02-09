"""
DialogueGen.py: Generate dialogue.json from world_plan.json.

Reads world_plan (areas + npc_plan), produces dialogue matching the Narrative/narrative.json
schema: premise + events (id, speaker, text, tags, choices, goto, effects).
Uses only world_plan["areas"] and world_plan["npc_plan"]. No terminal input.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from openai import OpenAI

client = OpenAI()

# -------------------------
# 1) Dialogue JSON schema for structured output
# -------------------------

DIALOGUE_SCHEMA: Dict[str, Any] = {
    "name": "dialogue_output",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "premise": {"type": "string"},
            "events": {
                "type": "array",
                "minItems": 45,
                "maxItems": 80,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "id": {"type": "string"},
                        "speaker": {"type": "string"},
                        "text": {"type": "string"},
                        "choices": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "text": {"type": "string"},
                                    "goto": {"type": "string"},
                                    "effects": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                },
                                "required": ["text", "goto", "effects"],
                            },
                        },
                        "goto": {"type": "string"},
                    },
                    "required": ["id", "speaker", "text", "choices", "goto"],
                },
            },
        },
        "required": ["premise", "events"],
    },
}

DIALOGUE_INSTRUCTIONS = """\
You are a writer for a premium indie psychological horror dialogue system.

Your task: Generate a single dialogue graph (premise + events) that will be played in order, with branches via choices and goto.

STORY TONE:
- Premium indie psychological horror. Escalating "polite wrongness" and identity gaslighting.
- Environmental shifts and wrongness are described by NARRATOR.
- Interiors (rooms, insides of buildings) appear ONLY as NARRATOR hallucinations (e.g. "Through the window you see..."). Do not introduce indoor areas or locations; all play happens outdoors or at thresholds.

STRUCTURAL RULES (MUST FOLLOW):
1. Output exactly 45 to 80 events. No fewer, no more.
2. Exactly 3 to 5 events must be "choice nodes" (events that have a "choices" array). Those events must NOT have "goto". Other events may have "goto" to jump to an event id, or no goto (linear next).
3. One event must have id exactly "merge_01". At least 2 other events (or choices from choice nodes) must have goto "merge_01" so that the graph converges there.
4. Place the "twist" (escalation, revelation, identity gaslighting peak) at around 60-75% through the event list. The event with id "merge_01" should be at or near this twist.
5. The final event (or the event that leads to the ending) must have exactly 2 choices (two ending options). That event must NOT have "goto".
6. One NPC at a time: Order events as NARRATOR (travel/setup) -> one NPC block (several events with speaker = that NPC's display_name) -> NARRATOR (transition) -> next NPC. Use ONLY the areas and NPCs provided; speaker for NPC lines must be exactly the NPC's display_name as given.
7. Every event where speaker is not "NARRATOR" must include "tags" with: "area" (that NPC's area_id), "needs" (array including that NPC's anchor_entity id; all ids must be from that area's entity list), "on_road": true.
8. No event may have both "choices" and "goto". Use either one or neither. For events with neither, set "choices": [] and "goto": "". For events with goto only, set "choices": []. For events with choices only, set "goto": "".
9. Every "goto" and every choice's "goto" must reference an event "id" that exists in the events list.
10. Add small flag effects on some choices: use "effects": ["set:curiosity"], ["set:defiant"], ["set:mercy"], or ["set:witness"] where appropriate. Use "effects": [] when no effect.
11. Every event must include "choices" and "goto". Use "choices": [] when the event has no choices. Use "goto": "" when the event has no goto. (Tags for NPC events will be added automatically.)

You will receive: (1) A list of areas with id and narrative. (2) A list of NPCs with display_name, area_id, anchor_entity.id, and dialogue_seed. Use ONLY these areas and NPCs. Use exact display_name as speaker for NPC lines. Use only entity ids from each area's entity list for tags.needs.

Return ONLY valid JSON matching the provided schema.
"""


def load_world_plan(path: str) -> dict:
    """Load world plan from JSON file. Same structure as world_plan.json from Worldplan.py."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_dialogue(dialogue: dict, path: str) -> None:
    """Write dialogue dict to JSON file (premise + events)."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dialogue, f, indent=2, ensure_ascii=False)


def _parse_dialogue_json(raw: str) -> Dict[str, Any]:
    """Parse API JSON; strip markdown fences and attempt repair on truncation."""
    text = raw.strip()
    if text.startswith("```"):
        first = text.find("\n")
        if first != -1 and ("json" in text[:first] or "JSON" in text[:first]):
            text = text[first + 1 :]
        text = text.strip()
        if text.endswith("```"):
            text = text[: text.rfind("```")].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        if e.pos is not None and e.pos > 500:
            truncated = text[: e.pos].rstrip()
            # Try closing events array and root: "... }" or "... },"
            if truncated.endswith("}"):
                try:
                    return json.loads(truncated + "]}")
                except json.JSONDecodeError:
                    pass
            if truncated.endswith("},"):
                try:
                    return json.loads(truncated[:-1] + "]}")
                except json.JSONDecodeError:
                    pass
            # Sometimes model omits comma between events: "}\n{" -> "},\n{"
            if "Expecting" in str(e) and "delimiter" in str(e):
                repaired = text.replace("}\n    {", "},\n    {").replace("}\n  {", "},\n  {")
                try:
                    return json.loads(repaired)
                except json.JSONDecodeError:
                    pass
            # Find last "}" that could be end of an event, then close array/root
            for i in range(len(truncated) - 1, -1, -1):
                if truncated[i] == "}":
                    try:
                        return json.loads(truncated[: i + 1] + "]}")
                    except json.JSONDecodeError:
                        continue
        raise


def _call_structured(model: str, instructions: str, user_text: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    resp = client.responses.create(
        model=model,
        instructions=instructions,
        input=user_text,
        text={
            "format": {
                "type": "json_schema",
                "name": schema["name"],
                "schema": schema["schema"],
                "strict": schema.get("strict", True),
            },
        },
    )
    return _parse_dialogue_json(resp.output_text)


def _build_user_text(world_plan: Dict[str, Any]) -> str:
    areas = world_plan.get("areas", [])
    npc_plan = world_plan.get("npc_plan", [])
    lines = ["AREAS (use only these; id and narrative):", ""]
    for a in areas:
        lines.append(f"- id: {a['id']}")
        lines.append(f"  narrative: {a.get('narrative', '')[:300]}")
        entity_ids = [e["id"] for e in a.get("entities", [])]
        lines.append(f"  entity_ids: {entity_ids}")
        lines.append("")
    lines.append("NPCs (use only these; speaker = display_name exactly):")
    for n in npc_plan:
        lines.append(f"- display_name: {n['display_name']!r}, area_id: {n['area_id']}, anchor_entity.id: {n['anchor_entity']['id']}, dialogue_seed: {n.get('dialogue_seed', '')[:150]}")
    return "\n".join(lines)


def _area_entity_ids(world_plan: Dict[str, Any], area_id: str) -> List[str]:
    for a in world_plan.get("areas", []):
        if a.get("id") == area_id:
            return [e["id"] for e in a.get("entities", [])]
    return []


def _npc_by_display_name(world_plan: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for n in world_plan.get("npc_plan", []):
        out[n["display_name"]] = n
    return out


def _postprocess_dialogue(dialogue: Dict[str, Any], world_plan: Dict[str, Any]) -> None:
    """Enforce tags for NPC events, merge_01, goto validity, no choices+goto, final 2 choices, event count."""
    events = dialogue.get("events", [])
    if not events:
        return
    npc_by_speaker = _npc_by_display_name(world_plan)

    # -1) Remove invalid events (empty id or empty text) and deduplicate ids
    seen_ids: Dict[str, int] = {}
    valid_events: List[Dict[str, Any]] = []
    for ev in events:
        eid = ev.get("id")
        text = ev.get("text")
        if not eid or (text is not None and str(text).strip() == ""):
            continue
        if eid in seen_ids:
            seen_ids[eid] += 1
            ev = dict(ev)
            ev["id"] = f"{eid}_{seen_ids[eid]}"
        else:
            seen_ids[eid] = 0
        valid_events.append(ev)
    events.clear()
    events.extend(valid_events)

    # 0) Strip empty optional fields so output matches Narrative format
    for ev in events:
        if ev.get("goto") == "":
            ev.pop("goto", None)
        if ev.get("choices") == []:
            ev.pop("choices", None)

    # 1) Tags for every NPC event: area, needs (from area entities), on_road true
    for ev in events:
        speaker = ev.get("speaker") or ""
        if speaker == "NARRATOR":
            continue
        npc = npc_by_speaker.get(speaker)
        if not npc:
            continue
        area_id = npc["area_id"]
        anchor_id = npc["anchor_entity"]["id"]
        entity_ids = _area_entity_ids(world_plan, area_id)
        needs = [anchor_id] if anchor_id in entity_ids else list(entity_ids[:3]) if entity_ids else [anchor_id]
        # Keep only ids that are in the area
        needs = [eid for eid in needs if eid in entity_ids] or (entity_ids[:1] if entity_ids else [anchor_id])
        ev["tags"] = {"area": area_id, "needs": needs, "on_road": True}

    # 2) No event has both choices and goto
    for ev in events:
        if ev.get("choices") and ev.get("goto"):
            ev.pop("goto", None)

    # 3) Ensure merge_01 exists and has >=2 inbound
    has_merge = any(e.get("id") == "merge_01" for e in events)
    merge_index = next((i for i, e in enumerate(events) if e.get("id") == "merge_01"), None)
    inbound = 0
    for ev in events:
        if ev.get("goto") == "merge_01":
            inbound += 1
        for c in ev.get("choices") or []:
            if c.get("goto") == "merge_01":
                inbound += 1
    if not has_merge:
        twist_idx = min(int(len(events) * 0.65), len(events) - 1)
        merge_ev = {
            "id": "merge_01",
            "speaker": "NARRATOR",
            "text": "The world tips. Something takes attendance. The boundary opens like an eye.",
        }
        events.insert(twist_idx, merge_ev)
        merge_index = twist_idx
    if inbound < 2 and merge_index is not None:
        # Point one or two gotos to merge_01 (e.g. event before merge)
        for i in range(len(events) - 1, -1, -1):
            if inbound >= 2:
                break
            ev = events[i]
            if ev.get("id") == "merge_01":
                continue
            if not ev.get("choices") and not ev.get("goto"):
                ev["goto"] = "merge_01"
                inbound += 1

    # 4) Fix invalid or empty gotos: point to existing id (merge_01 or next)
    valid_ids = {e["id"] for e in events if e.get("id")}  # Rebuild after any inserts
    fallback = "merge_01" if "merge_01" in valid_ids else (list(valid_ids)[0] if valid_ids else "merge_01")
    for ev in events:
        g = ev.get("goto")
        if g is not None and (g == "" or g not in valid_ids):
            ev["goto"] = fallback
        for c in ev.get("choices") or []:
            g = c.get("goto")
            if g is not None and (g == "" or g not in valid_ids):
                c["goto"] = fallback

    # 5) Final event: ensure exactly 2 ending choices if last event is a choice node
    if events:
        last = events[-1]
        if last.get("choices"):
            if len(last["choices"]) != 2:
                # Resize to exactly 2
                choices = last["choices"]
                if len(choices) > 2:
                    last["choices"] = choices[:2]
                else:
                    while len(last["choices"]) < 2:
                        target = "merge_01" if "merge_01" in valid_ids else (events[0]["id"] if events else "merge_01")
                        last["choices"].append({"text": "Continue.", "goto": target})
            if last.get("goto"):
                last.pop("goto", None)

    # 6) Clamp event count 45-80: add pad events, then add a single "end" so endings don't run into pads
    while len(events) < 45:
        events.append({
            "id": f"pad_{len(events)}",
            "speaker": "NARRATOR",
            "text": "The air shifts. Something watches from the edges of the street.",
        })
    if len(events) > 80:
        dialogue["events"] = events[:80]
    # 7) Add "end" event and point terminal events (no goto, before pads) to it so story stops cleanly
    end_id = "end"
    first_pad = next((i for i, e in enumerate(events) if e.get("id", "").startswith("pad_")), len(events))
    content_events = events[:first_pad]
    terminal = [e for e in content_events if not e.get("goto") and not e.get("choices")]
    if terminal and end_id not in valid_ids:
        for ev in terminal[-5:]:  # last few terminal events (the real endings)
            ev["goto"] = end_id
        events.append({
            "id": end_id,
            "speaker": "NARRATOR",
            "text": "—",
        })


def generate_dialogue(world_plan: Dict[str, Any], model: str = "gpt-4o") -> Dict[str, Any]:
    """
    Generate dialogue (premise + events) from world_plan. Uses only world_plan["areas"]
    and world_plan["npc_plan"]. Post-processes to enforce tags, merge_01, gotos, and constraints.
    """
    user_text = _build_user_text(world_plan)
    raw = _call_structured(
        model=model,
        instructions=DIALOGUE_INSTRUCTIONS,
        user_text=user_text,
        schema=DIALOGUE_SCHEMA,
    )
    dialogue = {"premise": raw.get("premise", ""), "events": raw.get("events", [])}
    _postprocess_dialogue(dialogue, world_plan)
    return dialogue


if __name__ == "__main__":
    wp = load_world_plan("world_plan.json")
    dialogue = generate_dialogue(wp, model="gpt-4o")
    save_dialogue(dialogue, "dialogue.json")
    print(f"Saved dialogue.json: premise + {len(dialogue.get('events', []))} events.")
