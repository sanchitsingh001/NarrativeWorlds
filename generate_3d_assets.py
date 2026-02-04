#!/usr/bin/env python3
"""
Generate 3D assets via Hunyuan API for entity groups from the v3 layout.

This script:
1. Loads asset group descriptions from asset_group_descriptions_*.json (groups[group].prompt)
2. Loads world_entity_layout_llm_v3_out.json to get areas and placements (id, group)
3. Collects unique groups that have a prompt; submits one Hunyuan job per group
4. Polls for completion every 60 seconds
5. Downloads GLB files as {group}.glb to mesh_dir/glb_files/
6. Copies GLBs to godot_world/models_library/ and godot_world/models_used/
7. Updates entity_models.json (placement_id -> res://models_used/{group}.glb for generated groups)

Batch mode: use --batch or BATCH=1 to skip confirmation and process all groups.
"""

import argparse
import json
import os
import shutil
import time
import urllib.request
from pathlib import Path
from typing import Dict, Optional, List, Set, Tuple

# Tencent Cloud SDK imports
from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
from tencentcloud.hunyuan.v20230901 import hunyuan_client, models

# Paths (can be overridden by env or CLI)
DEFAULT_V3_LAYOUT_PATH = Path(os.getenv("V3_LAYOUT_PATH", "godot_world/generated/world_entity_layout_llm_v3_out.json"))
DEFAULT_GROUP_DESCRIPTIONS_PATH = Path(os.getenv("ASSET_GROUP_DESCRIPTIONS_PATH", "asset_group_descriptions_ALL.json"))
GLB_OUTPUT_DIR = Path("mesh_dir/glb_files")
GODOT_DIR = Path("godot_world")
ENTITY_MODELS_PATH = Path("godot_world/generated/entity_models.json")

# Polling settings
POLL_INTERVAL_SECONDS = 60
MAX_POLL_ATTEMPTS = 60  # Max 1 hour of polling

# API settings
MAX_RETRIES = 3
MAX_PROMPT_LENGTH = 900  # Safe limit for Hunyuan API
MAX_CONCURRENT_JOBS = 2  # Hunyuan's concurrent job limit


def sanitize_prompt(prompt: str) -> str:
    """
    Sanitize and truncate prompt for Hunyuan API.
    """
    if len(prompt) > MAX_PROMPT_LENGTH:
        truncated = prompt[:MAX_PROMPT_LENGTH]
        last_period = truncated.rfind('.')
        if last_period > MAX_PROMPT_LENGTH * 0.7:
            truncated = truncated[:last_period + 1]
        prompt = truncated
    return prompt


def get_hunyuan_client():
    """Create and return a Hunyuan API client."""
    secret_id = os.getenv("TENCENTCLOUD_SECRET_ID")
    secret_key = os.getenv("TENCENTCLOUD_SECRET_KEY")

    if not secret_id or not secret_key:
        raise RuntimeError(
            "TENCENTCLOUD_SECRET_ID and TENCENTCLOUD_SECRET_KEY environment variables must be set"
        )

    cred = credential.Credential(secret_id, secret_key)
    http_profile = HttpProfile()
    http_profile.endpoint = "hunyuan.intl.tencentcloudapi.com"
    client_profile = ClientProfile()
    client_profile.httpProfile = http_profile
    client = hunyuan_client.HunyuanClient(cred, "ap-singapore", client_profile)
    return client


def submit_hunyuan_job(client, prompt: str, group_name: str = "") -> Optional[str]:
    """Submit a Hunyuan 3D generation job. Returns JobId or None."""
    original_len = len(prompt)
    prompt = sanitize_prompt(prompt)
    if len(prompt) < original_len:
        print(f"  (truncated prompt from {original_len} to {len(prompt)} chars)")

    for attempt in range(MAX_RETRIES):
        try:
            req = models.SubmitHunyuanTo3DProJobRequest()
            params = {"Prompt": prompt}
            req.from_json_string(json.dumps(params))
            resp = client.SubmitHunyuanTo3DProJob(req)
            result = json.loads(resp.to_json_string())
            job_id = result.get("JobId")
            if job_id:
                return job_id
            print(f"  Warning: No JobId in response: {result}")
        except TencentCloudSDKException as e:
            error_code = str(e).split("code:")[1].split(" ")[0] if "code:" in str(e) else ""
            if "InvalidParameter" in error_code:
                print(f"  Invalid prompt rejected by API (length={len(prompt)})")
                return None
            print(f"  Attempt {attempt + 1}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
    return None


def query_job_status(client, job_id: str) -> Dict:
    """Query Hunyuan job status."""
    for attempt in range(MAX_RETRIES):
        try:
            req = models.QueryHunyuanTo3DProJobRequest()
            params = {"JobId": job_id}
            req.from_json_string(json.dumps(params))
            resp = client.QueryHunyuanTo3DProJob(req)
            return json.loads(resp.to_json_string())
        except TencentCloudSDKException as e:
            print(f"  Query attempt {attempt + 1}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
    return {"Status": "ERROR", "ErrorMessage": "Failed to query job status"}


def download_glb(url: str, output_path: Path) -> bool:
    """Download a GLB file from URL to output_path."""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=120) as response:
            with open(output_path, 'wb') as f:
                f.write(response.read())
        return True
    except Exception as e:
        print(f"  Failed to download {url}: {e}")
        return False


def load_v3_layout(layout_path: Path) -> dict:
    """Load world_entity_layout_llm_v3_out.json."""
    if not layout_path.exists():
        raise FileNotFoundError(f"Layout not found: {layout_path}")
    with open(layout_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_group_descriptions(descriptions_path: Path) -> dict:
    """Load asset_group_descriptions_*.json; expect top-level 'groups' with group_name -> { prompt, ... }."""
    if not descriptions_path.exists():
        raise FileNotFoundError(f"Group descriptions not found: {descriptions_path}")
    with open(descriptions_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    groups = data.get("groups", data) if isinstance(data.get("groups"), dict) else {}
    return groups


def collect_groups_and_placements_from_v3(layout: dict) -> Tuple[Set[str], Dict[str, str]]:
    """
    From v3 layout (areas -> placements), collect unique group names and placement_id -> group.
    Returns (set of group names, dict placement_id -> group).
    """
    groups: Set[str] = set()
    placement_to_group: Dict[str, str] = {}
    areas = layout.get("areas", {})
    for area_id, area_blob in areas.items():
        if not isinstance(area_blob, dict):
            continue
        placements = area_blob.get("placements", {})
        if not isinstance(placements, dict):
            continue
        for instance_id, p in placements.items():
            group = (p.get("group") or p.get("id") or instance_id)
            if group:
                groups.add(group)
                placement_to_group[instance_id] = group
    return groups, placement_to_group


def get_groups_to_generate(
    layout_groups: Set[str],
    placement_to_group: Dict[str, str],
    group_descriptions: dict,
) -> List[Tuple[str, str]]:
    """
    Return list of (group_name, prompt) for groups that appear in layout and have a prompt.
    """
    out: List[Tuple[str, str]] = []
    for group_name in sorted(layout_groups):
        g = group_descriptions.get(group_name)
        if not g or not isinstance(g, dict):
            continue
        prompt = (g.get("prompt") or "").strip()
        if not prompt:
            continue
        out.append((group_name, prompt))
    return out


def load_existing_entity_models() -> dict:
    """Load existing entity_models.json if it exists."""
    if ENTITY_MODELS_PATH.exists():
        try:
            with open(ENTITY_MODELS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_entity_models(entity_models: dict) -> None:
    """Save entity models mapping to JSON."""
    ENTITY_MODELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ENTITY_MODELS_PATH, "w", encoding="utf-8") as f:
        json.dump(entity_models, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Generate 3D GLB assets via Hunyuan API from v3 layout and group descriptions.")
    parser.add_argument(
        "--layout",
        type=Path,
        default=DEFAULT_V3_LAYOUT_PATH,
        help="Path to world_entity_layout_llm_v3_out.json",
    )
    parser.add_argument(
        "--descriptions",
        type=Path,
        default=DEFAULT_GROUP_DESCRIPTIONS_PATH,
        help="Path to asset_group_descriptions_*.json (e.g. asset_group_descriptions_ALL.json)",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Non-interactive: process all groups with descriptions, no confirmation prompt.",
    )
    args = parser.parse_args()

    batch = args.batch or os.getenv("BATCH", "").strip().lower() in ("1", "true", "yes")

    print("=" * 60)
    print("Hunyuan 3D Asset Generation Pipeline (v3 layout + groups)")
    print("=" * 60)

    if not os.getenv("TENCENTCLOUD_SECRET_ID") or not os.getenv("TENCENTCLOUD_SECRET_KEY"):
        print("\nError: TENCENTCLOUD_SECRET_ID and TENCENTCLOUD_SECRET_KEY must be set")
        return

    # Load v3 layout
    try:
        layout = load_v3_layout(args.layout)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Ensure the layout is copied to godot_world/generated/ (e.g. by run_wrld.sh).")
        return

    # Load group descriptions
    try:
        group_descriptions = load_group_descriptions(args.descriptions)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Run generate_asset_descriptions.py --all first.")
        return

    layout_groups, placement_to_group = collect_groups_and_placements_from_v3(layout)
    groups_to_generate = get_groups_to_generate(layout_groups, placement_to_group, group_descriptions)

    if not groups_to_generate:
        print("\nNo groups with prompts found. Ensure --descriptions has groups with 'prompt' and layout has placements with 'group'.")
        return

    print(f"\nGroups with descriptions: {len(groups_to_generate)}")
    for group_name, _ in groups_to_generate:
        print(f"  - {group_name}")

    if not batch:
        confirm = input(f"\nSubmit {len(groups_to_generate)} jobs to Hunyuan? [y/N]: ").strip().lower()
        if confirm != "y":
            print("Cancelled.")
            return

    # Initialize client
    print("\nConnecting to Hunyuan API...")
    try:
        client = get_hunyuan_client()
    except Exception as e:
        print(f"Failed to create API client: {e}")
        return

    GLB_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_LIBRARY_DIR = GODOT_DIR / "models_library"
    MODELS_USED_DIR = GODOT_DIR / "models_used"
    MODELS_LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_USED_DIR.mkdir(parents=True, exist_ok=True)

    jobs: Dict[str, Dict] = {}
    pending = list(groups_to_generate)
    downloaded: List[str] = []

    def count_running() -> int:
        return sum(1 for j in jobs.values() if j["status"] == "RUN")

    def on_job_done(group_name: str, glb_url: str) -> bool:
        output_path = GLB_OUTPUT_DIR / f"{group_name}.glb"
        print(f"      Downloading {group_name}...", end=" ", flush=True)
        if not download_glb(glb_url, output_path):
            print("Failed")
            return False
        print("Saved")
        for dst_dir in (MODELS_LIBRARY_DIR, MODELS_USED_DIR):
            dst = dst_dir / f"{group_name}.glb"
            try:
                shutil.copy2(output_path, dst)
            except Exception as e:
                print(f"        (copy to {dst_dir} failed: {e})")
        downloaded.append(group_name)
        return True

    def poll_running():
        for group_name in [g for g, j in jobs.items() if j["status"] == "RUN"]:
            job_id = jobs[group_name]["job_id"]
            result = query_job_status(client, job_id)
            status = result.get("Status", "UNKNOWN")
            if status == "DONE":
                print(f"    {group_name}: DONE")
                jobs[group_name]["status"] = "DONE"
                for file_info in result.get("ResultFile3Ds", []):
                    if file_info.get("Type") == "GLB":
                        glb_url = file_info.get("Url")
                        jobs[group_name]["glb_url"] = glb_url
                        if glb_url:
                            on_job_done(group_name, glb_url)
                        break
            elif status in ("FAIL", "ERROR"):
                print(f"    {group_name}: {status}")
                jobs[group_name]["status"] = "FAIL"
                jobs[group_name]["error"] = result.get("ErrorMessage", "Unknown error")

    print(f"\nSubmitting jobs (max {MAX_CONCURRENT_JOBS} concurrent)...")
    while pending or count_running() > 0:
        while pending and count_running() < MAX_CONCURRENT_JOBS:
            group_name, prompt = pending.pop(0)
            print(f"  Submitting {group_name}...", end=" ", flush=True)
            job_id = submit_hunyuan_job(client, prompt, group_name)
            if job_id:
                print(f"JobId={job_id}")
                jobs[group_name] = {"job_id": job_id, "status": "RUN", "glb_url": None, "error": None}
            else:
                print("Failed")
                jobs[group_name] = {"job_id": None, "status": "FAIL", "glb_url": None, "error": "Submit failed"}

            time.sleep(0.5)

        if count_running() > 0:
            print(f"\n  Running: {count_running()}, queued: {len(pending)}, downloaded: {len(downloaded)}. Polling in {POLL_INTERVAL_SECONDS}s...")
            time.sleep(POLL_INTERVAL_SECONDS)
            poll_running()

    submitted = sum(1 for j in jobs.values() if j["job_id"])
    completed = sum(1 for j in jobs.values() if j["status"] == "DONE")
    print(f"\nJobs: {submitted} submitted, {completed} completed, {len(downloaded)} downloaded")

    if downloaded:
        entity_models = load_existing_entity_models()
        for placement_id, group in placement_to_group.items():
            if group in downloaded:
                entity_models[placement_id] = f"res://models_used/{group}.glb"
        save_entity_models(entity_models)
        print(f"Updated entity_models.json for {len(downloaded)} groups")

    if any(j.get("status") == "FAIL" for j in jobs.values()):
        print("\nFailed groups:")
        for g, j in jobs.items():
            if j.get("status") == "FAIL":
                print(f"  - {g}: {j.get('error', 'Unknown')}")

    print("\nNext: run generate_entity_models.py, then frontage_assets + optional Blender + consolidate_metadata.")


if __name__ == "__main__":
    main()
