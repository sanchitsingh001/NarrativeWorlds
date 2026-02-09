import bpy
import sys
import os
import math
import json
import base64
import urllib.request
import numpy as np
from mathutils import Vector

# ------------------------------------------------------------
# Args
# ------------------------------------------------------------
argv = sys.argv
argv = argv[argv.index("--") + 1:] if "--" in argv else []

if len(argv) < 2:
    raise SystemExit(
        "Usage:\n"
        "  blender -b -P batch_blender.py -- <in_dir> <out_dir> [--ask_vlm] [--frontage_assets <path>]\n"
        "  --frontage_assets <path>  JSON file with list of asset names (GLB stems) to process.\n"
        "                            Use this for entities with needs_frontage=true so they get\n"
        "                            front_yaw_deg for correct orientation (face the path).\n"
        "Example:\n"
        "  python mesh_dir/frontage_assets.py   # build frontage_assets.json from world layout\n"
        "  export DEEPINFRA_API_KEY=...\n"
        "  blender -b -P batch_blender.py -- glb_files renders --ask_vlm --frontage_assets frontage_assets.json"
    )

in_dir = argv[0]
out_dir = argv[1]
ask_vlm = ("--ask_vlm" in argv)

# Optional: only process GLBs used by needs_frontage entities (for correct face-the-path orientation)
frontage_assets_path = None
if "--frontage_assets" in argv:
    i = argv.index("--frontage_assets")
    if i + 1 < len(argv):
        frontage_assets_path = argv[i + 1]
frontage_set = None
if frontage_assets_path and os.path.isfile(frontage_assets_path):
    with open(frontage_assets_path, "r") as f:
        frontage_set = set(json.load(f))
    print("Frontage assets filter: %d asset(s)" % len(frontage_set))
elif frontage_assets_path:
    print("WARNING: --frontage_assets file not found:", frontage_assets_path)

os.makedirs(out_dir, exist_ok=True)

print("BATCH FRONT DETECTOR ✅")
print("IN :", in_dir)
print("OUT:", out_dir)
print("ask_vlm:", ask_vlm)

API_KEY = os.environ.get("DEEPINFRA_API_KEY", "")

# ------------------------------------------------------------
# Reset scene once
# ------------------------------------------------------------
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene

if scene.world is None:
    scene.world = bpy.data.worlds.new("World")

# ------------------------------------------------------------
# Choose fastest available render engine
# ------------------------------------------------------------
available_engines = {e.identifier for e in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items}
# Prefer Eevee, then Workbench, then Cycles
if "BLENDER_EEVEE" in available_engines:
    scene.render.engine = "BLENDER_EEVEE"
elif "BLENDER_WORKBENCH" in available_engines:
    scene.render.engine = "BLENDER_WORKBENCH"
else:
    scene.render.engine = "CYCLES"

print("Render engine:", scene.render.engine)

# ------------------------------------------------------------
# Global render settings (FAST)
# ------------------------------------------------------------
scene.render.image_settings.file_format = "PNG"
scene.render.film_transparent = False

VIEW_RES = 256
scene.render.resolution_x = VIEW_RES
scene.render.resolution_y = VIEW_RES

# Eevee knobs (only if Eevee)
if scene.render.engine == "BLENDER_EEVEE":
    ee = scene.eevee
    # Some properties may differ slightly by version; use getattr guards
    if hasattr(ee, "taa_render_samples"):
        ee.taa_render_samples = 8
    if hasattr(ee, "use_gtao"):
        ee.use_gtao = False
    if hasattr(ee, "use_bloom"):
        ee.use_bloom = False
    if hasattr(ee, "use_ssr"):
        ee.use_ssr = False
    if hasattr(ee, "use_volumetric_lights"):
        ee.use_volumetric_lights = False
    if hasattr(ee, "use_volumetric_shadows"):
        ee.use_volumetric_shadows = False

# If it fell back to Cycles, keep it minimal
if scene.render.engine == "CYCLES":
    scene.cycles.samples = 16
    if hasattr(scene.cycles, "use_denoising"):
        scene.cycles.use_denoising = False

# Color management
scene.view_settings.view_transform = "Standard"
scene.view_settings.look = "None"
scene.view_settings.exposure = 0.6
scene.view_settings.gamma = 1.0

# World background (dim gray)
scene.world.use_nodes = True
ntw = scene.world.node_tree
for n in list(ntw.nodes):
    ntw.nodes.remove(n)
outw = ntw.nodes.new("ShaderNodeOutputWorld")
bg = ntw.nodes.new("ShaderNodeBackground")
bg.inputs["Color"].default_value = (0.18, 0.18, 0.18, 1.0)
bg.inputs["Strength"].default_value = 0.35
ntw.links.new(bg.outputs["Background"], outw.inputs["Surface"])

# ------------------------------------------------------------
# Base scene objects: camera + lights
# ------------------------------------------------------------
# Remove all objects first
for o in list(scene.objects):
    bpy.data.objects.remove(o, do_unlink=True)

# Camera
cam_data = bpy.data.cameras.new("Camera")
cam_data.lens_unit = "FOV"
cam_data.angle = math.radians(35.0)
cam_data.clip_start = 0.01
cam_data.clip_end = 100000.0
cam = bpy.data.objects.new("Camera", cam_data)
scene.collection.objects.link(cam)
scene.camera = cam

def look_at(obj, target: Vector):
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()

# Lights (we will recreate per asset)
def clear_lights():
    for obj in list(scene.objects):
        if obj.type == "LIGHT":
            bpy.data.objects.remove(obj, do_unlink=True)

def add_area(name, energy, loc: Vector, size_val: float):
    ld = bpy.data.lights.new(name=name, type="AREA")
    ld.energy = energy
    ld.shape = "SQUARE"
    ld.size = size_val
    lo = bpy.data.objects.new(name=name, object_data=ld)
    scene.collection.objects.link(lo)
    lo.location = loc
    return lo

# ------------------------------------------------------------
# Helpers: clear imported, bounds, grid (no PIL)
# ------------------------------------------------------------
def clear_imported_meshes():
    for o in list(scene.objects):
        if o.type in ("MESH", "EMPTY", "ARMATURE"):
            bpy.data.objects.remove(o, do_unlink=True)
    # keep memory from exploding for lots of assets
    try:
        bpy.ops.outliner.orphans_purge(do_recursive=True)
    except Exception:
        pass

def compute_bounds(mesh_objs):
    mins = Vector((1e9, 1e9, 1e9))
    maxs = Vector((-1e9, -1e9, -1e9))
    for o in mesh_objs:
        for v in o.bound_box:
            w = o.matrix_world @ Vector(v)
            mins = Vector((min(mins.x, w.x), min(mins.y, w.y), min(mins.z, w.z)))
            maxs = Vector((max(maxs.x, w.x), max(maxs.y, w.y), max(maxs.z, w.z)))
    center = (mins + maxs) * 0.5
    size = (maxs - mins)
    return center, size

def load_png_rgba_float(path: str):
    img = bpy.data.images.load(path, check_existing=False)
    w, h = img.size
    arr = np.array(img.pixels[:], dtype=np.float32).reshape((h, w, 4))
    arr = np.flipud(arr)  # <-- IMPORTANT: convert bottom-up -> top-down
    bpy.data.images.remove(img, do_unlink=True)
    return arr

def save_rgba_float_to_png(arr: np.ndarray, path: str):
    h, w, _ = arr.shape
    out_img = bpy.data.images.new("grid_tmp", width=w, height=h, alpha=True)
    arr_out = np.flipud(arr)  # <-- convert back to Blender's bottom-up pixel order
    out_img.pixels = arr_out.reshape(-1).tolist()
    out_img.filepath_raw = path
    out_img.file_format = "PNG"
    out_img.save()
    bpy.data.images.remove(out_img, do_unlink=True)


def make_grid_2x2(view_paths, out_png):
    A = load_png_rgba_float(view_paths[0])
    B = load_png_rgba_float(view_paths[1])
    C = load_png_rgba_float(view_paths[2])
    D = load_png_rgba_float(view_paths[3])

    h, w, _ = A.shape
    grid = np.ones((h * 2, w * 2, 4), dtype=np.float32)
    grid[..., :3] = 1.0
    grid[..., 3] = 1.0

    grid[0:h, 0:w, :] = A
    grid[0:h, w:2*w, :] = B
    grid[h:2*h, 0:w, :] = C
    grid[h:2*h, w:2*w, :] = D

    save_rgba_float_to_png(grid, out_png)

def png_to_data_url(png_path: str) -> str:
    with open(png_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return "data:image/png;base64," + b64

def ask_deepinfra_front(grid_path: str) -> str:
    if not API_KEY:
        raise RuntimeError("DEEPINFRA_API_KEY not set")

    url = "https://api.deepinfra.com/v1/openai/chat/completions"
    prompt = (
        "You are given a 2x2 grid image of the same 3D structure from four yaw directions.\n"
        "Define panels by position:\n"
        "A = top-left, B = top-right, C = bottom-left, D = bottom-right.\n\n"
        "Task: Pick which panel shows the FRONT—the side where customers would approach.\n"
        "- For stalls and shops: the side with the main product display, sales counter, or entrance.\n"
        "- For buildings: the side with the main entrance or customer-facing facade.\n"
        "- Avoid picking a blank wall, sign-back, or storage side.\n"
        "If the object is non-directional or you cannot tell, answer 'NONE'.\n\n"
        "Reply with ONLY one of: A, B, C, D, NONE.\n"
    )

    payload = {
        "model": "Qwen/Qwen2.5-VL-32B-Instruct",
        "max_tokens": 32,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": png_to_data_url(grid_path)}},
            ],
        }],
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=120) as resp:
        j = json.loads(resp.read().decode("utf-8"))
    return j["choices"][0]["message"]["content"].strip()

# ------------------------------------------------------------
# Process each GLB (optionally only those in frontage_assets list)
# ------------------------------------------------------------
all_glbs = sorted([f for f in os.listdir(in_dir) if f.lower().endswith(".glb")])
if frontage_set is not None:
    glbs = [f for f in all_glbs if os.path.splitext(f)[0] in frontage_set]
    print("Found %d GLBs (%d match frontage_assets)" % (len(all_glbs), len(glbs)))
else:
    glbs = all_glbs
    print("Found", len(glbs), "GLBs")

mapping = {"A": 0, "B": 90, "C": 180, "D": 270}
yaws = [0, 90, 180, 270]

for idx, fname in enumerate(glbs, 1):
    glb_path = os.path.join(in_dir, fname)
    stem = os.path.splitext(fname)[0]
    asset_out = os.path.join(out_dir, stem)
    os.makedirs(asset_out, exist_ok=True)

    print(f"\n[{idx}/{len(glbs)}] Import:", glb_path)

    # Clean meshes & lights from previous asset
    clear_imported_meshes()
    clear_lights()

    bpy.ops.import_scene.gltf(filepath=glb_path)
    mesh_objs = [o for o in scene.objects if o.type == "MESH"]
    if not mesh_objs:
        print("  ❌ No meshes, skipping")
        continue

    center, size = compute_bounds(mesh_objs)
    max_dim = max(size.x, size.y, size.z)

    dist = max(2.0, max(size.x, size.y) * 2.2)
    light_size = max(1.0, max_dim * 0.8)

    add_area("Key",  800.0, Vector((center.x + dist, center.y + dist*0.4, center.z + dist*0.8)), light_size)
    add_area("Fill", 350.0, Vector((center.x - dist, center.y + dist*0.3, center.z + dist*0.6)), light_size)

    cam_height = center.z + 0.03 * size.z

    view_paths = []
    for yaw in yaws:
        ang = math.radians(yaw)
        cam.location = Vector((
            center.x + dist * math.sin(ang),
            center.y + dist * math.cos(ang),
            cam_height
        ))
        look_at(cam, Vector((center.x, center.y, cam_height)))

        out_path = os.path.join(asset_out, f"view_{yaw}.png")
        scene.render.filepath = out_path
        bpy.ops.render.render(write_still=True)
        view_paths.append(out_path)

    grid_path = os.path.join(asset_out, "grid_2x2.png")
    make_grid_2x2(view_paths, grid_path)
    print("  ✅ Rendered + grid:", grid_path)

    result = {
        "asset": fname,
        "grid": "grid_2x2.png",
        "views": {str(y): f"view_{y}.png" for y in yaws},
        "front_panel": None,
        "front_yaw_deg": None,
        # Bounding box dimensions (in Blender units, typically meters)
        "bbox_width": float(size.x),   # Along model's local X
        "bbox_depth": float(size.y),   # Along model's local Y (forward)
        "bbox_height": float(size.z),  # Along model's local Z (up)
    }

    if ask_vlm:
        try:
            raw_ans = ask_deepinfra_front(grid_path)
            print("  VLM raw:", raw_ans)
            # Clean VLM response: strip whitespace, take first character, uppercase
            ans = raw_ans.strip().upper()
            ans = ans[0] if ans else "NONE"
            print("  VLM parsed:", ans)
            result["front_panel"] = ans
            if ans in mapping:
                result["front_yaw_deg"] = mapping[ans]
        except Exception as e:
            print("  ❌ VLM failed:", repr(e))

    with open(os.path.join(asset_out, "front.json"), "w") as f:
        json.dump(result, f, indent=2)

print("\nDONE ✅")

