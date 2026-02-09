extends Node3D

# --- PLAYER-ANCHORED SCALE ---
@export var player_height_m: float = 1.8            # canonical human height
@export var grid_unit_in_player_heights: float = 1.0 # 1 JSON grid unit = 1.0 * player height (was 1.5)
@export var road_thickness_m: float = 0.08
@export var ground_thickness_m: float = 0.1  # Thin ground plane, just for visual

var tile_size_m: float = 1.0  # computed in _ready

const HEIGHT_SCALE := 1.0

# -----------------------------
# World source of truth (v3)
# -----------------------------
@export var v3_world_path := "res://generated/world_entity_layout_llm_v3_out.json"
@export var v3_auto_world_to_grid_scale: bool = true
@export var v3_world_to_grid_scale: float = 1.0 # used only when auto_scale is false
@export var v3_road_tile_overlap: float = 1.08 # >1.0 makes internal road tiles look continuous
@export var use_road_collision: bool = false # set true only if you really need road colliders

# Legacy world files (deprecated; no longer used for world structure)
@export var world_layout_path := "res://generated/world_layout.json"
@export var area_layouts_path := "res://generated/area_layouts.json"
@export var asset_metadata_path := "res://generated/asset_metadata.json"
@export var entity_models_path := "res://generated/entity_models.json"
@export var npc_models_path := "res://generated/npc_models.json"
@export var world_plan_path := "res://generated/world_plan.json"
@export var road_texture_mapping_path := "res://generated/road_texture_mapping.json"
@export var debug_orientation: bool = false  # Log entity_facing_yaw, asset_front_yaw, rotation_yaw per placed entity
@export var npc_model_scale: float = 1.5  # Scale NPC GLBs to human size (models ~1m, target ~1.7m)
@export var normalize_group_sizes: bool = true  # Use fixed size per group (ignores layout w/h) for consistent footprints
@export var group_size_overrides: Dictionary = {}  # group_name -> {"w": float, "h": float} (optional manual overrides)
@export var use_mesh_collision_for_entities: bool = true  # Use convex hull from mesh for entity collision; if false or fails, use box

var area_layouts := {}
var asset_metadata := {}  # Maps asset name -> {front_yaw_deg, bbox_*}
var entity_models := {}   # Maps Entity ID -> Model Path
var npc_models := {}      # Maps npc_id -> Model Path (res://models_used/npc_<npc_id>.glb when 3D generated)
var npc_plan: Array = []  # From world_plan.npc_plan: { npc_id, display_name, area_id, anchor_entity: { id } }
var road_texture_mapping := {}  # Maps area_id -> {texture_path, ...}
var normalized_group_sizes_cache := {}  # group_name -> {"w": float, "h": float} (computed once at load)

var _v3_world_to_grid_scale_runtime: float = 1.0

var _player_node: Node3D = null

# Platform detection (canonical, stable check)
var is_web := OS.has_feature("web")

func _init() -> void:
	# Debug: Log platform detection
	if OS.has_feature("web"):
		print("Platform: WEB (OS.has_feature('web') = true)")
	else:
		print("Platform: NATIVE (OS.has_feature('web') = false)")
		print("  OS.get_name() = ", OS.get_name())

# Cache for GLTF models to avoid reloading from disk 
var _model_cache := {} # Path -> { "doc": GLTFDocument, "state": GLTFState }
# Cache for Web-loaded scenes (PackedScene instances)
var _web_scene_cache := {} # Path -> PackedScene

# Material caching for performance (web optimization)
var _grass_material: ShaderMaterial = null
var _road_materials := {}  # Maps area_id -> StandardMaterial3D (cached per area)
var _noise_texture: Texture2D = null

# Grass: procedural rectangles (crossed quads). On by default; addon grass often fails (texture/singleton).
@export var use_grass_meshes: bool = true
@export var grass_density_per_m2: float = 60.0  # true instances per m² (try 60–120)
@export var grass_blade_height_m: float = 0.18  # blade height in meters (0.16–0.22)
@export var grass_blade_width_m: float = 0.05   # blade width in meters (0.04–0.06)
@export var grass_tilt_deg: float = 6.0        # random tilt variation in degrees (4–8)
@export var grass_max_instances: int = 25000   # cap per area (8000 used on web)
@export var grass_blade_texture_path: String = "res://textures/grass_blade.png"  # optional: white blade + alpha for alpha scissor

# SimpleGrassTextured addon: scatter grass (with wind). Requires addon + SimpleGrass autoload; off by default until addon resources work.
@export var use_simple_grass_textured: bool = false
@export var simple_grass_density: float = 0.12  # instances per m²; capped per area for performance
@export var simple_grass_max_per_area: int = 3500  # cap per area

func _ready() -> void:
	# Re-check platform at runtime (defensive)
	var runtime_is_web = OS.has_feature("web")
	if runtime_is_web != is_web:
		print("WARNING: Platform detection mismatch!")
		print("  Class-level is_web: ", is_web)
		print("  Runtime is_web: ", runtime_is_web)
		is_web = runtime_is_web  # Update to runtime value
	
	
	if is_web:
		print("=== WEB BUILD DETECTED ===")
		print("  All GLB loading will use ResourceLoader.load()")
		print("  GLTFDocument.generate_scene() will NOT be used")
	
	print("WorldLoader: _ready started")
	
	tile_size_m = player_height_m * grid_unit_in_player_heights
	
	# Load the world from v3 JSON (single source of truth for world structure)
	var v3_path = _resolve_path(v3_world_path)
	print("WorldLoader: Loading v3 world from ", v3_path)
	var v3_data = _load_json(v3_path)
	if v3_data == null:
		push_error("Failed to load v3 world: %s" % v3_path)
		return

	var built = _build_runtime_from_v3(v3_data)
	var world_data: Dictionary = built.get("world_data", {})
	area_layouts = built.get("area_layouts", {})
	print("WorldLoader: v3 world loaded. Areas: ", world_data.get("areas", []).size(), " | area_layouts: ", area_layouts.size())
	
	# Compute normalized group sizes (for consistent footprints across areas)
	if normalize_group_sizes:
		normalized_group_sizes_cache = _compute_normalized_group_sizes()
		print("WorldLoader: Normalized sizes for ", normalized_group_sizes_cache.size(), " groups")
	
	# Load asset metadata
	var meta = _load_json(_resolve_path(asset_metadata_path))
	if meta != null:
		asset_metadata = meta
	else:
		asset_metadata = {}

	# Load entity model mapping
	var models = _load_json(_resolve_path(entity_models_path))
	if models != null:
		entity_models = models
	else:
		entity_models = {}

	# Load NPC model mapping (generated when 3D generation runs; missing/empty when skipped)
	var npc_mod = _load_json(_resolve_path(npc_models_path))
	if npc_mod != null and npc_mod is Dictionary:
		npc_models = npc_mod
	else:
		npc_models = {}

	# Load road texture mapping
	var mapping = _load_json(_resolve_path(road_texture_mapping_path))
	if mapping != null:
		road_texture_mapping = mapping
		print("Loaded road texture mapping for ", road_texture_mapping.size(), " areas")
	else:
		road_texture_mapping = {}

	# Load npc_plan from world_plan for NPC placement
	var world_plan = _load_json(_resolve_path(world_plan_path))
	if world_plan != null and world_plan.has("npc_plan"):
		npc_plan = world_plan["npc_plan"]
		if npc_plan is Array:
			print("WorldLoader: Loaded npc_plan with ", npc_plan.size(), " NPCs")
		else:
			npc_plan = []
	else:
		npc_plan = []

	_spawn_world(world_data)
	_spawn_player(world_data)
	# Enable wind for SimpleGrassTextured grass (if addon is loaded)
	_setup_simple_grass_wind()

# -----------------------------
# v3 adapter (build runtime data from v3 JSON)
# -----------------------------

func _median(vals: Array) -> float:
	if vals.is_empty():
		return 1.0
	var s = vals.duplicate()
	s.sort()
	var n: int = s.size()
	var mid := int(n / 2)
	if n % 2 == 1:
		return float(s[mid])
	return (float(s[mid - 1]) + float(s[mid])) * 0.5

func _dir_to_yaw_deg(d: String) -> float:
	var dir := d.strip_edges().to_upper()
	match dir:
		"N":
			return 0.0
		"NE":
			return 45.0
		"E":
			return 90.0
		"SE":
			return 135.0
		"S":
			return 180.0
		"SW":
			return 225.0
		"W":
			return 270.0
		"NW":
			return 315.0
		_:
			return 180.0

func _edge_to_cardinal(edge: String) -> String:
	var e := edge.strip_edges().to_lower()
	if e == "top":
		return "N"
	if e == "bottom":
		return "S"
	if e == "left":
		return "W"
	if e == "right":
		return "E"
	return edge

func _build_runtime_from_v3(v3: Dictionary) -> Dictionary:
	var world_space: Dictionary = v3.get("world_space", {})
	var ws_areas: Dictionary = world_space.get("areas", {})
	var ws_connections: Array = world_space.get("connections", [])
	var v3_areas: Dictionary = v3.get("areas", {})

	# Compute a world->grid scale so the median v3 tile becomes ~1 grid unit.
	var tile_sizes: Array = []
	for aid in ws_areas.keys():
		var a_ws: Dictionary = ws_areas[aid]
		var ts := float(a_ws.get("tile_size_world", 0.0))
		if ts > 0.0:
			tile_sizes.append(ts)

	var scale := float(v3_world_to_grid_scale)
	if v3_auto_world_to_grid_scale:
		var med := _median(tile_sizes)
		if med > 0.0:
			scale = 1.0 / med
	_v3_world_to_grid_scale_runtime = scale
	print("WorldLoader: v3 world_to_grid_scale = ", _v3_world_to_grid_scale_runtime, " (auto=", v3_auto_world_to_grid_scale, ")")

	var out_world_areas: Array = []
	var out_world_gates: Array = []
	var out_connections: Array = []
	var out_area_layouts: Dictionary = {}

	# Connections are global world-space segments: {a:[x,y], b:[x,y]}
	for c in ws_connections:
		if c is Dictionary and c.has("a") and c.has("b"):
			var a = c.get("a", [])
			var b = c.get("b", [])
			if a is Array and b is Array and a.size() >= 2 and b.size() >= 2:
				out_connections.append({
					"polyline": [
						[float(a[0]) * scale, float(a[1]) * scale],
						[float(b[0]) * scale, float(b[1]) * scale]
					]
				})

	for aid in ws_areas.keys():
		var a_ws: Dictionary = ws_areas[aid]
		var rect: Dictionary = a_ws.get("rect", {})
		var rx := float(rect.get("x", 0.0))
		var ry := float(rect.get("y", 0.0))
		var rw := float(rect.get("w", 40.0))
		var rh := float(rect.get("h", 30.0))

		# World layout: areas in GLOBAL grid units
		out_world_areas.append({
			"area_id": str(aid),
			"x": rx * scale,
			"y": ry * scale,
			"w": rw * scale,
			"h": rh * scale
		})

		# Gates for spawn: GLOBAL grid units
		var gates_world: Array = a_ws.get("gates_world", [])
		var gate_i := 0
		for g in gates_world:
			if not (g is Dictionary):
				continue
			gate_i += 1
			out_world_gates.append({
				"area_id": str(aid),
				"gate_id": "gate_%d" % gate_i,
				"edge": _edge_to_cardinal(str(g.get("edge", ""))),
				"world_x": float(g.get("world_x", 0.0)) * scale,
				"world_y": float(g.get("world_y", 0.0)) * scale
			})

		# Per-area layouts: entity and road coordinates must be LOCAL to the area node.
		# Convert world-space coordinates -> local by subtracting the area rect origin.
		var intent: Dictionary = {}
		if v3_areas.has(aid):
			intent = (v3_areas[aid] as Dictionary).get("intent", {})

		var entities_local: Array = []
		var entities_world: Dictionary = a_ws.get("entities_world", {})
		for eid in entities_world.keys():
			var e: Dictionary = entities_world[eid]
			var ex := float(e.get("x", 0.0))
			var ey := float(e.get("y", 0.0))
			var ew := float(e.get("w", 1.0))
			var eh := float(e.get("h", 1.0))
			var dir := ""
			if intent.has(eid):
				dir = str((intent[eid] as Dictionary).get("dir", ""))
			var needs_frontage: bool = bool(e.get("needs_frontage", false))
			entities_local.append({
				"id": str(eid),
				"x": (ex - rx) * scale,
				"y": (ey - ry) * scale,
				"w": ew * scale,
				"h": eh * scale,
				"rotation_deg": _dir_to_yaw_deg(dir),
				"needs_frontage": needs_frontage
			})

		# v3 roads_world points appear to be TILE CENTERS in world-space.
		# We keep them as centers in LOCAL coords and render without +0.5 offsets.
		var road_tiles_world_local: Array = []
		var roads_world: Array = a_ws.get("roads_world", [])
		for t in roads_world:
			if t is Array and t.size() >= 2:
				road_tiles_world_local.append([
					(float(t[0]) - rx) * scale,
					(float(t[1]) - ry) * scale
				])

		out_area_layouts[str(aid)] = {
			"entities": entities_local,
			"road_tiles_world": road_tiles_world_local,
			# Keep these for compatibility with older code paths; they remain zero in v3 mode.
			"tilemap_min_x": 0.0,
			"tilemap_min_y": 0.0
		}

	return {
		"world_data": {
			"areas": out_world_areas,
			"gates": out_world_gates,
			"connections": out_connections
		},
		"area_layouts": out_area_layouts
	}

# Helper to resolve paths (check res:// then external)
func _resolve_path(path: String) -> String:
	if FileAccess.file_exists(path):
		return path
	
	# External fallback
	var rel_path = path.replace("res://", "")
	var base_dir = OS.get_executable_path().get_base_dir()
	var ext_path = ""
	
	if OS.get_name() == "macOS":
		ext_path = base_dir.path_join("../Resources/" + rel_path)
	else:
		ext_path = base_dir.path_join(rel_path)
	
	if FileAccess.file_exists(ext_path):
		return ext_path
	return path

# Load a model instance (Web-safe: uses ResourceLoader, Native: uses GLTFDocument)
func _load_model_instance(path: String) -> Node3D:
	"""
	Load a model instance from a GLB path.
	On Web: Uses ResourceLoader.load() to load pre-imported PackedScene
	On Native: Uses GLTFDocument for runtime loading (fallback)
	"""
	# Double-check platform at runtime (defensive)
	var runtime_is_web = OS.has_feature("web")
	
	if runtime_is_web:
		# Web code unchanged...
		if not _web_scene_cache.has(path):
			print("WEB MODE: Loading via ResourceLoader: ", path)
		
		if _web_scene_cache.has(path):
			var scene: PackedScene = _web_scene_cache[path]
			return scene.instantiate()
		
		if not ResourceLoader.exists(path):
			push_error("Resource does not exist: " + path)
			return null
		
		var scene = ResourceLoader.load(path)
		if scene == null: return null
		
		# ... (cache logic)
		_web_scene_cache[path] = scene as PackedScene
		return scene.instantiate()
	else:
		# Native: Use GLTFDocument
		var doc: GLTFDocument
		var state: GLTFState
		
		if _model_cache.has(path):
			var data = _model_cache[path]
			doc = data.doc
			state = data.state
		else:
			print("_load_model_instance: Using Native path for: ", path)
			doc = GLTFDocument.new()
			state = GLTFState.new()
			
			# Resolve path properly for Native (handle PCK vs Filesystem)
			var load_path = ProjectSettings.globalize_path(path)
			
			# If globalize doesn't yield a file, try manual resolution
			if not FileAccess.file_exists(load_path):
				load_path = _resolve_path(path)
				
			var err = doc.append_from_file(load_path, state)
			if err != OK:
				push_error("Failed to load model: %s error: %d" % [load_path, err])
				return null
			
			_model_cache[path] = { "doc": doc, "state": state }
		
		var instance = doc.generate_scene(state)
		if instance == null:
			push_error("Failed to generate scene from GLB: %s" % path)
			return null
		
		return instance

func _get_model_data_for_entity(entity_id: String) -> Dictionary:
	"""
	Get model data for entity (Native only - for GLTFDocument fallback).
	On Web, use _load_model_instance() instead.
	"""
	var path = entity_models.get(entity_id, "")
	if path == "":
		return {}
	
	# On Web, this function should not be used - use _load_model_instance() instead
	var runtime_is_web = OS.has_feature("web")
	if runtime_is_web:
		push_error("CRITICAL: _get_model_data_for_entity() called on Web!")
		push_error("  This should never happen - use _load_model_instance() instead")
		push_error("  Entity ID: ", entity_id, " Path: ", path)
		push_error("  Returning empty to prevent GLTFDocument usage")
		# Return empty to prevent GLTFDocument usage
		return {}
	
	# Check cache
	if _model_cache.has(path):
		return _model_cache[path]
	
	# CRITICAL: Double-check we're not on Web before using GLTFDocument
	var double_check_web = OS.has_feature("web")
	if double_check_web:
		push_error("CRITICAL ERROR: Attempted to use GLTFDocument in _get_model_data_for_entity() on Web!")
		push_error("  Path: ", path)
		push_error("  This will cause WASM heap corruption!")
		return {}
		
	# Load new (Native only)
	var doc = GLTFDocument.new()
	var state = GLTFState.new()
	var err = doc.append_from_file(ProjectSettings.globalize_path(path), state)
	if err != OK:
		push_error("Failed to load model: %s error: %d" % [path, err])
		return {}
	
	var data = { "doc": doc, "state": state }
	_model_cache[path] = data
	return data

func _get_asset_name_from_path(path: String) -> String:
	return path.get_file().get_basename()

func _get_grass_material() -> ShaderMaterial:
	"""
	Get or create the shared grass shader material.
	Reusing materials improves performance, especially for web export.
	"""
	if _grass_material != null:
		return _grass_material
	
	# Load shader
	var shader = load("res://shaders/grass.gdshader")
	if shader == null:
		push_error("Grass shader not found at res://shaders/grass.gdshader! Falling back to standard material.")
		return null
	
	# Create shader material
	_grass_material = ShaderMaterial.new()
	_grass_material.shader = shader
	
	# Load or generate noise texture
	if ResourceLoader.exists("res://textures/noise.png"):
		_noise_texture = load("res://textures/noise.png")
	else:
		# Fallback: generate noise texture programmatically
		_noise_texture = _generate_noise_texture()
		print("Generated noise texture (noise.png not found)")
	
	# Set shader parameters
	# Note: Shader multiplies colors, so we use moderate values for natural grass
	# These create a more realistic, varied grass appearance
	_grass_material.set_shader_parameter("color", Color(0.12, 0.30, 0.10))  # Dark green (natural grass)
	_grass_material.set_shader_parameter("color2", Color(0.18, 0.35, 0.12))  # Medium-dark green (lighter patches)
	_grass_material.set_shader_parameter("noise", _noise_texture)
	_grass_material.set_shader_parameter("noiseScale", 12.0)  # Smaller scale = more visible variation
	
	print("Grass shader material created successfully")
	print("  - Shader loaded: ", shader != null)
	print("  - Noise texture: ", _noise_texture != null)
	
	return _grass_material

func _get_road_material(area_id: String = "") -> StandardMaterial3D:
	"""
	Get or create the road material for a specific area.
	Each area can have its own texture based on road_texture_mapping.
	Reusing materials per area improves performance.
	"""
	# Use area_id for caching, fallback to "default" if not provided
	var cache_key = area_id if area_id != "" else "default"
	
	# Return cached material if available
	if _road_materials.has(cache_key):
		return _road_materials[cache_key]
	
	# Create new material for this area
	var road_material = StandardMaterial3D.new()
	road_material.albedo_color = Color(1.0, 1.0, 1.0)  # White - don't tint the texture, let it show natural colors
	road_material.metallic = 0.0  # Non-metallic
	road_material.roughness = 0.9  # Rough surface
	
	# Look up texture path from mapping if area_id provided
	var texture_path: String = ""
	if area_id != "" and road_texture_mapping.has(area_id):
		var mapping = road_texture_mapping[area_id]
		texture_path = mapping.get("texture_path", "")
		print("Loading road texture for area '", area_id, "': ", texture_path)
	else:
		# Fallback: use a default stone road texture so roads are always textured.
		texture_path = "res://roads/m_stone_road_2k/T_Stone_Road_BaseColor.png"
		print("No texture mapping found for area '", area_id, "', using default stone road texture: ", texture_path)
	
	var road_texture: Texture2D = null
	
	# Try loading texture if path is specified
	if texture_path != "":
		# CRITICAL: On Web, always use ResourceLoader to get compressed/imported textures
		# FileAccess + Image.load_from_file loads uncompressed images which use massive memory
		if is_web:
			# Web: Only use ResourceLoader (imported/compressed textures)
			if ResourceLoader.exists(texture_path):
				road_texture = load(texture_path)
			else:
				push_error("Road texture not found (not imported?): " + texture_path)
		else:
			# Native: Try ResourceLoader first, then fallback to FileAccess
			if ResourceLoader.exists(texture_path):
				road_texture = load(texture_path)
			
			# If that fails, try loading directly as Image from absolute path (Native only)
			if road_texture == null:
				var absolute_path = ProjectSettings.globalize_path(texture_path)
				var project_root = ProjectSettings.globalize_path("res://")
				var manual_path = project_root + texture_path.replace("res://", "")
				
				# Try both paths
				var paths_to_try = [absolute_path, manual_path]
				var img: Image = null
				
				for path in paths_to_try:
					if FileAccess.file_exists(path):
						# Try method 1: load_from_file
						img = Image.new()
						img.load_from_file(path)
						
						# If that didn't work, try method 2: load from buffer based on file extension
						if img.get_width() <= 0 or img.get_height() <= 0:
							var file = FileAccess.open(path, FileAccess.READ)
							if file != null:
								var buffer = file.get_buffer(file.get_length())
								file.close()
								img = Image.new()
								
								# Determine file type from extension
								var file_extension = path.get_extension().to_lower()
								var err: Error = ERR_FILE_CANT_READ
								
								if file_extension == "png":
									err = img.load_png_from_buffer(buffer)
								elif file_extension == "jpg" or file_extension == "jpeg":
									err = img.load_jpg_from_buffer(buffer)
								else:
									# Try PNG first, then JPG
									err = img.load_png_from_buffer(buffer)
									if err != OK:
										err = img.load_jpg_from_buffer(buffer)
								
								if err != OK:
									img = null
						
						if img != null and img.get_width() > 0 and img.get_height() > 0:
							# Ensure image is in correct format (RGB8 or RGBA8 for proper color)
							if img.get_format() != Image.FORMAT_RGB8 and img.get_format() != Image.FORMAT_RGBA8:
								img.convert(Image.FORMAT_RGBA8)
							
							var image_texture = ImageTexture.new()
							image_texture.set_image(img)
							road_texture = image_texture
							print("  - Successfully loaded road texture from: ", path)
							break
		
		if road_texture == null and texture_path != "":
			push_error("Failed to load road texture from: " + texture_path + " for area: " + area_id + ". Using fallback color.")
	
	# Apply texture if loaded
	if road_texture != null:
		road_material.albedo_texture = road_texture
	else:
		# If no texture loaded, use fallback color
		road_material.albedo_color = Color(0.5, 0.45, 0.4)  # Brown/tan road color (fallback)
	
	# Cache the material
	_road_materials[cache_key] = road_material
	
	return road_material

func _setup_simple_grass_wind() -> void:
	"""Set wind on SimpleGrass autoload so grass moves with the wind."""
	if not get_tree().root.has_node("SimpleGrass"):
		return
	var simple_grass = get_tree().root.get_node("SimpleGrass")
	if not simple_grass.has_method("set_wind_direction"):
		return
	simple_grass.set_wind_direction(Vector3(1.0, 0.0, 0.3).normalized())
	simple_grass.set_wind_strength(0.18)
	simple_grass.set_wind_turbulence(1.0)
	print("WorldLoader: SimpleGrass wind enabled")


func _spawn_simple_grass_textured(area_node: Node3D, area_id: String, area_w: float, area_h: float) -> void:
	"""Scatter SimpleGrassTextured grass randomly where there is no building or path. Uses wind from SimpleGrass singleton."""
	if not get_tree().root.has_node("SimpleGrass"):
		return  # Skip addon grass when singleton missing (avoids get_node error and texture load errors)
	# Try standard addon path first, then hash-named folder (Option A: use res://addons/simplegrasstextured/)
	var script_path: String = ""
	if ResourceLoader.exists("res://addons/simplegrasstextured/grass.gd"):
		script_path = "res://addons/simplegrasstextured/grass.gd"
	elif ResourceLoader.exists("res://SimpleGrassTextured-4125f7302a7a291b547343d267c8407af7dd7ba6/addons/simplegrasstextured/grass.gd"):
		script_path = "res://SimpleGrassTextured-4125f7302a7a291b547343d267c8407af7dd7ba6/addons/simplegrasstextured/grass.gd"
	if script_path.is_empty():
		return
	var grass_script = load(script_path) as GDScript
	if grass_script == null:
		return
	var grass_node: MultiMeshInstance3D = MultiMeshInstance3D.new()
	grass_node.set_script(grass_script)
	grass_node.name = "SimpleGrassTextured"
	grass_node.position = Vector3.ZERO

	# Build excluded grid: roads (tile centers -> cell) and entity rectangles
	var road_cells: Dictionary = {}  # key = "gx_gz"
	var entities: Array = []
	if area_layouts.has(area_id):
		var a: Dictionary = area_layouts[area_id]
		for t in a.get("road_tiles_world", []):
			if t is Array and t.size() >= 2:
				var cx := int(floorf(float(t[0])))
				var cz := int(floorf(float(t[1])))
				road_cells["%d_%d" % [cx, cz]] = true
		for e in a.get("entities", []):
			entities.append({
				"x": float(e.get("x", 0)),
				"y": float(e.get("y", 0)),
				"w": float(e.get("w", 1)),
				"h": float(e.get("h", 1))
			})

	var area_size_m := area_w * area_h * tile_size_m * tile_size_m
	var target_count := clampi(int(area_size_m * simple_grass_density), 0, simple_grass_max_per_area)
	if target_count <= 0:
		return

	var rng = RandomNumberGenerator.new()
	rng.randomize()
	var transforms: Array[Transform3D] = []
	var attempts := 0
	var max_attempts := target_count * 12

	while transforms.size() < target_count and attempts < max_attempts:
		attempts += 1
		var x_m := rng.randf() * area_w * tile_size_m
		var z_m := rng.randf() * area_h * tile_size_m
		var gx := x_m / tile_size_m
		var gz := z_m / tile_size_m
		# Skip if on road or inside entity footprint
		var ix := int(floorf(gx))
		var iz := int(floorf(gz))
		if road_cells.has("%d_%d" % [ix, iz]):
			continue
		var inside_entity := false
		for e in entities:
			if gx >= e.x and gx < e.x + e.w and gz >= e.y and gz < e.y + e.h:
				inside_entity = true
				break
		if inside_entity:
			continue
		var pos := Vector3(x_m, 0.0, z_m)
		var rot_y := rng.randf() * TAU
		var s := Vector3(
			rng.randf_range(0.85, 1.15),
			rng.randf_range(0.85, 1.15),
			rng.randf_range(0.85, 1.15)
		)
		var t := Transform3D()
		t = t.rotated(Vector3.UP, rot_y)
		t = t.scaled(s)
		t.origin = pos
		transforms.append(t)

	if transforms.is_empty():
		return

	area_node.add_child(grass_node)
	grass_node.temp_dist_min = 0.25
	grass_node.add_grass_batch(transforms)
	# Flush buffer into multimesh (addon only does this in _process, which is off at runtime)
	if grass_node.has_method("_update_multimesh"):
		grass_node.call("_update_multimesh")
	print("WorldLoader: SimpleGrassTextured scattered ", transforms.size(), " instances in ", area_id)


func _create_grass_blade_mesh(blade_height: float, blade_width: float) -> ArrayMesh:
	"""Procedural blade: two crossed vertical quads (X shape). UVs for alpha-cut texture; CULL_DISABLED + optional alpha scissor."""
	var h: float = blade_height
	var w: float = blade_width
	var hw: float = w * 0.5
	# Quad 1: vertical in XY plane (normal +Z). Quad 2: vertical in YZ plane (normal +X)
	var verts := PackedVector3Array([
		Vector3(-hw, 0, 0), Vector3(hw, 0, 0), Vector3(hw, h, 0), Vector3(-hw, h, 0),
		Vector3(0, 0, -hw), Vector3(0, 0, hw), Vector3(0, h, hw), Vector3(0, h, -hw)
	])
	var normals := PackedVector3Array([
		Vector3(0, 0, 1), Vector3(0, 0, 1), Vector3(0, 0, 1), Vector3(0, 0, 1),
		Vector3(1, 0, 0), Vector3(1, 0, 0), Vector3(1, 0, 0), Vector3(1, 0, 0)
	])
	# UVs: full quad = full texture (for alpha-cut blade PNG)
	var uvs := PackedVector2Array([
		Vector2(0, 0), Vector2(1, 0), Vector2(1, 1), Vector2(0, 1),
		Vector2(0, 0), Vector2(1, 0), Vector2(1, 1), Vector2(0, 1)
	])
	var indices := PackedInt32Array([0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7])
	var arrays: Array = []
	arrays.resize(Mesh.ARRAY_MAX)
	arrays[Mesh.ARRAY_VERTEX] = verts
	arrays[Mesh.ARRAY_NORMAL] = normals
	arrays[Mesh.ARRAY_TEX_UV] = uvs
	arrays[Mesh.ARRAY_INDEX] = indices
	var arr_mesh := ArrayMesh.new()
	arr_mesh.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES, arrays)
	var grass_mat := StandardMaterial3D.new()
	grass_mat.albedo_color = Color(0.20, 0.38, 0.13)  # Darker grass blade color
	grass_mat.roughness = 0.9
	grass_mat.metallic = 0.0
	grass_mat.cull_mode = BaseMaterial3D.CULL_DISABLED  # both sides visible = grass-ish, not rectangles
	if grass_blade_texture_path != "" and ResourceLoader.exists(grass_blade_texture_path):
		var tex: Texture2D = load(grass_blade_texture_path) as Texture2D
		if tex != null:
			grass_mat.albedo_texture = tex
			grass_mat.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA_SCISSOR
			grass_mat.alpha_scissor_threshold = 0.5
	arr_mesh.surface_set_material(0, grass_mat)
	return arr_mesh


func _spawn_grass_meshes(area_node: Node3D, area_id: String, area_w: float, area_h: float) -> void:
	"""Dense short ground-cover: procedural crossed-quad blade, true per-m² density, road/entity exclusion."""
	# Build excluded cells: roads (v3 uses road_tiles_world) and entity footprints
	var road_cells: Dictionary = {}
	var entities: Array = []
	if area_layouts.has(area_id):
		var a: Dictionary = area_layouts[area_id]
		for t in a.get("road_tiles_world", []):
			if t is Array and t.size() >= 2:
				var ix := int(floorf(float(t[0])))
				var iz := int(floorf(float(t[1])))
				road_cells["%d_%d" % [ix, iz]] = true
		for t in a.get("road_tiles", []):
			if t is Array and t.size() >= 2:
				road_cells["%d_%d" % [int(t[0]), int(t[1])]] = true
		for e in a.get("entities", []):
			entities.append({
				"x": float(e.get("x", 0)),
				"y": float(e.get("y", 0)),
				"w": float(e.get("w", 1)),
				"h": float(e.get("h", 1))
			})

	var area_size_m: float = area_w * area_h * tile_size_m * tile_size_m
	var target_count: int = int(area_size_m * grass_density_per_m2)
	var cap: int = grass_max_instances
	if is_web:
		cap = mini(grass_max_instances, 8000)
	target_count = clampi(target_count, 0, cap)
	if target_count == 0:
		return

	var grass_mesh: ArrayMesh = _create_grass_blade_mesh(grass_blade_height_m, grass_blade_width_m)

	var valid_positions: Array[Vector3] = []
	var rng = RandomNumberGenerator.new()
	rng.randomize()
	var attempts := 0
	var max_attempts := target_count * 12
	while valid_positions.size() < target_count and attempts < max_attempts:
		attempts += 1
		var x: float = rng.randf() * area_w * tile_size_m
		var z: float = rng.randf() * area_h * tile_size_m
		var gx: float = x / tile_size_m
		var gz: float = z / tile_size_m
		var ix := int(floorf(gx))
		var iz := int(floorf(gz))
		if road_cells.has("%d_%d" % [ix, iz]):
			continue
		var inside_entity := false
		for e in entities:
			if gx >= e.x and gx < e.x + e.w and gz >= e.y and gz < e.y + e.h:
				inside_entity = true
				break
		if not inside_entity:
			valid_positions.append(Vector3(x, 0.0, z))

	var grass_count: int = valid_positions.size()
	if grass_count == 0:
		return

	var mm_inst := MultiMeshInstance3D.new()
	var mm := MultiMesh.new()
	mm.transform_format = MultiMesh.TRANSFORM_3D
	mm.instance_count = grass_count
	mm.mesh = grass_mesh
	mm_inst.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF

	var tilt_rad: float = deg_to_rad(grass_tilt_deg)
	for i in range(grass_count):
		var pos: Vector3 = valid_positions[i]
		var yaw: float = rng.randf() * TAU
		var tilt_x: float = rng.randf_range(-tilt_rad, tilt_rad)
		var tilt_z: float = rng.randf_range(-tilt_rad, tilt_rad)
		var height_scale: float = rng.randf_range(0.8, 1.2)
		var transform := Transform3D()
		transform = transform.rotated(Vector3.UP, yaw)
		transform = transform.rotated(transform.basis.x, tilt_z)
		transform = transform.rotated(transform.basis.z, tilt_x)
		transform = transform.scaled(Vector3(1.0, height_scale, 1.0))
		transform.origin = pos
		mm.set_instance_transform(i, transform)

	mm_inst.multimesh = mm
	mm_inst.name = "GrassMeshes"
	area_node.add_child(mm_inst)
	print("Grass: ", grass_count, " blades in ", area_id, " (density ", grass_density_per_m2, "/m²)")

func _find_mesh_instance(node: Node) -> MeshInstance3D:
	"""Recursively find the first MeshInstance3D in a scene tree."""
	if node is MeshInstance3D:
		return node as MeshInstance3D
	
	for child in node.get_children():
		var result = _find_mesh_instance(child)
		if result != null:
			return result
	
	return null

func _print_node_tree(node: Node, depth: int) -> String:
	"""Helper function to print node tree for debugging."""
	var indent = ""
	for i in range(depth):
		indent += "  "
	var result = indent + node.get_class() + " (" + node.name + ")\n"
	for child in node.get_children():
		result += _print_node_tree(child, depth + 1)
	return result

func _generate_noise_texture(size: int = 256) -> ImageTexture:
	"""
	Generates a simple noise texture using FastNoiseLite.
	This is a fallback if no noise texture file is provided.
	"""
	var img = Image.create(size, size, false, Image.FORMAT_RGB8)
	
	# Create noise generator
	var noise = FastNoiseLite.new()
	noise.seed = randi()
	noise.frequency = 0.1
	noise.noise_type = FastNoiseLite.TYPE_PERLIN
	
	# Generate noise values
	for x in range(size):
		for y in range(size):
			var value = noise.get_noise_2d(x, y)
			# Normalize from -1..1 to 0..1
			value = (value + 1.0) / 2.0
			img.set_pixel(x, y, Color(value, value, value))
	
	# Create texture
	var texture = ImageTexture.new()
	texture.set_image(img)
	return texture


func _spawn_world(world_data: Dictionary) -> void:
	# Spawn each area as its own island (separate ground plane)
	for area in world_data.get("areas", []):
		_spawn_area(area)

	# Spawn roads
	for conn in world_data.get("connections", []):
		_draw_connection(conn)

func _get_story_start_area_id() -> String:
	"""Load dialogue.json and return the area id of the first event that has tags.area (story start)."""
	var path := _resolve_path("res://generated/dialogue.json")
	var data = _load_json(path)
	if data == null:
		return ""
	var evs: Array = data.get("events", [])
	for ev in evs:
		if ev is not Dictionary:
			continue
		var tags = ev.get("tags", {})
		if tags is Dictionary and tags.has("area"):
			return str(tags.get("area", ""))
	return ""

func _spawn_player(world_data: Dictionary) -> void:
	print("WorldLoader: _spawn_player called")
	var areas: Array = world_data.get("areas", [])
	if areas.is_empty():
		push_error("WorldLoader: No areas found in world data, cannot spawn player")
		return
	# Spawn at story start: first area mentioned in dialogue (edge/road of where story begins)
	var story_start_area_id := _get_story_start_area_id()
	var spawn_area = areas[0]
	if not story_start_area_id.is_empty():
		for a in areas:
			if a.get("area_id") == story_start_area_id:
				spawn_area = a
				print("WorldLoader: Spawning at story start area: ", story_start_area_id)
				break
	else:
		for a in areas:
			if a.get("area_id") == "mountain_road":
				spawn_area = a
				break
			
	var spawn_pos = Vector3.ZERO
	var found_gate = false
	
	# Try to find a gate for this area to spawn at (avoids spawning inside buildings)
	var gates = world_data.get("gates", [])
	for g in gates:
		if g.get("area_id") == spawn_area.get("area_id"):
			var gx = float(g.get("world_x", 0))
			var gy = float(g.get("world_y", 0))
			var spawn_y = player_height_m * 50.0 # Spawn MUCH higher to avoid tree tops
			spawn_pos = Vector3(gx, spawn_y, gy) * tile_size_m
			found_gate = true
			break
	
	# Fallback to center if no gate found
	if not found_gate:
		var x := float(spawn_area.get("x", 0))
		var y := float(spawn_area.get("y", 0))
		var w := float(spawn_area.get("w", 40))
		var h := float(spawn_area.get("h", 30))
		var spawn_y = player_height_m * 50.0 # Spawn MUCH higher to avoid tree tops
		spawn_pos = Vector3(x + w/2.0, spawn_y, y + h/2.0) * tile_size_m
	
	# Create player programmatically to avoid dependency errors with Player.tscn
	print("WorldLoader: Creating Player programmatically...")
	var player = CharacterBody3D.new()
	player.name = "Player"
	player.position = spawn_pos
	
	# Manually load and attach script
	var script_path = _resolve_path("res://scripts/player.gd")
	if FileAccess.file_exists(script_path):
		var script = load(script_path)
		if script:
			player.set_script(script)
			print("  ✓ Successfully attached player script from: ", script_path)
		else:
			push_error("Failed to load script resource from: " + script_path)
	else:
		push_error("Player script file not found at: " + script_path)
	
	# Add Collision Shape
	var collision = CollisionShape3D.new()
	collision.name = "CollisionShape3D"
	var capsule = CapsuleShape3D.new()
	capsule.radius = 0.3
	capsule.height = 1.8
	collision.shape = capsule
	collision.position.y = 0.9 # Half height
	player.add_child(collision)
	
	# Add Camera
	var camera = Camera3D.new()
	camera.name = "Camera3D"
	camera.position.y = 1.6 # Eye height
	camera.current = true
	player.add_child(camera)
	
	add_child(player)
	_player_node = player
	
	# Force _ready call if needed (though add_child should do it)
	if player.has_method("_ready"):
		# In Godot 4, add_child calls _ready automatically, but only if script is attached before adding
		# We attached script before add_child, so it should be fine.
		pass
		
	print("Player spawned at position: ", spawn_pos)
	print("Player node: ", player, " camera: ", camera)

func _spawn_area(area: Dictionary) -> void:
    # ... (function continues, no changes needed for now)

# ... (End of file, update _load_json)

	var area_id := str(area.get("area_id", "area"))
	var x := float(area.get("x", 0))
	var y := float(area.get("y", 0))
	var w := float(area.get("w", 40))
	var h := float(area.get("h", 30))

	var root := Node3D.new()
	root.name = area_id
	# In 3D, usually X is East/West, Z is North/South. Y is Up.
	# We map 2D (x,y) to 3D (x, 0, z)
	root.position = Vector3(x, 0, y) * tile_size_m
	root.add_to_group("generated_areas")
	add_child(root)

	# Each area is its own island with its own ground plane
	# Calculate the actual bounds from tilemap if available, otherwise use area dimensions
	var actual_min_x: float = 0.0
	var actual_min_y: float = 0.0
	var actual_w: float = w
	var actual_h: float = h
	
	if area_layouts.has(area_id):
		var a: Dictionary = area_layouts[area_id]
		var tilemap_grid: Array = a.get("tilemap_grid", [])
		if not tilemap_grid.is_empty():
			# Use tilemap bounds to ensure we cover everything
			var tilemap_min_x: float = float(a.get("tilemap_min_x", 0.0))
			var tilemap_min_y: float = float(a.get("tilemap_min_y", 0.0))
			var grid_w: int = int(a.get("tilemap_grid_w", w))
			var grid_h: int = int(a.get("tilemap_grid_h", h))
			
			# Calculate actual bounds: find the min/max extent
			# The tilemap extends from tilemap_min_x to tilemap_min_x + grid_w
			var tilemap_max_x: float = tilemap_min_x + float(grid_w)
			var tilemap_max_y: float = tilemap_min_y + float(grid_h)
			
			# The area extends from 0 to w (or h)
			# Find the overall bounds
			actual_min_x = min(0.0, tilemap_min_x)
			actual_min_y = min(0.0, tilemap_min_y)
			var actual_max_x: float = max(float(w), tilemap_max_x)
			var actual_max_y: float = max(float(h), tilemap_max_y)
			
			actual_w = actual_max_x - actual_min_x
			actual_h = actual_max_y - actual_min_y
	
	# Add extra margin around the area to create a complete island
	var island_margin: float = 5.0 * tile_size_m  # 5 tiles of extra grass around each area
	var island_w: float = actual_w * tile_size_m + island_margin * 2.0
	var island_h: float = actual_h * tile_size_m + island_margin * 2.0
	
	# Ground plane - Using MeshInstance3D with PlaneMesh for better shader support
	var ground_mesh := PlaneMesh.new()
	ground_mesh.size = Vector2(island_w, island_h)
	ground_mesh.subdivide_width = max(1, int(island_w / 2.0))  # Subdivide for better shader variation
	ground_mesh.subdivide_depth = max(1, int(island_h / 2.0))
	
	var ground := MeshInstance3D.new()
	ground.name = "IslandGround"
	ground.mesh = ground_mesh
	# PlaneMesh is horizontal (XZ plane) by default, positioned at Y=0
	# Center the island ground accounting for tilemap offsets
	# Position relative to area root (which is at x, y in world coords)
	var ground_center_x: float = (actual_min_x + actual_w / 2.0) * tile_size_m
	var ground_center_z: float = (actual_min_y + actual_h / 2.0) * tile_size_m
	ground.position = Vector3(ground_center_x, 0, ground_center_z)
	
	# Add collision for the entire island
	var collision := StaticBody3D.new()
	var collision_shape := CollisionShape3D.new()
	var box_shape := BoxShape3D.new()
	box_shape.size = Vector3(island_w, ground_thickness_m, island_h)
	collision_shape.shape = box_shape
	collision_shape.position = Vector3(0, -ground_thickness_m/2.0, 0)
	collision.add_child(collision_shape)
	ground.add_child(collision)
	
	# Use stylized grass shader material (reused for performance)
	var grass_mat = _get_grass_material()
	if grass_mat != null:
		ground.material_override = grass_mat
	else:
		# Fallback to standard material if shader fails
		var mat := StandardMaterial3D.new()
		mat.albedo_color = Color(0.14, 0.21, 0.14) # Darker green
		ground.material_override = mat
	
	root.add_child(ground)
	
	# Optionally add 3D grass meshes on top (if enabled and for web performance)
	if use_grass_meshes and grass_density_per_m2 > 0.0:
		_spawn_grass_meshes(root, area_id, w, h)

	# SimpleGrassTextured: scatter grass (with wind) where there is no building or path
	if use_simple_grass_textured and simple_grass_density > 0.0:
		_spawn_simple_grass_textured(root, area_id, w, h)

	# Label (3D Text)
	var label := Label3D.new()
	label.text = area_id
	label.position = Vector3(w/2.0, 2.0, h/2.0) * tile_size_m
	label.billboard = BaseMaterial3D.BILLBOARD_ENABLED
	label.font_size = 64
	root.add_child(label)

	if area_layouts.has(area_id):
		_spawn_tilemap_grid(root, area_id)
		_spawn_entities(root, area_id)
		_spawn_npcs_for_area(root, area_id)
		_spawn_internal_roads(root, area_id)

# Remove preload, we will load at runtime
# const BUILDING_MODEL = preload("res://Meshy_AI_House_T2_0111103525_texture.glb")

func _spawn_tilemap_grid(area_node: Node3D, area_id: String) -> void:
	"""
	Render the full tilemap grid matching the visualization PNGs.
	Uses exact coordinates from tilemap_grid with tilemap_min_x/min_y offsets.
	"""
	var a: Dictionary = area_layouts[area_id]
	var tilemap_grid: Array = a.get("tilemap_grid", [])
	
	if tilemap_grid.is_empty():
		return  # No tilemap data available
	
	var tilemap_min_x: float = float(a.get("tilemap_min_x", 0.0))
	var tilemap_min_y: float = float(a.get("tilemap_min_y", 0.0))
	var grid_w: int = int(a.get("tilemap_grid_w", 0))
	var grid_h: int = int(a.get("tilemap_grid_h", 0))
	
	if grid_w == 0 or grid_h == 0:
		return
	
	# Tile codes (matching visualize_tilemaps.py)
	const T_EMPTY = 0
	const T_ROAD = 1
	const T_BLOCKED = 2
	const T_GATE = 3
	const T_CENTER = 4
	
	# Group tiles by type for efficient MultiMesh rendering
	var tiles_by_type := {}
	tiles_by_type[T_EMPTY] = []
	tiles_by_type[T_ROAD] = []
	tiles_by_type[T_BLOCKED] = []
	tiles_by_type[T_GATE] = []
	tiles_by_type[T_CENTER] = []
	
	# Collect all tiles with their grid coordinates
	for y in range(grid_h):
		if y >= tilemap_grid.size():
			continue
		var row: Array = tilemap_grid[y]
		for x in range(grid_w):
			if x >= row.size():
				continue
			var tile_code: int = int(row[x])
			if tile_code in tiles_by_type:
				tiles_by_type[tile_code].append(Vector2(x, y))
	
	# Render each tile type with appropriate color/material
	# Note: Empty tiles and blocked tiles are skipped - the ground plane already covers them with grass
	# Blocked tiles (building footprints) are not rendered - grass ground plane covers them
	
	# Road tiles (brown/tan - roads are also rendered separately, but this shows exact grid)
	if not tiles_by_type[T_ROAD].is_empty():
		_render_tile_group(area_node, tiles_by_type[T_ROAD], tilemap_min_x, tilemap_min_y,
			Color(0.5, 0.45, 0.4), road_thickness_m)  # Match road material color
	
	# Gate markers (red, small raised markers)
	if not tiles_by_type[T_GATE].is_empty():
		_render_tile_group(area_node, tiles_by_type[T_GATE], tilemap_min_x, tilemap_min_y,
			Color(0.9, 0.2, 0.2), 0.3, 0.3)  # Red, raised, smaller size
	
	# Center marker (gold/yellow)
	if not tiles_by_type[T_CENTER].is_empty():
		_render_tile_group(area_node, tiles_by_type[T_CENTER], tilemap_min_x, tilemap_min_y,
			Color(0.9, 0.8, 0.2), 0.3, 0.3)  # Gold, raised, smaller size

func _render_tile_group(area_node: Node3D, tiles: Array, min_x: float, min_y: float, 
		color: Color, thickness: float, tile_scale: float = 1.0) -> void:
	"""
	Render a group of tiles using MultiMeshInstance3D for performance.
	tiles: Array of Vector2(grid_x, grid_y)
	min_x, min_y: World-to-grid transform offsets
	color: Material color
	thickness: Height of the tile
	tile_scale: Scale factor (1.0 = full tile, 0.3 = smaller marker)
	"""
	if tiles.is_empty():
		return
	
	var tile_count = tiles.size()
	var mm_inst := MultiMeshInstance3D.new()
	var mm := MultiMesh.new()
	
	mm.transform_format = MultiMesh.TRANSFORM_3D
	mm.instance_count = tile_count
	
	# Create the mesh
	var box_mesh := BoxMesh.new()
	var scaled_size = tile_size_m * tile_scale
	box_mesh.size = Vector3(scaled_size, thickness, scaled_size)
	var mat := StandardMaterial3D.new()
	mat.albedo_color = color
	mat.metallic = 0.0  # Non-metallic
	mat.roughness = 0.9  # Rough surface
	box_mesh.material = mat
	mm.mesh = box_mesh
	
	mm_inst.multimesh = mm
	area_node.add_child(mm_inst)
	
	# Setup Collision (One StaticBody with many Shapes for all tiles)
	var static_body := StaticBody3D.new()
	static_body.name = "TilemapCollision"
	area_node.add_child(static_body)
	
	# Shared Shape
	var shape := BoxShape3D.new()
	shape.size = box_mesh.size
	
	# Set transforms for each tile and add collision
	for i in range(tile_count):
		var tile_pos: Vector2 = tiles[i]
		var grid_x: float = tile_pos.x
		var grid_y: float = tile_pos.y
		
		# Convert grid coordinates to world coordinates using min_x, min_y
		# This matches the exact coordinate system from visualize_tilemaps.py
		var world_x: float = grid_x + min_x
		var world_z: float = grid_y + min_y
		
		# Position at tile center, with height offset
		var pos := Vector3((world_x + 0.5) * tile_size_m, thickness/2.0, (world_z + 0.5) * tile_size_m)
		var xform := Transform3D(Basis(), pos)
		mm.set_instance_transform(i, xform)
		
		# Add Collision Shape for this tile
		var coll := CollisionShape3D.new()
		coll.shape = shape # Share the resource!
		coll.position = pos
		static_body.add_child(coll)

func _spawn_internal_roads(area_node: Node3D, area_id: String) -> void:
	var a: Dictionary = area_layouts[area_id]

	# v3: roads expressed as WORLD-SPACE TILE CENTERS already converted to LOCAL coords.
	var tiles_world: Array = a.get("road_tiles_world", [])
	if not tiles_world.is_empty():
		var tile_count = tiles_world.size()
		var mm_inst := MultiMeshInstance3D.new()
		var mm := MultiMesh.new()

		mm.transform_format = MultiMesh.TRANSFORM_3D
		mm.instance_count = tile_count

		var box_mesh := BoxMesh.new()
		var s: float = tile_size_m * maxf(1.0, float(v3_road_tile_overlap))
		box_mesh.size = Vector3(s, road_thickness_m, s)
		box_mesh.material = _get_road_material(area_id)
		mm.mesh = box_mesh

		mm_inst.multimesh = mm
		area_node.add_child(mm_inst)

		for i in range(tile_count):
			var t = tiles_world[i]
			if not (t is Array) or t.size() < 2:
				continue
			var cx := float(t[0])
			var cz := float(t[1])

			# IMPORTANT: These are already tile centers (no +0.5 offset)
			var pos := Vector3(cx * tile_size_m, road_thickness_m/2.0, cz * tile_size_m)
			var xform := Transform3D(Basis(), pos)
			mm.set_instance_transform(i, xform)

		return

	# Try "road_tiles" first (new tile-based approach)
	var tiles: Array = a.get("road_tiles", [])
	
	if not tiles.is_empty():
		# Get tilemap coordinate offsets for exact positioning
		var tilemap_min_x: float = float(a.get("tilemap_min_x", 0.0))
		var tilemap_min_y: float = float(a.get("tilemap_min_y", 0.0))
		
		# OPTIMIZATION: Use MultiMeshInstance3D for rendering thousands of identical tiles effectively
		var tile_count = tiles.size()
		var mm_inst := MultiMeshInstance3D.new()
		var mm := MultiMesh.new()
		
		# Setup MultiMesh
		mm.transform_format = MultiMesh.TRANSFORM_3D
		# mm.color_format = MultiMesh.COLOR_NONE # Default is NONE (0)
		mm.instance_count = tile_count
		
		# Create the Mesh (shared for all instances)
		var box_mesh := BoxMesh.new()
		box_mesh.size = Vector3(tile_size_m, road_thickness_m, tile_size_m)
		# Use area-specific road material
		box_mesh.material = _get_road_material(area_id)
		mm.mesh = box_mesh
		
		mm_inst.multimesh = mm
		area_node.add_child(mm_inst)
		
		for i in range(tile_count):
			var t = tiles[i]
			var grid_x := float(t[0])
			var grid_y := float(t[1])
			
			# Convert grid coordinates to world coordinates using min_x, min_y
			# This matches the exact coordinate system from visualize_tilemaps.py
			var world_x: float = grid_x + tilemap_min_x
			var world_z: float = grid_y + tilemap_min_y
			
			# Calculate Transform
			# Position: world_x + 0.5 (center), y=road_thickness/2, world_z + 0.5
			var pos := Vector3((world_x + 0.5) * tile_size_m, road_thickness_m/2.0, (world_z + 0.5) * tile_size_m)
			var xform := Transform3D(Basis(), pos)
			
			# Set Visual Transform
			mm.set_instance_transform(i, xform)
		
		return

	# Fallback to old polyline method if tiles missing (backward compatibility)
	var roads: Array = a.get("internal_roads", [])
	for r in roads:
		var pts: Array = r.get("polyline", [])
		if pts.size() < 2:
			continue
			
		var path_node := Path3D.new()
		var curve := Curve3D.new()
		
		for p in pts:
			curve.add_point(Vector3(float(p[0]), road_thickness_m/2.0, float(p[1])) * tile_size_m)
			
		path_node.curve = curve
		area_node.add_child(path_node)
		
		var road := CSGPolygon3D.new()
		road.mode = CSGPolygon3D.MODE_PATH
		road.path_node = path_node.get_path()
		road.path_interval_type = CSGPolygon3D.PATH_INTERVAL_DISTANCE
		road.path_interval = 1.0
		road.path_simplify_angle = 0.0
		road.path_rotation = CSGPolygon3D.PATH_ROTATION_PATH
		road.path_local = false
		road.path_continuous_u = true
		road.path_u_distance = 1.0
		road.path_joined = false
		road.use_collision = true
		
		var width := 2.0 * tile_size_m
		var polygon := PackedVector2Array([
			Vector2(-width/2.0, 0),
			Vector2(width/2.0, 0),
			Vector2(width/2.0, road_thickness_m),
			Vector2(-width/2.0, road_thickness_m)
		])
		road.polygon = polygon
		
		# Use area-specific road material
		road.material = _get_road_material(area_id)
		
		path_node.add_child(road)

func _spawn_entities(area_node: Node3D, area_id: String) -> void:
	var a: Dictionary = area_layouts[area_id]
	var entities: Array = a.get("entities", [])
	
	# Get tilemap coordinate offsets for exact positioning
	var tilemap_min_x: float = float(a.get("tilemap_min_x", 0.0))
	var tilemap_min_y: float = float(a.get("tilemap_min_y", 0.0))
	
	var entities_by_model := {}  # Path -> Array of entity data
	var entities_no_glb: Array = []  # Entities with no GLB; spawn as cuboid
	
	for e in entities:
		var eid := str(e.get("id", e.get("name", "entity")))
		var path = entity_models.get(eid, "")
		if path != "":
			if not entities_by_model.has(path):
				entities_by_model[path] = []
			entities_by_model[path].append({
				"entity": e,
				"entity_id": eid,
				"path": path
			})
		else:
			entities_no_glb.append({"entity": e, "entity_id": eid, "path": ""})
	
	# Spawn entities with GLB when path exists (multimesh); placeholder box only when no path
	for model_path in entities_by_model.keys():
		var entity_list = entities_by_model[model_path]
		_spawn_entities_multimesh(area_node, area_id, model_path, entity_list, tilemap_min_x, tilemap_min_y)
	for item in entities_no_glb:
		_spawn_single_entity(area_node, area_id, item["entity"], item["entity_id"], item["path"], tilemap_min_x, tilemap_min_y)

const NPC_ENTITY_MARGIN := 0.25  # NPC center must be this far outside any entity rect

func _point_inside_entity(px: float, pz: float, ent: Dictionary) -> bool:
	var ex := float(ent.get("x", 0))
	var ey := float(ent.get("y", 0))
	var ew := float(ent.get("w", 1))
	var eh := float(ent.get("h", 1))
	return px >= ex and px < ex + ew and pz >= ey and pz < ey + eh

func _point_overlaps_entity(px: float, pz: float, ent: Dictionary) -> bool:
	var ex := float(ent.get("x", 0))
	var ey := float(ent.get("y", 0))
	var ew := float(ent.get("w", 1))
	var eh := float(ent.get("h", 1))
	return px >= ex - NPC_ENTITY_MARGIN and px < ex + ew + NPC_ENTITY_MARGIN and pz >= ey - NPC_ENTITY_MARGIN and pz < ey + eh + NPC_ENTITY_MARGIN

func _point_inside_any_entity(px: float, pz: float, entities: Array) -> bool:
	for ent in entities:
		if _point_overlaps_entity(px, pz, ent):
			return true
	return false

func _spawn_npcs_for_area(area_node: Node3D, area_id: String) -> void:
	"""Place NPC cuboids on the road or next to entity, outside entity footprints (npc_plan from world_plan)."""
	if npc_plan.is_empty():
		return
	var a: Dictionary = area_layouts.get(area_id, {})
	var entities: Array = a.get("entities", [])
	var road_tiles: Array = a.get("road_tiles_world", [])
	var tilemap_min_x: float = float(a.get("tilemap_min_x", 0.0))
	var tilemap_min_y: float = float(a.get("tilemap_min_y", 0.0))
	for npc in npc_plan:
		if str(npc.get("area_id", "")) != area_id:
			continue
		var anchor_ent = npc.get("anchor_entity", {})
		if anchor_ent is not Dictionary:
			continue
		var anchor_id: String = str(anchor_ent.get("id", ""))
		if anchor_id.is_empty():
			continue
		var e: Dictionary = {}
		for ent in entities:
			if str(ent.get("id", "")) == anchor_id:
				e = ent
				break
		if e.is_empty():
			continue
		var ex := float(e.get("x", 0))
		var ey := float(e.get("y", 0))
		var ew := float(e.get("w", 2))
		var eh := float(e.get("h", 2))
		var rotation_deg: float = float(e.get("rotation_deg", 180))
		var cx := ex + ew / 2.0
		var cz := ey + eh / 2.0
		var rad := deg_to_rad(rotation_deg)
		var dir_x := sin(rad)
		var dir_z := -cos(rad)
		var npc_grid_x: float
		var npc_grid_z: float
		var needs_frontage: bool = bool(e.get("needs_frontage", false))
		if needs_frontage and road_tiles.size() > 0:
			var ideal_gx := cx + 2.0 * dir_x
			var ideal_gz := cz + 2.0 * dir_z
			var best_dist := 1e10
			var best_tile: Array = []
			for t in road_tiles:
				if not (t is Array) or t.size() < 2:
					continue
				var tgx := float(t[0])
				var tgz := float(t[1])
				if _point_inside_any_entity(tgx, tgz, entities):
					continue
				var d := (tgx - ideal_gx) * (tgx - ideal_gx) + (tgz - ideal_gz) * (tgz - ideal_gz)
				if d < best_dist:
					best_dist = d
					best_tile = t
			if best_tile.size() >= 2:
				npc_grid_x = float(best_tile[0])
				npc_grid_z = float(best_tile[1])
			else:
				npc_grid_x = ideal_gx
				npc_grid_z = ideal_gz
			if _point_inside_any_entity(npc_grid_x, npc_grid_z, entities):
				var offset_min := maxf(ew, eh) / 2.0 + NPC_ENTITY_MARGIN
				var offset := offset_min
				for try in range(25):
					npc_grid_x = cx + offset * dir_x
					npc_grid_z = cz + offset * dir_z
					if not _point_inside_any_entity(npc_grid_x, npc_grid_z, entities):
						break
					offset += 0.5
				if _point_inside_any_entity(npc_grid_x, npc_grid_z, entities):
					continue
		else:
			if road_tiles.size() > 0:
				var best_road_dist := 1e10
				var rx := cx
				var rz := cz
				for t in road_tiles:
					if not (t is Array) or t.size() < 2:
						continue
					var tgx := float(t[0])
					var tgz := float(t[1])
					if _point_inside_any_entity(tgx, tgz, entities):
						continue
					var d := (tgx - cx) * (tgx - cx) + (tgz - cz) * (tgz - cz)
					if d < best_road_dist:
						best_road_dist = d
						rx = tgx
						rz = tgz
				if best_road_dist < 1e9:
					var dx := rx - cx
					var dz := rz - cz
					var len_sq := dx * dx + dz * dz
					if len_sq > 0.0001:
						var inv := 1.0 / sqrt(len_sq)
						dir_x = dx * inv
						dir_z = dz * inv
			var offset_min := maxf(ew, eh) / 2.0 + NPC_ENTITY_MARGIN
			var offset := offset_min
			var max_tries := 25
			for try in range(max_tries):
				npc_grid_x = cx + offset * dir_x
				npc_grid_z = cz + offset * dir_z
				if not _point_inside_any_entity(npc_grid_x, npc_grid_z, entities):
					break
				offset += 0.5
			if _point_inside_any_entity(npc_grid_x, npc_grid_z, entities):
				continue
		var world_x := (npc_grid_x + tilemap_min_x) * tile_size_m
		var world_z := (npc_grid_z + tilemap_min_y) * tile_size_m
		var npc_id := str(npc.get("npc_id", "npc"))
		var display_name := str(npc.get("display_name", "NPC"))
		var model_path: String = npc_models.get(npc_id, "")
		if model_path != "":
			# Spawn generated 3D NPC model (root at character center height for consistent E-to-talk distance)
			var root := Node3D.new()
			root.name = "npc_%s" % npc_id
			root.position = Vector3(world_x, 0.9, world_z)
			# Orient NPC so VLM-detected front faces the road (away from anchor)
			var asset_name := _get_asset_name_from_path(model_path)
			var asset_front_yaw: float = 0.0
			if asset_metadata.has(asset_name):
				var meta = asset_metadata[asset_name]
				if meta.has("front_yaw_deg") and meta["front_yaw_deg"] != null:
					asset_front_yaw = float(meta["front_yaw_deg"])
			var dx := cx - npc_grid_x
			var dz := cz - npc_grid_z
			# Face away from anchor so NPC faces the road / approach direction
			var toward_anchor_yaw: float = rad_to_deg(atan2(dx, -dz)) if (dx * dx + dz * dz) > 0.0001 else 0.0
			var desired_facing_yaw_deg: float = toward_anchor_yaw + 180.0
			var rotation_yaw_deg: float = desired_facing_yaw_deg - asset_front_yaw
			root.rotation.y = deg_to_rad(rotation_yaw_deg)
			var model_node = _load_model_instance(model_path)
			if model_node != null:
				model_node.position = Vector3(0, -0.9, 0)
				model_node.scale = Vector3(npc_model_scale, npc_model_scale, npc_model_scale)
				root.add_child(model_node)
			var npc_label := Label3D.new()
			npc_label.text = display_name
			npc_label.position = Vector3(0, 1.0, 0)  # Hover above head (~1.7m character)
			npc_label.billboard = BaseMaterial3D.BILLBOARD_ENABLED
			npc_label.font_size = 28
			npc_label.outline_render_priority = 0
			npc_label.modulate = Color(1, 1, 1)
			root.add_child(npc_label)
			var body := StaticBody3D.new()
			var shape := CollisionShape3D.new()
			var capsule := CapsuleShape3D.new()
			capsule.radius = 0.4
			capsule.height = 1.8
			shape.shape = capsule
			body.add_child(shape)
			root.add_child(body)
			root.add_to_group("npc")
			root.set_meta("display_name", display_name)
			root.set_meta("npc_id", npc_id)
			root.set_meta("description", str(npc.get("description", "")))
			area_node.add_child(root)
		else:
			# Placeholder box when 3D generation was skipped or no model
			var box := CSGBox3D.new()
			box.name = "npc_%s" % npc_id
			box.size = Vector3(0.3, 1.8, 0.3)
			box.position = Vector3(world_x, 0.9, world_z)
			box.use_collision = true
			var mat := StandardMaterial3D.new()
			mat.albedo_color = Color(0.25, 0.45, 0.7)
			box.material = mat
			area_node.add_child(box)
			var npc_label := Label3D.new()
			npc_label.text = display_name
			npc_label.position = Vector3(0, 1.1, 0)
			npc_label.billboard = BaseMaterial3D.BILLBOARD_ENABLED
			npc_label.font_size = 28
			npc_label.outline_render_priority = 0
			npc_label.modulate = Color(1, 1, 1)
			box.add_child(npc_label)
			box.add_to_group("npc")
			box.set_meta("display_name", display_name)
			box.set_meta("npc_id", npc_id)
			box.set_meta("description", str(npc.get("description", "")))

func _spawn_single_entity(area_node: Node3D, area_id: String, e: Dictionary, eid: String, path: String, tilemap_min_x: float, tilemap_min_y: float) -> void:
	var grid_x := float(e.get("x", 0))
	var grid_y := float(e.get("y", 0))
	var ew := float(e.get("w", 2))
	var eh := float(e.get("h", 2))
	
	# Convert grid coordinates to world coordinates
	var world_x: float = grid_x + tilemap_min_x
	var world_z: float = grid_y + tilemap_min_y
	
	# Get the entity's facing direction
	var entity_facing_yaw: float
	if e.has("rotation_deg"):
		entity_facing_yaw = float(e.get("rotation_deg", 180))
	else:
		entity_facing_yaw = float(e.get("facing_yaw_deg", 180))
	
	var height := clampf(sqrt(ew * eh) * 0.8, 2.0, 8.0)
	
	# Add a text label above the building
	var label := Label3D.new()
	label.text = eid
	label.position = Vector3((world_x + ew/2.0) * tile_size_m, (height + 1.0) * tile_size_m, (world_z + eh/2.0) * tile_size_m)
	label.billboard = BaseMaterial3D.BILLBOARD_ENABLED
	label.font_size = 48
	label.outline_render_priority = 0
	label.modulate = Color(1, 0, 0) if e.get("needs_frontage", false) else Color(1, 1, 1)
	area_node.add_child(label)
	
	# White cube placeholder for building/entity
	var box := CSGBox3D.new()
	box.name = eid
	box.size = Vector3(ew * 0.85, height, eh * 0.85) * tile_size_m
	box.position = Vector3((world_x + ew/2.0) * tile_size_m, (height/2.0) * tile_size_m, (world_z + eh/2.0) * tile_size_m)
	box.rotation.y = deg_to_rad(entity_facing_yaw)
	box.use_collision = true
	var mat := StandardMaterial3D.new()
	mat.albedo_color = Color(1, 1, 1)
	box.material = mat
	area_node.add_child(box)

func _spawn_entities_multimesh(area_node: Node3D, area_id: String, model_path: String, entity_list: Array, tilemap_min_x: float, tilemap_min_y: float) -> void:
	"""
	Optimized spawning using MultiMeshInstance3D for repeated models.
	This dramatically reduces memory usage on Web by sharing textures across instances.
	"""
	# Only print for large groups to reduce console spam
	if entity_list.size() >= 10:
		print("Using MultiMeshInstance3D for ", entity_list.size(), " instances of: ", model_path)
	
	# Load the model once to extract mesh (try resolved path if res:// fails)
	var instance = _load_model_instance(model_path)
	if instance == null:
		var resolved = _resolve_path(model_path)
		if resolved != model_path:
			instance = _load_model_instance(resolved)
	if instance == null:
		push_error("Failed to load model for MultiMesh: " + model_path)
		# No box fallback: skip these entities (GLB only, no white cubes)
		return
	
	# Find mesh in the loaded instance
	var mesh_instance = _find_mesh_instance(instance)
	if mesh_instance == null:
		push_error("Could not find MeshInstance3D in model: " + model_path)
		instance.queue_free()
		# No box fallback: skip these entities (GLB only, no white cubes)
		return
	
	var source_mesh = mesh_instance.mesh
	if source_mesh == null:
		push_error("Mesh is null in model: " + model_path)
		instance.queue_free()
		# No box fallback: skip these entities (GLB only, no white cubes)
		return
	
	# CRITICAL: Get the mesh instance's transform relative to the instance root
	# The GLB file may have the mesh rotated/scaled, and we need to preserve that
	# Get the transform from instance root to mesh_instance
	var instance_to_mesh_transform: Transform3D
	if mesh_instance.get_parent() == instance:
		# Mesh is direct child - use its transform directly
		instance_to_mesh_transform = mesh_instance.transform
	else:
		# Mesh is nested - calculate relative transform
		instance_to_mesh_transform = instance.transform.affine_inverse() * mesh_instance.global_transform
	
	# Extract rotation and scale from the original transform
	var original_basis = instance_to_mesh_transform.basis
	var original_rotation_euler = original_basis.get_euler()
	var original_scale = original_basis.get_scale()
	
	# Get world-space AABB for scale calculations (accounts for original transforms)
	var instance_world_aabb = _merged_world_aabb(instance)
	if instance_world_aabb.size.x <= 0.0001 or instance_world_aabb.size.z <= 0.0001 or instance_world_aabb.size.y <= 0.0001:
		# Fallback: use mesh local AABB and apply original scale
		var local_aabb = source_mesh.get_aabb()
		instance_world_aabb = AABB(local_aabb.position * original_scale, local_aabb.size * original_scale)
	
	# Get material from the mesh (shared across all instances)
	var material = source_mesh.surface_get_material(0)
	if material == null:
		material = mesh_instance.material_override
	if material == null:
		material = mesh_instance.get_surface_override_material(0)
	
	# CRITICAL: On Web, share the material directly (don't duplicate)
	# MultiMesh will share the material and textures across all instances automatically
	# Duplicating would create unnecessary copies
	
	instance.queue_free()  # We only needed it to extract the mesh
	
	# Create MultiMeshInstance3D
	var mm_inst := MultiMeshInstance3D.new()
	var mm := MultiMesh.new()
	mm.transform_format = MultiMesh.TRANSFORM_3D
	mm.instance_count = entity_list.size()
	mm.mesh = source_mesh
	# Use the material directly - MultiMesh shares it across all instances
	# This prevents texture duplication
	if material != null:
		mm.mesh.surface_set_material(0, material)  # Share material (and textures) across all instances
	
	mm_inst.multimesh = mm
	mm_inst.name = "MultiMesh_" + model_path.get_file().get_basename()
	area_node.add_child(mm_inst)
	
	# Create collision for MultiMesh instances
	# MultiMeshInstance3D doesn't support per-instance collision automatically,
	# so we create a StaticBody3D with individual CollisionShape3D nodes for each instance
	var static_body := StaticBody3D.new()
	static_body.name = "MultiMeshCollision_" + model_path.get_file().get_basename()
	area_node.add_child(static_body)
	
	# Base collision size from world-space AABB (accounts for original scale/rotation)
	# Individual instances will scale this based on their footprint
	var base_collision_size = instance_world_aabb.size
	
	# Mesh-derived collision shape (trimesh = exact visual mesh) so collision matches the visible geometry
	var mesh_collision_shape: Shape3D = null
	if use_mesh_collision_for_entities:
		mesh_collision_shape = source_mesh.create_trimesh_shape()
	
	# Set transforms for each instance
	for i in range(entity_list.size()):
		var item = entity_list[i]
		var e = item["entity"]
		var eid = item["entity_id"]
		
		var grid_x := float(e.get("x", 0))
		var grid_y := float(e.get("y", 0))
		var ew := float(e.get("w", 2))
		var eh := float(e.get("h", 2))
		
		# Normalize sizes per group if enabled (for consistent footprints across areas)
		if normalize_group_sizes:
			var group = e.get("group") or e.get("id") or ""
			if group_size_overrides.has(group):
				var override = group_size_overrides[group]
				ew = float(override.get("w", ew))
				eh = float(override.get("h", eh))
			elif normalized_group_sizes_cache.has(group):
				var normalized_size = normalized_group_sizes_cache[group]
				ew = normalized_size["w"]
				eh = normalized_size["h"]
		
		var world_x: float = grid_x + tilemap_min_x
		var world_z: float = grid_y + tilemap_min_y
		
		# Get rotation
		var entity_facing_yaw: float
		if e.has("rotation_deg"):
			entity_facing_yaw = float(e.get("rotation_deg", 180))
		else:
			entity_facing_yaw = float(e.get("facing_yaw_deg", 180))
		
		var asset_name = _get_asset_name_from_path(model_path)
		var asset_front_yaw: float = 0.0
		if asset_metadata.has(asset_name):
			var meta = asset_metadata[asset_name]
			if meta.has("front_yaw_deg") and meta["front_yaw_deg"] != null:
				asset_front_yaw = float(meta["front_yaw_deg"])
		
		var rotation_yaw_deg: float = entity_facing_yaw - asset_front_yaw
		if debug_orientation:
			print("Orientation ", eid, " | layout dir->yaw: ", entity_facing_yaw, " | asset front_yaw: ", asset_front_yaw, " | rotation_yaw: ", rotation_yaw_deg)
		
		# Calculate scale (simplified - assumes uniform scaling)
		var height := clampf(sqrt(ew * eh) * 0.8, 2.0, 8.0)
		var desired_w_m = ew * tile_size_m * 0.85
		var desired_d_m = eh * tile_size_m * 0.85
		var desired_h_m = height * tile_size_m
		
		# Get approximate scale from world-space AABB (accounts for original transforms)
		# For MultiMesh, we use uniform scaling based on average dimension
		var avg_dim = (desired_w_m + desired_d_m) / 2.0
		# Use world-space AABB size which already accounts for original scale/rotation
		var mesh_size = max(instance_world_aabb.size.x, instance_world_aabb.size.z)
		var scale_factor = avg_dim / mesh_size if mesh_size > 0 else 1.0
		var height_scale = desired_h_m / instance_world_aabb.size.y if instance_world_aabb.size.y > 0 else scale_factor
		# Calculate final scale: our desired scale divided by original scale
		# This gives us the additional scale we need to apply on top of the original
		var final_scale = Vector3(
			scale_factor / original_scale.x if original_scale.x > 0.001 else scale_factor,
			height_scale / original_scale.y if original_scale.y > 0.001 else height_scale,
			scale_factor / original_scale.z if original_scale.z > 0.001 else scale_factor
		)
		
		# Create transform: start with the original mesh instance transform, then apply our modifications
		# This preserves the model's original orientation from the GLB file
		var transform = instance_to_mesh_transform
		# 1. Apply our rotation (entity facing direction) on top of original rotation
		transform = transform.rotated(Vector3.UP, deg_to_rad(rotation_yaw_deg))
		# 2. Apply our scale (additional scale on top of original)
		transform = transform.scaled(final_scale)
		# 3. Set position (override original position, but preserve Y offset from original if needed)
		# The Y position will be adjusted by _fit_instance_to_footprint logic, but for MultiMesh
		# we need to account for the bottom of the AABB. For now, use 0 and let the scale handle it.
		var base_y = 0.0
		# If original transform had a Y offset, we might want to preserve it, but typically models
		# should be positioned at ground level, so we use 0.0
		transform.origin = Vector3((world_x + ew/2.0) * tile_size_m, base_y, (world_z + eh/2.0) * tile_size_m)
		
		mm.set_instance_transform(i, transform)
		
		# Add collision shape for this instance (mesh convex when available, else box)
		var coll := CollisionShape3D.new()
		if mesh_collision_shape != null:
			coll.shape = mesh_collision_shape
		else:
			var scaled_collision_size = base_collision_size * final_scale
			var instance_shape := BoxShape3D.new()
			instance_shape.size = scaled_collision_size
			coll.shape = instance_shape
		coll.transform = transform
		static_body.add_child(coll)
		
		# Add label for each entity (still needed for identification)
		var label := Label3D.new()
		label.text = eid
		label.position = Vector3((world_x + ew/2.0) * tile_size_m, (height + 1.0) * tile_size_m, (world_z + eh/2.0) * tile_size_m)
		label.billboard = BaseMaterial3D.BILLBOARD_ENABLED
		label.font_size = 48
		label.outline_render_priority = 0
		label.modulate = Color(1, 0, 0) if e.get("needs_frontage", false) else Color(1, 1, 1)
		area_node.add_child(label)
	
	print("  ✓ Created MultiMeshInstance3D with ", entity_list.size(), " instances")


func _draw_connection(conn: Dictionary) -> void:
	var pts: Array = conn.get("polyline", [])
	# v3 compatibility: {a:[x,y], b:[x,y]}
	if pts.is_empty() and conn.has("a") and conn.has("b"):
		pts = [conn.get("a"), conn.get("b")]
	if pts.size() < 2:
		return

	# Create a Path3D or just a CSGPolygon3D following the path
	# CSGPolygon3D in Path mode is easiest for "thick" roads
	
	var path_node := Path3D.new()
	var curve := Curve3D.new()
	
	for p in pts:
		# Lift slightly above ground to avoid z-fighting
		curve.add_point(Vector3(float(p[0]), road_thickness_m, float(p[1])) * tile_size_m)
		
	path_node.curve = curve
	add_child(path_node)
	
	var road := CSGPolygon3D.new()
	road.mode = CSGPolygon3D.MODE_PATH
	road.path_node = path_node.get_path()
	road.path_interval_type = CSGPolygon3D.PATH_INTERVAL_DISTANCE
	road.path_interval = 1.0
	road.path_simplify_angle = 0.0
	road.path_rotation = CSGPolygon3D.PATH_ROTATION_PATH
	road.path_local = false
	road.path_continuous_u = true
	road.path_u_distance = 1.0
	road.path_joined = false
	# Roads sit visually above the ground plane; we rely on the island ground collision
	# for walking, so keep road collision disabled to avoid catching on edges.
	road.use_collision = false
	
	# Cross section of the road (flat rectangle)
	var width := tile_size_m * maxf(1.0, float(v3_road_tile_overlap))
	var polygon := PackedVector2Array([
		Vector2(-width/2.0, 0),
		Vector2(width/2.0, 0),
		Vector2(width/2.0, road_thickness_m),
		Vector2(-width/2.0, road_thickness_m)
	])
	road.polygon = polygon
	
	# Use road material - try to get area_id from connection, otherwise use default
	var connection_area_id: String = ""
	if conn.has("from_area_id"):
		connection_area_id = conn.get("from_area_id", "")
	elif conn.has("to_area_id"):
		connection_area_id = conn.get("to_area_id", "")
	road.material = _get_road_material(connection_area_id)
	
	path_node.add_child(road)

func _add_collision_recursive(node: Node) -> void:
	if node is MeshInstance3D:
		var mi: MeshInstance3D = node as MeshInstance3D
		# Create trimesh collision (static body + collision shape)
		mi.create_trimesh_collision()
		
		# Optional: If you want to customize layer/mask, you can access the created StaticBody3D
		# which is usually added as a child named "StaticBody3D" or similar.
	
	for child in node.get_children():
		_add_collision_recursive(child)

func _share_textures_recursive(node: Node) -> void:
	"""
	Ensure textures are shared across instances to save memory on Web.
	This prevents each instance from having its own texture copy.
	"""
	if node is MeshInstance3D:
		var mi: MeshInstance3D = node as MeshInstance3D
		var mesh = mi.mesh
		if mesh != null:
			# Share materials across instances
			for surface_idx in range(mesh.get_surface_count()):
				var material = mesh.surface_get_material(surface_idx)
				if material != null and material is BaseMaterial3D:
					var base_mat = material as BaseMaterial3D
					# Force texture sharing by using the same resource
					# This is already handled by Godot's resource system, but we ensure it
					pass
	
	for child in node.get_children():
		_share_textures_recursive(child)

func _merged_world_aabb(root: Node) -> AABB:
	# Get AABB in the root node's local coordinate space
	# This recursively collects AABBs from all VisualInstance3D nodes
	var aabbs: Array[AABB] = []
	
	# If this node is a VisualInstance3D, get its local AABB
	if root is VisualInstance3D:
		var vi: VisualInstance3D = root as VisualInstance3D
		var local_aabb: AABB = vi.get_aabb()
		if local_aabb.size != Vector3.ZERO:
			aabbs.append(local_aabb)
	
	# Recursively get children's AABBs and transform them to this node's local space
	for n in root.get_children():
		if n is Node3D:
			var child_node: Node3D = n as Node3D
			var child_aabb := _merged_world_aabb(child_node)
			if child_aabb.size != Vector3.ZERO:
				# Transform child AABB corners to this node's local space
				var child_transform: Transform3D = child_node.transform
				var corners := [
					child_aabb.position,
					child_aabb.position + Vector3(child_aabb.size.x, 0, 0),
					child_aabb.position + Vector3(0, child_aabb.size.y, 0),
					child_aabb.position + Vector3(0, 0, child_aabb.size.z),
					child_aabb.position + Vector3(child_aabb.size.x, child_aabb.size.y, 0),
					child_aabb.position + Vector3(child_aabb.size.x, 0, child_aabb.size.z),
					child_aabb.position + Vector3(0, child_aabb.size.y, child_aabb.size.z),
					child_aabb.position + child_aabb.size
				]
				var min_point: Vector3 = child_transform * corners[0]
				var max_point: Vector3 = min_point
				for corner in corners:
					var transformed_corner: Vector3 = child_transform * corner
					min_point = min_point.min(transformed_corner)
					max_point = max_point.max(transformed_corner)
				var transformed_aabb := AABB(min_point, max_point - min_point)
				aabbs.append(transformed_aabb)
	
	# Merge all AABBs
	if aabbs.is_empty():
		return AABB()
	
	var merged := aabbs[0]
	for i in range(1, aabbs.size()):
		merged = merged.merge(aabbs[i])
	
	return merged

func _global_aabb_of_node(root: Node3D) -> AABB:
	# Returns AABB of root (and all VisualInstance3D descendants) in global space.
	var local_aabb := _merged_world_aabb(root)
	if local_aabb.size.x <= 0.0001 and local_aabb.size.y <= 0.0001 and local_aabb.size.z <= 0.0001:
		return AABB()
	var xform: Transform3D = root.global_transform
	var corners := [
		local_aabb.position,
		local_aabb.position + Vector3(local_aabb.size.x, 0, 0),
		local_aabb.position + Vector3(0, local_aabb.size.y, 0),
		local_aabb.position + Vector3(0, 0, local_aabb.size.z),
		local_aabb.position + Vector3(local_aabb.size.x, local_aabb.size.y, 0),
		local_aabb.position + Vector3(local_aabb.size.x, 0, local_aabb.size.z),
		local_aabb.position + Vector3(0, local_aabb.size.y, local_aabb.size.z),
		local_aabb.position + local_aabb.size
	]
	var min_p: Vector3 = xform * corners[0]
	var max_p: Vector3 = min_p
	for c in corners:
		var g: Vector3 = xform * c
		min_p = min_p.min(g)
		max_p = max_p.max(g)
	return AABB(min_p, max_p - min_p)

func _fit_instance_to_footprint(instance: Node3D, desired_w_m: float, desired_d_m: float, desired_h_m: float, ground_y_global: float) -> void:
	# Get size at current scale (AABB is in instance's local space, before scaling)
	var aabb := _merged_world_aabb(instance)
	if aabb.size.x <= 0.0001 or aabb.size.z <= 0.0001 or aabb.size.y <= 0.0001:
		return

	# Non-Uniform scaling to fit within desired footprint AND height
	var sx = desired_w_m / aabb.size.x
	var sz = desired_d_m / aabb.size.z
	var sy = desired_h_m / aabb.size.y
	
	instance.scale = Vector3(sx, sy, sz)

	# Position so the bottom of the model sits on the ground. Use global AABB because
	# the instance may be rotated (local Y != world Y); local-space bottom would be wrong.
	var global_aabb := _global_aabb_of_node(instance)
	if global_aabb.size.y <= 0.0001:
		return
	var global_bottom_y := global_aabb.position.y
	# Move instance so the global bottom of the mesh equals ground; small offset to avoid z-fighting
	instance.position.y += (ground_y_global - global_bottom_y) + 0.01

func _compute_normalized_group_sizes() -> Dictionary:
	"""
	Compute normalized (w, h) for each group by finding the most common size across all placements.
	Returns dict: group_name -> {"w": float, "h": float}
	"""
	var group_size_counts: Dictionary = {}  # group -> {"w_h" -> count}
	
	# Collect all placements from all areas
	for area_id in area_layouts:
		var area_layout = area_layouts[area_id]
		var entities = area_layout.get("entities", [])
		for ent in entities:
			var group := ""
			if ent.has("group") and typeof(ent["group"]) == TYPE_STRING:
				group = ent["group"]
			elif ent.has("id") and typeof(ent["id"]) == TYPE_STRING:
				group = ent["id"]

			if group.is_empty():
				continue
			var w = float(ent.get("w", 2))
			var h = float(ent.get("h", 2))
			var key = "%d_%d" % [int(w), int(h)]
			
			if not group_size_counts.has(group):
				group_size_counts[group] = {}
			if not group_size_counts[group].has(key):
				group_size_counts[group][key] = {"count": 0, "w": w, "h": h}
			group_size_counts[group][key]["count"] += 1
	
	# For each group, pick most common size
	var result: Dictionary = {}
	for group in group_size_counts:
		var size_counts = group_size_counts[group]
		var max_count = 0
		var best_size: Dictionary = {}
		for key in size_counts:
			var entry = size_counts[key]
			if entry["count"] > max_count:
				max_count = entry["count"]
				best_size = {"w": entry["w"], "h": entry["h"]}
		if not best_size.is_empty():
			result[group] = best_size
	
	return result

func _get_normalized_group_size(group: String, current_placement: Dictionary) -> Dictionary:
	"""Get normalized size for a group from cache (computed at load time)."""
	if group.is_empty() or not normalized_group_sizes_cache.has(group):
		return {}
	return normalized_group_sizes_cache[group]

func _load_json(path: String):
	if FileAccess.file_exists(path):
		var txt := FileAccess.get_file_as_string(path)
		var parsed: Variant = JSON.parse_string(txt)
		if typeof(parsed) != TYPE_DICTIONARY:
			push_error("Invalid JSON (expected object): %s" % path)
			return null
		return parsed

	# Fallback: Check external path (for exported builds where JSONs weren't packed)
	var rel_path = path.replace("res://", "")
	var base_dir = OS.get_executable_path().get_base_dir()
	var ext_path = ""
	
	if OS.get_name() == "macOS":
		# On macOS, executable is in Contents/MacOS, resources are in Contents/Resources
		ext_path = base_dir.path_join("../Resources/" + rel_path)
	else:
		# On Windows/Linux, resources are usually next to executable
		ext_path = base_dir.path_join(rel_path)
	
	if FileAccess.file_exists(ext_path):
		print("Found external file: ", ext_path)
		var txt := FileAccess.get_file_as_string(ext_path)
		var parsed: Variant = JSON.parse_string(txt)
		if typeof(parsed) != TYPE_DICTIONARY:
			push_error("Invalid JSON (expected object): %s" % ext_path)
			return null
		return parsed

	push_error("Missing file: %s (checked external: %s)" % [path, ext_path])
	return null
