extends CharacterBody3D

const SPEED = 5.0
const BOOST_SPEED = 50.0
const JUMP_VELOCITY = 4.5
const MOUSE_SENSITIVITY = 0.003

# Get the gravity from the project settings to be synced with RigidBody nodes.
var gravity: float = ProjectSettings.get_setting("physics/3d/default_gravity")

var camera: Camera3D = null

@export var player_height_m: float = 1.8
@export var eye_height_ratio: float = 0.92
@export var capsule_radius_m: float = 0.3

func _ready() -> void:
	print("PLAYER SCRIPT: _ready() called")
	add_to_group("player")
	camera = get_node_or_null("Camera3D")
	_setup_input_map()
	_apply_player_scale()
	print("PLAYER SCRIPT: actions setup complete. Camera: ", camera)
	
	if OS.has_feature("web"):
		Input.mouse_mode = Input.MOUSE_MODE_VISIBLE
	else:
		Input.mouse_mode = Input.MOUSE_MODE_CAPTURED

var fly_mode: bool = false
var _last_frame_pos := Vector3.ZERO
var _stuck_frames := 0
const NPC_INTERACT_RANGE_M := 3.0
var _near_npc: Node = null  # Closest NPC in range (group "npc"), for E-to-talk

func _setup_input_map() -> void:
	# Programmatically add WASD actions if they don't exist
	if not InputMap.has_action("move_forward"):
		InputMap.add_action("move_forward")
		var ev = InputEventKey.new()
		ev.keycode = KEY_W
		InputMap.action_add_event("move_forward", ev)
		
	if not InputMap.has_action("move_backward"):
		InputMap.add_action("move_backward")
		var ev = InputEventKey.new()
		ev.keycode = KEY_S
		InputMap.action_add_event("move_backward", ev)
		
	if not InputMap.has_action("move_left"):
		InputMap.add_action("move_left")
		var ev = InputEventKey.new()
		ev.keycode = KEY_A
		InputMap.action_add_event("move_left", ev)
		
	if not InputMap.has_action("move_right"):
		InputMap.add_action("move_right")
		var ev = InputEventKey.new()
		ev.keycode = KEY_D
		InputMap.action_add_event("move_right", ev)
		
	if not InputMap.has_action("jump"):
		InputMap.add_action("jump")
		var ev = InputEventKey.new()
		ev.keycode = KEY_SPACE
		InputMap.action_add_event("jump", ev)
		
	if not InputMap.has_action("sprint"):
		InputMap.add_action("sprint")
		var ev = InputEventKey.new()
		ev.keycode = KEY_SHIFT
		InputMap.action_add_event("sprint", ev)

	# Fly controls
	if not InputMap.has_action("fly_toggle"):
		InputMap.add_action("fly_toggle")
		var ev = InputEventKey.new()
		ev.keycode = KEY_F
		InputMap.action_add_event("fly_toggle", ev)

	if not InputMap.has_action("fly_up"):
		InputMap.add_action("fly_up")
		var ev = InputEventKey.new()
		ev.keycode = KEY_E
		InputMap.action_add_event("fly_up", ev)

	if not InputMap.has_action("fly_down"):
		InputMap.add_action("fly_down")
		var ev = InputEventKey.new()
		ev.keycode = KEY_Q
		InputMap.action_add_event("fly_down", ev)

	if not InputMap.has_action("interact"):
		InputMap.add_action("interact")
		var ev = InputEventKey.new()
		ev.keycode = KEY_E
		InputMap.action_add_event("interact", ev)

func _apply_player_scale() -> void:
	if camera:
		camera.position.y = player_height_m * eye_height_ratio
		camera.current = true

	var cs: CollisionShape3D = $CollisionShape3D
	if not cs:
		return
		
	if cs.shape is CapsuleShape3D:
		var cap: CapsuleShape3D = cs.shape
		cap.radius = capsule_radius_m
		cap.height = max(0.1, player_height_m - 2.0 * capsule_radius_m)
		cs.position.y = player_height_m / 2.0

func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventKey and event.pressed and event.keycode == KEY_R:
		print("DEBUG: Force respawn/unstuck")
		position += Vector3(0, 5, 0)
		velocity = Vector3.ZERO

	if OS.has_feature("web") and Input.mouse_mode != Input.MOUSE_MODE_CAPTURED:
		if event is InputEventMouseButton and event.pressed:
			if event.button_index == MOUSE_BUTTON_LEFT or event.button_index == MOUSE_BUTTON_RIGHT:
				Input.mouse_mode = Input.MOUSE_MODE_CAPTURED
				return
	
	if event is InputEventMouseMotion:
		if Input.mouse_mode == Input.MOUSE_MODE_CAPTURED and camera != null:
			rotate_y(-event.relative.x * MOUSE_SENSITIVITY)
			camera.rotate_x(-event.relative.y * MOUSE_SENSITIVITY)
			camera.rotation.x = clamp(camera.rotation.x, deg_to_rad(-90), deg_to_rad(90))
	
	if event.is_action_pressed("ui_cancel"):
		if Input.mouse_mode == Input.MOUSE_MODE_CAPTURED:
			Input.mouse_mode = Input.MOUSE_MODE_VISIBLE
		else:
			Input.mouse_mode = Input.MOUSE_MODE_CAPTURED
			
	if event.is_action_pressed("fly_toggle"):
		fly_mode = not fly_mode
		print("Fly mode: ", "ON" if fly_mode else "OFF")

	# Dialogue: E or Space to advance when in dialogue; E to start when near NPC
	if DialogueManager != null:
		if DialogueManager.has_active_dialogue():
			if event.is_action_pressed("interact") or event.is_action_pressed("ui_accept"):
				DialogueManager.advance()
				get_viewport().set_input_as_handled()
				return
		elif _near_npc != null and event.is_action_pressed("interact"):
			var display_name = _near_npc.get_meta("display_name", "") if _near_npc.has_meta("display_name") else ""
			if not display_name.is_empty():
				# Rotate NPC to face the player when starting dialogue
				var npc_3d = _near_npc as Node3D
				if npc_3d != null:
					var look_target = Vector3(global_position.x, npc_3d.global_position.y, global_position.z)
					npc_3d.look_at(look_target, Vector3.UP)
					npc_3d.rotate_y(PI)  # Model faces +Z; look_at uses -Z
				DialogueManager.start_dialogue(display_name)
				get_viewport().set_input_as_handled()
				return

var frame_count = 0

func _physics_process(delta: float) -> void:
	# Update closest NPC in range for E-to-talk
	var npcs = get_tree().get_nodes_in_group("npc")
	var best_dist := NPC_INTERACT_RANGE_M
	_near_npc = null
	for n in npcs:
		if n is Node3D:
			var d := global_position.distance_to((n as Node3D).global_position)
			if d < best_dist:
				best_dist = d
				_near_npc = n

	frame_count += 1
	if frame_count % 60 == 0:
		var input = Input.get_vector("move_left", "move_right", "move_forward", "move_backward")
		print("PLAYER LOOP: Input: %v | Vel: %v | Pos: %v | Mode: %s | OnFloor: %s" % [input, velocity, position, "FLY" if fly_mode else "WALK", is_on_floor()])

	var current_speed = SPEED
	if Input.is_action_pressed("sprint"):
		current_speed = BOOST_SPEED

	var input_dir := Input.get_vector("move_left", "move_right", "move_forward", "move_backward")

	if fly_mode:
		# Flying Movement (No Gravity, 6DOF) - TRUE NOCLIP
		var direction = Vector3.ZERO
		
		# Move relative to Camera look direction
		if camera == null:
			return
		
		var cam_basis = camera.global_transform.basis
		direction += -cam_basis.z * -input_dir.y # Forward/Back
		direction += cam_basis.x * input_dir.x # Left/Right
		
		# Vertical movement
		if Input.is_action_pressed("fly_up"):
			direction.y += 1.0
		if Input.is_action_pressed("fly_down"):
			direction.y -= 1.0
		
		if direction.length() > 0:
			direction = direction.normalized()
			velocity = direction * current_speed
		else:
			velocity = velocity.move_toward(Vector3.ZERO, current_speed)
		
		# Noclip Movement (Bypass Physics) - NO move_and_slide!
		position += velocity * delta
			
	else:
		# Walking Movement (Standard FPS with Physics)
		if not is_on_floor():
			velocity.y -= gravity * delta

		if Input.is_action_pressed("jump") and is_on_floor():
			velocity.y = JUMP_VELOCITY

		var direction := (transform.basis * Vector3(input_dir.x, 0, input_dir.y)).normalized()
		
		if direction:
			velocity.x = direction.x * current_speed
			velocity.z = direction.z * current_speed
		else:
			velocity.x = move_toward(velocity.x, 0, current_speed)
			velocity.z = move_toward(velocity.z, 0, current_speed)

		# Physics movement - ONLY in walk mode
		move_and_slide()
	
		# Stuck Debugging and Auto-Unstuck (Only in Walk Mode)
		if get_slide_collision_count() > 0:
			if frame_count % 60 == 0:
				var collider = get_slide_collision(0).get_collider()
				print("PLAYER COLLISION: Hitting ", collider.name if collider else "null")
				
			# If trying to move but position isn't changing significantly
			if velocity.length() > 1.0 and (position - _last_frame_pos).length() < 0.01:
				_stuck_frames += 1
				if _stuck_frames > 60: # Stuck for 1 second
					print("DEBUG: Auto-unstuck triggered! Teleporting up.")
					position.y += 10.0
					_stuck_frames = 0
			else:
				_stuck_frames = 0
				
		_last_frame_pos = position
