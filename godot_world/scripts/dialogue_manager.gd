extends Node
## DialogueManager autoload: loads dialogue.json, shows bottom text box, runs dialogue graph (goto/choices).
## Call start_dialogue(display_name) when player talks to NPC; advance() on Next/Space/Enter.

const DIALOGUE_PATH := "res://generated/dialogue.json"
const NPC_INTERACT_RANGE_M := 3.0

var dialogue_data: Dictionary = {}
var events: Array = []
var event_by_id: Dictionary = {}
var npc_first_event: Dictionary = {}  # display_name -> first event id for that speaker

var dialogue_active: bool = false
var current_event_id: String = ""
var _dialogue_start_npc_name: String = ""  # NPC we're talking to; end when we would show another NPC

# UI nodes (created in _ready)
var _layer: CanvasLayer = null
var _panel: PanelContainer = null
var _vbox: VBoxContainer = null
var _speaker_label: Label = null
var _text_label: Label = null
var _choices_container: HBoxContainer = null
var _next_btn: Button = null
var _hint_label: Label = null

# Arrow-key choice selection: when current event has choices
var _current_choice_goto_ids: Array = []  # [goto_id, ...]
var _selected_choice_index: int = 0

# Ending sequence: black screen + narrator lines, then quit
var _ending_mode: bool = false
var _ending_layer: CanvasLayer = null
var _ending_label: Label = null
var _ending_hint: Label = null


func _ready() -> void:
	_load_dialogue()
	_build_ui()
	_hide_ui()


func _input(event: InputEvent) -> void:
	if _ending_mode:
		if event.is_action_pressed("ui_accept") or event.is_action_pressed("ui_select"):
			_ending_advance()
			get_viewport().set_input_as_handled()
		return
	if not dialogue_active or _current_choice_goto_ids.is_empty():
		return
	var n := _current_choice_goto_ids.size()
	if n == 0:
		return
	if event.is_action_pressed("ui_left") or event.is_action_pressed("ui_up"):
		_selected_choice_index = (_selected_choice_index - 1 + n) % n
		_update_choice_highlight()
		get_viewport().set_input_as_handled()
	elif event.is_action_pressed("ui_right") or event.is_action_pressed("ui_down"):
		_selected_choice_index = (_selected_choice_index + 1) % n
		_update_choice_highlight()
		get_viewport().set_input_as_handled()


func _load_dialogue() -> void:
	var path := DIALOGUE_PATH
	if not FileAccess.file_exists(path):
		path = _resolve_path(path)
	var f = FileAccess.open(path, FileAccess.READ)
	if f == null:
		push_error("DialogueManager: Could not open dialogue.json at " + path)
		return
	var json_text := f.get_as_text()
	f.close()
	var json := JSON.new()
	var err := json.parse(json_text)
	if err != OK:
		push_error("DialogueManager: Failed to parse dialogue.json: " + json.get_error_message())
		return
	dialogue_data = json.data
	events = dialogue_data.get("events", [])
	event_by_id.clear()
	npc_first_event.clear()
	_ending_event_cache.clear()  # Clear cache when dialogue is reloaded
	for i in range(events.size()):
		var ev = events[i]
		var eid := str(ev.get("id", ""))
		if eid.is_empty():
			continue
		event_by_id[eid] = ev
		var speaker := str(ev.get("speaker", ""))
		if speaker.is_empty():
			continue
		if not npc_first_event.has(speaker):
			npc_first_event[speaker] = eid
	print("DialogueManager: Loaded ", events.size(), " events, ", npc_first_event.size(), " speakers")


func _resolve_path(res_path: String) -> String:
	if FileAccess.file_exists(res_path):
		return res_path
	var rel = res_path.replace("res://", "")
	var base_dir := OS.get_executable_path().get_base_dir()
	var ext_path := base_dir.path_join(rel)
	if FileAccess.file_exists(ext_path):
		return ext_path
	return res_path


# Cache for ending event detection to avoid repeated traversal
var _ending_event_cache: Dictionary = {}  # event_id -> bool


func _is_ending_event(event_id: String) -> bool:
	"""
	Check if an event should start the ending sequence (black screen).
	True if event_id is "end" OR if the event has tags.ending = true.
	"""
	if event_id == "end":
		return true
	
	var ev: Dictionary = event_by_id.get(event_id, {})
	var tags: Dictionary = ev.get("tags", {}) if ev.get("tags", {}) is Dictionary else {}
	return tags.get("ending", false)


func _build_ui() -> void:
	_layer = CanvasLayer.new()
	_layer.name = "DialogueLayer"
	add_child(_layer)

	_panel = PanelContainer.new()
	_panel.name = "DialoguePanel"
	_panel.set_anchors_preset(Control.PRESET_FULL_RECT)
	_panel.anchor_top = 0.75
	_panel.anchor_bottom = 1.0
	_panel.anchor_left = 0.0
	_panel.anchor_right = 1.0
	_panel.offset_left = 20
	_panel.offset_top = 20
	_panel.offset_right = -20
	_panel.offset_bottom = -20
	var style := StyleBoxFlat.new()
	style.bg_color = Color(0.12, 0.12, 0.18, 0.95)
	style.set_border_width_all(2)
	style.border_color = Color(0.4, 0.4, 0.5)
	style.set_corner_radius_all(8)
	_panel.add_theme_stylebox_override("panel", style)
	_layer.add_child(_panel)

	_vbox = VBoxContainer.new()
	_vbox.name = "VBox"
	_vbox.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT, Control.PRESET_MODE_MINSIZE, 8)
	_vbox.add_theme_constant_override("separation", 12)
	_panel.add_child(_vbox)

	_speaker_label = Label.new()
	_speaker_label.name = "SpeakerLabel"
	_speaker_label.text = ""
	_speaker_label.add_theme_font_size_override("font_size", 22)
	_speaker_label.add_theme_color_override("font_color", Color(0.9, 0.85, 0.6))
	_vbox.add_child(_speaker_label)

	_text_label = Label.new()
	_text_label.name = "TextLabel"
	_text_label.text = ""
	_text_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	_text_label.custom_minimum_size.y = 60
	_text_label.add_theme_font_size_override("font_size", 18)
	_vbox.add_child(_text_label)

	_choices_container = HBoxContainer.new()
	_choices_container.name = "ChoicesContainer"
	_choices_container.add_theme_constant_override("separation", 12)
	_vbox.add_child(_choices_container)

	_next_btn = Button.new()
	_next_btn.name = "NextButton"
	_next_btn.text = "Next (E / Space)"
	_next_btn.pressed.connect(_on_next_pressed)
	_vbox.add_child(_next_btn)

	_hint_label = Label.new()
	_hint_label.name = "HintLabel"
	_hint_label.text = "Arrow keys: select  •  E / Space: confirm"
	_hint_label.add_theme_font_size_override("font_size", 14)
	_hint_label.add_theme_color_override("font_color", Color(0.7, 0.7, 0.8))
	_vbox.add_child(_hint_label)
	_hint_label.visible = false


func _hide_ui() -> void:
	if _layer:
		_layer.visible = false
	_hide_ending_ui()
	dialogue_active = false
	_ending_mode = false
	current_event_id = ""
	_dialogue_start_npc_name = ""
	_current_choice_goto_ids.clear()
	_selected_choice_index = 0


func _show_ui() -> void:
	if _layer:
		_layer.visible = true
	dialogue_active = true


func _display_current() -> void:
	if current_event_id.is_empty() or not event_by_id.has(current_event_id):
		end_dialogue()
		return
	var ev: Dictionary = event_by_id[current_event_id]
	
	# Check if this is an ending event FIRST (before speaker validation)
	# This ensures ending events are shown on black screen, not in normal dialogue UI
	if _is_ending_event(current_event_id):
		_show_ending_sequence(ev)
		return
	
	# Terminal ending: black screen + narrator line(s), then quit
	var tags: Dictionary = ev.get("tags", {}) if ev.get("tags", {}) is Dictionary else {}
	if current_event_id == "end" or tags.get("ending", false):
		_show_ending_sequence(ev)
		return
	var speaker := str(ev.get("speaker", ""))
	# End conversation when we would show another NPC (scope dialogue to the one we started with)
	if speaker != "NARRATOR" and speaker != _dialogue_start_npc_name:
		# Safety net: if this event leads to "end", transition to ending sequence instead of ending
		if str(ev.get("goto", "")) == "end":
			current_event_id = "end"
			var end_ev: Dictionary = event_by_id.get("end", {})
			_show_ending_sequence(end_ev)
			return
		end_dialogue()
		return
	var text := str(ev.get("text", ""))
	_speaker_label.text = speaker
	_text_label.text = text

	# Clear choice buttons
	for c in _choices_container.get_children():
		c.queue_free()

	var choices: Array = ev.get("choices", [])
	_current_choice_goto_ids.clear()
	_selected_choice_index = 0
	if choices is Array and choices.size() > 0:
		_next_btn.visible = false
		_hint_label.visible = true
		for ch in choices:
			var choice: Dictionary = ch if ch is Dictionary else {}
			var btn := Button.new()
			btn.text = str(choice.get("text", ""))
			var goto_id := str(choice.get("goto", ""))
			_current_choice_goto_ids.append(goto_id)
			btn.pressed.connect(_on_choice_pressed.bind(goto_id))
			_choices_container.add_child(btn)
		_update_choice_highlight()
	else:
		_next_btn.visible = true
		_hint_label.visible = false
		var goto_id := str(ev.get("goto", ""))
		if goto_id.is_empty():
			_next_btn.text = "Close (E / Space)"
		else:
			_next_btn.text = "Next (E / Space)"


func _update_choice_highlight() -> void:
	var children = _choices_container.get_children()
	for i in range(children.size()):
		var btn = children[i] as Button
		if btn:
			btn.modulate = Color(1.25, 1.2, 0.85) if i == _selected_choice_index else Color(1, 1, 1)
			if i == _selected_choice_index:
				btn.grab_focus()

func _on_next_pressed() -> void:
	advance()


func _on_choice_pressed(goto_id: String) -> void:
	if goto_id.is_empty():
		end_dialogue()
		return
	current_event_id = goto_id
	_display_current()


func has_active_dialogue() -> bool:
	return dialogue_active or _ending_mode


func start_dialogue(npc_display_name: String) -> void:
	var first_id = npc_first_event.get(npc_display_name, "")
	if first_id.is_empty():
		push_error("DialogueManager: No dialogue for speaker: " + npc_display_name)
		return
	_dialogue_start_npc_name = npc_display_name
	current_event_id = first_id
	_show_ui()
	_display_current()


func advance() -> void:
	if _ending_mode:
		_ending_advance()
		return
	if not dialogue_active or current_event_id.is_empty():
		return
	# When showing choices, E/Space confirms the currently selected option
	if _current_choice_goto_ids.size() > 0:
		var goto_id: String = str(_current_choice_goto_ids[_selected_choice_index])
		_on_choice_pressed(goto_id)
		return
	var ev: Dictionary = event_by_id.get(current_event_id, {})
	var goto_id: String = str(ev.get("goto", ""))
	if goto_id.is_empty():
		end_dialogue()
		return
	current_event_id = goto_id
	_display_current()


func end_dialogue() -> void:
	_hide_ui()


func _show_ending_sequence(ev: Dictionary) -> void:
	# Hide normal dialogue panel; show fullscreen black + narrator text
	if _layer:
		_layer.visible = false
	_ending_mode = true
	if _ending_layer == null:
		_ending_layer = CanvasLayer.new()
		_ending_layer.name = "EndingLayer"
		_ending_layer.layer = 1000
		add_child(_ending_layer)
		var black := ColorRect.new()
		black.name = "EndingBlack"
		black.set_anchors_preset(Control.PRESET_FULL_RECT)
		black.set_anchor(SIDE_LEFT, 0.0)
		black.set_anchor(SIDE_TOP, 0.0)
		black.set_anchor(SIDE_RIGHT, 1.0)
		black.set_anchor(SIDE_BOTTOM, 1.0)
		black.offset_left = 0
		black.offset_top = 0
		black.offset_right = 0
		black.offset_bottom = 0
		black.color = Color.BLACK
		black.mouse_filter = Control.MOUSE_FILTER_IGNORE
		_ending_layer.add_child(black)
		var center := CenterContainer.new()
		center.name = "EndingCenter"
		center.set_anchors_preset(Control.PRESET_FULL_RECT)
		center.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT, Control.PRESET_MODE_MINSIZE, 40)
		_ending_layer.add_child(center)
		var vbox := VBoxContainer.new()
		vbox.add_theme_constant_override("separation", 24)
		center.add_child(vbox)
		_ending_label = Label.new()
		_ending_label.name = "EndingLabel"
		_ending_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
		_ending_label.vertical_alignment = VERTICAL_ALIGNMENT_CENTER
		_ending_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
		_ending_label.custom_minimum_size.x = 600
		_ending_label.add_theme_font_size_override("font_size", 28)
		_ending_label.add_theme_color_override("font_color", Color.WHITE)
		vbox.add_child(_ending_label)
		_ending_hint = Label.new()
		_ending_hint.name = "EndingHint"
		_ending_hint.text = "Press E or Space to continue"
		_ending_hint.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
		_ending_hint.add_theme_font_size_override("font_size", 16)
		_ending_hint.add_theme_color_override("font_color", Color(0.6, 0.6, 0.6))
		vbox.add_child(_ending_hint)
	_ending_layer.visible = true
	_ending_label.text = str(ev.get("text", ""))


func _hide_ending_ui() -> void:
	if _ending_layer:
		_ending_layer.visible = false


func _ending_advance() -> void:
	if not _ending_mode or current_event_id.is_empty():
		return
	var ev: Dictionary = event_by_id.get(current_event_id, {})
	var goto_id: String = str(ev.get("goto", ""))
	if goto_id.is_empty():
		_quit_or_the_end()
		return
	current_event_id = goto_id
	ev = event_by_id.get(current_event_id, {})
	if ev.is_empty():
		_quit_or_the_end()
		return
	_ending_label.text = str(ev.get("text", ""))


func _quit_or_the_end() -> void:
	_ending_mode = false
	_hide_ending_ui()
	if _ending_layer != null and _ending_label != null:
		_ending_layer.visible = true
		_ending_label.text = "The End"
		if _ending_hint:
			_ending_hint.text = ""
	if OS.has_feature("web"):
		pass
	else:
		# Show "The End" on black for ~1.5 s, then quit
		var t := get_tree().create_timer(1.5)
		t.timeout.connect(_on_ending_quit_timeout)


func _on_ending_quit_timeout() -> void:
	get_tree().quit()
