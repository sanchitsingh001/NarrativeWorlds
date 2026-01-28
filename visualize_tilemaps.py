"""
Tilemap compiler for area layouts.

Converts intent graphs (from area_layouts.json) into actual walkable road tiles.
Uses utilities from visualize_graphs.py for position computation.
"""

import json
import math
import hashlib
from pathlib import Path
from collections import deque
import heapq

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# ==================== Imports from visualize_graphs ====================
from visualize_graphs import (
    compute_graph_positions,
    resolve_overlaps,
    compute_bounding_box,
    place_gates_on_boundary,
    get_boundary_point_in_direction,
    get_entity_rect,
    pick_footprint,
    SIZE_HINT_TO_FOOTPRINT,
    DEFAULT_FOOTPRINT,
    _DIR_TO_VEC,
    _DIST_STEPS,
)

# ==================== Constants ====================

# Tile codes for visualization
T_EMPTY = 0
T_ROAD = 1
T_BLOCKED = 2
T_GATE = 3
T_CENTER = 4
T_FOOTPATH = 5

# Corner overlay codes (separate layer from road_mask)
# Used for filling triangles INTO empty tiles (concave/inner corners)
C_NONE = 0
C_NE = 1   # Triangle fills top-right corner
C_NW = 2   # Triangle fills top-left corner
C_SE = 3   # Triangle fills bottom-right corner
C_SW = 4   # Triangle fills bottom-left corner

# Road cutout codes (for cutting corners OFF road tiles - convex/outer corners)
# These mark which corner of a road tile should be "cut off" to smooth the outer edge
# Uses same values as corner codes for consistency
CUT_NONE = 0
CUT_NE = 1   # Cut top-right corner off road tile
CUT_NW = 2   # Cut top-left corner off road tile
CUT_SE = 3   # Cut bottom-right corner off road tile
CUT_SW = 4   # Cut bottom-left corner off road tile

# Direction vectors for 4-neighbor (grid style) and 8-neighbor (organic/trails)
DIR_4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # E, W, N, S
DIR_8 = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)]

# Road style configuration
ROAD_STYLE_CONFIG = {
    "grid": {
        "neighbors": 4,          # 4-neighbor only
        "turn_penalty": 100.0,   # Heavy penalty for turns
        "merge_bonus": 0.25,     # Cost multiplier for existing roads
        "directions": DIR_4,
    },
    "organic": {
        "neighbors": 8,          # 8-neighbor allowed
        "turn_penalty": 10.0,    # Mild turn penalty
        "merge_bonus": 0.25,
        "directions": DIR_8,
    },
    "trails": {
        "neighbors": 8,          # 8-neighbor
        "turn_penalty": 2.0,     # Minimal turn penalty
        "merge_bonus": 0.25,
        "directions": DIR_8,
    },
    "none": None,  # Skip road generation
}

# Road width by kind
ROAD_WIDTH = {
    "trunk_road": 5,
    "road": 3,
    "footpath": 1,
}

# Grid size hints
_SIZE_TO_GRID = {
    "small": (60, 50),
    "medium": (90, 70),
    "large": (130, 95),
    "huge": (180, 130),
}


# ==================== Coordinate Transform ====================

def world_to_grid(wx: float, wy: float, min_x: float, min_y: float) -> tuple[int, int]:
    """Convert world coordinates to grid indices."""
    gx = int(round(wx - min_x))
    gy = int(round(wy - min_y))
    return (gx, gy)


def grid_to_world(gx: int, gy: int, min_x: float, min_y: float) -> tuple[float, float]:
    """Convert grid indices back to world coordinates."""
    wx = float(gx) + min_x
    wy = float(gy) + min_y
    return (wx, wy)


def in_bounds(x: int, y: int, grid_w: int, grid_h: int) -> bool:
    """Check if grid coordinates are within bounds."""
    return 0 <= x < grid_w and 0 <= y < grid_h


# ==================== A* Pathfinding with Direction State ====================

def a_star_with_direction(
    start: tuple[int, int],
    goal: tuple[int, int],
    grid_w: int,
    grid_h: int,
    blocked: list[list[bool]],
    road_mask: list[list[bool]],
    road_style: str,
    edge_margin: int = 2
) -> list[tuple[int, int]]:
    """
    A* pathfinding with direction as part of state for proper turn penalties.
    
    State: (x, y, incoming_dir_idx)
    This allows turn penalties to work correctly for grid-style roads.
    """
    config = ROAD_STYLE_CONFIG.get(road_style)
    if config is None:
        # Fallback to organic style if road_style not found
        config = ROAD_STYLE_CONFIG["organic"]
    
    directions = config["directions"]
    turn_penalty = config["turn_penalty"]
    merge_bonus = config["merge_bonus"]
    
    sx, sy = start
    gx, gy = goal
    
    if not in_bounds(sx, sy, grid_w, grid_h) or not in_bounds(gx, gy, grid_w, grid_h):
        return []
    
    if start == goal:
        return [start]
    
    # Compute edge margin dynamically - allow paths near boundaries if start/goal are there
    start_near_edge = (sx < 5 or sy < 5 or sx >= grid_w - 5 or sy >= grid_h - 5)
    goal_near_edge = (gx < 5 or gy < 5 or gx >= grid_w - 5 or gy >= grid_h - 5)
    effective_edge_margin = 1 if (start_near_edge or goal_near_edge) else edge_margin
    
    def heuristic(x: int, y: int) -> float:
        return abs(x - gx) + abs(y - gy)
    
    def is_turn(dir_idx: int, new_dir_idx: int) -> bool:
        """Check if moving from dir_idx to new_dir_idx is a turn."""
        if dir_idx < 0:  # No previous direction
            return False
        return dir_idx != new_dir_idx
    
    def compute_cost(x: int, y: int, prev_dir_idx: int, new_dir_idx: int) -> float:
        """Compute cost to move to (x, y) from direction new_dir_idx."""
        if blocked[y][x]:
            return float('inf')
        
        # Base cost (lower if road already exists - merge bonus)
        base = merge_bonus if road_mask[y][x] else 1.0
        
        # Edge penalty to prevent roads hugging boundaries (but reduced if routing to edge)
        edge_pen = 0.0
        if x < effective_edge_margin or y < effective_edge_margin or \
           x >= grid_w - effective_edge_margin or y >= grid_h - effective_edge_margin:
            edge_pen = 20.0  # Reduced from 100 to allow edge routing
        
        # Turn penalty
        turn_pen = 0.0
        if is_turn(prev_dir_idx, new_dir_idx):
            turn_pen = turn_penalty
        
        # Clearance penalty to avoid tight gaps near obstacles (reduces pinch points)
        clearance_pen = 0.0
        for cdx, cdy in DIR_4:
            cx, cy = x + cdx, y + cdy
            if in_bounds(cx, cy, grid_w, grid_h) and blocked[cy][cx]:
                clearance_pen += 5.0  # Penalty for being near obstacles
        
        # Diagonal movement costs more (sqrt(2) vs 1)
        dx, dy = directions[new_dir_idx]
        move_cost = math.sqrt(dx * dx + dy * dy)
        
        return base * move_cost + edge_pen + turn_pen + clearance_pen
    
    # Priority queue: (f_score, g_score, x, y, dir_idx)
    # dir_idx = -1 means "no incoming direction" (start)
    open_heap: list[tuple[float, float, int, int, int]] = []
    heapq.heappush(open_heap, (heuristic(sx, sy), 0.0, sx, sy, -1))
    
    # visited[(x, y, dir_idx)] = best g_score
    visited: dict[tuple[int, int, int], float] = {}
    came_from: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    
    while open_heap:
        f, g, x, y, dir_idx = heapq.heappop(open_heap)
        
        # Check if we reached goal (any direction is fine)
        if (x, y) == (gx, gy):
            # Reconstruct path
            path = [(x, y)]
            state = (x, y, dir_idx)
            while state in came_from:
                state = came_from[state]
                path.append((state[0], state[1]))
            path.reverse()
            return path
        
        state = (x, y, dir_idx)
        if state in visited and visited[state] <= g:
            continue
        visited[state] = g
        
        # Explore neighbors
        for new_dir_idx, (dx, dy) in enumerate(directions):
            nx, ny = x + dx, y + dy
            if not in_bounds(nx, ny, grid_w, grid_h):
                continue
            
            cost = compute_cost(nx, ny, dir_idx, new_dir_idx)
            if cost >= float('inf'):
                continue
            
            ng = g + cost
            new_state = (nx, ny, new_dir_idx)
            
            if new_state not in visited or ng < visited[new_state]:
                came_from[new_state] = state
                heapq.heappush(open_heap, (ng + heuristic(nx, ny), ng, nx, ny, new_dir_idx))
    
    return []  # No path found


# ==================== Road Routing ====================

def normalize_vec(dx: float, dy: float) -> tuple[float, float]:
    """Normalize a vector to unit length."""
    length = math.sqrt(dx * dx + dy * dy)
    if length < 0.001:
        return (1.0, 0.0)
    return (dx / length, dy / length)


def find_walkable_near(
    target: tuple[int, int],
    blocked: list[list[bool]],
    grid_w: int,
    grid_h: int,
    max_search: int = 10
) -> tuple[int, int]:
    """Find a walkable tile near the target, searching outward."""
    tx, ty = target
    
    # Check target first
    if in_bounds(tx, ty, grid_w, grid_h) and not blocked[ty][tx]:
        return (tx, ty)
    
    # Search in expanding rings
    for radius in range(1, max_search + 1):
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if abs(dx) != radius and abs(dy) != radius:
                    continue
                nx, ny = tx + dx, ty + dy
                if in_bounds(nx, ny, grid_w, grid_h) and not blocked[ny][nx]:
                    return (nx, ny)
    
    # Fallback to target
    return target


def get_routing_endpoints(
    from_pos: tuple[float, float],
    from_size: tuple[int, int],
    to_pos: tuple[float, float],
    to_size: tuple[int, int],
    min_x: float,
    min_y: float,
    blocked: list[list[bool]] | None = None,
    grid_w: int = 0,
    grid_h: int = 0
) -> tuple[tuple[int, int], tuple[int, int]]:
    """
    Get routing start and end points on building boundaries, stepped outside.
    Routes from boundary points, not centers, to avoid hitting buildings.
    
    If blocked grid is provided, ensures endpoints are on walkable tiles.
    """
    # Direction from "from" center to "to" center
    dx = to_pos[0] - from_pos[0]
    dy = to_pos[1] - from_pos[1]
    dir_vec = normalize_vec(dx, dy)
    
    # From rect: (center_x, center_y, width, height)
    from_rect = (from_pos[0], from_pos[1], from_size[0], from_size[1])
    to_rect = (to_pos[0], to_pos[1], to_size[0], to_size[1])
    
    # Get boundary points
    from_boundary = get_boundary_point_in_direction(from_rect, dir_vec)
    # Opposite direction for "to" entity
    to_boundary = get_boundary_point_in_direction(to_rect, (-dir_vec[0], -dir_vec[1]))
    
    # Step outside the boundary (use smaller step to stay close to buildings)
    step_dist = 1.5
    start_world = (from_boundary[0] + dir_vec[0] * step_dist, 
                   from_boundary[1] + dir_vec[1] * step_dist)
    end_world = (to_boundary[0] - dir_vec[0] * step_dist, 
                 to_boundary[1] - dir_vec[1] * step_dist)
    
    # Convert to grid coords
    start_grid = world_to_grid(start_world[0], start_world[1], min_x, min_y)
    end_grid = world_to_grid(end_world[0], end_world[1], min_x, min_y)
    
    # If blocked grid provided, find walkable tiles near the targets
    if blocked is not None and grid_w > 0 and grid_h > 0:
        start_grid = find_walkable_near(start_grid, blocked, grid_w, grid_h)
        end_grid = find_walkable_near(end_grid, blocked, grid_w, grid_h)
    
    return start_grid, end_grid


def dedupe_paths(paths: list[dict]) -> list[dict]:
    """
    Deduplicate paths treating them as undirected edges.
    Keeps the first occurrence of each unique edge.
    """
    seen: set[tuple[str, str]] = set()
    deduped: list[dict] = []
    for p in paths:
        from_id = p.get("from", "")
        to_id = p.get("to", "")
        edge = tuple(sorted([from_id, to_id]))
        if edge not in seen:
            seen.add(edge)
            deduped.append(p)
    return deduped


def widen_road_path(
    path: list[tuple[int, int]],
    width: int,
    road_mask: list[list[bool]],
    blocked: list[list[bool]],
    grid_w: int,
    grid_h: int
) -> None:
    """
    Widen a road path, respecting obstacles.
    Only sets road_mask[y][x] = True if not blocked[y][x].
    """
    half = width // 2
    for (x, y) in path:
        for dy in range(-half, half + 1):
            for dx in range(-half, half + 1):
                nx, ny = x + dx, y + dy
                if in_bounds(nx, ny, grid_w, grid_h):
                    if not blocked[ny][nx]:  # CRITICAL: respect obstacles
                        road_mask[ny][nx] = True
def fill_diagonal_gaps(
    road_mask: list[list[bool]],
    blocked: list[list[bool]],
    grid_w: int,
    grid_h: int,
    passes: int = 2
) -> int:
    """
    Fill 'white triangle' gaps along diagonals by closing 2x2 patterns.

    If we see:
      R .        . R
      . R   or   R .

    then we fill the other two corners (if not blocked) to remove staircase gaps.
    """
    filled = 0
    for _ in range(passes):
        changed = 0
        for y in range(grid_h - 1):
            for x in range(grid_w - 1):
                a = road_mask[y][x]
                b = road_mask[y][x + 1]
                c = road_mask[y + 1][x]
                d = road_mask[y + 1][x + 1]

                # Diagonal \ pattern: a and d are road, b and c are empty
                if a and d and (not b) and (not c):
                    if not blocked[y][x + 1]:
                        road_mask[y][x + 1] = True
                        changed += 1
                    if not blocked[y + 1][x]:
                        road_mask[y + 1][x] = True
                        changed += 1

                # Diagonal / pattern: b and c are road, a and d are empty
                elif b and c and (not a) and (not d):
                    if not blocked[y][x]:
                        road_mask[y][x] = True
                        changed += 1
                    if not blocked[y + 1][x + 1]:
                        road_mask[y + 1][x + 1] = True
                        changed += 1

        filled += changed
        if changed == 0:
            break
    return filled


def compute_diagonal_corner_overlays(
    road_mask: list[list[bool]],
    blocked: list[list[bool]],
    grid_w: int,
    grid_h: int
) -> list[list[int]]:
    """
    Returns overlay[y][x] = one of C_*.
    Place a triangle overlay in empty tiles that are the "missing half" next to a diagonal edge.
    
    This smooths the visual staircase effect on diagonal roads by marking where
    triangle sprites should be placed instead of full square tiles.
    
    Handles INNER (concave) corners where an empty tile is surrounded by 
    an L-shaped road pattern. This fills the "white triangle" gaps inside
    the road boundary.
    
    Note: OUTER (convex) corners on the road boundary require a different
    approach - using cutout overlays ON road tiles, not filling empty tiles.
    That would need a separate overlay layer (e.g., road_cutouts).
    """
    overlay = [[C_NONE for _ in range(grid_w)] for _ in range(grid_h)]

    def is_road(x: int, y: int) -> bool:
        return 0 <= x < grid_w and 0 <= y < grid_h and road_mask[y][x]

    def is_free(x: int, y: int) -> bool:
        return 0 <= x < grid_w and 0 <= y < grid_h and (not blocked[y][x])

    for y in range(grid_h):
        for x in range(grid_w):
            if road_mask[y][x]:
                continue  # overlay only into empty tiles

            if not is_free(x, y):
                continue

            # ============================================================
            # INNER (CONCAVE) CORNERS
            # Pattern: L-shaped road around this empty tile
            # 
            # Example:  . X    where X is the empty tile getting a triangle
            #           R R    R = road tiles forming an L-shape
            #
            # The triangle fills the gap to make the inner corner smooth.
            # ============================================================

            # SW corner fill at (x,y): roads to West and South, no road at SW diagonal
            if is_road(x - 1, y) and is_road(x, y - 1) and (not is_road(x - 1, y - 1)):
                overlay[y][x] = C_SW
                continue

            # SE corner fill: roads to East and South, no road at SE diagonal
            if is_road(x + 1, y) and is_road(x, y - 1) and (not is_road(x + 1, y - 1)):
                overlay[y][x] = C_SE
                continue

            # NW corner fill: roads to West and North, no road at NW diagonal
            if is_road(x - 1, y) and is_road(x, y + 1) and (not is_road(x - 1, y + 1)):
                overlay[y][x] = C_NW
                continue

            # NE corner fill: roads to East and North, no road at NE diagonal
            if is_road(x + 1, y) and is_road(x, y + 1) and (not is_road(x + 1, y + 1)):
                overlay[y][x] = C_NE
                continue

    return overlay


def compute_road_cutouts(
    road_mask: list[list[bool]],
    grid_w: int,
    grid_h: int
) -> list[list[int]]:
    """
    Returns cutout[y][x] = one of CUT_*.
    Marks road tiles that need a corner "cut off" to smooth the outer (convex) boundary.
    
    This handles the OUTER staircase effect where road tiles at the boundary
    have a square corner that should appear diagonal.
    
    Pattern: A road tile at the corner of a diagonal edge, where cutting the
    corner would make the edge appear smoother.
    
    Example:  R R .    The road tile at (1,1) has its NE corner exposed
              R X .    X = road tile that needs NE corner cut
              . . .    The staircase step could be smoothed by cutting NE corner
    """
    cutout = [[CUT_NONE for _ in range(grid_w)] for _ in range(grid_h)]

    def is_road(x: int, y: int) -> bool:
        return 0 <= x < grid_w and 0 <= y < grid_h and road_mask[y][x]

    for y in range(grid_h):
        for x in range(grid_w):
            if not road_mask[y][x]:
                continue  # cutouts only apply to road tiles

            # ============================================================
            # OUTER (CONVEX) CORNER CUTOUTS
            # Pattern: road tile at corner with empty diagonal neighbor
            # 
            # For a road tile to need a corner cut, it must:
            # 1. Have NO road in the diagonal direction
            # 2. Have NO road in BOTH cardinal directions adjacent to that diagonal
            #
            # Example for NE cutout:
            #   . .     No road at (x+1, y+1), (x+1, y), or (x, y+1)
            #   R .     This road tile R needs NE corner cut
            # ============================================================

            # NE corner cut: no road at NE, E, or N
            if (not is_road(x + 1, y + 1)) and (not is_road(x + 1, y)) and (not is_road(x, y + 1)):
                cutout[y][x] = CUT_NE
                continue

            # NW corner cut: no road at NW, W, or N
            if (not is_road(x - 1, y + 1)) and (not is_road(x - 1, y)) and (not is_road(x, y + 1)):
                cutout[y][x] = CUT_NW
                continue

            # SE corner cut: no road at SE, E, or S
            if (not is_road(x + 1, y - 1)) and (not is_road(x + 1, y)) and (not is_road(x, y - 1)):
                cutout[y][x] = CUT_SE
                continue

            # SW corner cut: no road at SW, W, or S
            if (not is_road(x - 1, y - 1)) and (not is_road(x - 1, y)) and (not is_road(x, y - 1)):
                cutout[y][x] = CUT_SW
                continue

    return cutout


def route_all_roads(
    layout: dict,
    positions: dict[str, tuple[float, float]],
    gate_positions: dict[str, tuple[float, float]],
    anchor_id: str,
    blocked: list[list[bool]],
    road_mask: list[list[bool]],
    road_style: str,
    min_x: float,
    min_y: float,
    grid_w: int,
    grid_h: int
) -> list[tuple[str, list[tuple[int, int]]]]:
    """
    Route all roads for an area layout using three-phase approach.
    
    Phases (in order):
    1. Gate -> anchor (center) trunk roads first (arteries)
    2. Inter-gate connections for multi-gate areas (ring roads)
    3. Building connections from main_connectivity/paths
    """
    placements = layout.get("placements", []) or []
    placement_map = {p["id"]: p for p in placements if p.get("id")}
    
    # Get road plan edges from main_connectivity or paths
    road_plan = layout.get("road_plan") or {}
    building_edges = []
    
    for e in road_plan.get("main_connectivity", []) or []:
        if isinstance(e, dict):
            building_edges.append({
                "from": e.get("from"),
                "to": e.get("to"),
                "kind": e.get("kind", "road")
            })
    
    # If no main_connectivity, use paths
    if not building_edges:
        paths = layout.get("paths", []) or []
        building_edges = dedupe_paths(paths)
    else:
        building_edges = dedupe_paths(building_edges)
    
    road_paths: list[tuple[str, list[tuple[int, int]]]] = []
    
    def get_entity_pos_and_size(entity_id: str) -> tuple[tuple[float, float], tuple[int, int], bool]:
        """
        Get position and footprint size for an entity.
        Returns (position, size, is_gate).
        """
        # Check gates first - they have no blocked footprint
        if entity_id in gate_positions:
            return (gate_positions[entity_id], (1, 1), True)  # Gates are points
        
        # Check positions
        if entity_id in positions:
            pos = positions[entity_id]
            # Get size from placement
            if entity_id in placement_map:
                size = pick_footprint(placement_map[entity_id])
            elif entity_id == anchor_id:
                size = DEFAULT_FOOTPRINT
            else:
                size = (4, 4)
            return (pos, size, False)
        
        # Anchor fallback
        if anchor_id in positions:
            return (positions[anchor_id], DEFAULT_FOOTPRINT, False)
        
        return ((0.0, 0.0), (4, 4), False)
    
    def route_edge(edge: dict) -> tuple[str, list[tuple[int, int]]] | None:
        """Route a single edge and return (kind, path) or None if failed."""
        from_id = edge.get("from", "")
        to_id = edge.get("to", "")
        kind = edge.get("kind", "road")
        
        from_pos, from_size, from_is_gate = get_entity_pos_and_size(from_id)
        to_pos, to_size, to_is_gate = get_entity_pos_and_size(to_id)
        
        # For gates, route directly from the gate position
        # For buildings, use boundary points
        if from_is_gate and to_is_gate:
            # Both are gates - route directly
            start = world_to_grid(from_pos[0], from_pos[1], min_x, min_y)
            goal = world_to_grid(to_pos[0], to_pos[1], min_x, min_y)
        elif from_is_gate:
            # From gate to building
            start = world_to_grid(from_pos[0], from_pos[1], min_x, min_y)
            # Get boundary point on target building facing the gate
            dx = from_pos[0] - to_pos[0]
            dy = from_pos[1] - to_pos[1]
            dir_vec = normalize_vec(dx, dy)
            to_rect = (to_pos[0], to_pos[1], to_size[0], to_size[1])
            to_boundary = get_boundary_point_in_direction(to_rect, dir_vec)
            goal = world_to_grid(to_boundary[0] + dir_vec[0] * 1.5, 
                                 to_boundary[1] + dir_vec[1] * 1.5, min_x, min_y)
        elif to_is_gate:
            # From building to gate
            goal = world_to_grid(to_pos[0], to_pos[1], min_x, min_y)
            # Get boundary point on source building facing the gate
            dx = to_pos[0] - from_pos[0]
            dy = to_pos[1] - from_pos[1]
            dir_vec = normalize_vec(dx, dy)
            from_rect = (from_pos[0], from_pos[1], from_size[0], from_size[1])
            from_boundary = get_boundary_point_in_direction(from_rect, dir_vec)
            start = world_to_grid(from_boundary[0] + dir_vec[0] * 1.5,
                                  from_boundary[1] + dir_vec[1] * 1.5, min_x, min_y)
        else:
            # Both are buildings - use boundary routing
            start, goal = get_routing_endpoints(
                from_pos, from_size, to_pos, to_size, min_x, min_y,
                blocked, grid_w, grid_h
            )
        
        # Find walkable tiles near start/goal
        start = find_walkable_near(start, blocked, grid_w, grid_h)
        goal = find_walkable_near(goal, blocked, grid_w, grid_h)
        
        # Clamp to grid bounds only (let edge penalty handle boundary avoidance)
        start = (max(0, min(grid_w - 1, start[0])), max(0, min(grid_h - 1, start[1])))
        goal = (max(0, min(grid_w - 1, goal[0])), max(0, min(grid_h - 1, goal[1])))
        
        # Route using A*
        path = a_star_with_direction(
            start, goal, grid_w, grid_h, blocked, road_mask, road_style
        )
        
        if path:
            # Widen based on road kind
            width = ROAD_WIDTH.get(kind, 3)
            widen_road_path(path, width, road_mask, blocked, grid_w, grid_h)
            return (kind, path)
        return None
    
    # ============ PHASE 1: Gate -> Center trunk roads (arteries) ============
    # These are the main roads that connect gates to the center anchor
    gate_ids = list(gate_positions.keys())
    for gid in gate_ids:
        edge = {"from": gid, "to": anchor_id, "kind": "trunk_road"}
        result = route_edge(edge)
        if result:
            road_paths.append(result)
    
    # ============ PHASE 2: Inter-gate connections (ring roads) ============
    # For multi-gate areas, connect adjacent gates to form a perimeter road
    if len(gate_ids) >= 2:
        # Sort gates by their position (clockwise around center) for natural ring
        anchor_pos = positions.get(anchor_id, (0.0, 0.0))
        def gate_angle(gid):
            gpos = gate_positions[gid]
            dx = gpos[0] - anchor_pos[0]
            dy = gpos[1] - anchor_pos[1]
            return math.atan2(dy, dx)
        
        sorted_gates = sorted(gate_ids, key=gate_angle)
        
        # Connect each gate to the next (forming a ring)
        for i in range(len(sorted_gates)):
            gid1 = sorted_gates[i]
            gid2 = sorted_gates[(i + 1) % len(sorted_gates)]
            edge = {"from": gid1, "to": gid2, "kind": "road"}
            result = route_edge(edge)
            if result:
                road_paths.append(result)
    
    # ============ PHASE 3: Building connections (from main_connectivity/paths) ============
    # Filter out edges that are already covered by Phase 1 or 2
    routed_pairs = set()
    for gid in gate_ids:
        routed_pairs.add(tuple(sorted([gid, anchor_id])))
    for i in range(len(gate_ids)):
        gid1 = gate_ids[i]
        gid2 = gate_ids[(i + 1) % len(gate_ids)]
        routed_pairs.add(tuple(sorted([gid1, gid2])))
    
    for edge in building_edges:
        from_id = edge.get("from", "")
        to_id = edge.get("to", "")
        pair = tuple(sorted([from_id, to_id]))
        if pair in routed_pairs:
            continue  # Already routed in Phase 1 or 2
        
        result = route_edge(edge)
        if result:
            road_paths.append(result)
            routed_pairs.add(pair)
    
    return road_paths


# ==================== Frontage Spur Routing ====================

def find_best_perimeter_tile(
    rect_grid: tuple[int, int, int, int],
    target: tuple[int, int],
    blocked: list[list[bool]],
    grid_w: int,
    grid_h: int
) -> tuple[int, int]:
    """Find the best walkable perimeter tile closest to target."""
    x, y, w, h = rect_grid
    tx, ty = target
    
    # Get all perimeter tiles just outside the building
    candidates = []
    for xx in range(x - 1, x + w + 1):
        candidates.append((xx, y - 1))      # Top
        candidates.append((xx, y + h))      # Bottom
    for yy in range(y, y + h):
        candidates.append((x - 1, yy))      # Left
        candidates.append((x + w, yy))      # Right
    
    # Find closest walkable tile to target
    best = None
    best_dist = float('inf')
    for cx, cy in candidates:
        if not in_bounds(cx, cy, grid_w, grid_h) or blocked[cy][cx]:
            continue
        dist = abs(cx - tx) + abs(cy - ty)
        if dist < best_dist:
            best_dist = dist
            best = (cx, cy)
    
    return best if best else (x, y - 1)


def route_frontage_spurs(
    positions: dict[str, tuple[float, float]],
    placements: list[dict],
    road_mask: list[list[bool]],
    blocked: list[list[bool]],
    road_style: str,
    min_x: float,
    min_y: float,
    grid_w: int,
    grid_h: int
) -> list[tuple[str, list[tuple[int, int]]]]:
    """
    Route spur roads from frontage buildings that lack adjacent roads.
    For each needs_frontage=True placement without road adjacency:
    1. Find nearest road tile to building perimeter
    2. A* route from building boundary to that road tile
    3. Widen to minimum walkable width (2 tiles)
    4. If A* fails, try direct line carving as fallback
    """
    spur_paths = []
    
    def carve_direct_line(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
        """Bresenham-style line carving as fallback when A* fails."""
        x0, y0 = start
        x1, y1 = end
        path = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        while True:
            if in_bounds(x0, y0, grid_w, grid_h):
                path.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        return path
    
    for p in placements:
        pid = p.get("id")
        if not pid or not p.get("needs_frontage") or pid not in positions:
            continue
        
        pos = positions[pid]
        gx, gy = world_to_grid(pos[0], pos[1], min_x, min_y)
        pw, ph = pick_footprint(p)
        rect_grid = (gx - pw // 2, gy - ph // 2, pw, ph)
        
        # Skip if already has adjacent road
        if has_adjacent_road(rect_grid, road_mask, grid_w, grid_h, max_distance=2):
            continue
        
        # Find nearest road tile
        road_tile = find_nearest_road_tile(rect_grid, road_mask, grid_w, grid_h)
        if not road_tile:
            continue
        
        # Find best building perimeter tile closest to road
        start = find_best_perimeter_tile(rect_grid, road_tile, blocked, grid_w, grid_h)
        if not start:
            continue
        
        # Route spur from building to road
        path = a_star_with_direction(
            start, road_tile, grid_w, grid_h,
            blocked, road_mask, road_style
        )
        
        # Fallback: if A* fails, try direct line carving
        if not path:
            path = carve_direct_line(start, road_tile)
        
        if path:
            widen_road_path(path, 2, road_mask, blocked, grid_w, grid_h)  # min walkable width
            spur_paths.append(("footpath", path))
    
    return spur_paths


# ==================== Gate Connectivity Enforcement ====================

def find_road_components(
    road_mask: list[list[bool]],
    grid_w: int,
    grid_h: int,
    road_style: str = "organic"
) -> tuple[list[list[int]], int]:
    """
    Find connected components of road tiles using BFS.
    Returns (component_grid, num_components) where component_grid[y][x] = component_id (0 = no road).
    """
    config = ROAD_STYLE_CONFIG.get(road_style, ROAD_STYLE_CONFIG["organic"])
    directions = config["directions"] if config else DIR_4
    
    component_grid = [[0 for _ in range(grid_w)] for _ in range(grid_h)]
    num_components = 0
    
    for start_y in range(grid_h):
        for start_x in range(grid_w):
            if not road_mask[start_y][start_x] or component_grid[start_y][start_x] != 0:
                continue
            
            # BFS from this unvisited road tile
            num_components += 1
            queue = deque([(start_x, start_y)])
            component_grid[start_y][start_x] = num_components
            
            while queue:
                x, y = queue.popleft()
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if in_bounds(nx, ny, grid_w, grid_h) and road_mask[ny][nx] and component_grid[ny][nx] == 0:
                        component_grid[ny][nx] = num_components
                        queue.append((nx, ny))
    
    return component_grid, num_components


def find_nearest_road_in_component(
    point: tuple[int, int],
    component_grid: list[list[int]],
    target_component: int,
    grid_w: int,
    grid_h: int,
    max_search: int = 50
) -> tuple[int, int] | None:
    """Find the nearest road tile in a specific component."""
    px, py = point
    
    for radius in range(max_search):
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if abs(dx) != radius and abs(dy) != radius:
                    continue
                nx, ny = px + dx, py + dy
                if in_bounds(nx, ny, grid_w, grid_h) and component_grid[ny][nx] == target_component:
                    return (nx, ny)
    return None


def ensure_gate_connectivity(
    gate_tiles: list[tuple[int, int]],
    road_mask: list[list[bool]],
    blocked: list[list[bool]],
    grid_w: int,
    grid_h: int,
    road_style: str
) -> list[tuple[str, list[tuple[int, int]]]]:
    """
    Post-pass: ensure all gates are in one connected road component.
    
    Algorithm:
    1. Find road connected components using BFS
    2. For each gate, find its nearest road tile and map to a component
    3. If gates are in different components, find closest pair of tiles between components
    4. Route between them with A* (or direct carve if A* fails)
    5. Repeat until all gate-reachable components are merged
    
    Returns list of (kind, path) for any new roads added.
    """
    if len(gate_tiles) <= 1:
        return []
    
    new_paths = []
    max_iterations = len(gate_tiles) + 5  # Safety limit
    
    def carve_direct_line(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
        """Bresenham-style line carving as fallback when A* fails."""
        x0, y0 = start
        x1, y1 = end
        path = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        while True:
            if in_bounds(x0, y0, grid_w, grid_h):
                path.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        return path
    
    for iteration in range(max_iterations):
        # Find current road components
        component_grid, num_components = find_road_components(road_mask, grid_w, grid_h, road_style)
        
        if num_components <= 1:
            break  # All roads are already connected (or no roads)
        
        # Map each gate to its nearest road component
        gate_components: dict[int, list[tuple[int, int]]] = {}  # component_id -> list of gates
        for gx, gy in gate_tiles:
            # Find nearest road tile to this gate
            best_comp = 0
            best_dist = float('inf')
            for radius in range(30):
                found = False
                for dx in range(-radius, radius + 1):
                    for dy in range(-radius, radius + 1):
                        nx, ny = gx + dx, gy + dy
                        if in_bounds(nx, ny, grid_w, grid_h) and component_grid[ny][nx] > 0:
                            dist = abs(dx) + abs(dy)
                            if dist < best_dist:
                                best_dist = dist
                                best_comp = component_grid[ny][nx]
                                found = True
                if found and radius > 0:
                    break  # Found at this radius, no need to search further
            
            if best_comp > 0:
                if best_comp not in gate_components:
                    gate_components[best_comp] = []
                gate_components[best_comp].append((gx, gy))
        
        # If all gates are in the same component (or no gates reached roads), we're done
        if len(gate_components) <= 1:
            break
        
        # Find two components with gates and connect them
        comp_ids = list(gate_components.keys())
        comp1, comp2 = comp_ids[0], comp_ids[1]
        
        # Find closest pair of road tiles between the two components
        best_pair = None
        best_dist = float('inf')
        
        for y in range(grid_h):
            for x in range(grid_w):
                if component_grid[y][x] == comp1:
                    # Find nearest tile in comp2
                    nearest = find_nearest_road_in_component((x, y), component_grid, comp2, grid_w, grid_h)
                    if nearest:
                        dist = abs(x - nearest[0]) + abs(y - nearest[1])
                        if dist < best_dist:
                            best_dist = dist
                            best_pair = ((x, y), nearest)
        
        if not best_pair:
            break  # Can't find connection points
        
        start, goal = best_pair
        
        # Try A* first
        path = a_star_with_direction(start, goal, grid_w, grid_h, blocked, road_mask, road_style)
        
        # Fallback to direct line if A* fails
        if not path:
            path = carve_direct_line(start, goal)
        
        if path:
            widen_road_path(path, 2, road_mask, blocked, grid_w, grid_h)
            new_paths.append(("road", path))
    
    return new_paths


# ==================== Pinch Point Widening ====================

def detect_pinch_points(
    road_mask: list[list[bool]],
    grid_w: int,
    grid_h: int
) -> list[tuple[int, int]]:
    """
    Detect single-tile corridor points (pinch points) in the road network.
    A tile is a pinch if it's a road but has road neighbors only in opposing directions
    (e.g., only E-W or only N-S) with no perpendicular neighbors.
    """
    pinch_points = []
    
    for y in range(grid_h):
        for x in range(grid_w):
            if not road_mask[y][x]:
                continue
            
            # Count neighbors in cardinal directions
            has_n = y > 0 and road_mask[y-1][x]
            has_s = y < grid_h - 1 and road_mask[y+1][x]
            has_e = x < grid_w - 1 and road_mask[y][x+1]
            has_w = x > 0 and road_mask[y][x-1]
            
            # Check for pinch patterns (corridor with no perpendicular exits)
            is_vertical_corridor = (has_n or has_s) and not (has_e or has_w)
            is_horizontal_corridor = (has_e or has_w) and not (has_n or has_s)
            
            # Also check if this is a dead-end or isolated (even worse)
            neighbor_count = sum([has_n, has_s, has_e, has_w])
            
            if is_vertical_corridor or is_horizontal_corridor or neighbor_count <= 1:
                pinch_points.append((x, y))
    
    return pinch_points


def widen_pinch_points(
    road_mask: list[list[bool]],
    blocked: list[list[bool]],
    grid_w: int,
    grid_h: int,
    min_width: int = 2
) -> int:
    """
    Post-pass to widen single-tile corridors (pinch points) for playability.
    
    Algorithm:
    1. Detect all pinch points
    2. For each pinch, try to add adjacent road tiles in perpendicular directions
    3. Only add if the target tile is not blocked
    
    Returns the number of pinch points that were widened.
    """
    pinch_points = detect_pinch_points(road_mask, grid_w, grid_h)
    widened_count = 0
    
    for x, y in pinch_points:
        # Determine which direction the corridor runs
        has_n = y > 0 and road_mask[y-1][x]
        has_s = y < grid_h - 1 and road_mask[y+1][x]
        has_e = x < grid_w - 1 and road_mask[y][x+1]
        has_w = x > 0 and road_mask[y][x-1]
        
        # For vertical corridors (N-S), try to widen E or W
        if (has_n or has_s) and not (has_e and has_w):
            widened = False
            # Try east first
            if x < grid_w - 1 and not blocked[y][x+1] and not road_mask[y][x+1]:
                road_mask[y][x+1] = True
                widened = True
            # Try west if east didn't work
            if not widened and x > 0 and not blocked[y][x-1] and not road_mask[y][x-1]:
                road_mask[y][x-1] = True
                widened = True
            if widened:
                widened_count += 1
        
        # For horizontal corridors (E-W), try to widen N or S
        elif (has_e or has_w) and not (has_n and has_s):
            widened = False
            # Try south first
            if y < grid_h - 1 and not blocked[y+1][x] and not road_mask[y+1][x]:
                road_mask[y+1][x] = True
                widened = True
            # Try north if south didn't work
            if not widened and y > 0 and not blocked[y-1][x] and not road_mask[y-1][x]:
                road_mask[y-1][x] = True
                widened = True
            if widened:
                widened_count += 1
        
        # For dead-ends or isolated tiles, try any direction
        else:
            neighbor_count = sum([has_n, has_s, has_e, has_w])
            if neighbor_count <= 1:
                for dx, dy in DIR_4:
                    nx, ny = x + dx, y + dy
                    if in_bounds(nx, ny, grid_w, grid_h) and not blocked[ny][nx] and not road_mask[ny][nx]:
                        road_mask[ny][nx] = True
                        widened_count += 1
                        break
    
    return widened_count


# ==================== Road-Safe Constrained Relaxation ====================

# Drift caps by entity kind (in tiles)
DRIFT_CAPS = {
    "landmark": 2,
    "building": 8,
    "prop": 15,
    "nature": 10,
}
DEFAULT_DRIFT_CAP = 5


def rect_intersects_road(
    rect_grid: tuple[int, int, int, int],
    road_mask: list[list[bool]],
    grid_w: int,
    grid_h: int,
    clearance: int = 1
) -> bool:
    """
    Check if a building rectangle (x, y, w, h) intersects any road tile
    or comes within 'clearance' tiles of a road.
    Returns True if collision detected.
    """
    x, y, w, h = rect_grid
    for yy in range(max(0, y - clearance), min(grid_h, y + h + clearance)):
        for xx in range(max(0, x - clearance), min(grid_w, x + w + clearance)):
            if road_mask[yy][xx]:
                return True
    return False


def constrained_relaxation(
    positions: dict[str, tuple[float, float]],
    placements: list[dict],
    road_mask: list[list[bool]],
    blocked: list[list[bool]],
    anchor_id: str,
    min_x: float,
    min_y: float,
    grid_w: int,
    grid_h: int,
    iterations: int = 10,
    road_clearance: int = 1
) -> dict[str, tuple[float, float]]:
    """
    Road-safe constrained relaxation to align buildings with roads naturally.
    
    ROAD-SAFE INVARIANT: A nudge is only accepted if the new building rect
    does NOT intersect road_mask tiles (dilated by road_clearance).
    If it would, reject the nudge or reduce step size until valid.
    
    Drift caps by kind:
    - landmarks: 0-2 tiles
    - buildings: 3-8 tiles
    - props: 8-15 tiles
    
    Per iteration:
    1. For each needs_frontage building not on road: snap toward nearest road (setback 1-2)
    2. For each building causing pinch: nudge 1 tile away
    3. Add tiny random jitter (+-1 tile) for organic feel
    4. BEFORE accepting any nudge: check rect_intersects_road(new_rect, road_mask, road_clearance)
       - If intersects, reject or halve step size and retry
    """
    new_positions = positions.copy()
    
    # Track original positions for drift cap enforcement
    original_positions = {pid: pos for pid, pos in positions.items()}
    
    # Build placement lookup
    placement_map = {p.get("id"): p for p in placements if p.get("id")}
    
    def get_drift_cap(placement: dict) -> float:
        kind = (placement.get("kind") or "").lower()
        return float(DRIFT_CAPS.get(kind, DEFAULT_DRIFT_CAP))
    
    def compute_new_rect(pid: str, new_pos: tuple[float, float]) -> tuple[int, int, int, int]:
        """Compute grid rect for entity at new position."""
        gx, gy = world_to_grid(new_pos[0], new_pos[1], min_x, min_y)
        if pid in placement_map:
            pw, ph = pick_footprint(placement_map[pid])
        else:
            pw, ph = DEFAULT_FOOTPRINT
        return (gx - pw // 2, gy - ph // 2, pw, ph)
    
    def is_valid_nudge(pid: str, new_pos: tuple[float, float], drift_cap: float) -> bool:
        """Check if nudge is valid: within drift cap and doesn't intersect roads."""
        # Check drift cap
        if pid in original_positions:
            orig = original_positions[pid]
            drift = math.sqrt((new_pos[0] - orig[0])**2 + (new_pos[1] - orig[1])**2)
            if drift > drift_cap:
                return False
        
        # Check road intersection
        rect = compute_new_rect(pid, new_pos)
        if rect_intersects_road(rect, road_mask, grid_w, grid_h, road_clearance):
            return False
        
        return True
    
    def try_nudge(pid: str, direction: tuple[float, float], base_step: float, drift_cap: float) -> tuple[float, float]:
        """Try to nudge entity in direction, halving step until valid or giving up."""
        current_pos = new_positions[pid]
        step = base_step
        
        for _ in range(5):  # Max 5 halvings
            new_pos = (current_pos[0] + direction[0] * step, 
                       current_pos[1] + direction[1] * step)
            
            if is_valid_nudge(pid, new_pos, drift_cap):
                return new_pos
            
            step /= 2
            if step < 0.5:
                break
        
        return current_pos  # No valid nudge found
    
    # Use deterministic "randomness" based on entity ID for jitter
    def stable_jitter(pid: str, magnitude: float = 1.0) -> tuple[float, float]:
        h = int(hashlib.md5(pid.encode("utf-8")).hexdigest(), 16)
        jx = ((h % 100) / 50.0 - 1.0) * magnitude
        jy = (((h // 100) % 100) / 50.0 - 1.0) * magnitude
        return (jx, jy)
    
    for iteration in range(iterations):
        changes_made = 0
        
        for p in placements:
            pid = p.get("id")
            if not pid or pid not in new_positions or pid == anchor_id:
                continue
            
            drift_cap = get_drift_cap(p)
            pos = new_positions[pid]
            gx, gy = world_to_grid(pos[0], pos[1], min_x, min_y)
            pw, ph = pick_footprint(p)
            rect_grid = (gx - pw // 2, gy - ph // 2, pw, ph)
            
            # 1. For frontage buildings not on road: snap toward nearest road
            if p.get("needs_frontage"):
                road_tile = find_nearest_road_tile(rect_grid, road_mask, grid_w, grid_h)
                if road_tile:
                    rx, ry = road_tile
                    # Direction from building center to road
                    dx = rx - gx
                    dy = ry - gy
                    dist = math.sqrt(dx*dx + dy*dy)
                    if dist > 2:  # Only nudge if road is more than 2 tiles away
                        direction = (dx / dist, dy / dist)
                        new_pos = try_nudge(pid, direction, 2.0, drift_cap)
                        if new_pos != pos:
                            new_positions[pid] = new_pos
                            changes_made += 1
                            continue
            
            # 2. Add tiny jitter for organic feel (only on last iteration)
            if iteration == iterations - 1:
                jx, jy = stable_jitter(pid, 0.5)
                new_pos = (pos[0] + jx, pos[1] + jy)
                if is_valid_nudge(pid, new_pos, drift_cap):
                    new_positions[pid] = new_pos
                    changes_made += 1
        
        # Early termination if no changes
        if changes_made == 0 and iteration > 0:
            break
    
    return new_positions


# ==================== Occupancy Grid ====================

def create_occupancy_grid(
    positions: dict[str, tuple[float, float]],
    placements: list[dict],
    anchor_id: str,
    min_x: float,
    min_y: float,
    grid_w: int,
    grid_h: int,
    margin: int = 1
) -> list[list[bool]]:
    """
    Create occupancy grid marking building footprints as blocked.
    Uses world_to_grid transform for coordinate conversion.
    
    Uses variable margins based on entity size and frontage needs:
    - Small frontage props (area <= 4, needs_frontage): pad=0 (roads can get close)
    - Large buildings (area > 16): pad=2
    - Everything else: pad=1
    """
    blocked = [[False for _ in range(grid_w)] for _ in range(grid_h)]
    
    def mark_rect(x: int, y: int, w: int, h: int, pad: int = 1):
        """Mark a rectangle as blocked with padding."""
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(grid_w, x + w + pad)
        y1 = min(grid_h, y + h + pad)
        for yy in range(y0, y1):
            for xx in range(x0, x1):
                blocked[yy][xx] = True
    
    # Mark anchor (smaller margin for anchor to allow roads to reach it)
    if anchor_id in positions:
        ax, ay = positions[anchor_id]
        gx, gy = world_to_grid(ax, ay, min_x, min_y)
        aw, ah = DEFAULT_FOOTPRINT
        # Mark from top-left corner with minimal margin
        mark_rect(gx - aw // 2, gy - ah // 2, aw, ah, 0)
    
    # Mark all placements with variable padding based on size and frontage
    for p in placements:
        pid = p.get("id")
        if not pid or pid not in positions:
            continue
        px, py = positions[pid]
        gx, gy = world_to_grid(px, py, min_x, min_y)
        pw, ph = pick_footprint(p)
        area = pw * ph
        
        # Variable padding based on size and frontage need
        # CRITICAL: needs_frontage buildings cap at pad=1 so roads can reach them
        if p.get("needs_frontage"):
            pad = 0 if area <= 4 else 1  # Cap at 1 for ALL frontage buildings
        elif area > 16:
            pad = 2  # Large non-frontage buildings keep margin
        else:
            pad = 1  # Default for medium entities
        
        mark_rect(gx - pw // 2, gy - ph // 2, pw, ph, pad)
    
    return blocked


# ==================== Frontage Snapping ====================

def find_nearest_road_tile(
    rect_grid: tuple[int, int, int, int],
    road_mask: list[list[bool]],
    grid_w: int,
    grid_h: int
) -> tuple[int, int] | None:
    """
    Find the nearest road tile to a building rectangle.
    rect_grid: (x, y, w, h) in grid coordinates (x, y is top-left corner)
    """
    x, y, w, h = rect_grid
    
    # Get perimeter tiles
    perimeter = []
    for xx in range(x, x + w):
        perimeter.append((xx, y - 1))      # Top
        perimeter.append((xx, y + h))      # Bottom
    for yy in range(y, y + h):
        perimeter.append((x - 1, yy))      # Left
        perimeter.append((x + w, yy))      # Right
    
    # Find nearest road tile by BFS-like search
    best_road = None
    best_dist = float('inf')
    
    # Check perimeter first (distance 1)
    for px, py in perimeter:
        if in_bounds(px, py, grid_w, grid_h) and road_mask[py][px]:
            return (px, py)
    
    # Expand search radius
    cx, cy = x + w // 2, y + h // 2
    for radius in range(1, 30):
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if abs(dx) != radius and abs(dy) != radius:
                    continue
                nx, ny = cx + dx, cy + dy
                if in_bounds(nx, ny, grid_w, grid_h) and road_mask[ny][nx]:
                    dist = abs(dx) + abs(dy)
                    if dist < best_dist:
                        best_dist = dist
                        best_road = (nx, ny)
        if best_road:
            return best_road
    
    return None


def snap_frontage_to_road(
    entity_id: str,
    positions: dict[str, tuple[float, float]],
    placements: list[dict],
    road_mask: list[list[bool]],
    min_x: float,
    min_y: float,
    grid_w: int,
    grid_h: int,
    setback: int = 1
) -> tuple[float, float]:
    """
    Snap a frontage building to be adjacent to the nearest road.
    Returns the new world position.
    """
    if entity_id not in positions:
        return positions.get(entity_id, (0.0, 0.0))
    
    # Get current position and size
    pos = positions[entity_id]
    placement = next((p for p in placements if p.get("id") == entity_id), None)
    if not placement:
        return pos
    
    pw, ph = pick_footprint(placement)
    gx, gy = world_to_grid(pos[0], pos[1], min_x, min_y)
    
    # Current rect in grid coords (top-left x, y, width, height)
    rect_grid = (gx - pw // 2, gy - ph // 2, pw, ph)
    
    # Find nearest road
    road_tile = find_nearest_road_tile(rect_grid, road_mask, grid_w, grid_h)
    if not road_tile:
        return pos  # No road found, keep original position
    
    rx, ry = road_tile
    
    # Determine which side of the building the road is on
    # and compute translation to snap building to road
    cx, cy = gx, gy  # Building center
    
    # Direction from building center to road
    dx = rx - cx
    dy = ry - cy
    
    # Compute how much to move the building
    # We want the building edge to be 'setback' tiles from the road
    move_x = 0.0
    move_y = 0.0
    
    if abs(dx) > abs(dy):
        # Road is primarily to the left or right
        if dx > 0:
            # Road is to the right, move building right
            target_x = rx - pw // 2 - setback
            move_x = target_x - cx
        else:
            # Road is to the left, move building left
            target_x = rx + pw // 2 + setback
            move_x = target_x - cx
    else:
        # Road is primarily above or below
        if dy > 0:
            # Road is below, move building down
            target_y = ry - ph // 2 - setback
            move_y = target_y - cy
        else:
            # Road is above, move building up
            target_y = ry + ph // 2 + setback
            move_y = target_y - cy
    
    # Limit movement to avoid drastic repositioning
    max_move = 5.0
    move_x = max(-max_move, min(max_move, move_x))
    move_y = max(-max_move, min(max_move, move_y))
    
    # Compute new world position
    new_pos = (pos[0] + move_x, pos[1] + move_y)
    return new_pos


def snap_all_frontage(
    positions: dict[str, tuple[float, float]],
    placements: list[dict],
    road_mask: list[list[bool]],
    min_x: float,
    min_y: float,
    grid_w: int,
    grid_h: int
) -> dict[str, tuple[float, float]]:
    """Snap all frontage buildings to nearest roads."""
    new_positions = positions.copy()
    
    for p in placements:
        pid = p.get("id")
        if not pid or not p.get("needs_frontage"):
            continue
        
        new_pos = snap_frontage_to_road(
            pid, new_positions, placements, road_mask,
            min_x, min_y, grid_w, grid_h
        )
        new_positions[pid] = new_pos
    
    return new_positions


# ==================== Diagonal Frontage Rotation ====================

def detect_diagonal_road_adjacency(
    road_mask: list[list[bool]],
    x: int, y: int,
    w: int, h: int,
    grid_w: int,
    grid_h: int
) -> tuple[bool, tuple[int, int] | None]:
    """
    Detect if a building has diagonal road adjacency.
    Returns (has_diagonal, road_vec) where road_vec is (1,1), (1,-1), (-1,1), or (-1,-1).
    
    x, y: top-left corner of building in grid coords
    w, h: building width and height
    """
    # First, determine which edge faces the road (using existing logic)
    edge_road_count = {"N": 0, "S": 0, "E": 0, "W": 0}
    
    # Count road tiles adjacent to each edge
    for xx in range(x, x + w):
        if y > 0 and in_bounds(xx, y - 1, grid_w, grid_h) and road_mask[y - 1][xx]:
            edge_road_count["N"] += 1
        if in_bounds(xx, y + h, grid_w, grid_h) and road_mask[y + h][xx]:
            edge_road_count["S"] += 1
    
    for yy in range(y, y + h):
        if in_bounds(x + w, yy, grid_w, grid_h) and road_mask[yy][x + w]:
            edge_road_count["E"] += 1
        if x > 0 and in_bounds(x - 1, yy, grid_w, grid_h) and road_mask[yy][x - 1]:
            edge_road_count["W"] += 1
    
    # Find the edge with the most road tiles (frontage edge)
    best_edge = max(edge_road_count, key=edge_road_count.get)
    if edge_road_count[best_edge] == 0:
        return (False, None)
    
    # Get frontage edge tiles
    frontage_tiles = []
    if best_edge == "N":
        frontage_tiles = [(xx, y - 1) for xx in range(x, x + w) if y > 0]
    elif best_edge == "S":
        frontage_tiles = [(xx, y + h) for xx in range(x, x + w)]
    elif best_edge == "E":
        frontage_tiles = [(x + w, yy) for yy in range(y, y + h)]
    elif best_edge == "W":
        frontage_tiles = [(x - 1, yy) for yy in range(y, y + h) if x > 0]
    
    # Check diagonal neighbors of frontage edge tiles
    diagonal_dirs = [(1, 1), (1, -1), (-1, 1), (-1, -1)]  # NE, SE, NW, SW
    diagonal_counts = {(1, 1): 0, (1, -1): 0, (-1, 1): 0, (-1, -1): 0}
    
    for fx, fy in frontage_tiles:
        for dx, dy in diagonal_dirs:
            nx, ny = fx + dx, fy + dy
            if in_bounds(nx, ny, grid_w, grid_h) and road_mask[ny][nx]:
                diagonal_counts[(dx, dy)] += 1
    
    # Find dominant diagonal direction
    max_count = max(diagonal_counts.values())
    if max_count == 0:
        return (False, None)
    
    # Get the dominant direction(s)
    dominant_dirs = [d for d, count in diagonal_counts.items() if count == max_count]
    
    # If multiple, prefer based on frontage edge
    if len(dominant_dirs) > 1:
        # Prefer directions that make sense with the frontage edge
        if best_edge == "N":
            # Prefer NW or NE
            preferred = [d for d in dominant_dirs if d[1] > 0]  # North component
            if preferred:
                dominant_dirs = preferred
        elif best_edge == "S":
            # Prefer SW or SE
            preferred = [d for d in dominant_dirs if d[1] < 0]  # South component
            if preferred:
                dominant_dirs = preferred
        elif best_edge == "E":
            # Prefer NE or SE
            preferred = [d for d in dominant_dirs if d[0] > 0]  # East component
            if preferred:
                dominant_dirs = preferred
        elif best_edge == "W":
            # Prefer NW or SW
            preferred = [d for d in dominant_dirs if d[0] < 0]  # West component
            if preferred:
                dominant_dirs = preferred
    
    # Return first dominant direction
    road_vec = dominant_dirs[0]
    return (True, road_vec)


def choose_diagonal_rotation(road_vec: tuple[int, int]) -> tuple[int, tuple[int, int]]:
    """
    Choose rotation angle based on road direction vector.
    Returns (rotation_deg, facing_vec).
    """
    dx, dy = road_vec
    
    # Normalize to signs
    sign_dx = 1 if dx > 0 else -1 if dx < 0 else 0
    sign_dy = 1 if dy > 0 else -1 if dy < 0 else 0
    facing_vec = (sign_dx, sign_dy)
    
    # Choose rotation: (1,1) or (-1,-1) -> 45°, (1,-1) or (-1,1) -> 135°
    if (dx > 0 and dy > 0) or (dx < 0 and dy < 0):
        rotation_deg = 45
    else:
        rotation_deg = 135
    
    return (rotation_deg, facing_vec)


def get_rotated_footprint(
    placement: dict,
    rotation_deg: int,
    pos: tuple[float, float],
    min_x: float,
    min_y: float
) -> tuple[int, int, int, int]:
    """
    Compute footprint after rotation.
    Returns (x0, y0, w, h) in grid coordinates.
    
    For diagonal rotations (45°/135°), keep axis-aligned bounding box with optional buffer.
    For cardinal rotations (90°/270°), swap w/h.
    """
    pw, ph = pick_footprint(placement)
    
    # Convert position to grid
    gx, gy = world_to_grid(pos[0], pos[1], min_x, min_y)
    
    if rotation_deg in (45, 135):
        # Diagonal rotation: keep axis-aligned, optionally enlarge by 1 tile for safety
        new_w = pw + 1
        new_h = ph + 1
    elif rotation_deg in (90, 270):
        # Cardinal rotation: swap dimensions
        new_w = ph
        new_h = pw
    else:
        # 0° or 180°: no change
        new_w = pw
        new_h = ph
    
    # Top-left corner
    x0 = gx - new_w // 2
    y0 = gy - new_h // 2
    
    return (x0, y0, new_w, new_h)


def get_footprint_tiles(x0: int, y0: int, w: int, h: int, grid_w: int, grid_h: int) -> set[tuple[int, int]]:
    """Get set of all grid tiles occupied by a footprint."""
    tiles = set()
    for yy in range(y0, y0 + h):
        for xx in range(x0, x0 + w):
            if in_bounds(xx, yy, grid_w, grid_h):
                tiles.add((xx, yy))
    return tiles


def check_footprint_overlap(
    footprint_tiles: set[tuple[int, int]],
    occupied_tiles: set[tuple[int, int]],
    road_mask: list[list[bool]],
    grid_w: int,
    grid_h: int
) -> tuple[bool, list[tuple[int, int]]]:
    """
    Check if footprint overlaps with occupied tiles (excluding roads).
    Returns (has_overlap, overlapping_tiles).
    """
    overlapping = []
    for tile in footprint_tiles:
        x, y = tile
        if not in_bounds(x, y, grid_w, grid_h):
            continue
        # Roads don't count as overlap
        if road_mask[y][x]:
            continue
        # Check if tile is occupied by another building/prop
        if tile in occupied_tiles:
            overlapping.append(tile)
    
    return (len(overlapping) > 0, overlapping)


def find_nearest_road_distance(
    pos: tuple[float, float],
    road_mask: list[list[bool]],
    min_x: float,
    min_y: float,
    grid_w: int,
    grid_h: int,
    max_search: int = 10
) -> float:
    """Find distance from position to nearest road tile."""
    gx, gy = world_to_grid(pos[0], pos[1], min_x, min_y)
    
    for radius in range(max_search + 1):
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if abs(dx) != radius and abs(dy) != radius:
                    continue
                nx, ny = gx + dx, gy + dy
                if in_bounds(nx, ny, grid_w, grid_h) and road_mask[ny][nx]:
                    return math.sqrt(dx*dx + dy*dy)
    
    return float('inf')


def resolve_diagonal_overlap(
    pid: str,
    placement: dict,
    pos: tuple[float, float],
    rotation_deg: int,
    facing_vec: tuple[int, int],
    occupied_tiles: set[tuple[int, int]],
    road_mask: list[list[bool]],
    min_x: float,
    min_y: float,
    grid_w: int,
    grid_h: int,
    anchor_pos: tuple[float, float]
) -> tuple[tuple[float, float], int, bool, str]:
    """
    Resolve overlap by sliding along road direction, then nudging away.
    Returns (new_pos, final_rotation_deg, was_adjusted, reason).
    """
    tangent = facing_vec
    dx, dy = tangent
    
    # 6A. Slide along road direction
    for k in range(1, 13):  # Try k in [1..12]
        for direction in [1, -1]:
            offset = k * direction
            new_pos = (pos[0] + dx * offset, pos[1] + dy * offset)
            
            # Check if still near road
            road_dist = find_nearest_road_distance(new_pos, road_mask, min_x, min_y, grid_w, grid_h)
            if road_dist > 2.0:
                continue
            
            # Recompute footprint
            x0, y0, w, h = get_rotated_footprint(placement, rotation_deg, new_pos, min_x, min_y)
            footprint_tiles = get_footprint_tiles(x0, y0, w, h, grid_w, grid_h)
            
            # Check overlap
            has_overlap, _ = check_footprint_overlap(footprint_tiles, occupied_tiles, road_mask, grid_w, grid_h)
            if not has_overlap:
                return (new_pos, rotation_deg, True, f"slid_{direction * k}_along_road")
    
    # 6B. Nudge away from road normal
    # Compute normal vectors
    if tangent == (1, 1) or tangent == (-1, -1):
        normals = [(1, -1), (-1, 1)]
    else:  # (1, -1) or (-1, 1)
        normals = [(1, 1), (-1, -1)]
    
    best_candidate = None
    best_score = (float('inf'), float('inf'), float('inf'))  # (overlap_count, dist_from_anchor, dist_to_road)
    
    for normal in normals:
        ndx, ndy = normal
        for n in range(1, 5):  # Try n in [1..4]
            new_pos = (pos[0] + ndx * n, pos[1] + ndy * n)
            
            # Recompute footprint
            x0, y0, w, h = get_rotated_footprint(placement, rotation_deg, new_pos, min_x, min_y)
            footprint_tiles = get_footprint_tiles(x0, y0, w, h, grid_w, grid_h)
            
            # Check overlap
            has_overlap, overlapping = check_footprint_overlap(footprint_tiles, occupied_tiles, road_mask, grid_w, grid_h)
            overlap_count = len(overlapping)
            
            # Compute distances
            dist_from_anchor = math.sqrt((new_pos[0] - anchor_pos[0])**2 + (new_pos[1] - anchor_pos[1])**2)
            road_dist = find_nearest_road_distance(new_pos, road_mask, min_x, min_y, grid_w, grid_h)
            
            score = (overlap_count, dist_from_anchor, road_dist)
            if score < best_score:
                best_score = score
                best_candidate = (new_pos, n, normal)
    
    if best_candidate and best_score[0] == 0:  # No overlap
        new_pos, n, normal = best_candidate
        return (new_pos, rotation_deg, True, f"nudged_{n}_away_from_road")
    
    # 6C. Fallback: revert to cardinal rotation
    # Use original facing_yaw logic to determine cardinal direction
    gx, gy = world_to_grid(pos[0], pos[1], min_x, min_y)
    pw, ph = pick_footprint(placement)
    x0, y0 = gx - pw // 2, gy - ph // 2
    facing_yaw = compute_facing_yaw(road_mask, x0, y0, pw, ph, grid_w, grid_h)
    
    return (pos, facing_yaw, False, "reverted_to_cardinal_rotation")


def apply_diagonal_rotations(
    placements: list[dict],
    positions: dict[str, tuple[float, float]],
    road_mask: list[list[bool]],
    blocked: list[list[bool]],
    min_x: float,
    min_y: float,
    grid_w: int,
    grid_h: int
) -> tuple[dict[str, tuple[float, float]], dict[str, dict], list[str]]:
    """
    Apply diagonal rotations to frontage assets adjacent to diagonal roads.
    Returns (updated_positions, rotation_metadata, debug_logs).
    
    rotation_metadata: pid -> {rotation_deg, facing_vec, anchor_pos, was_adjusted, reason}
    """
    updated_positions = positions.copy()
    rotation_metadata = {}
    debug_logs = []
    
    # Build occupied tiles set (from blocked grid, excluding roads)
    occupied_tiles = set()
    for y in range(grid_h):
        for x in range(grid_w):
            if blocked[y][x] and not road_mask[y][x]:
                occupied_tiles.add((x, y))
    
    # Process frontage assets
    for p in placements:
        pid = p.get("id")
        if not pid or not p.get("needs_frontage") or pid not in positions:
            continue
        
        pos = positions[pid]
        pw, ph = pick_footprint(p)
        gx, gy = world_to_grid(pos[0], pos[1], min_x, min_y)
        x0, y0 = gx - pw // 2, gy - ph // 2
        
        # Detect diagonal road adjacency
        has_diagonal, road_vec = detect_diagonal_road_adjacency(
            road_mask, x0, y0, pw, ph, grid_w, grid_h
        )
        
        if not has_diagonal or road_vec is None:
            # No diagonal road, use existing facing logic
            facing_yaw = compute_facing_yaw(road_mask, x0, y0, pw, ph, grid_w, grid_h)
            rotation_metadata[pid] = {
                "rotation_deg": facing_yaw,
                "facing_vec": None,
                "anchor_pos": pos,
                "was_adjusted": False,
                "reason": "no_diagonal_road"
            }
            continue
        
        # Choose diagonal rotation
        rotation_deg, facing_vec = choose_diagonal_rotation(road_vec)
        
        # Compute rotated footprint
        x0_rot, y0_rot, w_rot, h_rot = get_rotated_footprint(p, rotation_deg, pos, min_x, min_y)
        footprint_tiles = get_footprint_tiles(x0_rot, y0_rot, w_rot, h_rot, grid_w, grid_h)
        
        # Remove old footprint from occupied (before checking overlap)
        old_footprint = get_footprint_tiles(x0, y0, pw, ph, grid_w, grid_h)
        for tile in old_footprint:
            if tile in occupied_tiles:
                occupied_tiles.remove(tile)
        
        # Check for overlap with updated occupied_tiles
        has_overlap, overlapping = check_footprint_overlap(footprint_tiles, occupied_tiles, road_mask, grid_w, grid_h)
        
        if has_overlap:
            # Resolve overlap
            new_pos, final_rotation, was_adjusted, reason = resolve_diagonal_overlap(
                pid, p, pos, rotation_deg, facing_vec,
                occupied_tiles, road_mask, min_x, min_y, grid_w, grid_h, pos
            )
            
            updated_positions[pid] = new_pos
            rotation_deg = final_rotation
            
            # Recompute footprint from final position
            final_x0, final_y0, final_w, final_h = get_rotated_footprint(p, rotation_deg, new_pos, min_x, min_y)
            final_footprint = get_footprint_tiles(final_x0, final_y0, final_w, final_h, grid_w, grid_h)
        else:
            # No overlap, use original position and rotated footprint
            new_pos = pos
            updated_positions[pid] = new_pos  # Ensure position is set
            final_footprint = footprint_tiles
            was_adjusted = False
            reason = "no_overlap"
        
        # Add final footprint to occupied tiles (for all rotated assets)
        for tile in final_footprint:
            if not road_mask[tile[1]][tile[0]]:
                occupied_tiles.add(tile)
            
        # Store metadata
        rotation_metadata[pid] = {
            "rotation_deg": rotation_deg,
            "facing_vec": facing_vec if rotation_deg in (45, 135) else None,
            "anchor_pos": pos,
            "was_adjusted": was_adjusted,
            "reason": reason
        }
        
        if was_adjusted:
            debug_logs.append(
                f"{pid}: original={pos}, rotation={rotation_deg}, "
                f"overlap_reason={reason}, final={updated_positions[pid]}, adjusted={was_adjusted}"
            )
        else:
            debug_logs.append(f"{pid}: diagonal rotation {rotation_deg}°, no overlap")
    
    return (updated_positions, rotation_metadata, debug_logs)


def validate_diagonal_placements(
    placements: list[dict],
    rotation_metadata: dict[str, dict]
) -> tuple[list[str], dict[str, int]]:
    """
    Validate diagonal placements and return statistics.
    Returns (issues, stats).
    """
    issues = []
    stats = {
        "diagonal_rotated": 0,
        "resolved_overlaps": 0,
        "failures": 0,
        "total_frontage": 0
    }
    
    for p in placements:
        pid = p.get("id")
        if not pid or not p.get("needs_frontage"):
            continue
        
        stats["total_frontage"] += 1
        
        if pid not in rotation_metadata:
            continue
        
        meta = rotation_metadata[pid]
        rotation_deg = meta.get("rotation_deg", 0)
        
        if rotation_deg in (45, 135):
            stats["diagonal_rotated"] += 1
        
        if meta.get("was_adjusted", False):
            stats["resolved_overlaps"] += 1
            reason = meta.get("reason", "")
            if "reverted" in reason:
                stats["failures"] += 1
                issues.append(f"{pid}: failed to resolve overlap, reverted to cardinal rotation")
    
    return (issues, stats)


# ==================== Facing Direction ====================

def compute_facing_yaw(
    road_mask: list[list[bool]],
    x: int, y: int,
    w: int, h: int,
    grid_w: int,
    grid_h: int
) -> int:
    """
    Compute which direction a building should face based on road adjacency.
    Returns yaw in degrees: N=0, E=90, S=180, W=270
    The building's front will face the edge with the most adjacent road tiles.
    
    x, y: top-left corner of building in grid coords
    w, h: building width and height
    """
    edge_road_count = {"N": 0, "S": 0, "E": 0, "W": 0}
    
    # Count road tiles adjacent to each edge
    # North edge (y - 1, above the building top)
    for xx in range(x, x + w):
        if y > 0 and in_bounds(xx, y - 1, grid_w, grid_h) and road_mask[y - 1][xx]:
            edge_road_count["N"] += 1
    
    # South edge (y + h, below the building bottom)
    for xx in range(x, x + w):
        if in_bounds(xx, y + h, grid_w, grid_h) and road_mask[y + h][xx]:
            edge_road_count["S"] += 1
    
    # East edge (x + w, right of the building)
    for yy in range(y, y + h):
        if in_bounds(x + w, yy, grid_w, grid_h) and road_mask[yy][x + w]:
            edge_road_count["E"] += 1
    
    # West edge (x - 1, left of the building)
    for yy in range(y, y + h):
        if x > 0 and in_bounds(x - 1, yy, grid_w, grid_h) and road_mask[yy][x - 1]:
            edge_road_count["W"] += 1
    
    # Find the edge with the most road tiles
    best_edge = max(edge_road_count, key=edge_road_count.get)
    
    # If no road adjacency at all, default to South
    if edge_road_count[best_edge] == 0:
        best_edge = "S"
    
    edge_to_yaw = {"N": 0, "E": 90, "S": 180, "W": 270}
    return edge_to_yaw[best_edge]


# ==================== Validation ====================

def all_connected_bfs(
    road_mask: list[list[bool]],
    gate_tiles: list[tuple[int, int]],
    grid_w: int,
    grid_h: int,
    road_style: str = "organic"
) -> bool:
    """Check if all gate tiles are connected via road tiles using BFS.
    Uses DIR_8 for organic/trails styles, DIR_4 for grid style."""
    if not gate_tiles:
        return True
    
    # Use direction set matching road style
    config = ROAD_STYLE_CONFIG.get(road_style, ROAD_STYLE_CONFIG["organic"])
    directions = config["directions"] if config else DIR_4
    
    # Search radius for finding roads near gates (gates are on boundary, roads inside)
    search_radius = 15
    
    if len(gate_tiles) == 1:
        # Single gate - just check if it's near a road
        gx, gy = gate_tiles[0]
        for radius in range(search_radius):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    nx, ny = gx + dx, gy + dy
                    if in_bounds(nx, ny, grid_w, grid_h) and road_mask[ny][nx]:
                        return True
        return False
    
    # Find first road tile near any gate (search wider radius)
    start = None
    for gx, gy in gate_tiles:
        for radius in range(search_radius):
            if start:
                break
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if abs(dx) != radius and abs(dy) != radius:
                        continue
                    nx, ny = gx + dx, gy + dy
                    if in_bounds(nx, ny, grid_w, grid_h) and road_mask[ny][nx]:
                        start = (nx, ny)
                        break
        if start:
            break
    
    if not start:
        return False
    
    # BFS from start using appropriate direction set
    visited = set()
    queue = deque([start])
    visited.add(start)
    
    while queue:
        x, y = queue.popleft()
        for dx, dy in directions:  # Use road_style-matched directions
            nx, ny = x + dx, y + dy
            if in_bounds(nx, ny, grid_w, grid_h) and (nx, ny) not in visited:
                if road_mask[ny][nx]:
                    visited.add((nx, ny))
                    queue.append((nx, ny))
    
    # Check all gate tiles are reachable (within search_radius of a visited road tile)
    for gx, gy in gate_tiles:
        if not in_bounds(gx, gy, grid_w, grid_h):
            continue
        
        reachable = False
        # Check within search_radius tiles
        for radius in range(search_radius):
            if reachable:
                break
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    nx, ny = gx + dx, gy + dy
                    if (nx, ny) in visited:
                        reachable = True
                        break
        
        if not reachable:
            return False
    
    return True


def has_adjacent_road(
    rect_grid: tuple[int, int, int, int],
    road_mask: list[list[bool]],
    grid_w: int,
    grid_h: int,
    max_distance: int = 2
) -> bool:
    """
    Check if a building rectangle has any road tile within max_distance.
    Default max_distance=2 allows for a 1-tile gap (sidewalk/setback).
    """
    x, y, w, h = rect_grid
    
    # Check expanded perimeter (within max_distance)
    for dist in range(1, max_distance + 1):
        # Check all perimeter at this distance
        for xx in range(x - dist, x + w + dist):
            # Top rows
            for yy in range(max(0, y - dist), y):
                if in_bounds(xx, yy, grid_w, grid_h) and road_mask[yy][xx]:
                    return True
            # Bottom rows
            for yy in range(y + h, min(grid_h, y + h + dist)):
                if in_bounds(xx, yy, grid_w, grid_h) and road_mask[yy][xx]:
                    return True
        
        for yy in range(y, y + h):
            # Left columns
            for xx in range(max(0, x - dist), x):
                if in_bounds(xx, yy, grid_w, grid_h) and road_mask[yy][xx]:
                    return True
            # Right columns
            for xx in range(x + w, min(grid_w, x + w + dist)):
                if in_bounds(xx, yy, grid_w, grid_h) and road_mask[yy][xx]:
                    return True
    
    return False


def is_single_tile_corridor(
    x: int, y: int,
    road_mask: list[list[bool]],
    grid_w: int,
    grid_h: int
) -> bool:
    """
    Check if a road tile is a single-tile corridor (pinch point).
    A tile is a pinch if it's a road but has no road neighbors in perpendicular directions.
    """
    # Count neighbors in cardinal directions
    neighbors = []
    for dx, dy in DIR_4:
        nx, ny = x + dx, y + dy
        if in_bounds(nx, ny, grid_w, grid_h) and road_mask[ny][nx]:
            neighbors.append((dx, dy))
    
    if len(neighbors) < 2:
        return True  # Dead end or isolated
    
    if len(neighbors) == 2:
        # Check if neighbors are opposite (corridor) and no perpendicular neighbors
        d1, d2 = neighbors
        # If they're opposite, it's a corridor
        if (d1[0] + d2[0] == 0 and d1[1] + d2[1] == 0):
            # Check perpendicular
            if d1[0] == 0:  # Vertical corridor, check horizontal
                has_horiz = False
                for dx in [-1, 1]:
                    nx = x + dx
                    if in_bounds(nx, y, grid_w, grid_h) and road_mask[y][nx]:
                        has_horiz = True
                return not has_horiz
            else:  # Horizontal corridor, check vertical
                has_vert = False
                for dy in [-1, 1]:
                    ny = y + dy
                    if in_bounds(x, ny, grid_w, grid_h) and road_mask[ny][x]:
                        has_vert = True
                return not has_vert
    
    return False


def validate_playability(
    road_mask: list[list[bool]],
    gate_positions: dict[str, tuple[float, float]],
    positions: dict[str, tuple[float, float]],
    placements: list[dict],
    min_x: float,
    min_y: float,
    grid_w: int,
    grid_h: int,
    road_style: str = "organic"
) -> list[str]:
    """
    Validate playability of the generated tilemap.
    Returns a list of issues (empty if valid).
    """
    issues = []
    
    # 1. Gate connectivity (use road_style to match BFS directions with road routing)
    gate_tiles = []
    for gid, (gx, gy) in gate_positions.items():
        grid_pos = world_to_grid(gx, gy, min_x, min_y)
        gate_tiles.append(grid_pos)
    
    if gate_tiles and not all_connected_bfs(road_mask, gate_tiles, grid_w, grid_h, road_style):
        issues.append("Not all gates are connected via roads")
    
    # 2. Frontage adjacency
    for p in placements:
        pid = p.get("id")
        if not pid or not p.get("needs_frontage"):
            continue
        if pid not in positions:
            continue
        
        pos = positions[pid]
        gx, gy = world_to_grid(pos[0], pos[1], min_x, min_y)
        pw, ph = pick_footprint(p)
        rect_grid = (gx - pw // 2, gy - ph // 2, pw, ph)
        
        if not has_adjacent_road(rect_grid, road_mask, grid_w, grid_h):
            issues.append(f"Building '{pid}' has no road frontage")
    
    # 3. Road width pinching (sample check, not exhaustive)
    pinch_count = 0
    for y in range(grid_h):
        for x in range(grid_w):
            if road_mask[y][x]:
                if is_single_tile_corridor(x, y, road_mask, grid_w, grid_h):
                    pinch_count += 1
    
    if pinch_count > 10:
        issues.append(f"Road has {pinch_count} pinch points (single-tile corridors)")
    
    return issues


# ==================== Main Compiler ====================

def compile_area_to_grid(layout: dict) -> dict:
    """
    Full pipeline to compile an area layout to a tilemap grid.
    
    Pipeline (improved with arteries-first road network):
    1. Compute positions in world space
    2. Resolve overlaps
    3. Compute bounding box and grid dimensions
    4. Place gates on boundary
    5. Create occupancy grid (fixed padding for frontage buildings)
    6. Route roads in phases:
       - Phase 1: Gate->Center trunk roads (arteries)
       - Phase 2: Inter-gate connections (ring roads)
       - Phase 3: Building connections from main_connectivity
    7. Route frontage spurs (width=2 for playability)
    8. Ensure gate connectivity (post-pass to merge disconnected components)
    9. Widen pinch points (post-pass to fix single-tile corridors)
    10. Road-safe constrained relaxation (nudge buildings toward roads)
    11. Rebuild occupancy grid (safety check after relaxation)
    12. Compute placed rectangles with facing
    13. Validate
    """
    anchor_id = layout["anchor"]["id"]
    placements = [p for p in layout.get("placements", []) if p.get("id")]
    gates = layout.get("gates", []) or []
    meta = layout.get("meta") or {}
    road_style = meta.get("road_style", "organic")
    
    # 1. Compute positions in world space
    positions = compute_graph_positions(layout)
    
    # 2. Resolve overlaps
    positions = resolve_overlaps(positions, placements, anchor_id, min_spacing=1.5, max_iterations=20)
    
    # 3. Compute tight bounding box around buildings with 5 tile border
    bbox = compute_bounding_box(positions, placements, anchor_id, padding=5.0)
    min_x, min_y, max_x, max_y, width, height = bbox
    grid_w, grid_h = int(width) + 2, int(height) + 2
    
    # 4. Place gates on the tighter boundary (world coords)
    gate_positions = place_gates_on_boundary(gates, bbox)
    
    # 5. Create occupancy grid (with fixed padding for frontage buildings)
    blocked = create_occupancy_grid(
        positions, placements, anchor_id,
        min_x, min_y, grid_w, grid_h, margin=2
    )
    road_mask = [[False for _ in range(grid_w)] for _ in range(grid_h)]
    
    # 6. Route main roads (3-phase: gate->center, inter-gate, building connections)
    road_paths = []
    if road_style != "none":
        road_paths = route_all_roads(
            layout, positions, gate_positions, anchor_id,
            blocked, road_mask, road_style,
            min_x, min_y, grid_w, grid_h
        )
        
        # 7. Route frontage spurs (width=2 for playability)
        spur_paths = route_frontage_spurs(
            positions, placements, road_mask, blocked,
            road_style, min_x, min_y, grid_w, grid_h
        )
        road_paths.extend(spur_paths)
        
        # 8. Ensure gate connectivity (post-pass)
        gate_tiles = [world_to_grid(gx, gy, min_x, min_y) for gx, gy in gate_positions.values()]
        connectivity_paths = ensure_gate_connectivity(
            gate_tiles, road_mask, blocked, grid_w, grid_h, road_style
        )
        road_paths.extend(connectivity_paths)
        
        # 9. Widen pinch points (post-pass)
        widen_pinch_points(road_mask, blocked, grid_w, grid_h, min_width=2)
        
        fill_diagonal_gaps(road_mask, blocked, grid_w, grid_h, passes=2)

    # 9b. Compute diagonal corner overlays for smooth diagonal rendering
    corner_overlay = compute_diagonal_corner_overlays(road_mask, blocked, grid_w, grid_h)
    
    # 9c. Compute road cutouts for outer (convex) corners
    road_cutouts = compute_road_cutouts(road_mask, grid_w, grid_h)

    # 10. Road-safe constrained relaxation (replaces snap_all_frontage)
    positions = constrained_relaxation(
        positions, placements, road_mask, blocked, anchor_id,
        min_x, min_y, grid_w, grid_h,
        iterations=10, road_clearance=1
    )
    
    # 11. Rebuild occupancy grid after relaxation (safety check)
    blocked = create_occupancy_grid(
        positions, placements, anchor_id,
        min_x, min_y, grid_w, grid_h, margin=2
    )
    
    # 12. Apply diagonal rotations for frontage assets
    rotation_metadata = {}
    diagonal_logs = []
    if road_style != "none":
        positions, rotation_metadata, diagonal_logs = apply_diagonal_rotations(
            placements, positions, road_mask, blocked,
            min_x, min_y, grid_w, grid_h
        )
        
        # Rebuild occupancy grid after diagonal adjustments
        blocked = create_occupancy_grid(
            positions, placements, anchor_id,
            min_x, min_y, grid_w, grid_h, margin=2
        )
    
    # 13. Compute placed rectangles with facing and rotation metadata
    placed_rects = []
    for p in placements:
        pid = p.get("id")
        if not pid or pid not in positions:
            continue
        pos = positions[pid]
        gx, gy = world_to_grid(pos[0], pos[1], min_x, min_y)
        pw, ph = pick_footprint(p)
        
        # Top-left corner
        x0, y0 = gx - pw // 2, gy - ph // 2
        
        # Get rotation metadata if available
        if pid in rotation_metadata:
            meta = rotation_metadata[pid]
            rotation_deg = meta.get("rotation_deg", 0)
            facing_vec = meta.get("facing_vec")
            anchor_pos = meta.get("anchor_pos", pos)
            was_adjusted = meta.get("was_adjusted", False)
        else:
            # Fallback to existing facing logic
            rotation_deg = compute_facing_yaw(road_mask, x0, y0, pw, ph, grid_w, grid_h)
            facing_vec = None
            anchor_pos = pos
            was_adjusted = False
        
        # Extended format: (pid, x0, y0, pw, ph, rotation_deg, facing_vec, anchor_pos, was_adjusted)
        placed_rects.append((pid, x0, y0, pw, ph, rotation_deg, facing_vec, anchor_pos, was_adjusted))
    
    # 14. Validate diagonal placements
    diagonal_issues, diagonal_stats = validate_diagonal_placements(placements, rotation_metadata)
    if diagonal_logs:
        print(f"  Diagonal rotation logs: {len(diagonal_logs)} entries")
        for log in diagonal_logs[:5]:  # Show first 5
            print(f"    {log}")
    if diagonal_stats:
        print(f"  Diagonal rotation stats: {diagonal_stats['diagonal_rotated']} rotated, "
              f"{diagonal_stats['resolved_overlaps']} overlaps resolved, "
              f"{diagonal_stats['failures']} failures")
    
    # 15. Validate
    issues = validate_playability(
        road_mask, gate_positions, positions, placements,
        min_x, min_y, grid_w, grid_h, road_style
    )
    issues.extend(diagonal_issues)
    
    # Convert gate positions to grid
    gates_grid = []
    for gid, (gx, gy) in gate_positions.items():
        grid_pos = world_to_grid(gx, gy, min_x, min_y)
        gates_grid.append((gid, grid_pos[0], grid_pos[1]))
    
    # Anchor position in grid
    anchor_grid = world_to_grid(positions[anchor_id][0], positions[anchor_id][1], min_x, min_y)
    
    return {
        "grid_w": grid_w,
        "grid_h": grid_h,
        "min_x": min_x,
        "min_y": min_y,
        "anchor_id": anchor_id,
        "center_xy": anchor_grid,
        "gates": gates_grid,
        "road": road_mask,
        "corner_overlay": corner_overlay,
        "road_cutouts": road_cutouts,
        "blocked": blocked,
        "placed_rects": placed_rects,
        "placements": placed_rects,  # Alias for compose_world.py
        "road_paths": road_paths,
        "positions": positions,
        "issues": issues,
        "rotation_metadata": rotation_metadata,  # For debugging
        "diagonal_stats": diagonal_stats,  # For statistics
    }


# ==================== Visualization ====================

def visualize_tilemaps(area_layouts_file: str = "area_layouts.json", output_dir: str = "graphs_tilemap"):
    """
    Produce tilemap visualizations for all areas.
    Shows roads, buildings, gates, and validation issues.
    """
    with open(area_layouts_file) as f:
        layouts = json.load(f)
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    for area_id, layout in layouts.items():
        print(f"Compiling {area_id}...")
        
        compiled = compile_area_to_grid(layout)
        
        grid_w = compiled["grid_w"]
        grid_h = compiled["grid_h"]
        road_mask = compiled["road"]
        blocked = compiled["blocked"]
        cx, cy = compiled["center_xy"]
        anchor_id = compiled["anchor_id"]
        issues = compiled.get("issues", [])
        
        # Build tile code map for visualization
        tiles = [[T_EMPTY for _ in range(grid_w)] for _ in range(grid_h)]
        for y in range(grid_h):
            for x in range(grid_w):
                if blocked[y][x]:
                    tiles[y][x] = T_BLOCKED
                if road_mask[y][x]:
                    tiles[y][x] = T_ROAD
        
        # Mark gates and center
        for gid, gx, gy in compiled["gates"]:
            if in_bounds(gx, gy, grid_w, grid_h):
                tiles[gy][gx] = T_GATE
        if in_bounds(cx, cy, grid_w, grid_h):
            tiles[cy][cx] = T_CENTER
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 9))
        
        title = f"Tilemap: {area_id}"
        if issues:
            title += f" [{len(issues)} issues]"
        ax.set_title(title, fontsize=12, fontweight="bold")
        
        ax.set_xlim(0, grid_w)
        ax.set_ylim(0, grid_h)
        ax.set_aspect("equal")
        ax.set_xlabel("tiles")
        ax.set_ylabel("tiles")
        
        # Grid lines
        ax.set_xticks(range(0, grid_w + 1, 10))
        ax.set_yticks(range(0, grid_h + 1, 10))

        ax.grid(False, linewidth=0.3, alpha=0.25)
        
        # Custom colormap
        cmap_colors = [
            (0.95, 0.95, 0.95),  # T_EMPTY - light gray
            (0.7, 0.6, 0.5),     # T_ROAD - brown/tan
            (0.3, 0.3, 0.3),     # T_BLOCKED - dark gray
            (0.9, 0.2, 0.2),     # T_GATE - red
            (0.9, 0.8, 0.2),     # T_CENTER - gold
            (0.6, 0.5, 0.4),     # T_FOOTPATH - lighter brown
        ]
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(cmap_colors)
        
        # Draw tile layer
        ax.imshow(tiles, origin="lower", extent=(0, grid_w, 0, grid_h),
                  interpolation="nearest", alpha=0.85, cmap=cmap, vmin=0, vmax=5)
        
        # Draw corner overlay triangles for smooth diagonal roads
        corner_overlay = compiled.get("corner_overlay")
        if corner_overlay:
            road_color = (0.7, 0.6, 0.5)  # Same tan/brown as T_ROAD
            for y in range(grid_h):
                for x in range(grid_w):
                    c = corner_overlay[y][x]
                    if c == C_NONE:
                        continue
                    # Define triangle vertices based on corner type
                    # Using origin="lower", so y=0 is at bottom
                    if c == C_NE:
                        # Top-right corner fill: top-left, top-right, bottom-right
                        verts = [(x, y + 1), (x + 1, y + 1), (x + 1, y)]
                    elif c == C_NW:
                        # Top-left corner fill: bottom-left, top-left, top-right
                        verts = [(x, y), (x, y + 1), (x + 1, y + 1)]
                    elif c == C_SE:
                        # Bottom-right corner fill: bottom-left, bottom-right, top-right
                        verts = [(x, y), (x + 1, y), (x + 1, y + 1)]
                    elif c == C_SW:
                        # Bottom-left corner fill: bottom-left, bottom-right, top-left
                        verts = [(x, y), (x + 1, y), (x, y + 1)]
                    else:
                        continue
                    triangle = patches.Polygon(
                        verts, closed=True,
                        facecolor=road_color, edgecolor=road_color,
                        alpha=0.85, linewidth=0
                    )
                    ax.add_patch(triangle)
        
        # Draw road cutout triangles (cut corners off road tiles for smooth outer edges)
        # Check each corner of each road tile independently to handle multiple cutouts per tile
        bg_color = (0.95, 0.95, 0.95)  # Same as T_EMPTY background
        
        def is_road_at(rx: int, ry: int) -> bool:
            return 0 <= rx < grid_w and 0 <= ry < grid_h and road_mask[ry][rx]
        
        for y in range(grid_h):
            for x in range(grid_w):
                if not road_mask[y][x]:
                    continue
                
                # Check each corner independently - a tile can have multiple cutouts
                # NE corner cut: no road at NE, E, or N
                if (not is_road_at(x + 1, y + 1)) and (not is_road_at(x + 1, y)) and (not is_road_at(x, y + 1)):
                    verts = [(x, y + 1), (x + 1, y + 1), (x + 1, y)]
                    ax.add_patch(patches.Polygon(verts, closed=True, facecolor=bg_color, edgecolor=bg_color, alpha=1.0, linewidth=0))
                
                # NW corner cut: no road at NW, W, or N
                if (not is_road_at(x - 1, y + 1)) and (not is_road_at(x - 1, y)) and (not is_road_at(x, y + 1)):
                    verts = [(x, y), (x, y + 1), (x + 1, y + 1)]
                    ax.add_patch(patches.Polygon(verts, closed=True, facecolor=bg_color, edgecolor=bg_color, alpha=1.0, linewidth=0))
                
                # SE corner cut: no road at SE, E, or S
                if (not is_road_at(x + 1, y - 1)) and (not is_road_at(x + 1, y)) and (not is_road_at(x, y - 1)):
                    verts = [(x, y), (x + 1, y), (x + 1, y + 1)]
                    ax.add_patch(patches.Polygon(verts, closed=True, facecolor=bg_color, edgecolor=bg_color, alpha=1.0, linewidth=0))
                
                # SW corner cut: no road at SW, W, or S
                if (not is_road_at(x - 1, y - 1)) and (not is_road_at(x - 1, y)) and (not is_road_at(x, y - 1)):
                    verts = [(x, y), (x + 1, y), (x, y + 1)]
                    ax.add_patch(patches.Polygon(verts, closed=True, facecolor=bg_color, edgecolor=bg_color, alpha=1.0, linewidth=0))
        
        # Draw building rectangles
        placement_map = {p[0]: p for p in compiled["placed_rects"]}
        placements_data = layout.get("placements", []) or []
        
        for placement in compiled["placed_rects"]:
            pid, x0, y0, rw, rh = placement[:5]
            
            # Handle extended format: (pid, x0, y0, pw, ph, rotation_deg, facing_vec, anchor_pos, was_adjusted)
            if len(placement) >= 6:
                rotation_deg = placement[5]
                facing_vec = placement[6] if len(placement) > 6 else None
                anchor_pos = placement[7] if len(placement) > 7 else None
                was_adjusted = placement[8] if len(placement) > 8 else False
            else:
                # Backward compatibility: old format with facing_yaw
                rotation_deg = placement[5] if len(placement) > 5 else 180
                facing_vec = None
                anchor_pos = None
                was_adjusted = False
            
            # Find original placement data
            p_data = next((p for p in placements_data if p.get("id") == pid), {})
            needs_frontage = p_data.get("needs_frontage", False)
            
            # Color: blue for frontage, gray for others, yellow dot if adjusted
            edge_color = "blue" if needs_frontage else "gray"
            rect = patches.Rectangle(
                (x0, y0), rw, rh,
                fill=False, linewidth=1.2,
                edgecolor=edge_color
            )
            ax.add_patch(rect)
            
            # Show marker if position was adjusted
            if was_adjusted and anchor_pos:
                # Draw yellow dot at anchor position
                anchor_gx, anchor_gy = world_to_grid(anchor_pos[0], anchor_pos[1], 
                                                     compiled["min_x"], compiled["min_y"])
                ax.plot(anchor_gx, anchor_gy, 'yo', markersize=4)
                # Draw line from anchor to current position
                center_x, center_y = x0 + rw / 2, y0 + rh / 2
                ax.plot([anchor_gx, center_x], [anchor_gy, center_y], 
                       'y--', linewidth=1, alpha=0.5)
                # Show displacement magnitude
                displacement = math.sqrt((center_x - anchor_gx)**2 + (center_y - anchor_gy)**2)
                ax.text(center_x + 1, center_y + 1, f"Δ{displacement:.1f}", 
                       fontsize=4, color="yellow", weight="bold")
            
            # Show facing direction
            if rotation_deg in (45, 135) and facing_vec:
                # Diagonal rotation: draw arrow in facing direction
                center_x, center_y = x0 + rw / 2, y0 + rh / 2
                dx, dy = facing_vec
                arrow_length = min(rw, rh) / 2
                ax.arrow(center_x, center_y, dx * arrow_length, dy * arrow_length,
                        head_width=1.5, head_length=1.5, fc="orange", ec="orange", linewidth=1.5)
                ax.text(x0 + 0.5, y0 + rh - 1, f"{pid} {rotation_deg}°", fontsize=5, color="orange")
            else:
                # Cardinal rotation: use arrow symbol
                facing_arrow = {0: "↑", 90: "→", 180: "↓", 270: "←"}.get(rotation_deg, "?")
                ax.text(x0 + 0.5, y0 + rh - 1, f"{pid} {facing_arrow}", fontsize=5)
            
            # Frontage indicator (for cardinal directions)
            if needs_frontage and rotation_deg not in (45, 135):
                # Draw pink line on the facing edge
                if rotation_deg == 0:  # North
                    ax.plot([x0, x0 + rw], [y0, y0], color="hotpink", linewidth=2)
                elif rotation_deg == 180:  # South
                    ax.plot([x0, x0 + rw], [y0 + rh, y0 + rh], color="hotpink", linewidth=2)
                elif rotation_deg == 90:  # East
                    ax.plot([x0 + rw, x0 + rw], [y0, y0 + rh], color="hotpink", linewidth=2)
                elif rotation_deg == 270:  # West
                    ax.plot([x0, x0], [y0, y0 + rh], color="hotpink", linewidth=2)
            elif needs_frontage and rotation_deg in (45, 135):
                # For diagonal, show diagonal line on the frontage edge
                center_x, center_y = x0 + rw / 2, y0 + rh / 2
                if facing_vec:
                    dx, dy = facing_vec
                    # Draw diagonal line from center in facing direction
                    line_length = min(rw, rh) / 2
                    ax.plot([center_x, center_x + dx * line_length], 
                           [center_y, center_y + dy * line_length],
                           color="hotpink", linewidth=2)
        
        # Mark center
        ax.text(cx + 0.5, cy + 0.5, f"{anchor_id}\n(center)", fontsize=7,
                fontweight="bold", ha="center", va="center")
        
        # Mark gates
        for gid, gx, gy in compiled["gates"]:
            ax.text(gx, gy + 2, gid, fontsize=6, color="red", ha="center")
        
        # Show issues as text overlay
        if issues:
            issue_text = "\n".join(issues[:5])  # Show first 5
            ax.text(0.02, 0.98, f"Issues:\n{issue_text}", transform=ax.transAxes,
                    fontsize=7, verticalalignment="top", color="red",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        
        # Save
        outpath = Path(output_dir) / f"tilemap_{area_id}.png"
        fig.savefig(outpath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {outpath}")
        
        if issues:
            for issue in issues:
                print(f"  Warning: {issue}")




if __name__ == "__main__":
    visualize_tilemaps()
