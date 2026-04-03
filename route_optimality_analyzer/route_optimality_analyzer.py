"""
RouteOptimalityAnalyzer
=======================
Computes Route Optimality Rate (ROR) for museum visitor routes.

Given a museum layout annotation file, this class:
  1. Builds a walkable mask (floor area minus forbidden zones and exhibit bodies)
  2. Finds the shortest feasible path between any two points via Dijkstra
  3. Solves the optimal exhibit-visit ordering via Held-Karp TSP (≤15 stops)
     or nearest-neighbour heuristic (>15 stops)
  4. Validates that an actual route visits all required exhibits within a
     proximity threshold and without colliding with exhibit bodies
  5. Computes per-route optimality ratio and aggregate ROR across many routes
  6. Produces a detailed multi-panel visualization

Usage
-----
    analyzer = RouteOptimalityAnalyzer("museum_layout_annotations.json")

    result = analyzer.evaluate_route(
        actual_route_px   = 4200.0,          # total skeleton distance
        start             = (668, 2790),      # entrance centre
        end               = (920, 2791),      # exit centre
        exhibit_numbers   = ["3", "7", "12"], # exhibits to visit
        proximity_threshold = 40,
    )

    analyzer.visualize(result, output_path="ror_visualization.png")

    # Aggregate across many routes:
    ror = analyzer.calculate_ror([result1, result2, result3])
"""

import json
import math
import heapq
import itertools
from collections import deque

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgba


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _combinations(items, r):
    """Pure-Python combinations to avoid itertools dependency in hot path."""
    items = list(items)
    n = len(items)
    if r > n:
        return
    indices = list(range(r))
    yield [items[i] for i in indices]
    while True:
        for i in reversed(range(r)):
            if indices[i] != i + n - r:
                break
        else:
            return
        indices[i] += 1
        for j in range(i + 1, r):
            indices[j] = indices[j - 1] + 1
        yield [items[i] for i in indices]


# ──────────────────────────────────────────────────────────────────────────────
# Main class
# ──────────────────────────────────────────────────────────────────────────────

class RouteOptimalityAnalyzer:
    """
    Computes Route Optimality Rate (ROR) against the shortest feasible
    path that respects museum layout constraints and required exhibit visits.
    """

    def __init__(self, annotations_file: str, path_scale: float = 0.25):
        """
        Parameters
        ----------
        annotations_file : path to the JSON produced by the annotation tool
        path_scale       : downsample factor for pathfinding (0.25 = 25% of
                           original resolution).  Smaller = faster but less
                           precise.  Full-resolution Dijkstra on a 1770×2991
                           image with 100 exhibit pairs is very slow; 0.25×
                           reduces the search space by 16× with minimal error.
        """
        with open(annotations_file, "r") as f:
            self.ann = json.load(f)

        self.path_scale   = path_scale
        self.exhibits     = self.ann.get("exhibits", [])
        self.entrances    = self.ann.get("entrances", [])
        self.exits        = self.ann.get("exits", [])
        self.floor_areas  = self.ann.get("floor_areas", [])
        self.forbidden    = self.ann.get("forbidden_areas", [])
        self.walls        = self.ann.get("walls", [])

        # Scale factor from annotation JSON metadata (mm per px)
        dist_ann = self.ann.get("distance_in_mm", [])
        self.mm_per_px = dist_ann[0]["mm_per_px"] if dist_ann else 1.0

        # Infer canvas size from floor area bounds
        fa = self.floor_areas[0]["coordinates"] if self.floor_areas else None
        if fa:
            self.canvas_w = int(fa["x"] + fa["width"])  + 50
            self.canvas_h = int(fa["y"] + fa["height"]) + 50
        else:
            self.canvas_w, self.canvas_h = 1800, 3000

        # Build walkable mask once (full-res for drawing, scaled for pathfinding)
        self._full_walkable  = None   # built lazily
        self._small_walkable = None   # scaled version for Dijkstra
        self._pixel_set      = None   # set of walkable pixels at path_scale

        print(f"[RouteOptimalityAnalyzer] Loaded {len(self.exhibits)} exhibits")
        print(f"  Canvas    : {self.canvas_w} × {self.canvas_h} px")
        print(f"  Scale     : {self.mm_per_px:.4f} mm/px")
        print(f"  Path scale: {path_scale}× (pathfinding resolution)")

    # ──────────────────────────────────────────────────────────────────────────
    # Walkable mask
    # ──────────────────────────────────────────────────────────────────────────

    def _build_walkable_mask(self, collision_margin: int = 3) -> np.ndarray:
        """
        Binary uint8 mask at full resolution.
        Walkable = 255, blocked = 0.

        Layers applied in order:
          1. Paint floor areas as walkable
          2. Block forbidden areas
          3. Block wall polylines (thick stroke)
          4. Block exhibit interiors + collision_margin
        """
        mask = np.zeros((self.canvas_h, self.canvas_w), dtype=np.uint8)

        # ── 1. Floor areas ────────────────────────────────────────────
        for fa in self.floor_areas:
            if fa["shape"] == "rectangle":
                c = fa["coordinates"]
                x, y = int(c["x"]), int(c["y"])
                x2   = x + int(c["width"])
                y2   = y + int(c["height"])
                mask[y:y2, x:x2] = 255

        # ── 2. Forbidden areas ────────────────────────────────────────
        for fb in self.forbidden:
            if fb["shape"] == "rectangle":
                c = fb["coordinates"]
                x, y = int(c["x"]), int(c["y"])
                x2   = x + int(c["width"])
                y2   = y + int(c["height"])
                mask[y:y2, x:x2] = 0

        # ── 3. Walls (thick polylines) ────────────────────────────────
        for wall in self.walls:
            if wall["shape"] == "polyline":
                pts = wall["coordinates"]["points"]
                arr = np.array([[int(p["x"]), int(p["y"])] for p in pts],
                               dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(mask, [arr], isClosed=False,
                              color=0, thickness=6)

        # ── 4. Exhibit bodies + collision margin ──────────────────────
        for ex in self.exhibits:
            c = ex["coordinates"]
            if ex["shape"] == "circle":
                cx = int(c["center_x"])
                cy = int(c["center_y"])
                r  = int(c["radius"]) + collision_margin
                cv2.circle(mask, (cx, cy), r, 0, thickness=-1)
            elif ex["shape"] == "rectangle":
                x  = int(c["x"])   - collision_margin
                y  = int(c["y"])   - collision_margin
                x2 = int(c["x"]  + c["width"])  + collision_margin
                y2 = int(c["y"]  + c["height"]) + collision_margin
                cv2.rectangle(mask, (x, y), (x2, y2), 0, thickness=-1)

        return mask

    def _get_walkable(self) -> np.ndarray:
        if self._full_walkable is None:
            self._full_walkable = self._build_walkable_mask()
        return self._full_walkable

    def _get_small_walkable(self):
        """Return (small_mask, pixel_set) at path_scale resolution."""
        if self._small_walkable is None:
            full = self._get_walkable()
            s    = self.path_scale
            sw   = max(1, int(full.shape[1] * s))
            sh   = max(1, int(full.shape[0] * s))
            small = cv2.resize(full, (sw, sh), interpolation=cv2.INTER_NEAREST)
            self._small_walkable = small
            ys, xs = np.where(small > 0)
            self._pixel_set = set(zip(xs.tolist(), ys.tolist()))
        return self._small_walkable, self._pixel_set

    def _to_small(self, pt):
        """Scale a full-res (x, y) to path_scale coordinates."""
        s = self.path_scale
        return (max(0, int(round(pt[0] * s))),
                max(0, int(round(pt[1] * s))))

    def _to_full(self, pt):
        """Scale a path_scale (x, y) back to full-res."""
        s = self.path_scale
        return (int(round(pt[0] / s)), int(round(pt[1] / s)))

    # ──────────────────────────────────────────────────────────────────────────
    # Exhibit waypoints
    # ──────────────────────────────────────────────────────────────────────────

    def _exhibit_center(self, ex) -> tuple:
        c = ex["coordinates"]
        if ex["shape"] == "circle":
            return (int(c["center_x"]), int(c["center_y"]))
        elif ex["shape"] == "rectangle":
            return (int(c["x"] + c["width"]  / 2),
                    int(c["y"] + c["height"] / 2))
        return None

    def _exhibit_radius(self, ex) -> float:
        c = ex["coordinates"]
        if ex["shape"] == "circle":
            return float(c["radius"])
        elif ex["shape"] == "rectangle":
            return max(c["width"], c["height"]) / 2.0
        return 20.0

    def _snap_to_walkable(self, pt, pixel_set, xs_arr, ys_arr):
        """Return the nearest walkable pixel to pt."""
        px, py = pt
        if (px, py) in pixel_set:
            return (px, py)
        dists = np.sqrt((xs_arr - px)**2 + (ys_arr - py)**2)
        idx   = int(np.argmin(dists))
        return (int(xs_arr[idx]), int(ys_arr[idx]))

    def _exhibit_waypoint(self, ex, proximity_threshold: float):
        """
        Find the closest walkable pixel to the exhibit that is within
        proximity_threshold (full-res px).  Returns full-res (x, y) or None.
        """
        _, pset = self._get_small_walkable()
        ys, xs  = np.where(self._small_walkable > 0)
        xs_arr, ys_arr = xs.astype(float), ys.astype(float)

        center_full = self._exhibit_center(ex)
        if center_full is None:
            return None
        cx, cy = self._to_small(center_full)

        dists = np.sqrt((xs_arr - cx)**2 + (ys_arr - cy)**2)
        idx   = int(np.argmin(dists))
        dist_small = float(dists[idx])
        dist_full  = dist_small / self.path_scale

        if dist_full > proximity_threshold:
            return None
        return self._to_full((int(xs[idx]), int(ys[idx])))

    # ──────────────────────────────────────────────────────────────────────────
    # Dijkstra on walkable mask
    # ──────────────────────────────────────────────────────────────────────────

    def _dijkstra(self, start_full, end_full) -> tuple:
        """
        Shortest walkable path from start to end.

        Returns (distance_px_full_res, path_full_res_list) or (None, None).
        Path is a list of full-res (x, y) coords sampled every few nodes.
        """
        _, pset = self._get_small_walkable()
        ys, xs  = np.where(self._small_walkable > 0)
        xs_arr, ys_arr = xs.astype(float), ys.astype(float)

        a = self._snap_to_walkable(self._to_small(start_full), pset, xs_arr, ys_arr)
        b = self._snap_to_walkable(self._to_small(end_full),   pset, xs_arr, ys_arr)

        if a == b:
            return 0.0, [start_full]

        dist  = {a: 0.0}
        prev  = {a: None}
        heap  = [(0.0, a)]

        while heap:
            d, (cx, cy) = heapq.heappop(heap)
            if (cx, cy) == b:
                break
            if d > dist.get((cx, cy), float("inf")):
                continue
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nb = (cx + dx, cy + dy)
                    if nb not in pset:
                        continue
                    nd = d + math.sqrt(dx*dx + dy*dy)
                    if nd < dist.get(nb, float("inf")):
                        dist[nb] = nd
                        prev[nb] = (cx, cy)
                        heapq.heappush(heap, (nd, nb))

        if b not in dist:
            return None, None

        # Reconstruct path
        path_small = []
        node = b
        while node is not None:
            path_small.append(node)
            node = prev[node]
        path_small.reverse()

        # Convert distance and path back to full-res scale
        dist_full = dist[b] / self.path_scale
        path_full = [self._to_full(p) for p in path_small]

        return round(dist_full, 2), path_full

    # ──────────────────────────────────────────────────────────────────────────
    # Pairwise distance matrix
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_pairwise(self, waypoints: list) -> tuple:
        """
        Compute all pairwise Dijkstra distances between waypoints.

        waypoints : list of (label, (x, y)) in full-res coords

        Returns
        -------
        dist_matrix : {(label_a, label_b): float}  — symmetric
        path_matrix : {(label_a, label_b): list}   — list of full-res coords
        """
        dist_matrix = {}
        path_matrix = {}

        labels = [lbl for lbl, _ in waypoints]
        coords = {lbl: pt for lbl, pt in waypoints}

        for i in range(len(waypoints)):
            for j in range(i + 1, len(waypoints)):
                la, lb = labels[i], labels[j]
                d, path = self._dijkstra(coords[la], coords[lb])
                dist_matrix[(la, lb)] = d
                dist_matrix[(lb, la)] = d
                path_matrix[(la, lb)] = path
                path_matrix[(lb, la)] = list(reversed(path)) if path else None
                status = f"{d:.1f} px" if d is not None else "unreachable"
                print(f"  {la:20s} ↔ {lb:20s} : {status}")

        return dist_matrix, path_matrix

    # ──────────────────────────────────────────────────────────────────────────
    # Path stitching
    # ──────────────────────────────────────────────────────────────────────────

    def _stitch_segments(self, path_order: list, path_matrix: dict) -> list:
        """
        Concatenate per-leg coordinate lists into one continuous chain.

        Reuses the last point of each leg as the first point of the next,
        eliminating pixel-level gaps caused by independent Dijkstra snapping.

        Parameters
        ----------
        path_order  : ordered list of waypoint labels e.g. ['START', 'Exhibit 7', 'END']
        path_matrix : {(label_a, label_b): [full-res (x,y), ...]}

        Returns
        -------
        list of full-res (x, y) forming one unbroken polyline
        """
        full_path = []
        for i in range(len(path_order) - 1):
            a, b = path_order[i], path_order[i + 1]
            seg  = path_matrix.get((a, b)) or []
            if not seg:
                continue
            if full_path:
                # Overwrite the last stored point with seg[0] so the
                # junction is seamless, then append the rest of the segment
                full_path[-1] = seg[0]
                full_path.extend(seg[1:])
            else:
                full_path.extend(seg)
        return full_path

    # ──────────────────────────────────────────────────────────────────────────
    # TSP solver (Held-Karp DP for N≤15, nearest-neighbour heuristic beyond)
    # ──────────────────────────────────────────────────────────────────────────

    def _solve_tsp(self, dist_matrix: dict, start: str,
                   end: str, stops: list) -> dict:
        n = len(stops)

        def edge(a, b):
            v = dist_matrix.get((a, b))
            return v if v is not None else float("inf")

        if n == 0:
            return {"ordered_path": [start, end],
                    "total_distance": edge(start, end),
                    "method": "direct"}

        if n <= 15:
            # ── Held-Karp exact DP ────────────────────────────────────
            dp   = {}
            prev = {}

            for s in stops:
                key       = (frozenset([s]), s)
                dp[key]   = edge(start, s)
                prev[key] = start

            for size in range(2, n + 1):
                for combo in _combinations(stops, size):
                    visited = frozenset(combo)
                    for cur in visited:
                        remaining = visited - {cur}
                        best_d, best_p = float("inf"), None
                        for p in remaining:
                            pk = (remaining, p)
                            c  = dp.get(pk, float("inf")) + edge(p, cur)
                            if c < best_d:
                                best_d, best_p = c, p
                        key = (visited, cur)
                        dp[key]   = best_d
                        prev[key] = best_p

            all_vis = frozenset(stops)
            best_total, best_last = float("inf"), None
            for last in stops:
                key  = (all_vis, last)
                cost = dp.get(key, float("inf")) + edge(last, end)
                if cost < best_total:
                    best_total, best_last = cost, last

            # Reconstruct
            path    = [end]
            visited = all_vis
            cur     = best_last
            while cur != start:
                path.append(cur)
                key      = (visited, cur)
                previous = prev.get(key, start)
                visited  = visited - {cur}
                cur      = previous
            path.append(start)
            path.reverse()

            return {"ordered_path"  : path,
                    "total_distance": round(best_total, 2),
                    "method"        : "held-karp"}

        else:
            # ── Nearest-neighbour heuristic ───────────────────────────
            unvisited = set(stops)
            path, total, cur = [start], 0.0, start
            while unvisited:
                nxt    = min(unvisited, key=lambda s: edge(cur, s))
                total += edge(cur, nxt)
                path.append(nxt)
                unvisited.remove(nxt)
                cur = nxt
            total += edge(cur, end)
            path.append(end)

            return {"ordered_path"  : path,
                    "total_distance": round(total, 2),
                    "method"        : "nearest-neighbour"}

    # ──────────────────────────────────────────────────────────────────────────
    # Shortest feasible path with required waypoints
    # ──────────────────────────────────────────────────────────────────────────

    def shortest_feasible_with_waypoints(self, start: tuple, end: tuple,
                                          exhibit_numbers: list,
                                          proximity_threshold: float = 40.0) -> dict:
        """
        Find the shortest walkable route from start to end that passes
        within proximity_threshold of every exhibit in exhibit_numbers.

        Returns dict with ordered_path, total_distance, method, legs,
        path_segments (list of coordinate lists for drawing).
        """
        print(f"\n[Shortest Feasible Path]")
        print(f"  Exhibits to visit : {exhibit_numbers}")
        print(f"  Proximity         : {proximity_threshold} px")

        # ── Resolve exhibit waypoints ──────────────────────────────────
        exhibit_map  = {}   # label → full-res (x, y)
        skipped      = []

        ex_lookup = {str(ex["exhibit_number"]): ex for ex in self.exhibits}

        for num in exhibit_numbers:
            key = str(num)
            ex  = ex_lookup.get(key)
            if ex is None:
                print(f"  ⚠️  Exhibit {num}: not found in annotations — skipped")
                skipped.append(num)
                continue
            wp = self._exhibit_waypoint(ex, proximity_threshold)
            if wp is None:
                print(f"  ⚠️  Exhibit {num}: no walkable pixel within "
                      f"{proximity_threshold} px — skipped")
                skipped.append(num)
                continue
            exhibit_map[f"Exhibit {num}"] = wp

        if not exhibit_map:
            print("  No reachable exhibits — computing direct START→END")
            d, path = self._dijkstra(start, end)
            return {"ordered_path"  : ["START", "END"],
                    "total_distance": d,
                    "method"        : "direct",
                    "legs"          : [{"from": "START", "to": "END",
                                        "distance": d}],
                    "path_segments" : [path] if path else [],
                    "skipped"       : skipped}

        # ── Build waypoint list & solve TSP ───────────────────────────
        exhibit_labels = list(exhibit_map.keys())
        waypoints = (
            [("START", start)]
            + [(lbl, pt) for lbl, pt in exhibit_map.items()]
            + [("END", end)]
        )

        print(f"\n  Computing {len(waypoints)*(len(waypoints)-1)//2} pairwise distances…")
        dist_matrix, path_matrix = self._compute_pairwise(waypoints)

        tsp = self._solve_tsp(dist_matrix, "START", "END", exhibit_labels)

        # ── Build per-leg breakdown ────────────────────────────────────
        path  = tsp["ordered_path"]
        legs  = []
        segs  = []
        total = 0.0

        for i in range(len(path) - 1):
            a, b = path[i], path[i+1]
            d    = dist_matrix.get((a, b), 0.0) or 0.0
            seg  = path_matrix.get((a, b))
            total += d
            legs.append({"from": a, "to": b, "distance": round(d, 2)})
            if seg:
                segs.append(seg)
            print(f"    {a:20s} → {b:20s} : {d:.1f} px")

        print(f"  {'─'*55}")
        print(f"  Optimal order    : {' → '.join(path)}")
        print(f"  Total distance   : {total:.1f} px  [{tsp['method']}]")

        stitched = self._stitch_segments(path, path_matrix)

        return {"ordered_path"    : path,
                "total_distance"  : round(total, 2),
                "method"          : tsp["method"],
                "legs"            : legs,
                "path_segments"   : segs,
                "stitched_path"   : stitched,
                "exhibit_waypoints": exhibit_map,
                "skipped"         : skipped}

    # ──────────────────────────────────────────────────────────────────────────
    # Exhibit visit validation
    # ──────────────────────────────────────────────────────────────────────────

    def validate_exhibit_visits(self, skeleton_points: list,
                                 exhibit_numbers: list,
                                 proximity_threshold: float = 40.0) -> dict:
        """
        Check whether skeleton_points (list of full-res (x,y)) passes within
        proximity_threshold of each exhibit without entering its body.

        Returns dict: visited, missed, visit_details, all_visited.
        """
        if not skeleton_points:
            return {"visited": [], "missed": list(exhibit_numbers),
                    "visit_details": {}, "all_visited": False}

        pts = np.array(skeleton_points, dtype=float)   # (N, 2)
        ex_lookup = {str(ex["exhibit_number"]): ex for ex in self.exhibits}

        print(f"\n[Exhibit Visit Validation]  threshold={proximity_threshold} px")
        visited, missed, details = [], [], {}

        for num in exhibit_numbers:
            ex = ex_lookup.get(str(num))
            if ex is None:
                missed.append(num)
                continue

            cx, cy  = self._exhibit_center(ex)
            ex_r    = self._exhibit_radius(ex)
            dists   = np.sqrt((pts[:, 0] - cx)**2 + (pts[:, 1] - cy)**2)
            min_d   = float(np.min(dists))
            closest = tuple(pts[int(np.argmin(dists))].astype(int))

            in_prox    = min_d <= proximity_threshold
            no_collide = min_d >= ex_r
            is_visited = in_prox and no_collide

            details[num] = {
                "centre"          : (cx, cy),
                "closest_route_pt": closest,
                "min_distance"    : round(min_d, 2),
                "exhibit_radius"  : round(ex_r, 2),
                "in_proximity"    : in_prox,
                "no_collision"    : no_collide,
                "visited"         : is_visited,
            }

            if is_visited:
                visited.append(num)
                print(f"  ✅ Exhibit {num:>3s}: closest={min_d:.1f} px")
            elif not in_prox:
                missed.append(num)
                print(f"  ❌ Exhibit {num:>3s}: too far  ({min_d:.1f} px > "
                      f"{proximity_threshold} px threshold)")
            else:
                missed.append(num)
                print(f"  ❌ Exhibit {num:>3s}: collision ({min_d:.1f} px < "
                      f"exhibit radius {ex_r:.1f} px)")

        return {"visited"      : visited,
                "missed"       : missed,
                "visit_details": details,
                "all_visited"  : len(missed) == 0}

    # ──────────────────────────────────────────────────────────────────────────
    # Single-route optimality
    # ──────────────────────────────────────────────────────────────────────────

    def calculate_route_optimality(self, actual_distance_px: float,
                                    shortest_result: dict,
                                    tolerance: float = 0.15) -> dict:
        """
        Compare actual route distance against the shortest feasible path.

        Parameters
        ----------
        actual_distance_px : total arc length of the drawn route skeleton
        shortest_result    : dict returned by shortest_feasible_with_waypoints()
        tolerance          : acceptable overhead fraction (default 15%)

        Returns dict: shortest_feasible, actual_distance, ratio,
                       overhead_pct, is_optimal.
        """
        shortest = shortest_result.get("total_distance")
        if shortest is None or shortest == 0:
            print("  ⚠️  Cannot compute optimality — shortest path unavailable")
            return None

        ratio      = actual_distance_px / shortest
        overhead   = (ratio - 1.0) * 100.0
        is_optimal = ratio <= (1.0 + tolerance)

        # Convert px distances to metres using scale annotation
        def to_m(px): return round(px * self.mm_per_px / 1000.0, 2)

        print(f"\n[Route Optimality]")
        print(f"  Shortest feasible : {shortest:.1f} px  ({to_m(shortest)} m)")
        print(f"  Actual route      : {actual_distance_px:.1f} px  "
              f"({to_m(actual_distance_px)} m)")
        print(f"  Optimality ratio  : {ratio:.3f}  ({overhead:+.1f}% overhead)")
        print(f"  Tolerance         : ±{tolerance*100:.0f}%")
        print(f"  {'✅ OPTIMAL' if is_optimal else '❌ SUBOPTIMAL'}")

        return {"shortest_feasible_px" : round(shortest, 2),
                "shortest_feasible_m"  : to_m(shortest),
                "actual_distance_px"   : round(actual_distance_px, 2),
                "actual_distance_m"    : to_m(actual_distance_px),
                "optimality_ratio"     : round(ratio, 4),
                "overhead_pct"         : round(overhead, 2),
                "is_optimal"           : is_optimal,
                "tolerance"            : tolerance}

    # ──────────────────────────────────────────────────────────────────────────
    # Aggregate ROR
    # ──────────────────────────────────────────────────────────────────────────

    def calculate_ror(self, optimality_results: list) -> dict:
        """
        Compute Route Optimality Rate across multiple evaluated routes.

        Parameters
        ----------
        optimality_results : list of dicts from calculate_route_optimality()
                             (None entries are excluded as unevaluable)

        Returns dict: ror, n_evaluated, n_optimal, mean_ratio, mean_overhead_pct.
        """
        valid = [r for r in optimality_results if r is not None]
        if not valid:
            print("  ⚠️  No valid results — cannot compute ROR")
            return None

        n_opt = sum(1 for r in valid if r["is_optimal"])
        ror   = n_opt / len(valid)
        mean_ratio    = sum(r["optimality_ratio"] for r in valid) / len(valid)
        mean_overhead = sum(r["overhead_pct"]      for r in valid) / len(valid)

        print(f"\n{'='*55}")
        print(f"  ROUTE OPTIMALITY RATE (ROR)")
        print(f"{'='*55}")
        print(f"  Routes evaluated : {len(valid)}")
        print(f"  Optimal routes   : {n_opt}")
        print(f"  ROR              : {ror:.1%}")
        print(f"  Mean ratio       : {mean_ratio:.3f}")
        print(f"  Mean overhead    : {mean_overhead:+.1f}%")
        print(f"{'='*55}")

        return {"ror"              : round(ror, 4),
                "n_evaluated"      : len(valid),
                "n_optimal"        : n_opt,
                "mean_ratio"       : round(mean_ratio, 4),
                "mean_overhead_pct": round(mean_overhead, 2)}

    # ──────────────────────────────────────────────────────────────────────────
    # Convenience: evaluate a single route end-to-end
    # ──────────────────────────────────────────────────────────────────────────

    def evaluate_route(self, actual_route_px: float, start: tuple,
                        end: tuple, exhibit_numbers: list,
                        proximity_threshold: float = 40.0,
                        tolerance: float = 0.15,
                        skeleton_points: list = None) -> dict:
        """
        Full evaluation pipeline for one route.

        Parameters
        ----------
        actual_route_px     : total drawn route length in pixels
        start / end         : full-res (x, y) endpoints
        exhibit_numbers     : exhibit numbers the route should visit
        proximity_threshold : max px from exhibit centre to count as visited
        tolerance           : ROR tolerance (default 15%)
        skeleton_points     : optional list of (x, y) for exhibit visit
                              validation (if None, validation is skipped)

        Returns a merged result dict suitable for visualize().
        """
        shortest = self.shortest_feasible_with_waypoints(
            start, end, exhibit_numbers, proximity_threshold)

        optimality = self.calculate_route_optimality(
            actual_route_px, shortest, tolerance)

        visit_validation = None
        if skeleton_points is not None:
            visit_validation = self.validate_exhibit_visits(
                skeleton_points, exhibit_numbers, proximity_threshold)

        return {"start"            : start,
                "end"              : end,
                "exhibit_numbers"  : exhibit_numbers,
                "actual_route_px"  : actual_route_px,
                "shortest"         : shortest,
                "optimality"       : optimality,
                "visit_validation" : visit_validation,
                "proximity_threshold": proximity_threshold}

    # ──────────────────────────────────────────────────────────────────────────
    # Visualization
    # ──────────────────────────────────────────────────────────────────────────

    def visualize(self, eval_result: dict, output_path: str = None,
                  display_scale: float = 0.30) -> None:
        """
        Render a 3-panel visualization:
          Left   — walkable mask with all annotations overlaid
          Centre — optimal path stitched from Dijkstra segments
          Right  — ROR summary dashboard

        Parameters
        ----------
        eval_result   : dict from evaluate_route()
        output_path   : save to file if provided; otherwise plt.show()
        display_scale : downsample factor for rendering (0.3 = 30%)
        """
        print("\n[Visualizing…]")
        full  = self._get_walkable()
        s     = display_scale
        dw    = max(1, int(full.shape[1] * s))
        dh    = max(1, int(full.shape[0] * s))

        def sc(x): return int(round(x * s))

        # ── Build display canvas ──────────────────────────────────────
        # White background
        canvas = np.ones((dh, dw, 3), dtype=np.uint8) * 245

        # Walkable area (light blue-grey)
        walk_small = cv2.resize(full, (dw, dh), interpolation=cv2.INTER_NEAREST)
        canvas[walk_small > 0] = [220, 232, 245]

        # Floor area outline
        for fa in self.floor_areas:
            if fa["shape"] == "rectangle":
                c = fa["coordinates"]
                pt1 = (sc(c["x"]),              sc(c["y"]))
                pt2 = (sc(c["x"]+c["width"]),   sc(c["y"]+c["height"]))
                cv2.rectangle(canvas, pt1, pt2, (150, 180, 220), 1)

        # Forbidden areas (red hatch)
        for fb in self.forbidden:
            if fb["shape"] == "rectangle":
                c = fb["coordinates"]
                x1, y1 = sc(c["x"]), sc(c["y"])
                x2, y2 = sc(c["x"]+c["width"]), sc(c["y"]+c["height"])
                cv2.rectangle(canvas, (x1,y1), (x2,y2), (220, 180, 180), -1)
                cv2.rectangle(canvas, (x1,y1), (x2,y2), (200, 80, 80), 1)

        # Walls
        for wall in self.walls:
            if wall["shape"] == "polyline":
                pts = wall["coordinates"]["points"]
                arr = np.array([[sc(p["x"]), sc(p["y"])] for p in pts],
                               dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(canvas, [arr], False, (60, 60, 80), 2)

        # Exhibits
        visited_nums = set()
        if eval_result.get("visit_validation"):
            visited_nums = set(str(v) for v in
                               eval_result["visit_validation"]["visited"])

        target_nums = set(str(n) for n in eval_result.get("exhibit_numbers", []))

        for ex in self.exhibits:
            num = str(ex.get("exhibit_number", ""))
            c   = ex["coordinates"]
            cx, cy = sc(int(c["center_x"])), sc(int(c["center_y"]))
            r   = max(3, sc(int(c["radius"])))

            if num in visited_nums:
                color, lc = (80, 200, 120), (30, 140, 60)   # green = visited
            elif num in target_nums:
                color, lc = (255, 160, 60),  (200, 100, 20) # orange = target
            else:
                color, lc = (200, 200, 210), (140, 140, 160)# grey = other

            cv2.circle(canvas, (cx, cy), r, color, -1)
            cv2.circle(canvas, (cx, cy), r, lc,    1)

            # Label only target exhibits
            if num in target_nums:
                fs = max(0.25, s * 0.55)
                cv2.putText(canvas, num, (cx - r - 2, cy - r - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, fs, (80, 40, 0), 1,
                            cv2.LINE_AA)

        # Entrance / exit
        for ent in self.entrances:
            if ent["shape"] == "rectangle":
                c = ent["coordinates"]
                cv2.rectangle(canvas,
                              (sc(c["x"]),              sc(c["y"])),
                              (sc(c["x"]+c["width"]),   sc(c["y"]+c["height"])),
                              (30, 180, 60), 2)

        for ext in self.exits:
            if ext["shape"] == "rectangle":
                c = ext["coordinates"]
                cv2.rectangle(canvas,
                              (sc(c["x"]),              sc(c["y"])),
                              (sc(c["x"]+c["width"]),   sc(c["y"]+c["height"])),
                              (200, 200, 0), 2)

        # START / END
        start = eval_result["start"]
        end   = eval_result["end"]
        cv2.circle(canvas, (sc(start[0]), sc(start[1])),
                   max(6, sc(15)), (0, 200, 80), -1)
        cv2.circle(canvas, (sc(end[0]),   sc(end[1])),
                   max(6, sc(15)), (200, 40, 40), -1)

        # ── Build optimal-path overlay ────────────────────────────────
        path_canvas = canvas.copy()

        # Draw the full route as one continuous stitched polyline
        stitched = eval_result["shortest"].get("stitched_path", [])
        if len(stitched) > 1:
            pts = np.array([[sc(p[0]), sc(p[1])] for p in stitched],
                           dtype=np.int32).reshape(-1, 1, 2)
            # White shadow for contrast against dark backgrounds
            cv2.polylines(path_canvas, [pts], False, (255, 255, 255), 5)
            # Route line on top
            cv2.polylines(path_canvas, [pts], False, (255, 140, 0),   2)

        # Waypoint dots: one per exhibit stop in TSP order
        path_order = eval_result["shortest"].get("ordered_path", [])
        wp_map     = eval_result["shortest"].get("exhibit_waypoints", {})
        for step, lbl in enumerate(path_order):
            if lbl not in wp_map:
                continue
            pt = wp_map[lbl]
            # Filled white dot with dark border
            cv2.circle(path_canvas, (sc(pt[0]), sc(pt[1])),
                       max(5, sc(11)), (40, 40, 60),   -1)
            cv2.circle(path_canvas, (sc(pt[0]), sc(pt[1])),
                       max(5, sc(11)), (255, 255, 255),  2)
            # Step number inside the dot
            num_str = lbl.replace("Exhibit ", "")
            fs = max(0.25, s * 0.45)
            tw, th = cv2.getTextSize(num_str, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)[0]
            cv2.putText(path_canvas, num_str,
                        (sc(pt[0]) - tw//2, sc(pt[1]) + th//2),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 220, 100), 1, cv2.LINE_AA)

        # Re-draw START / END on top of route
        cv2.circle(path_canvas, (sc(start[0]), sc(start[1])),
                   max(6, sc(15)), (0, 200, 80), -1)
        cv2.circle(path_canvas, (sc(start[0]), sc(start[1])),
                   max(6, sc(15)), (255, 255, 255), 2)
        cv2.circle(path_canvas, (sc(end[0]),   sc(end[1])),
                   max(6, sc(15)), (200, 40, 40), -1)
        cv2.circle(path_canvas, (sc(end[0]),   sc(end[1])),
                   max(6, sc(15)), (255, 255, 255), 2)

        # ── Matplotlib figure ─────────────────────────────────────────
        fig = plt.figure(figsize=(20, 12), facecolor="#1a1a2e")

        ax1 = fig.add_axes([0.01, 0.05, 0.30, 0.88])
        ax2 = fig.add_axes([0.34, 0.05, 0.30, 0.88])
        ax3 = fig.add_axes([0.67, 0.05, 0.31, 0.88])

        for ax in (ax1, ax2):
            ax.imshow(cv2.cvtColor(canvas      if ax is ax1 else path_canvas,
                                   cv2.COLOR_BGR2RGB))
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor("#4a4a6a"); sp.set_linewidth(1.5)

        # Titles
        ax1.set_title("Museum Layout\n& Walkable Space",
                      color="white", fontsize=12, fontweight="bold", pad=8)
        ax2.set_title("Optimal Route\n(Dijkstra + TSP)",
                      color="white", fontsize=12, fontweight="bold", pad=8)

        # ── Dashboard panel ───────────────────────────────────────────
        ax3.set_facecolor("#12122a")
        ax3.set_xlim(0, 1); ax3.set_ylim(0, 1)
        ax3.set_xticks([]); ax3.set_yticks([])
        ax3.set_title("Route Optimality Dashboard",
                      color="white", fontsize=12, fontweight="bold", pad=8)
        for sp in ax3.spines.values():
            sp.set_edgecolor("#4a4a6a"); sp.set_linewidth(1.5)

        opt = eval_result.get("optimality", {}) or {}
        vv  = eval_result.get("visit_validation", {}) or {}

        # ── Gauge: optimality ratio ────────────────────────────────────
        ratio     = opt.get("optimality_ratio", 1.0)
        tol       = opt.get("tolerance",        0.15)
        is_opt    = opt.get("is_optimal",       False)
        gauge_col = "#2ecc71" if is_opt else "#e74c3c"

        theta1, theta2 = 180, 180 - min(ratio / 2.0, 1.0) * 180
        wedge = mpatches.Wedge((0.5, 0.72), 0.18, theta2, theta1,
                               width=0.06, facecolor=gauge_col, alpha=0.9)
        ax3.add_patch(wedge)
        # Background arc
        bg = mpatches.Wedge((0.5, 0.72), 0.18, 0, 180,
                            width=0.06, facecolor="#2a2a4a", zorder=0)
        ax3.add_patch(bg)
        wedge.set_zorder(1)

        ax3.text(0.5, 0.69, f"{ratio:.3f}×",
                 ha="center", va="center", color=gauge_col,
                 fontsize=22, fontweight="bold")
        ax3.text(0.5, 0.62, "Optimality Ratio  (1.00 = perfect)",
                 ha="center", va="center", color="#aaaacc", fontsize=8)

        status_txt = "✓  OPTIMAL" if is_opt else "✗  SUBOPTIMAL"
        ax3.text(0.5, 0.57,
                 status_txt,
                 ha="center", va="center",
                 color=gauge_col, fontsize=13, fontweight="bold")

        # ── Key metrics table ──────────────────────────────────────────
        def row(y, label, value, vc="#e8e8ff"):
            ax3.text(0.08, y, label, ha="left",  va="center",
                     color="#aaaacc",  fontsize=9)
            ax3.text(0.92, y, value, ha="right", va="center",
                     color=vc, fontsize=9, fontweight="bold")

        y0 = 0.50
        ax3.axhline(y0 + 0.02, xmin=0.04, xmax=0.96,
                    color="#3a3a5a", linewidth=0.8)

        row(y0 - 0.00, "Shortest feasible",
            f"{opt.get('shortest_feasible_px',0):.0f} px  "
            f"({opt.get('shortest_feasible_m',0):.1f} m)")
        row(y0 - 0.06, "Actual route",
            f"{opt.get('actual_distance_px',0):.0f} px  "
            f"({opt.get('actual_distance_m',0):.1f} m)")
        row(y0 - 0.12, "Overhead",
            f"{opt.get('overhead_pct',0):+.1f}%",
            vc=gauge_col)
        row(y0 - 0.18, "Tolerance",
            f"±{tol*100:.0f}%")
        row(y0 - 0.24, "TSP method",
            eval_result["shortest"].get("method", "—").replace("-", "\u2011"))

        # ── Exhibit visit summary ──────────────────────────────────────
        ax3.axhline(y0 - 0.29, xmin=0.04, xmax=0.96,
                    color="#3a3a5a", linewidth=0.8)
        ax3.text(0.5, y0 - 0.33, "Exhibit Visits",
                 ha="center", va="center",
                 color="#ccccee", fontsize=10, fontweight="bold")

        visited = vv.get("visited", [])
        missed  = vv.get("missed",  [])
        total_t = len(eval_result.get("exhibit_numbers", []))

        if vv:
            all_ok  = vv.get("all_visited", False)
            vv_col  = "#2ecc71" if all_ok else "#e67e22"
            ax3.text(0.5, y0 - 0.39,
                     f"{len(visited)} / {total_t} visited",
                     ha="center", va="center",
                     color=vv_col, fontsize=12, fontweight="bold")

            # Per-exhibit status dots
            nums  = [str(n) for n in eval_result.get("exhibit_numbers", [])]
            ncols = min(10, len(nums))
            nrows = math.ceil(len(nums) / ncols)
            xs_d  = np.linspace(0.08, 0.92, ncols)
            for idx, num in enumerate(nums):
                xi = idx % ncols
                yi = idx // ncols
                xp = xs_d[xi]
                yp = y0 - 0.46 - yi * 0.055
                c2 = "#2ecc71" if num in [str(v) for v in visited] else "#e74c3c"
                circ = plt.Circle((xp, yp), 0.018,
                                  color=c2, transform=ax3.transData, zorder=3)
                ax3.add_patch(circ)
                ax3.text(xp, yp, str(num),
                         ha="center", va="center",
                         color="white", fontsize=6, fontweight="bold", zorder=4)
        else:
            ax3.text(0.5, y0 - 0.39,
                     f"{total_t} exhibits targeted\n(no skeleton provided for validation)",
                     ha="center", va="center",
                     color="#888899", fontsize=9)

        # ── Legend ─────────────────────────────────────────────────────
        legend_y = 0.04
        legend_items = [
            (mpatches.Patch(color="#dce8f5"),          "Walkable floor"),
            (mpatches.Patch(color="#dc5050", alpha=0.6),"Forbidden area"),
            (Line2D([0],[0], color="#3c3c50",  lw=2),   "Wall"),
            (plt.Circle((0,0), 0.1, color="#50c878"),   "Visited exhibit"),
            (plt.Circle((0,0), 0.1, color="#ffa030"),   "Target exhibit"),
            (plt.Circle((0,0), 0.1, color="#c8c8d2"),   "Other exhibit"),
            (plt.Circle((0,0), 0.1, color="#00c850"),   "START"),
            (plt.Circle((0,0), 0.1, color="#c82828"),   "END"),
        ]
        fig.legend(
            [h for h, _ in legend_items],
            [l for _, l in legend_items],
            loc="lower center",
            ncol=8,
            fontsize=7.5,
            facecolor="#1a1a2e",
            edgecolor="#4a4a6a",
            labelcolor="white",
            framealpha=0.9,
            bbox_to_anchor=(0.5, -0.01),
        )

        fig.suptitle("Route Optimality Rate (ROR) Analysis",
                     color="white", fontsize=15, fontweight="bold", y=0.98)

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            print(f"  ✅ Saved to {output_path}")
        else:
            plt.show()
        plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Demo
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    ann_path = sys.argv[1] if len(sys.argv) > 1 \
               else "/mnt/user-data/uploads/museum_layout_annotations.json"

    analyzer = RouteOptimalityAnalyzer(ann_path, path_scale=0.25)

    # ── Entrance / exit centres from annotations ───────────────────────
    ann       = analyzer.ann
    ent_c     = ann["entrances"][0]["coordinates"]
    ext_c     = ann["exits"][0]["coordinates"]
    start_pt  = (int(ent_c["x"] + ent_c["width"]  / 2),
                 int(ent_c["y"] + ent_c["height"] / 2))
    end_pt    = (int(ext_c["x"] + ext_c["width"]  / 2),
                 int(ext_c["y"] + ext_c["height"] / 2))

    # ── Demo: visit exhibits 3, 7, 10, 13 ─────────────────────────────
    TARGET_EXHIBITS     = ["3", "7", "10", "13"]
    PROXIMITY_THRESHOLD = 55   # px

    print(f"\nDemo: START={start_pt}  END={end_pt}")
    print(f"      Visiting exhibits: {TARGET_EXHIBITS}")

    result = analyzer.evaluate_route(
        actual_route_px     = 5800.0,   # simulated actual route
        start               = start_pt,
        end                 = end_pt,
        exhibit_numbers     = TARGET_EXHIBITS,
        proximity_threshold = PROXIMITY_THRESHOLD,
        tolerance           = 0.20,
    )

    analyzer.visualize(result, output_path="/mnt/user-data/outputs/ror_visualization.png")

    # ── Simulate ROR across 3 routes ──────────────────────────────────
    shortest_dist = result["shortest"]["total_distance"]
    sim_routes = [
        shortest_dist * 1.05,   # near-optimal
        shortest_dist * 1.18,   # within 20% tolerance
        shortest_dist * 1.45,   # suboptimal
    ]
    opt_results = [
        analyzer.calculate_route_optimality(d, result["shortest"], tolerance=0.20)
        for d in sim_routes
    ]
    ror = analyzer.calculate_ror(opt_results)
    print(f"\nSimulated ROR across 3 routes: {ror['ror']:.1%}")