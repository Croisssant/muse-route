"""
Visualize Route Detection Process - Debug Script
Shows step-by-step how the route is detected and endpoints are identified
"""

import cv2
import numpy as np
from PIL import Image as PILImage, ImageDraw, ImageFont
import json
import argparse
import math
from pathlib import Path


class RouteDetectionVisualizer:
    def __init__(self, annotations_file):
        """Initialize with annotations for context."""
        self.annotations_file = annotations_file
        
        # Load annotations
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        
        self.entrances = self.annotations.get('entrances', [])
        self.exits = self.annotations.get('exits', [])
        self.walls = self.annotations.get('walls', [])
        self.floor_areas = self.annotations.get('floor_areas', [])
        
        # Calculate museum bounds
        self.museum_bounds = self._calculate_museum_bounds()
        
        print(f"Loaded annotations:")
        print(f"  Entrances: {len(self.entrances)}")
        print(f"  Exits: {len(self.exits)}")
        if self.museum_bounds:
            print(f"  Museum bounds: X=[{self.museum_bounds['min_x']:.0f}, {self.museum_bounds['max_x']:.0f}], "
                  f"Y=[{self.museum_bounds['min_y']:.0f}, {self.museum_bounds['max_y']:.0f}]")
    
    def _calculate_museum_bounds(self):
        """Calculate bounding box from all annotations."""
        if not self.walls:
            return None
        
        all_x = []
        all_y = []
        
        for wall in self.walls:
            if wall['shape'] == 'polyline':
                for point in wall['coordinates']['points']:
                    all_x.append(point['x'])
                    all_y.append(point['y'])
        
        if not all_x or not all_y:
            return None
        
        return {
            'min_x': min(all_x),
            'max_x': max(all_x),
            'min_y': min(all_y),
            'max_y': max(all_y)
        }
    
    def _distance(self, p1, p2):
        """Calculate Euclidean distance between two points."""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def _find_route_endpoints(self, points, neighbor_distance=5):
        """
        OLD METHOD: Find route endpoints by identifying points with few neighbors.
        Returns list of (point, neighbor_count) tuples.
        This is kept for comparison purposes.
        """
        if len(points) < 3:
            return [(p, 0) for p in points]
        
        endpoint_data = []
        
        for point in points:
            # Count neighbors within threshold distance
            neighbors = sum(1 for p in points 
                          if p != point and self._distance(p, point) <= neighbor_distance)
            
            # Points with 1 or 2 neighbors are likely endpoints
            if neighbors <= 2:
                endpoint_data.append((point, neighbors))
        
        return endpoint_data
    
    def _clean_skeleton(self, skeleton):
        """
        Clean skeleton to remove small branches and spurs that create false endpoints.
        """
        # Apply morphological opening to remove small protrusions
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(skeleton, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Optionally re-skeletonize to ensure it's still thin
        cleaned = self._skeletonize(cleaned)
        
        return cleaned
    
    def _merge_endpoint_clusters(self, endpoints, min_distance=50):
        """
        Merge nearby endpoints into clusters and keep only one per cluster.
        This removes false endpoints that are close together.
        """
        if len(endpoints) <= 2:
            return endpoints
        
        # Sort by position for consistent ordering
        endpoints = sorted(endpoints, key=lambda ep: (ep[0], ep[1]))
        
        clusters = []
        used = set()
        
        for i, ep1 in enumerate(endpoints):
            if i in used:
                continue
            
            cluster = [ep1]
            used.add(i)
            
            for j, ep2 in enumerate(endpoints):
                if j in used or i == j:
                    continue
                
                if self._distance(ep1, ep2) < min_distance:
                    cluster.append(ep2)
                    used.add(j)
            
            # Keep the center of the cluster
            center_x = sum(p[0] for p in cluster) / len(cluster)
            center_y = sum(p[1] for p in cluster) / len(cluster)
            clusters.append((int(center_x), int(center_y)))
        
        return clusters
    
    def _filter_edge_endpoints(self, endpoints, image_shape, edge_margin=50):
        """
        Remove endpoints near image edges (likely artifacts from image differences).
        
        Args:
            endpoints: List of endpoint coordinates
            image_shape: Shape of the image (h, w)
            edge_margin: Pixels from edge to consider as border (default: 50)
        
        Returns:
            Filtered list of endpoints away from edges
        """
        if not endpoints:
            return endpoints
        
        h, w = image_shape[:2]
        filtered = []
        removed = []
        
        for ep in endpoints:
            x, y = ep
            # Keep only if not near any edge
            if (edge_margin < x < w - edge_margin and 
                edge_margin < y < h - edge_margin):
                filtered.append(ep)
            else:
                removed.append(ep)
        
        if removed:
            print(f"      Filtered out {len(removed)} endpoints near image edges:")
            for ep in removed:
                x, y = ep
                edge = []
                if x <= edge_margin: edge.append("left")
                if x >= w - edge_margin: edge.append("right")
                if y <= edge_margin: edge.append("top")
                if y >= h - edge_margin: edge.append("bottom")
                print(f"        {ep} (near {'/'.join(edge)} edge)")
        
        # If all endpoints were filtered, return originals as fallback
        return filtered if filtered else endpoints
    
    def _build_adjacency(self, points):
        """Build 8-connected adjacency dict from a list of (x, y) skeleton points."""
        point_set = set(points)
        adjacency = {p: [] for p in point_set}
        for (x, y) in point_set:
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nb = (x + dx, y + dy)
                    if nb in point_set:
                        adjacency[(x, y)].append(nb)
        return adjacency

    def _find_degree1_endpoints(self, adjacency):
        """Return all nodes with exactly 1 neighbour (true skeleton terminals)."""
        return [p for p, nbs in adjacency.items() if len(nbs) == 1]

    def _walk_path(self, start, adjacency):
        """
        Brute-force longest-path DFS on the skeleton graph (which is a tree
        after skeletonisation — no cycles).

        Explicit stack carries (current_node, path_so_far, visited_frozenset).
        At every junction ALL unvisited branches are pushed; dead-end paths
        compete by length and the longest wins.

        Returns the ordered pixel list of the longest path from `start`.
        ordered[0] == start, ordered[-1] == true far terminal end.
        """
        best = [start]
        stack = [(start, [start], frozenset([start]))]
        while stack:
            node, path, visited = stack.pop()
            nbs = [nb for nb in adjacency[node] if nb not in visited]
            if not nbs:
                if len(path) > len(best):
                    best = path
                continue
            for nb in nbs:
                stack.append((nb, path + [nb], visited | {nb}))
        return best

    def _order_path_from_start(self, points, start_point, skeleton=None):
        """Thin wrapper kept for visualisation reuse; delegates to _walk_path."""
        return self._walk_path_from_nearest_d1(points, start_point)

    def _walk_path_from_nearest_d1(self, points, hint_start):
        """
        Build adjacency, snap hint_start to nearest degree-1 node, then run
        _walk_path.  Returns the ordered pixel list (start … end).
        """
        if len(points) <= 1:
            return list(points)
        adjacency = self._build_adjacency(points)
        endpoints_d1 = self._find_degree1_endpoints(adjacency)
        if endpoints_d1:
            walk_start = min(endpoints_d1, key=lambda p: self._distance(p, hint_start))
        else:
            walk_start = min(points, key=lambda p: self._distance(p, hint_start))
        return self._walk_path(walk_start, adjacency)
    
    def _find_route_endpoints_improved(self, points, neighbor_distance=10, image_shape=None, edge_margin=50, skeleton=None):
        """
        Find START and END of the route and store the full ordered path.

        START = degree-1 skeleton node closest to the entrance annotation.
        END   = ordered[-1] from the brute-force longest-path walk.

        Stores self._ordered_path for reuse by _create_visualization so that
        cyan index labels use exactly the same ordering as the endpoints.

        Returns [(start, 0), (end, 0)].
        """
        if len(points) < 3:
            self._ordered_path = list(points)
            return [(p, 0) for p in points]

        if not self.entrances:
            self._ordered_path = list(points)
            return [(points[0], 0), (points[-1], 0)]

        entrance = self.entrances[0]
        if entrance['shape'] != 'rectangle':
            self._ordered_path = list(points)
            return [(points[0], 0), (points[-1], 0)]

        coords = entrance['coordinates']
        entrance_center = (
            coords['x'] + coords['width']  / 2,
            coords['y'] + coords['height'] / 2,
        )

        print(f"      Finding endpoints via brute-force longest-path walk...")

        # Build graph and find degree-1 terminals
        adjacency    = self._build_adjacency(points)
        endpoints_d1 = self._find_degree1_endpoints(adjacency)
        print(f"      Degree-1 endpoints found: {len(endpoints_d1)}")

        # START = degree-1 node closest to entrance
        if endpoints_d1:
            walk_start = min(endpoints_d1, key=lambda p: self._distance(p, entrance_center))
        else:
            walk_start = min(points,       key=lambda p: self._distance(p, entrance_center))

        print(f"      START: {walk_start} ({self._distance(walk_start, entrance_center):.1f}px from entrance)")

        # Walk longest path from START — end is ordered[-1]
        ordered = self._walk_path(walk_start, adjacency)
        end_point = ordered[-1]

        print(f"      END  : {end_point}")
        print(f"      Path : {len(ordered)} pixels ordered start→end")

        # Store for visualization
        self._ordered_path = ordered

        return [(walk_start, 0), (end_point, 0)]

    def _extract_route_component(self, skeleton):
        """
        Extract the route component from skeleton based on proximity to entrance.
        This removes disconnected artifacts like edge lines.
        
        Returns:
            tuple: (cleaned_skeleton, num_components, kept_component_size)
        """
        # Find all connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            skeleton, connectivity=8
        )
        
        if num_labels <= 1:  # Only background
            return skeleton, 0, cv2.countNonZero(skeleton)
        
        # Calculate entrance center if available
        if not self.entrances:
            # Fallback: keep largest component
            largest_label = 1
            largest_size = 0
            for label in range(1, num_labels):
                size = stats[label, cv2.CC_STAT_AREA]
                if size > largest_size:
                    largest_size = size
                    largest_label = label
            
            route_component = np.zeros_like(skeleton)
            route_component[labels == largest_label] = 255
            
            print(f"    Found {num_labels-1} components, kept largest ({largest_size} pixels)")
            return route_component, num_labels-1, largest_size
        
        # Use entrance to select correct component
        entrance = self.entrances[0]
        if entrance['shape'] == 'rectangle':
            coords = entrance['coordinates']
            entrance_center = (
                coords['x'] + coords['width'] / 2,
                coords['y'] + coords['height'] / 2
            )
        
        # Strategy: Keep largest component within reasonable distance of entrance
        # This balances size (likely the route) with proximity (should start near entrance)
        
        max_entrance_distance = 1500  # Maximum distance from entrance to consider
        min_component_size = 500  # Minimum size to be considered a route
        
        # Find large components within reasonable distance of entrance
        candidates = []
        
        for label in range(1, num_labels):
            size = stats[label, cv2.CC_STAT_AREA]
            centroid = centroids[label]
            dist = self._distance((centroid[0], centroid[1]), entrance_center)
            
            if size >= min_component_size and dist <= max_entrance_distance:
                candidates.append((label, size, dist))
        
        if not candidates:
            # Fallback: if no good candidates, keep largest component
            largest_label = 1
            largest_size = 0
            for label in range(1, num_labels):
                size = stats[label, cv2.CC_STAT_AREA]
                if size > largest_size:
                    largest_size = size
                    largest_label = label
            best_label = largest_label
            component_size = largest_size
            min_dist = self._distance((centroids[best_label][0], centroids[best_label][1]), entrance_center)
            print(f"    No valid candidates, keeping largest component ({component_size} pixels, {min_dist:.1f}px from entrance)")
        else:
            # Among candidates, keep the largest one
            candidates.sort(key=lambda x: x[1], reverse=True)  # Sort by size descending
            best_label, component_size, min_dist = candidates[0]
            
            print(f"    Found {len(candidates)} valid candidates (>={min_component_size}px, <={max_entrance_distance}px from entrance)")
            print(f"    Kept largest candidate: component #{best_label} ({component_size} pixels, {min_dist:.1f}px from entrance)")
        
        # Keep only the route component
        route_component = np.zeros_like(skeleton)
        route_component[labels == best_label] = 255
        component_size = stats[best_label, cv2.CC_STAT_AREA]
        
        print(f"    Found {num_labels-1} components")
        print(f"    Kept component #{best_label} (closest to entrance: {min_dist:.1f}px, size: {component_size} pixels)")
        
        # Show what was filtered out
        for label in range(1, num_labels):
            if label != best_label:
                size = stats[label, cv2.CC_STAT_AREA]
                centroid_dist = self._distance((centroids[label][0], centroids[label][1]), entrance_center)
                print(f"      Filtered component #{label}: {size} pixels (distance: {centroid_dist:.1f}px from entrance)")
        
        return route_component, num_labels-1, component_size
    
    def _skeletonize(self, mask):
        """Apply skeletonization to get route centerline."""
        skeleton = np.zeros(mask.shape, np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        temp_mask = mask.copy()
        
        iteration = 0
        while True:
            eroded = cv2.erode(temp_mask, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(temp_mask, temp)
            skeleton = cv2.bitwise_or(skeleton, temp)
            temp_mask = eroded.copy()
            
            iteration += 1
            if cv2.countNonZero(temp_mask) == 0:
                break
        
        print(f"    Skeletonization completed in {iteration} iterations")
        return skeleton
    
    def extract_and_visualize(self, route_image_path, original_image_path, 
                             difference_threshold=10, output_path='route_detection_debug.png',
                             endpoints_only=False, marker_size=10):
        """
        Extract route and create comprehensive visualization of the process.
        
        Args:
            endpoints_only: If True, generate only Panel 8 (endpoints) instead of full grid
            marker_size: Size of START/END marker circles in pixels
        """
        print("\n" + "="*70)
        print("ROUTE DETECTION VISUALIZATION")
        print("="*70)
        
        # Load images
        print("\n[1/7] Loading images...")
        route_pil = PILImage.open(str(route_image_path))
        original_pil = PILImage.open(str(original_image_path))
        
        # Convert to RGB
        if route_pil.mode == 'RGBA':
            route_pil = route_pil.convert('RGB')
        if original_pil.mode == 'RGBA':
            original_pil = original_pil.convert('RGB')
        
        route_img = np.array(route_pil)
        original_img = np.array(original_pil)
        
        # Convert to BGR for OpenCV
        route_img = cv2.cvtColor(route_img, cv2.COLOR_RGB2BGR)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
        
        print(f"    Route image: {route_img.shape[:2]} (H x W)")
        print(f"    Original image: {original_img.shape[:2]} (H x W)")
        
        # Compute difference
        print("\n[2/7] Computing image difference...")
        difference = cv2.absdiff(route_img, original_img)
        gray_diff = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold
        print(f"\n[3/7] Applying threshold (threshold={difference_threshold})...")
        _, mask_threshold = cv2.threshold(gray_diff, difference_threshold, 255, cv2.THRESH_BINARY)
        pixel_count = cv2.countNonZero(mask_threshold)
        print(f"    Detected {pixel_count} changed pixels")
        
        # Apply ROI mask if available (with asymmetric margins to exclude edge artifacts)
        mask_roi = mask_threshold.copy()
        if self.museum_bounds:
            print(f"\n[4/7] Applying museum ROI mask (asymmetric margins to exclude edges)...")
            roi_mask = np.zeros(mask_threshold.shape, dtype=np.uint8)
            
            # Use asymmetric margins - generous on museum sides, tight on right edge
            left_margin = 200
            right_margin = 0      # NO margin on right to exclude artifact line
            top_margin = 200
            bottom_margin = 200
            
            x_min = max(0, int(self.museum_bounds['min_x'] - left_margin))
            x_max = min(mask_threshold.shape[1], int(self.museum_bounds['max_x'] + right_margin))
            y_min = max(0, int(self.museum_bounds['min_y'] - top_margin))
            y_max = min(mask_threshold.shape[0], int(self.museum_bounds['max_y'] + bottom_margin))
            
            roi_mask[y_min:y_max, x_min:x_max] = 255
            mask_roi = cv2.bitwise_and(mask_threshold, roi_mask)
            roi_pixel_count = cv2.countNonZero(mask_roi)
            
            print(f"    ROI bounds: X=[{x_min}, {x_max}], Y=[{y_min}, {y_max}]")
            print(f"    Margins: left={left_margin}, right={right_margin}, top={top_margin}, bottom={bottom_margin}")
            print(f"    After ROI: {roi_pixel_count} pixels")
        else:
            print("\n[4/7] Skipping ROI mask (no museum bounds)")
        
        # Morphological operations
        print("\n[5/7] Applying morphological operations...")
        kernel_small = np.ones((2, 2), np.uint8)
        mask_closed = cv2.morphologyEx(mask_roi, cv2.MORPH_CLOSE, kernel_small, iterations=1)
        mask_opened = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel_small, iterations=1)
        morph_pixel_count = cv2.countNonZero(mask_opened)
        print(f"    After morphology: {morph_pixel_count} pixels")
        
        # Skeletonization
        print("\n[6/7] Skeletonizing route...")
        skeleton = self._skeletonize(mask_opened)
        skeleton_pixel_count = cv2.countNonZero(skeleton)
        print(f"    Skeleton: {skeleton_pixel_count} pixels (centerline)")
        
        # Extract points
        points = np.column_stack(np.where(skeleton > 0))
        if len(points) == 0:
            print("ERROR: No skeleton points found!")
            return
        
        points = [(p[1], p[0]) for p in points]  # Convert to (x, y)
        print(f"    Extracted {len(points)} skeleton points")
        
        # Find endpoints
        print("\n[7/7] Detecting endpoints...")
        endpoint_data_new = self._find_route_endpoints_improved(
            points, neighbor_distance=10, 
            image_shape=skeleton.shape, edge_margin=100
        )
        endpoints_new = [ep[0] for ep in endpoint_data_new]

        # START and END come directly from _find_route_endpoints_improved.
        # That method already snapped to the correct degree-1 nodes and ran the
        # brute-force longest-path walk — no further re-derivation needed.
        start_point_new = endpoints_new[0] if endpoints_new else points[0]
        end_point_new   = endpoints_new[1] if len(endpoints_new) > 1 else points[-1]

        print(f"    START: {start_point_new}")
        print(f"    END  : {end_point_new}")
        
        # Create visualization
        print("\n" + "="*70)
        print("Creating visualization...")
        print("="*70)
        
        self._create_visualization(
            route_img, original_img, gray_diff, mask_threshold, 
            mask_roi, mask_closed, mask_opened, skeleton,
            points, start_point_new, end_point_new,
            output_path, endpoints_only, marker_size
        )
    
    def _create_visualization(self, route_img, original_img, gray_diff, 
                             mask_threshold, mask_roi, mask_closed, mask_opened, skeleton,
                             points, start_point, end_point, output_path, endpoints_only=False, marker_size=10):
        """Create visualization - either full 8-panel grid or just Panel 8 (endpoints)."""
        
        # Convert masks to RGB for visualization
        def mask_to_rgb(mask, color_false=(0,0,0), color_true=(255,255,255)):
            rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
            rgb[mask > 0] = color_true
            rgb[mask == 0] = color_false
            return rgb
        
        # Prepare panels (resize for display if needed)
        h, w = route_img.shape[:2]
        scale = 1.0
        max_dim = 800
        if h > max_dim or w > max_dim:
            scale = max_dim / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
        else:
            new_h, new_w = h, w
        
        def resize_img(img):
            if scale != 1.0:
                return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            return img
        
        # Create panels
        panel1 = resize_img(route_img.copy())
        panel2 = resize_img(original_img.copy())
        panel3 = resize_img(cv2.cvtColor(gray_diff, cv2.COLOR_GRAY2BGR))
        panel4 = resize_img(mask_to_rgb(mask_threshold))
        panel5 = resize_img(mask_to_rgb(mask_roi))
        panel6 = resize_img(mask_to_rgb(mask_opened))
        panel7 = resize_img(mask_to_rgb(skeleton, color_true=(0,255,0)))
        
        # Panel 8: Endpoint detection with START, END, and path index markers
        panel8 = resize_img(mask_to_rgb(skeleton, color_false=(30,30,30), color_true=(100,100,100)))
        
        # Draw path indices every 1000 points to show ordering
        # Reuse self._ordered_path set by _find_route_endpoints_improved so the
        # cyan labels reflect exactly the same start→end ordering as the markers.
        if len(points) > 100:
            ordered_for_vis = getattr(self, '_ordered_path', None)
            if ordered_for_vis is None:
                ordered_for_vis = self._walk_path_from_nearest_d1(list(points), start_point)
            font_small = cv2.FONT_HERSHEY_SIMPLEX
            for i in range(0, len(ordered_for_vis), 1000):
                pt = ordered_for_vis[i]
                pt_scaled = (int(pt[0] * scale), int(pt[1] * scale))
                cv2.circle(panel8, pt_scaled, 3, (255, 255, 0), -1)
                cv2.putText(panel8, str(i), (pt_scaled[0]+5, pt_scaled[1]-5),
                           font_small, 0.4, (255, 255, 0), 1)
        
        # Draw START point in bright green (using marker_size)
        marker_scaled = int(marker_size * scale)
        start_scaled = (int(start_point[0] * scale), int(start_point[1] * scale))
        cv2.circle(panel8, start_scaled, marker_scaled, (0, 255, 0), -1)
        cv2.circle(panel8, start_scaled, marker_scaled + 2, (255, 255, 255), 2)
        # cv2.putText(panel8, "START(0)", (start_scaled[0]+marker_scaled+5, start_scaled[1]), 
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Draw END point in bright red (using marker_size)
        end_scaled = (int(end_point[0] * scale), int(end_point[1] * scale))
        cv2.circle(panel8, end_scaled, marker_scaled, (0, 0, 255), -1)
        cv2.circle(panel8, end_scaled, marker_scaled + 2, (255, 255, 255), 2)
        # cv2.putText(panel8, "END", (end_scaled[0]+marker_scaled+5, end_scaled[1]), 
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Draw entrance/exit boxes
        if self.entrances:
            for entrance in self.entrances:
                if entrance['shape'] == 'rectangle':
                    coords = entrance['coordinates']
                    x1 = int(coords['x'] * scale)
                    y1 = int(coords['y'] * scale)
                    x2 = int((coords['x'] + coords['width']) * scale)
                    y2 = int((coords['y'] + coords['height']) * scale)
                    cv2.rectangle(panel8, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        if self.exits:
            for exit_ann in self.exits:
                if exit_ann['shape'] == 'rectangle':
                    coords = exit_ann['coordinates']
                    x1 = int(coords['x'] * scale)
                    y1 = int(coords['y'] * scale)
                    x2 = int((coords['x'] + coords['width']) * scale)
                    y2 = int((coords['y'] + coords['height']) * scale)
                    cv2.rectangle(panel8, (x1, y1), (x2, y2), (255, 255, 0), 2)
        
        # Add labels to panels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        def add_label(img, text, color=(255, 255, 255)):
            cv2.putText(img, text, (10, 30), font, font_scale, (0, 0, 0), thickness+2)
            cv2.putText(img, text, (10, 30), font, font_scale, color, thickness)
        
        # If endpoints_only, just save panel8
        if endpoints_only:
            print("Generating endpoints-only view (Panel 8)...")
            # Convert to PIL and save directly
            final_pil = PILImage.fromarray(cv2.cvtColor(panel8, cv2.COLOR_BGR2RGB))
            final_pil.save(output_path)
            print(f"\n✅ Endpoints visualization saved to: {output_path}")
            print(f"   Image size: {final_pil.size[0]} x {final_pil.size[1]} pixels")
            return
        
        add_label(panel1, "1. Route Image")
        add_label(panel2, "2. Original Image")
        add_label(panel3, "3. Difference (Gray)")
        add_label(panel4, f"4. Threshold ({cv2.countNonZero(mask_threshold)} px)")
        add_label(panel5, f"5. ROI Mask ({cv2.countNonZero(mask_roi)} px)")
        add_label(panel6, f"6. Morphology ({cv2.countNonZero(mask_opened)} px)")
        add_label(panel7, f"7. Skeleton ({cv2.countNonZero(skeleton)} px)")
        add_label(panel8, f"8. Endpoints Detected", color=(100, 255, 100))
        
        # Arrange in 4x2 grid
        row1 = np.hstack([panel1, panel2])
        row2 = np.hstack([panel3, panel4])
        row3 = np.hstack([panel5, panel6])
        row4 = np.hstack([panel7, panel8])
        
        grid = np.vstack([row1, row2, row3, row4])
        
        # Add extra space at bottom for legend (200 pixels)
        legend_space = np.ones((200, grid.shape[1], 3), dtype=np.uint8) * 40  # Dark gray background
        final = np.vstack([grid, legend_space])
        
        # Convert to PIL for adding detailed legend
        final_pil = PILImage.fromarray(cv2.cvtColor(final, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(final_pil)
        
        try:
            title_font = ImageFont.truetype("arial.ttf", 36)
            text_font = ImageFont.truetype("arial.ttf", 20)
            small_font = ImageFont.truetype("arial.ttf", 16)
        except:
            title_font = ImageFont.load_default()
            text_font = ImageFont.load_default()
            small_font = ImageFont.load_default()
        
        # Add title at top
        title = "Route Detection Process - Step by Step"
        title_width = draw.textlength(title, font=title_font) if hasattr(draw, 'textlength') else len(title) * 20
        title_x = (grid.shape[1] - title_width) // 2
        draw.text((title_x, 10), title, fill=(255, 255, 0), font=title_font)
        
        # Add legend in the dedicated space at bottom
        legend_y = grid.shape[0] + 20  # Start 20px into the legend space
        legend_x = 20
        
        draw.rectangle([legend_x-10, legend_y-10, legend_x+700, legend_y+170], 
                      fill=(20, 20, 20), outline=(255, 255, 255), width=2)
        
        draw.text((legend_x, legend_y), "Endpoint Detection Results", fill=(255, 255, 255), font=text_font)
        legend_y += 30
        
        # Green circle
        draw.ellipse([legend_x, legend_y, legend_x+20, legend_y+20], fill=(0, 255, 0))
        draw.text((legend_x+30, legend_y), f"= START at {start_point} ({self._distance(start_point, (self.entrances[0]['coordinates']['x'] + self.entrances[0]['coordinates']['width']/2, self.entrances[0]['coordinates']['y'] + self.entrances[0]['coordinates']['height']/2)):.1f}px from entrance)", fill=(255, 255, 255), font=small_font)
        legend_y += 30
        
        # Red circle
        draw.ellipse([legend_x, legend_y, legend_x+20, legend_y+20], fill=(255, 0, 0))
        draw.text((legend_x+30, legend_y), f"= END at {end_point} ({self._distance(start_point, end_point):.1f}px from START)", fill=(255, 255, 255), font=small_font)
        legend_y += 30
        
        # Boxes
        draw.rectangle([legend_x, legend_y, legend_x+20, legend_y+20], outline=(0, 255, 0), width=2)
        draw.text((legend_x+30, legend_y), "= Entrance", fill=(255, 255, 255), font=small_font)
        
        if self.exits:
            draw.rectangle([legend_x+250, legend_y, legend_x+270, legend_y+20], outline=(255, 255, 0), width=2)
            draw.text((legend_x+280, legend_y), "= Exit", fill=(255, 255, 255), font=small_font)
        
        # Save
        final_pil.save(output_path)
        print(f"\n✅ Visualization saved to: {output_path}")
        print(f"   Image size: {final_pil.size[0]} x {final_pil.size[1]} pixels")
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print("="*70)
        print(f"Review panel 8 to see if START (green) and END (red) are correct.")
        print(f"START should be near entrance, END should be at the other end of route.")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize route detection process step-by-step',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python visualize_route_detection_process.py \\
    --route-image valid_route.png \\
    --original-image museum_layout_01.png \\
    --annotations aligned_annotations.json \\
    --output route_detection_debug.png
        """
    )
    
    parser.add_argument('--route-image', required=True, help='Image with drawn route')
    parser.add_argument('--original-image', required=True, help='Original floor plan (no route)')
    parser.add_argument('--annotations', required=True, help='JSON file with annotations')
    parser.add_argument('--difference-threshold', type=int, default=10,
                       help='Pixel difference threshold (default: 10)')
    parser.add_argument('--output', default='route_detection_debug.png',
                       help='Output visualization image (default: route_detection_debug.png)')
    parser.add_argument('--endpoints-only', action='store_true',
                       help='Generate only the endpoints detection panel (Panel 8) instead of full 8-panel grid')
    parser.add_argument('--marker-size', type=int, default=10,
                       help='Size of START/END marker circles in pixels (default: 10)')
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = RouteDetectionVisualizer(args.annotations)
    
    # Extract and visualize
    visualizer.extract_and_visualize(
        route_image_path=args.route_image,
        original_image_path=args.original_image,
        difference_threshold=args.difference_threshold,
        output_path=args.output,
        endpoints_only=args.endpoints_only,
        marker_size=args.marker_size
    )


if __name__ == '__main__':
    main()
