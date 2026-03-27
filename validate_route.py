"""
Route Validation Script for Museum Floor Plans
Validates visitor routes against floor plan annotations (walls, exhibits)
"""

import cv2
import numpy as np
import json
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import math

class RouteValidator:
    def __init__(self, annotations_file, proximity_threshold=25):
        """
        Initialize the route validator.
        
        Args:
            annotations_file: Path to JSON file with wall and exhibit annotations
            proximity_threshold: Distance in pixels for exhibit visit detection
        """
        self.annotations_file = annotations_file
        self.proximity_threshold = proximity_threshold
        
        # Load annotations
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        
        # Extract walls and exhibits
        self.walls = self.annotations.get('walls', [])
        self.exhibits = self.annotations.get('exhibits', [])
        self.entrances = self.annotations.get('entrances', [])
        self.exits = self.annotations.get('exits', [])
        self.floor_areas = self.annotations.get('floor_areas', [])
        
        # Calculate museum bounds from annotations
        self.museum_bounds = self._calculate_museum_bounds()
        
        print(f"Loaded {len(self.walls)} wall(s) and {len(self.exhibits)} exhibit(s)")
        print(f"Loaded {len(self.entrances)} entrance(s), {len(self.exits)} exit(s), and {len(self.floor_areas)} floor area(s)")
        if self.museum_bounds:
            print(f"Museum bounds: X=[{self.museum_bounds['min_x']}, {self.museum_bounds['max_x']}], "
                  f"Y=[{self.museum_bounds['min_y']}, {self.museum_bounds['max_y']}]")
    
    def _calculate_museum_bounds(self):
        """Calculate bounding box from all annotations."""
        if not self.walls and not self.exhibits:
            return None
        
        all_x = []
        all_y = []
        
        # Collect wall points
        for wall in self.walls:
            if wall['shape'] == 'polyline':
                for point in wall['coordinates']['points']:
                    all_x.append(point['x'])
                    all_y.append(point['y'])
        
        # Collect exhibit points
        for exhibit in self.exhibits:
            if exhibit['shape'] == 'circle':
                cx = exhibit['coordinates']['center_x']
                cy = exhibit['coordinates']['center_y']
                r = exhibit['coordinates']['radius']
                all_x.extend([cx - r, cx + r])
                all_y.extend([cy - r, cy + r])
        
        if not all_x or not all_y:
            return None
        
        return {
            'min_x': min(all_x),
            'max_x': max(all_x),
            'min_y': min(all_y),
            'max_y': max(all_y)
        }
    
    def extract_route_by_difference(self, route_image_path, original_image_path, 
                                    difference_threshold=10, min_route_pixels=50):
        """
        Extract route points by comparing route image with original floor plan.
        
        Args:
            route_image_path: Path to image with drawn route
            original_image_path: Path to original floor plan (no route)
            difference_threshold: Minimum pixel difference to detect route (0-255)
            min_route_pixels: Minimum number of route pixels to consider valid
        
        Returns:
            List of (x, y) tuples representing route points
        """
        print(f"\nExtracting route by image difference...")
        print(f"  Route image: {route_image_path}")
        print(f"  Original image: {original_image_path}")
        
        # Load images using PIL to handle RGBA properly
        from PIL import Image as PILImage
        route_pil = PILImage.open(str(route_image_path))
        original_pil = PILImage.open(str(original_image_path))
        
        # Convert to RGB (ignore alpha channel if present)
        if route_pil.mode == 'RGBA':
            route_pil = route_pil.convert('RGB')
        if original_pil.mode == 'RGBA':
            original_pil = original_pil.convert('RGB')
        
        # Convert to numpy arrays (RGB format)
        route_img = np.array(route_pil)
        original_img = np.array(original_pil)
        
        # Convert from RGB to BGR for OpenCV
        route_img = cv2.cvtColor(route_img, cv2.COLOR_RGB2BGR)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
        
        if route_img is None or original_img is None:
            raise ValueError(f"Could not load images")
        
        # Check dimensions match exactly
        if route_img.shape != original_img.shape:
            raise ValueError(
                f"Image dimensions don't match!\n"
                f"  Original: {original_img.shape[:2]} (H x W)\n"
                f"  Route: {route_img.shape[:2]} (H x W)\n"
                f"Please ensure both images have identical dimensions."
            )
        
        print(f"  Image dimensions: {route_img.shape[:2]} (H x W) ✓")
        print(f"  Channels: {route_img.shape[2]} (RGB only, alpha ignored)")
        
        # Compute absolute difference
        difference = cv2.absdiff(route_img, original_img)
        
        # Convert to grayscale
        gray_diff = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to create binary mask
        _, mask = cv2.threshold(gray_diff, difference_threshold, 255, cv2.THRESH_BINARY)
        
        # Check if we detected enough pixels
        route_pixel_count = cv2.countNonZero(mask)
        print(f"  Detected {route_pixel_count} changed pixels")
        
        if route_pixel_count < min_route_pixels:
            raise ValueError(
                f"Too few route pixels detected ({route_pixel_count} < {min_route_pixels}).\n"
                f"Possible issues:\n"
                f"  - No route drawn on image\n"
                f"  - Route too faint (try lowering --difference-threshold)\n"
                f"  - Images are identical"
            )
        
        # Apply ROI mask to focus only on museum area (if bounds available)
        if self.museum_bounds:
            mask = self._apply_museum_roi_mask(mask)
        
        # Check if we still have enough pixels after ROI filtering
        roi_pixel_count = cv2.countNonZero(mask)
        print(f"  After ROI filtering: {roi_pixel_count} pixels remain")
        
        if roi_pixel_count < min_route_pixels:
            raise ValueError(
                f"Too few route pixels after ROI filtering ({roi_pixel_count} < {min_route_pixels}).\n"
                f"The route may not be drawn in the museum area."
            )
        
        # Apply lighter morphological operations to preserve route structure
        # Use smaller kernel to avoid over-erosion
        kernel_small = np.ones((2, 2), np.uint8)
        kernel_medium = np.ones((3, 3), np.uint8)
        
        # Close small gaps
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_small, iterations=1)
        # Remove small noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
        
        after_morph = cv2.countNonZero(mask)
        print(f"  After morphological operations: {after_morph} pixels remain")
        
        # Use skeletonization to extract thin centerline from thick route
        # This gives a clean 1-pixel-wide path that follows the actual route
        print(f"  Applying skeletonization to extract route centerline...")
        skeleton = self._skeletonize(mask)
        skeleton_pixels = cv2.countNonZero(skeleton)
        print(f"  Skeleton extracted: {skeleton_pixels} pixels (centerline)")
        
        # Extract points from skeleton and determine endpoints
        route_points, endpoints = self._order_route_points(skeleton)
        
        print(f"  Extracted {len(route_points)} ordered route points from centerline")
        
        # Store endpoints and skeleton for validation and visualization
        self._determined_endpoints = endpoints
        self._route_skeleton = skeleton
        
        # Validate and filter route points based on museum bounds
        if route_points and self.museum_bounds:
            filtered_points = self._filter_route_by_bounds(route_points)
            return filtered_points
        
        return route_points
    
    def _apply_museum_roi_mask(self, mask, margin=200):
        """
        Apply Region of Interest mask to focus only on museum area.
        This filters out border artifacts (like those added by MS Paint).
        
        Args:
            mask: Binary mask of detected differences
            margin: Extra margin in pixels around museum bounds
        
        Returns:
            Filtered mask with only museum ROI
        """
        print(f"  Applying ROI mask to focus on museum area (margin={margin}px)...")
        
        # Create ROI mask
        roi_mask = np.zeros(mask.shape, dtype=np.uint8)
        
        # Define ROI bounds with margin (convert to int for array indexing)
        x_min = max(0, int(self.museum_bounds['min_x'] - margin))
        x_max = min(mask.shape[1], int(self.museum_bounds['max_x'] + margin))
        y_min = max(0, int(self.museum_bounds['min_y'] - margin))
        y_max = min(mask.shape[0], int(self.museum_bounds['max_y'] + margin))
        
        # Set ROI area to white (255)
        roi_mask[y_min:y_max, x_min:x_max] = 255
        
        # Apply ROI mask to difference mask
        masked = cv2.bitwise_and(mask, roi_mask)
        
        # Count pixels before and after
        before_count = cv2.countNonZero(mask)
        after_count = cv2.countNonZero(masked)
        filtered_count = before_count - after_count
        
        print(f"     ROI bounds: X=[{x_min}, {x_max}], Y=[{y_min}, {y_max}]")
        print(f"     Filtered out {filtered_count} pixels outside museum area")
        print(f"     Kept {after_count} pixels within museum ROI")
        
        if after_count < 50:
            print(f"     ⚠️  WARNING: Very few pixels remain after ROI filtering!")
            print(f"     This might mean the route is not drawn in the museum area.")
        
        return masked
    
    def _filter_route_by_bounds(self, route_points, margin=200):
        """
        Filter route points to only include those within museum bounds.
        
        Args:
            route_points: List of (x, y) tuples
            margin: Extra margin in pixels around museum bounds
        
        Returns:
            Filtered list of route points
        """
        if not route_points:
            return route_points
        
        # Calculate route bounds
        route_xs = [p[0] for p in route_points]
        route_ys = [p[1] for p in route_points]
        route_bounds = {
            'min_x': min(route_xs),
            'max_x': max(route_xs),
            'min_y': min(route_ys),
            'max_y': max(route_ys)
        }
        
        print(f"\n  🔍 Route Detection Analysis:")
        print(f"     Detected route bounds: X=[{route_bounds['min_x']}, {route_bounds['max_x']}], "
              f"Y=[{route_bounds['min_y']}, {route_bounds['max_y']}]")
        print(f"     Museum bounds:         X=[{self.museum_bounds['min_x']}, {self.museum_bounds['max_x']}], "
              f"Y=[{self.museum_bounds['min_y']}, {self.museum_bounds['max_y']}]")
        
        # Expanded museum bounds with margin
        expanded_bounds = {
            'min_x': self.museum_bounds['min_x'] - margin,
            'max_x': self.museum_bounds['max_x'] + margin,
            'min_y': self.museum_bounds['min_y'] - margin,
            'max_y': self.museum_bounds['max_y'] + margin
        }
        
        # Filter points
        filtered_points = [
            p for p in route_points
            if (expanded_bounds['min_x'] <= p[0] <= expanded_bounds['max_x'] and
                expanded_bounds['min_y'] <= p[1] <= expanded_bounds['max_y'])
        ]
        
        removed_count = len(route_points) - len(filtered_points)
        
        if removed_count > 0:
            print(f"     ⚠️  WARNING: Removed {removed_count} points outside museum area")
            print(f"     This suggests the route was detected in the wrong location!")
            print(f"     Possible causes:")
            print(f"       - Extra content differs between images (watermarks, labels, etc.)")
            print(f"       - Route drawn outside the museum area")
            print(f"       - Image format/compression differences")
        
        if len(filtered_points) == 0:
            raise ValueError(
                f"❌ No valid route points found within museum area!\n"
                f"   Detected route at: X=[{route_bounds['min_x']}-{route_bounds['max_x']}], "
                f"Y=[{route_bounds['min_y']}-{route_bounds['max_y']}]\n"
                f"   Museum is at:      X=[{self.museum_bounds['min_x']}-{self.museum_bounds['max_x']}], "
                f"Y=[{self.museum_bounds['min_y']}-{self.museum_bounds['max_y']}]\n\n"
                f"   The detected 'route' is completely outside the museum area.\n"
                f"   This means the algorithm is detecting differences in the wrong part of the image.\n\n"
                f"   Troubleshooting steps:\n"
                f"   1. Verify both images are the EXACT same base image\n"
                f"   2. Check for watermarks, signatures, or labels that differ\n"
                f"   3. Ensure the route is actually drawn on the museum area\n"
                f"   4. Try saving both images in the same format (PNG recommended)\n"
                f"   5. Check if there's extra whitespace or content at image edges"
            )
        
        print(f"     ✅ Kept {len(filtered_points)} valid route points within museum area")
        
        return filtered_points
    
    def _skeletonize(self, mask):
        """Apply skeletonization to get route centerline."""
        skeleton = np.zeros(mask.shape, np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        
        while True:
            eroded = cv2.erode(mask, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(mask, temp)
            skeleton = cv2.bitwise_or(skeleton, temp)
            mask = eroded.copy()
            
            if cv2.countNonZero(mask) == 0:
                break
        
        return skeleton
    
    def _order_route_points(self, skeleton):
        """Extract and order points from skeletonized route using entrance/exit boxes."""
        # Find all non-zero points
        points = np.column_stack(np.where(skeleton > 0))
        
        if len(points) == 0:
            return [], {'start': None, 'end': None}
        
        # Convert to (x, y) format
        points = [(p[1], p[0]) for p in points]  # Convert to (x, y)
        
        # Find START and END based on entrance/exit boxes
        endpoints_result = self._find_route_endpoints_by_boxes(points)
        
        # Return the full list of points and the determined endpoints
        return points, endpoints_result
    
    def _find_route_endpoints_by_boxes(self, points):
        """
        Find START and END points based on entrance/exit bounding boxes.
        
        Args:
            points: List of (x, y) route pixels
        
        Returns:
            Dictionary with 'start' and 'end' points, or None if not found
        """
        result = {
            'start': None,
            'end': None
        }
        
        # Find entrance points
        if not self.entrances:
            return result
        
        entrance = self.entrances[0]
        if entrance['shape'] != 'rectangle':
            return result
        
        entrance_coords = entrance['coordinates']
        entrance_center = (
            entrance_coords['x'] + entrance_coords['width'] / 2,
            entrance_coords['y'] + entrance_coords['height'] / 2
        )
        
        # Find all route points within entrance box
        entrance_points = []
        for point in points:
            if self.point_in_rectangle(point, entrance_coords):
                entrance_points.append(point)
        
        if entrance_points:
            # Choose the point closest to entrance center as START
            start_point = min(entrance_points, key=lambda p: self._distance(p, entrance_center))
            result['start'] = start_point
        
        # Find exit points
        if not self.exits:
            return result
        
        exit_ann = self.exits[0]
        if exit_ann['shape'] != 'rectangle':
            return result
        
        exit_coords = exit_ann['coordinates']
        exit_center = (
            exit_coords['x'] + exit_coords['width'] / 2,
            exit_coords['y'] + exit_coords['height'] / 2
        )
        
        # Find all route points within exit box
        exit_points = []
        for point in points:
            if self.point_in_rectangle(point, exit_coords):
                exit_points.append(point)
        
        if exit_points:
            # Choose the point closest to exit center as END
            end_point = min(exit_points, key=lambda p: self._distance(p, exit_center))
            result['end'] = end_point
        
        return result
    
    
    def _distance(self, p1, p2):
        """Calculate Euclidean distance between two points."""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def point_in_polygon(self, point, polygon_points):
        """
        Check if a point is inside a polygon using ray casting algorithm.
        
        Args:
            point: (x, y) tuple
            polygon_points: List of (x, y) tuples forming polygon
        
        Returns:
            True if point is inside or on boundary, False otherwise
        """
        x, y = point
        n = len(polygon_points)
        inside = False
        
        p1x, p1y = polygon_points[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon_points[i % n]
            
            # Check if point is on the edge
            if self._point_on_segment(point, (p1x, p1y), (p2x, p2y)):
                return True  # On boundary is valid
            
            # Ray casting algorithm: cast a ray from point to the right
            # Count how many times it intersects the polygon edges
            # If odd number of intersections, point is inside
            if ((p1y > y) != (p2y > y)) and (x < (p2x - p1x) * (y - p1y) / (p2y - p1y) + p1x):
                inside = not inside
            
            p1x, p1y = p2x, p2y
        
        return inside
    
    def _point_on_segment(self, point, seg_start, seg_end, tolerance=2):
        """Check if point is on line segment within tolerance."""
        px, py = point
        x1, y1 = seg_start
        x2, y2 = seg_end
        
        # Calculate distance from point to line segment
        dx = x2 - x1
        dy = y2 - y1
        
        if dx == 0 and dy == 0:
            return self._distance(point, seg_start) <= tolerance
        
        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
        
        nearest_x = x1 + t * dx
        nearest_y = y1 + t * dy
        
        dist = math.sqrt((px - nearest_x)**2 + (py - nearest_y)**2)
        return dist <= tolerance
    
    def point_in_rectangle(self, point, rect_coords):
        """
        Check if a point is inside a rectangle.
        
        Args:
            point: (x, y) tuple
            rect_coords: Dictionary with x, y, width, height
        
        Returns:
            True if point is inside rectangle
        """
        px, py = point
        x, y = rect_coords['x'], rect_coords['y']
        width, height = rect_coords['width'], rect_coords['height']
        
        return (x <= px <= x + width) and (y <= py <= y + height)
    
    def validate_route(self, route_points):
        """
        Validate route against walls, exhibits, floor areas, entrance/exit.
        
        Validation rules:
        1. Route must be within floor_area (if defined)
        2. Route must not intersect any walls (interior or exterior)
        3. Route must start from an entrance
        4. Route must exit from an exit (or entrance if no exit exists)
        5. Route must not collide with exhibits
        
        Args:
            route_points: List of (x, y) tuples
        
        Returns:
            Dictionary with validation results
        """
        print("\nValidating route...")
        
        violations = {
            'wall_crossings': [],
            'exhibit_collisions': [],
            'out_of_bounds': [],
            'floor_area_violations': [],
            'entrance_violations': [],
            'exit_violations': []
        }
        
        exhibit_visits = {}
        for exhibit in self.exhibits:
            exhibit_id = exhibit.get('exhibit_number', exhibit['id'])
            exhibit_visits[exhibit_id] = {
                'visited': False,
                'closest_distance': float('inf')
            }
        
        # Get all wall polygons (not just the first one - check ALL walls)
        wall_polygons = []
        for wall in self.walls:
            if wall['shape'] == 'polyline':
                polygon = [(p['x'], p['y']) for p in wall['coordinates']['points']]
                wall_polygons.append(polygon)
        
        # Get floor area rectangles
        floor_area_rects = []
        for floor_area in self.floor_areas:
            if floor_area['shape'] == 'rectangle':
                floor_area_rects.append(floor_area['coordinates'])
        
        # Get determined START/END from entrance/exit boxes
        endpoints = getattr(self, '_determined_endpoints', {'start': None, 'end': None})
        
        # Validate START point (must be in an entrance box)
        if endpoints['start'] is None:
            violations['entrance_violations'].append({
                'point': None,
                'reason': 'No route pixels found in entrance bounding box'
            })
            print(f"  ❌ No route pixels found in entrance bounding box")
        else:
            print(f"  ✅ Route START found in entrance box: {endpoints['start']}")
        
        # Validate END point (must be in exit box)
        if endpoints['end'] is None:
            exit_type = "exit" if self.exits else "entrance (no exit defined)"
            violations['exit_violations'].append({
                'point': None,
                'reason': f'No route pixels found in {exit_type} bounding box'
            })
            print(f"  ❌ No route pixels found in {exit_type} bounding box")
        else:
            print(f"  ✅ Route END found in exit box: {endpoints['end']}")
        
        # Validate each point in the route
        for i, point in enumerate(route_points):
            # Check if point is within floor area (if floor areas are defined)
            if floor_area_rects:
                in_floor_area = False
                for floor_rect in floor_area_rects:
                    if self.point_in_rectangle(point, floor_rect):
                        in_floor_area = True
                        break
                
                if not in_floor_area:
                    violations['floor_area_violations'].append({
                        'point': [int(point[0]), int(point[1])],
                        'index': int(i),
                        'reason': 'outside designated floor area'
                    })
            
            # Check if point is going THROUGH ANY wall (inside wall polygon)
            # Note: Wall polygons trace the wall structure itself, not interior space
            # Valid routes should be OUTSIDE wall polygons (in walkable space)
            for wall_idx, wall_polygon in enumerate(wall_polygons):
                if self.point_in_polygon(point, wall_polygon):
                    violations['wall_crossings'].append({
                        'point': [int(point[0]), int(point[1])],
                        'index': int(i),
                        'wall_number': wall_idx + 1,
                        'reason': f'passing through wall #{wall_idx + 1}'
                    })
                    break  # Only report once per point
            
            # Check exhibit collisions and visits
            for exhibit in self.exhibits:
                if exhibit['shape'] == 'circle':
                    center = (exhibit['coordinates']['center_x'], 
                             exhibit['coordinates']['center_y'])
                    radius = exhibit['coordinates']['radius']
                    exhibit_id = exhibit.get('exhibit_number', exhibit['id'])
                    
                    dist = self._distance(point, center)
                    
                    # Update closest distance
                    if dist < exhibit_visits[exhibit_id]['closest_distance']:
                        exhibit_visits[exhibit_id]['closest_distance'] = float(dist)  # Convert to Python float
                    
                    # Check collision (route passes through exhibit)
                    if dist < radius:
                        violations['exhibit_collisions'].append({
                            'exhibit_id': exhibit_id,
                            'point': [int(point[0]), int(point[1])],  # Convert to Python int
                            'index': int(i),  # Convert to Python int
                            'distance': float(round(dist, 2))  # Convert to Python float
                        })
                    
                    # Check visit (within proximity threshold)
                    elif dist < radius + self.proximity_threshold:
                        exhibit_visits[exhibit_id]['visited'] = True
        
        # Compile results
        visited_exhibits = [eid for eid, data in exhibit_visits.items() if data['visited']]
        
        total_violations = (len(violations['wall_crossings']) + 
                          len(violations['exhibit_collisions']) + 
                          len(violations['out_of_bounds']) +
                          len(violations['floor_area_violations']) +
                          len(violations['entrance_violations']) +
                          len(violations['exit_violations']))
        
        is_valid = total_violations == 0
        
        result = {
            'validation_summary': {
                'is_valid': is_valid,
                'total_violations': total_violations,
                'route_length_pixels': len(route_points),
                'exhibits_visited': visited_exhibits
            },
            'violations': violations,
            'exhibit_visits': exhibit_visits
        }
        
        print(f"\nValidation Result: {'✅ VALID' if is_valid else '❌ INVALID'}")
        print(f"Total violations: {total_violations}")
        print(f"Exhibits visited: {len(visited_exhibits)}/{len(self.exhibits)}")
        
        return result
    
    def visualize_validation(self, original_image_path, route_points, validation_result, output_path):
        """
        Create annotated image showing validation results.
        
        Args:
            original_image_path: Path to original image
            route_points: List of route points
            validation_result: Validation result dictionary
            output_path: Path to save annotated image
        """
        print(f"\nCreating validation visualization...")
        
        # Load image with OpenCV to draw skeleton pixels
        img_cv = cv2.imread(str(original_image_path))
        if img_cv is None:
            # Fallback to PIL
            from PIL import Image as PILImage
            img_pil = PILImage.open(original_image_path)
            if img_pil.mode == 'RGBA':
                img_pil = img_pil.convert('RGB')
            img_cv = np.array(img_pil)
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
        
        # Get skeleton and draw it properly
        skeleton = getattr(self, '_route_skeleton', None)
        is_valid = validation_result['validation_summary']['is_valid']
        
        if skeleton is not None:
            # Draw actual skeleton pixels in green if valid, orange if invalid
            route_color_bgr = (0, 255, 0) if is_valid else (0, 150, 255)  # Green or Orange in BGR
            img_cv[skeleton > 0] = route_color_bgr
            print(f"  Drew {cv2.countNonZero(skeleton)} skeleton pixels")
        else:
            print(f"  ⚠️ WARNING: No skeleton available, skipping route visualization")
        
        # Convert to PIL for drawing text and markers
        img = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        
        # Try to load font
        try:
            font = ImageFont.truetype("arial.ttf", 16)
            font_small = ImageFont.truetype("arial.ttf", 12)
        except:
            font = ImageFont.load_default()
            font_small = ImageFont.load_default()
        
        # Draw visit zones (proximity threshold circles) around all exhibits FIRST
        print(f"  Drawing visit zones (radius + {self.proximity_threshold}px) around exhibits...")
        for exhibit in self.exhibits:
            if exhibit['shape'] == 'circle':
                exhibit_id = exhibit.get('exhibit_number', exhibit['id'])
                center = (int(exhibit['coordinates']['center_x']), 
                         int(exhibit['coordinates']['center_y']))
                radius = exhibit['coordinates']['radius']
                visit_zone_radius = int(radius + self.proximity_threshold)
                
                # Determine if this exhibit was visited
                was_visited = validation_result['exhibit_visits'][exhibit_id]['visited']
                
                if was_visited:
                    # Draw yellow/green visit zone for visited exhibits
                    draw.ellipse([center[0]-visit_zone_radius, center[1]-visit_zone_radius,
                                center[0]+visit_zone_radius, center[1]+visit_zone_radius],
                               outline=(200, 200, 0), width=2)  # Yellow outline
                else:
                    # Draw gray visit zone for unvisited exhibits
                    draw.ellipse([center[0]-visit_zone_radius, center[1]-visit_zone_radius,
                                center[0]+visit_zone_radius, center[1]+visit_zone_radius],
                               outline=(150, 150, 150), width=1)  # Gray outline
        
        # Mark violations
        violations = validation_result['violations']
        
        # Out of bounds points
        for violation in violations['out_of_bounds']:
            point = tuple(violation['point'])
            draw.ellipse([point[0]-5, point[1]-5, point[0]+5, point[1]+5], 
                        fill=(255, 0, 0), outline=(0, 0, 0), width=2)
            draw.text((point[0]+8, point[1]-8), "OUT", fill=(255, 0, 0), font=font_small)
        
        # Exhibit collisions
        for violation in violations['exhibit_collisions']:
            point = tuple(violation['point'])
            draw.ellipse([point[0]-7, point[1]-7, point[0]+7, point[1]+7], 
                        fill=(255, 0, 255), outline=(0, 0, 0), width=2)
            draw.text((point[0]+10, point[1]-10), f"HIT {violation['exhibit_id']}", 
                     fill=(255, 0, 255), font=font_small)
        
        # Highlight visited exhibits
        for exhibit in self.exhibits:
            if exhibit['shape'] == 'circle':
                exhibit_id = exhibit.get('exhibit_number', exhibit['id'])
                center = (exhibit['coordinates']['center_x'], 
                         exhibit['coordinates']['center_y'])
                radius = exhibit['coordinates']['radius']
                
                if validation_result['exhibit_visits'][exhibit_id]['visited']:
                    # Draw yellow highlight for visited
                    draw.ellipse([center[0]-radius-5, center[1]-radius-5,
                                center[0]+radius+5, center[1]+radius+5],
                               outline=(255, 255, 0), width=3)
                    draw.text((center[0]+radius+10, center[1]), f"✓ {exhibit_id}", 
                             fill=(0, 200, 0), font=font)
        
        # Add legend
        legend_x, legend_y = 20, 20
        legend_bg = [(legend_x-10, legend_y-10), (legend_x+280, legend_y+120)]
        draw.rectangle(legend_bg, fill=(255, 255, 255, 230), outline=(0, 0, 0), width=2)
        
        status = "VALID ✅" if is_valid else "INVALID ❌"
        draw.text((legend_x, legend_y), f"Route Validation: {status}", fill=(0, 0, 0), font=font)
        draw.text((legend_x, legend_y+25), f"Violations: {validation_result['validation_summary']['total_violations']}", 
                 fill=(0, 0, 0), font=font_small)
        draw.text((legend_x, legend_y+45), f"Exhibits Visited: {len(validation_result['validation_summary']['exhibits_visited'])}/{len(self.exhibits)}", 
                 fill=(0, 0, 0), font=font_small)
        
        # Legend symbols
        draw.ellipse([legend_x, legend_y+65, legend_x+10, legend_y+75], outline=(200, 200, 0), width=2)
        draw.text((legend_x+15, legend_y+65), "= Visit zone (visited)", fill=(0, 0, 0), font=font_small)
        draw.ellipse([legend_x, legend_y+85, legend_x+10, legend_y+95], outline=(150, 150, 150), width=1)
        draw.text((legend_x+15, legend_y+85), "= Visit zone (not visited)", fill=(0, 0, 0), font=font_small)
        draw.ellipse([legend_x, legend_y+105, legend_x+10, legend_y+115], fill=(255, 0, 0))
        draw.text((legend_x+15, legend_y+105), "= Out of Bounds", fill=(0, 0, 0), font=font_small)
        
        # Save
        img.save(output_path)
        print(f"Validation visualization saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description='Validate museum visitor route using automatic detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python validate_route.py \\
    --route-image my_route.png \\
    --original-image museum_layout_01.png \\
    --annotations museum_layout_01_annotated.json

  python validate_route.py \\
    --route-image visitor_path.png \\
    --original-image museum_layout_01.png \\
    --annotations museum_layout_01_annotated.json \\
    --proximity-threshold 30 \\
    --difference-threshold 15
        """
    )
    
    parser.add_argument('--route-image', required=True, 
                       help='Image with drawn route')
    parser.add_argument('--original-image', required=True, 
                       help='Original floor plan image (without route)')
    parser.add_argument('--annotations', required=True, 
                       help='JSON file with wall and exhibit annotations')
    parser.add_argument('--proximity-threshold', type=int, default=25, 
                       help='Distance for exhibit visit detection in pixels (default: 25)')
    parser.add_argument('--difference-threshold', type=int, default=10,
                       help='Minimum pixel difference to detect route (default: 10)')
    parser.add_argument('--min-route-pixels', type=int, default=50,
                       help='Minimum route pixels to consider valid (default: 50)')
    parser.add_argument('--output-json', default='route_validation_report.json', 
                       help='Output JSON report file (default: route_validation_report.json)')
    parser.add_argument('--output-image', default='route_validation_visual.png', 
                       help='Output annotated image (default: route_validation_visual.png)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Museum Route Validation System")
    print("Automatic Route Detection via Image Differencing")
    print("=" * 60)
    
    # Initialize validator
    validator = RouteValidator(
        annotations_file=args.annotations,
        proximity_threshold=args.proximity_threshold
    )
    
    # Extract route by comparing with original image
    try:
        route_points = validator.extract_route_by_difference(
            route_image_path=args.route_image,
            original_image_path=args.original_image,
            difference_threshold=args.difference_threshold,
            min_route_pixels=args.min_route_pixels
        )
    except ValueError as e:
        print(f"\n❌ Error: {e}")
        return
    
    if not route_points:
        print("❌ Error: No route points extracted. Please check your images.")
        return
    
    # Validate route
    validation_result = validator.validate_route(route_points)
    
    # Save JSON report
    with open(args.output_json, 'w') as f:
        json.dump(validation_result, f, indent=2)
    print(f"\n📄 JSON report saved to: {args.output_json}")
    
    # Create visualization (use original image as base)
    validator.visualize_validation(
        original_image_path=args.original_image,
        route_points=route_points,
        validation_result=validation_result,
        output_path=args.output_image
    )
    
    print("\n" + "=" * 60)
    print("✅ Validation Complete!")
    print("=" * 60)

if __name__ == '__main__':
    main()
