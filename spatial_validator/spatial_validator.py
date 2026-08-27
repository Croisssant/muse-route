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

class SpatialValidator:
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
        self.galleries = self.annotations.get('galleries', [])
        self.forbidden_areas = self.annotations.get('forbidden_areas', [])
        
        # Gallery configurations (can be set from main.py)
        self.gallery_configs = {}
        
        # Calculate museum bounds from annotations
        self.museum_bounds = self._calculate_museum_bounds()
        
        print(f"Loaded {len(self.walls)} wall(s) and {len(self.exhibits)} exhibit(s)")
        print(f"Loaded {len(self.entrances)} entrance(s), {len(self.exits)} exit(s), and {len(self.floor_areas)} floor area(s)")
        print(f"Loaded {len(self.galleries)} gallery/galleries and {len(self.forbidden_areas)} forbidden area(s)")
        if self.museum_bounds:
            print(f"Museum bounds: X=[{self.museum_bounds['min_x']}, {self.museum_bounds['max_x']}], "
                  f"Y=[{self.museum_bounds['min_y']}, {self.museum_bounds['max_y']}]")
    
    def set_gallery_configurations(self, gallery_configs):
        """
        Set gallery access configurations.
        
        Args:
            gallery_configs: Dict mapping gallery_name to type ('must_see', 'restricted', 'normal')
                           Example: {'gallery_room_1': 'must_see', 'gallery_open_space_1': 'normal'}
        """
        self.gallery_configs = gallery_configs
        print(f"\nGallery configurations set:")
        for gallery_name, config_type in gallery_configs.items():
            print(f"  - {gallery_name}: {config_type}")
    
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
    

    def _distance(self, p1, p2):
        """Calculate Euclidean distance between two points."""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def _segments_intersect(self, p1, p2, p3, p4):
        """
        Check if line segment (p1, p2) intersects with segment (p3, p4).
        Uses the cross-product method for line intersection.
        
        Args:
            p1: (x, y) first point of segment 1
            p2: (x, y) second point of segment 1
            p3: (x, y) first point of segment 2
            p4: (x, y) second point of segment 2
        
        Returns:
            True if segments intersect, False otherwise
        """
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        
        if abs(denom) < 1e-10:
            return False  # Lines are parallel
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
        
        # Intersection occurs if both t and u are in [0, 1]
        return 0 <= t <= 1 and 0 <= u <= 1
    
    def _line_crosses_polygon(self, p1, p2, polygon):
        """
        Check if line segment (p1, p2) crosses through a polygon.
        Uses line-segment intersection for each polygon edge.
        
        Args:
            p1: (x, y) starting point
            p2: (x, y) ending point
            polygon: List of (x, y) tuples forming polygon vertices
        
        Returns:
            True if line crosses polygon, False otherwise
        """
        n = len(polygon)
        for i in range(n):
            edge_start = polygon[i]
            edge_end = polygon[(i + 1) % n]
            
            if self._segments_intersect(p1, p2, edge_start, edge_end):
                return True
        
        return False
    
    def _line_intersects_wall(self, point1, point2):
        """
        Check if line segment from point1 to point2 intersects any wall.
        This is used for line-of-sight validation to ensure exhibits are only
        visible when there's no wall blocking the view.
        
        Args:
            point1: (x, y) starting point (typically route point)
            point2: (x, y) ending point (typically exhibit center)
        
        Returns:
            True if line intersects any wall polygon, False otherwise
        """
        for wall in self.walls:
            if wall['shape'] == 'polyline':
                wall_points = [(p['x'], p['y']) for p in wall['coordinates']['points']]
                
                # Check if line segment crosses the wall polygon
                if self._line_crosses_polygon(point1, point2, wall_points):
                    return True
        
        return False
    
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
    
    def validate_route(self, route_points, endpoints=None):
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
            endpoints: Optional dict with 'start' and 'end' points from route extraction
        
        Returns:
            Dictionary with validation results
        """
        print("\nValidating route...")
        
        # Set route data if provided (from RouteExtractor)
        if endpoints is not None:
            self._determined_endpoints = endpoints
        
        violations = {
            'wall_crossings': [],
            'exhibit_collisions': [],
            'out_of_bounds': [],
            'floor_area_violations': [],
            'entrance_violations': [],
            'exit_violations': [],
            'forbidden_area_violations': [],
            'gallery_violations': [],
            'must_see_gallery_violations': [],
        }
        
        exhibit_visits = {}
        for exhibit in self.exhibits:
            exhibit_id = exhibit.get('exhibit_number', exhibit['id'])
            exhibit_visits[exhibit_id] = {
                'visited': False,
                'closest_distance': float('inf')
            }
        
        # Track gallery visits for validation
        gallery_visits = {}
        for gallery in self.galleries:
            gallery_name = gallery.get('gallery_name', gallery.get('id', 'unknown'))
            gallery_visits[gallery_name] = False
        
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
        
        # Validate START point (must exist AND be in an entrance box)
        if endpoints['start'] is None:
            violations['entrance_violations'].append({
                'point': None,
                'reason': 'No START point detected'
            })
            print(f"  ❌ No START point detected")
        else:
            # Check if START is actually inside entrance box
            if self.entrances:
                entrance = self.entrances[0]
                shape = entrance.get('shape', 'rectangle')
                
                if shape == 'rectangle':
                    ec = entrance['coordinates']
                    if self.point_in_rectangle(endpoints['start'], ec):
                        print(f"  ✅ START {endpoints['start']} is inside entrance box")
                    else:
                        violations['entrance_violations'].append({
                            'point': endpoints['start'],
                            'reason': 'START point is outside entrance bounding box'
                        })
                        dist_to_entrance = self._distance(endpoints['start'], 
                                                         (ec['x'] + ec['width']/2, ec['y'] + ec['height']/2))
                        print(f"  ❌ START {endpoints['start']} is outside entrance box (distance: {dist_to_entrance:.1f} px)")
                elif shape == 'polygon':
                    polygon_points = [(p['x'], p['y']) for p in entrance['coordinates']['points']]
                    if self.point_in_polygon(endpoints['start'], polygon_points):
                        print(f"  ✅ START {endpoints['start']} is inside entrance polygon")
                    else:
                        violations['entrance_violations'].append({
                            'point': endpoints['start'],
                            'reason': 'START point is outside entrance polygon'
                        })
                        # Calculate center of polygon for distance
                        center_x = sum(p[0] for p in polygon_points) / len(polygon_points)
                        center_y = sum(p[1] for p in polygon_points) / len(polygon_points)
                        dist_to_entrance = self._distance(endpoints['start'], (center_x, center_y))
                        print(f"  ❌ START {endpoints['start']} is outside entrance polygon (distance: {dist_to_entrance:.1f} px)")
                else:
                    print(f"  ⚠️ Unsupported entrance shape: {shape}")
            else:
                print(f"  ⚠️ No entrance box defined for validation")
        
        # Validate END point (must exist AND be in exit box)
        if endpoints['end'] is None:
            violations['exit_violations'].append({
                'point': None,
                'reason': f'No END point detected'
            })
            print(f"  ❌ No END point detected")
        else:
            # Check if END is actually inside exit box
            if self.exits:
                exit_shape = self.exits[0]
                shape = exit_shape.get('shape', 'rectangle')
                
                if shape == 'rectangle':
                    xc = exit_shape['coordinates']
                    if self.point_in_rectangle(endpoints['end'], xc):
                        print(f"  ✅ END {endpoints['end']} is inside exit box")
                    else:
                        violations['exit_violations'].append({
                            'point': endpoints['end'],
                            'reason': 'END point is outside exit bounding box'
                        })
                        dist_to_exit = self._distance(endpoints['end'], 
                                                      (xc['x'] + xc['width']/2, xc['y'] + xc['height']/2))
                        print(f"  ❌ END {endpoints['end']} is outside exit box (distance: {dist_to_exit:.1f} px)")
                elif shape == 'polygon':
                    polygon_points = [(p['x'], p['y']) for p in exit_shape['coordinates']['points']]
                    if self.point_in_polygon(endpoints['end'], polygon_points):
                        print(f"  ✅ END {endpoints['end']} is inside exit polygon")
                    else:
                        violations['exit_violations'].append({
                            'point': endpoints['end'],
                            'reason': 'END point is outside exit polygon'
                        })
                        # Calculate center of polygon for distance
                        center_x = sum(p[0] for p in polygon_points) / len(polygon_points)
                        center_y = sum(p[1] for p in polygon_points) / len(polygon_points)
                        dist_to_exit = self._distance(endpoints['end'], (center_x, center_y))
                        print(f"  ❌ END {endpoints['end']} is outside exit polygon (distance: {dist_to_exit:.1f} px)")
                else:
                    print(f"  ⚠️ Unsupported exit shape: {shape}")
            else:
                print(f"  ⚠️ No exit box defined for validation")
        
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
                    
                    # Check visit (within proximity threshold AND clear line-of-sight)
                    elif dist < radius + self.proximity_threshold:
                        # Verify line-of-sight: only mark as visited if no wall blocks the view
                        if not self._line_intersects_wall(point, center):
                            exhibit_visits[exhibit_id]['visited'] = True
                        # else: within range but blocked by wall - don't mark as visited
            
            # Check if route passes through forbidden areas
            for forbidden_area in self.forbidden_areas:
                if forbidden_area['shape'] == 'rectangle':
                    if self.point_in_rectangle(point, forbidden_area['coordinates']):
                        violations['forbidden_area_violations'].append({
                            'point': [int(point[0]), int(point[1])],
                            'index': int(i),
                            'reason': 'route passes through forbidden area'
                        })
                        break  # Only report once per point
            
            # Check gallery violations based on configuration
            for gallery in self.galleries:
                if gallery['shape'] == 'rectangle':
                    gallery_name = gallery.get('gallery_name', gallery.get('id', 'unknown'))
                    gallery_type = self.gallery_configs.get(gallery_name, 'normal')
                    
                    if self.point_in_rectangle(point, gallery['coordinates']):
                        # Mark gallery as visited
                        gallery_visits[gallery_name] = True
                        
                        # Violation if gallery is restricted
                        if gallery_type == 'restricted':
                            violations['gallery_violations'].append({
                                'gallery_name': gallery_name,
                                'point': [int(point[0]), int(point[1])],
                                'index': int(i),
                                'reason': f'route enters restricted gallery "{gallery_name}"'
                            })
                            break  # Only report once per point
        
        # Check if must_see galleries were visited
        for gallery_name, gallery_type in self.gallery_configs.items():
            if gallery_type == 'must_see' and not gallery_visits.get(gallery_name, False):
                violations['must_see_gallery_violations'].append({
                    'gallery_name': gallery_name,
                    'point': None,
                    'index': None,
                    'reason': f'route did not visit must-see gallery "{gallery_name}"'
                })
        
        # Compile results
        visited_exhibits = [eid for eid, data in exhibit_visits.items() if data['visited']]
        
        total_violations = (len(violations['wall_crossings']) + 
                          len(violations['exhibit_collisions']) + 
                          len(violations['out_of_bounds']) +
                          len(violations['floor_area_violations']) +
                          len(violations['entrance_violations']) +
                          len(violations['exit_violations']) +
                          len(violations['forbidden_area_violations']) +
                          len(violations['gallery_violations']) +
                          len(violations['must_see_gallery_violations']))
        
        is_valid = total_violations == 0
        
        print(f"\nValidation Result: {'✅ VALID' if is_valid else '❌ INVALID'}")
        print(f"Exhibits visited: {len(visited_exhibits)}/{len(self.exhibits)}")
        print(f"Total violations: {total_violations}")
        violation_reasons = []
        for key, value in violations.items():
            num_violations = len(value)
            if num_violations > 0:
                print(f"  - {key}: {num_violations}")
                violation_reasons.append(key)


        result = {
            'validation_summary': {
                'is_valid': is_valid,
                'total_violations': total_violations,
                'violation_reasons': violation_reasons,
                'num_route_pixels': len(route_points),
                'exhibits_visited': visited_exhibits,
                'gallery_visits': gallery_visits,
            },
            'violations': violations,
            'exhibit_visits': exhibit_visits,
        }
        
        return result
    
    def visualize_validation(self, original_image_path, route_points, validation_result, output_path=None, must_visit_exhibits=None, show_all_zones=False):
        """
        Create annotated image showing validation results.
        
        Args:
            original_image_path: Path to original image
            route_points: List of route points
            validation_result: Validation result dictionary
            output_path: Path to save annotated image
            must_visit_exhibits: List of exhibit IDs that are required to visit (from config's specific_exhibit_to_cover)
            show_all_zones: If True, show all exhibit zones (debug mode). If False, only show visited and unvisited must-visit zones
        """
        print(f"\nCreating validation visualization...")
        
        # Load image with OpenCV to draw route pixels
        img_cv = cv2.imread(str(original_image_path))
        if img_cv is None:
            # Fallback to PIL
            from PIL import Image as PILImage
            img_pil = PILImage.open(original_image_path)
            if img_pil.mode == 'RGBA':
                img_pil = img_pil.convert('RGB')
            img_cv = np.array(img_pil)
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
        
        # Create violation lookup for each route point
        violations = validation_result['violations']
        point_violations = {}  # Maps point index to list of violation types
        
        # Index wall crossings
        for violation in violations['wall_crossings']:
            idx = violation['index']
            if idx not in point_violations:
                point_violations[idx] = []
            point_violations[idx].append('wall')
        
        # Index exhibit collisions
        for violation in violations['exhibit_collisions']:
            idx = violation['index']
            if idx not in point_violations:
                point_violations[idx] = []
            point_violations[idx].append('exhibit')
        
        # Index floor area violations
        for violation in violations['floor_area_violations']:
            idx = violation['index']
            if idx not in point_violations:
                point_violations[idx] = []
            point_violations[idx].append('floor_area')
        
        # Index forbidden area violations
        for violation in violations['forbidden_area_violations']:
            idx = violation['index']
            if idx not in point_violations:
                point_violations[idx] = []
            point_violations[idx].append('forbidden_area')
        
        # Index gallery violations (only those with specific point index)
        for violation in violations['gallery_violations']:
            if violation['index'] is not None:
                idx = violation['index']
                if idx not in point_violations:
                    point_violations[idx] = []
                point_violations[idx].append('gallery')
        
        # Draw route points with color-coded violations
        if route_points and len(route_points) > 0:
            violation_counts = {'wall': 0, 'exhibit': 0, 'floor_area': 0, 'forbidden_area': 0, 'gallery': 0, 'valid': 0}
            
            for i, point in enumerate(route_points):
                x, y = int(point[0]), int(point[1])
                
                # Bounds check
                if not (0 <= x < img_cv.shape[1] and 0 <= y < img_cv.shape[0]):
                    continue
                
                # Determine color based on violations
                if i in point_violations:
                    violation_types = point_violations[i]
                    
                    # Priority: wall > forbidden_area > gallery > exhibit > floor_area
                    if 'wall' in violation_types:
                        color = (0, 0, 255)  # Red for wall crossings
                        violation_counts['wall'] += 1
                    elif 'forbidden_area' in violation_types:
                        color = (0, 100, 255)  # Dark orange for forbidden areas
                        violation_counts['forbidden_area'] += 1
                    elif 'gallery' in violation_types:
                        color = (128, 0, 128)  # Purple for gallery violations
                        violation_counts['gallery'] += 1
                    elif 'exhibit' in violation_types:
                        color = (255, 0, 255)  # Magenta for exhibit collisions
                        violation_counts['exhibit'] += 1
                    elif 'floor_area' in violation_types:
                        color = (0, 165, 255)  # Orange for floor area violations
                        violation_counts['floor_area'] += 1
                    else:
                        color = (0, 128, 0)  # Deep green for better contrast (shouldn't happen)
                        violation_counts['valid'] += 1
                else:
                    # No violations - deep green for high contrast with entrance
                    color = (0, 128, 0)  # Deep green in BGR
                    violation_counts['valid'] += 1
                
                # Draw point with determined color
                cv2.circle(img_cv, (x, y), radius=1, color=color, thickness=-1)
            
            total = len(route_points)
            print(f"  Drew {total} route points:")
            print(f"    - {violation_counts['valid']} valid points ({100*violation_counts['valid']/total:.1f}%) [green]")
            print(f"    - {violation_counts['wall']} wall crossings ({100*violation_counts['wall']/total:.1f}%) [red]")
            print(f"    - {violation_counts['forbidden_area']} forbidden area violations ({100*violation_counts['forbidden_area']/total:.1f}%) [dark orange]")
            print(f"    - {violation_counts['gallery']} gallery violations ({100*violation_counts['gallery']/total:.1f}%) [purple]")
            print(f"    - {violation_counts['exhibit']} exhibit collisions ({100*violation_counts['exhibit']/total:.1f}%) [magenta]")
            print(f"    - {violation_counts['floor_area']} floor area violations ({100*violation_counts['floor_area']/total:.1f}%) [orange]")
        else:
            print(f"  ⚠️ WARNING: No route points available, skipping route visualization")
        
        # Draw START and END point markers if available
        endpoints = getattr(self, '_determined_endpoints', {'start': None, 'end': None})
        
        if endpoints['start']:
            sx, sy = int(endpoints['start'][0]), int(endpoints['start'][1])
            # Draw large green circle for START
            cv2.circle(img_cv, (sx, sy), radius=15, color=(0, 255, 0), thickness=-1)
            # Draw black outline
            cv2.circle(img_cv, (sx, sy), radius=17, color=(0, 0, 0), thickness=2)
        
        if endpoints['end']:
            ex, ey = int(endpoints['end'][0]), int(endpoints['end'][1])
            # Draw large red circle for END
            cv2.circle(img_cv, (ex, ey), radius=15, color=(0, 0, 255), thickness=-1)
            # Draw black outline
            cv2.circle(img_cv, (ex, ey), radius=17, color=(0, 0, 0), thickness=2)
        
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
        
        # Try to load bold font for START/END labels
        try:
            font_bold = ImageFont.truetype("arialbd.ttf", 20)  # Arial Bold, 20pt
        except:
            try:
                font_bold = ImageFont.truetype("arial.ttf", 20)  # Fallback to regular Arial
            except:
                font_bold = ImageFont.load_default()  # Fallback to default
        
        # Add labels for START and END
        if endpoints['start']:
            sx, sy = int(endpoints['start'][0]), int(endpoints['start'][1])
            draw.text((sx + 20, sy - 10), "START", fill=(0, 0, 0), font=font_bold)
        
        if endpoints['end']:
            ex, ey = int(endpoints['end'][0]), int(endpoints['end'][1])
            draw.text((ex + 20, ey - 10), "END", fill=(0, 0, 0), font=font_bold)
        
        # Draw visit zones (proximity threshold circles) around exhibits
        # Behavior depends on show_all_zones flag (debug mode)
        if show_all_zones:
            print(f"  Drawing ALL visit zones (radius + {self.proximity_threshold}px) [DEBUG MODE]...")
        else:
            print(f"  Drawing selective visit zones (radius + {self.proximity_threshold}px) [NORMAL MODE]...")
        
        # Convert must_visit_exhibits to set for faster lookup, ensuring consistent types
        must_visit_set = set()
        if must_visit_exhibits:
            for eid in must_visit_exhibits:
                # Convert to string for consistent comparison with exhibit IDs
                must_visit_set.add(str(eid) if not isinstance(eid, str) else eid)
        
        for exhibit in self.exhibits:
            if exhibit['shape'] == 'circle':
                exhibit_id = exhibit.get('exhibit_number', exhibit['id'])
                # Ensure exhibit_id is string for comparison
                exhibit_id_str = str(exhibit_id) if not isinstance(exhibit_id, str) else exhibit_id
                
                center = (int(exhibit['coordinates']['center_x']), 
                         int(exhibit['coordinates']['center_y']))
                radius = exhibit['coordinates']['radius']
                visit_zone_radius = int(radius + self.proximity_threshold)
                
                # Determine if this exhibit was visited
                was_visited = validation_result['exhibit_visits'][exhibit_id]['visited']
                is_must_visit = exhibit_id_str in must_visit_set
                
                if show_all_zones:
                    # DEBUG MODE: Show all zones (current behavior)
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
                else:
                    # NORMAL MODE: Only show visited and unvisited must-visit zones
                    if was_visited:
                        # Draw yellow/green visit zone for visited exhibits
                        draw.ellipse([center[0]-visit_zone_radius, center[1]-visit_zone_radius,
                                    center[0]+visit_zone_radius, center[1]+visit_zone_radius],
                                   outline=(200, 200, 0), width=2)  # Yellow outline
                    elif is_must_visit:
                        # Draw RED visit zone for unvisited must-visit exhibits
                        draw.ellipse([center[0]-visit_zone_radius, center[1]-visit_zone_radius,
                                    center[0]+visit_zone_radius, center[1]+visit_zone_radius],
                                   outline=(255, 0, 0), width=3)  # RED outline with thicker border
                    # else: Skip drawing zone for other unvisited exhibits
        
        # Highlight visited exhibits and unvisited must-visit exhibits
        for exhibit in self.exhibits:
            if exhibit['shape'] == 'circle':
                exhibit_id = exhibit.get('exhibit_number', exhibit['id'])
                exhibit_id_str = str(exhibit_id) if not isinstance(exhibit_id, str) else exhibit_id
                center = (exhibit['coordinates']['center_x'], 
                         exhibit['coordinates']['center_y'])
                radius = exhibit['coordinates']['radius']
                
                was_visited = validation_result['exhibit_visits'][exhibit_id]['visited']
                is_must_visit = exhibit_id_str in must_visit_set
                
                if was_visited:
                    # Draw yellow highlight for visited exhibits
                    draw.ellipse([center[0]-radius-5, center[1]-radius-5,
                                center[0]+radius+5, center[1]+radius+5],
                               outline=(255, 255, 0), width=3)
                elif not show_all_zones and is_must_visit:
                    # Draw RED highlight for unvisited must-visit exhibits (only in normal mode)
                    draw.ellipse([center[0]-radius-5, center[1]-radius-5,
                                center[0]+radius+5, center[1]+radius+5],
                               outline=(255, 0, 0), width=3)
        
        # Highlight unvisited must-see galleries with overlay and label
        gallery_visits = validation_result['validation_summary']['gallery_visits']
        
        # Create a semi-transparent overlay for galleries
        overlay = Image.new('RGBA', img.size, (255, 255, 255, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        
        for gallery in self.galleries:
            if gallery['shape'] == 'rectangle':
                gallery_name = gallery.get('gallery_name', gallery.get('id', 'unknown'))
                gallery_type = self.gallery_configs.get(gallery_name, 'normal')
                
                # Check if this is a must-see gallery that was NOT visited
                if gallery_type == 'must_see' and not gallery_visits.get(gallery_name, False):
                    coords = gallery['coordinates']
                    x, y = coords['x'], coords['y']
                    width, height = coords['width'], coords['height']
                    
                    # Draw semi-transparent red overlay
                    overlay_draw.rectangle(
                        [(x, y), (x + width, y + height)],
                        fill=(255, 100, 100, 80),  # Semi-transparent red
                        outline=(220, 50, 50, 255),  # Solid red border
                        width=3
                    )
                    
                    # Calculate center X position for text
                    center_x = x + width // 2
                    
                    # Prepare text - only "NOT VISITED"
                    text = "NOT VISITED"
                    
                    # Draw text with background for better visibility
                    # Use bold font if available
                    try:
                        text_font = ImageFont.truetype("arialbd.ttf", 16)
                    except:
                        try:
                            text_font = ImageFont.truetype("arial.ttf", 16)
                        except:
                            text_font = font
                    
                    # Get text bounding box
                    bbox = overlay_draw.textbbox((0, 0), text, font=text_font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                    
                    # Position text above the bounding box
                    text_pos = (center_x - text_width // 2, y - text_height - 10)
                    
                    # Draw text with black outline for visibility
                    for offset_x, offset_y in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                        overlay_draw.text(
                            (text_pos[0] + offset_x, text_pos[1] + offset_y),
                            text, fill=(0, 0, 0, 255), font=text_font
                        )
                    
                    # Draw main text in white
                    overlay_draw.text(text_pos, text, fill=(255, 255, 255, 255), font=text_font)
        
        # Composite the overlay with the main image
        img = img.convert('RGBA')
        img = Image.alpha_composite(img, overlay)
        img = img.convert('RGB')
        draw = ImageDraw.Draw(img)
        
        # Add legend with color-coded route explanations
        legend_x, legend_y = 20, 20
        is_valid = validation_result['validation_summary']['is_valid']
        legend_bg = [(legend_x-10, legend_y-10), (legend_x+300, legend_y+200)]
        draw.rectangle(legend_bg, fill=(255, 255, 255, 230), outline=(0, 0, 0), width=2)
        
        status = "VALID ✅" if is_valid else "INVALID ❌"
        draw.text((legend_x, legend_y), f"Route Validation: {status}", fill=(0, 0, 0), font=font)
        draw.text((legend_x, legend_y+25), f"Violations: {validation_result['validation_summary']['total_violations']}", 
                 fill=(0, 0, 0), font=font_small)
        draw.text((legend_x, legend_y+45), f"Exhibits Visited: {len(validation_result['validation_summary']['exhibits_visited'])}/{len(self.exhibits)}", 
                 fill=(0, 0, 0), font=font_small)
        
        # Route color legend
        y_offset = legend_y + 70
        draw.ellipse([legend_x, y_offset, legend_x+10, y_offset+10], fill=(0, 128, 0))
        draw.text((legend_x+15, y_offset), "= Valid route point", fill=(0, 0, 0), font=font_small)
        
        y_offset += 20
        draw.ellipse([legend_x, y_offset, legend_x+10, y_offset+10], fill=(255, 0, 0))
        draw.text((legend_x+15, y_offset), "= Wall crossing", fill=(0, 0, 0), font=font_small)
        
        y_offset += 20
        draw.ellipse([legend_x, y_offset, legend_x+10, y_offset+10], fill=(255, 100, 0))
        draw.text((legend_x+15, y_offset), "= Forbidden area", fill=(0, 0, 0), font=font_small)
        
        y_offset += 20
        draw.ellipse([legend_x, y_offset, legend_x+10, y_offset+10], fill=(128, 0, 128))
        draw.text((legend_x+15, y_offset), "= Gallery violation", fill=(0, 0, 0), font=font_small)
        
        y_offset += 20
        draw.ellipse([legend_x, y_offset, legend_x+10, y_offset+10], fill=(255, 0, 255))
        draw.text((legend_x+15, y_offset), "= Exhibit collision", fill=(0, 0, 0), font=font_small)
        
        y_offset += 20
        draw.ellipse([legend_x, y_offset, legend_x+10, y_offset+10], fill=(255, 165, 0))
        draw.text((legend_x+15, y_offset), "= Floor area violation", fill=(0, 0, 0), font=font_small)
        
        y_offset += 20
        draw.ellipse([legend_x, y_offset, legend_x+10, y_offset+10], outline=(200, 200, 0), width=2)
        draw.text((legend_x+15, y_offset), "= Exhibit visited", fill=(0, 0, 0), font=font_small)
        
        # Add red zone legend entry in normal mode
        if not show_all_zones and must_visit_exhibits and len(must_visit_exhibits) > 0:
            y_offset += 20
            draw.ellipse([legend_x, y_offset, legend_x+10, y_offset+10], outline=(255, 0, 0), width=3)
            draw.text((legend_x+15, y_offset), "= Missed must-visit exhibit", fill=(0, 0, 0), font=font_small)
        
        # Save
        if output_path:
            img.save(output_path)
            print(f"Validation visualization saved to: {output_path}")

        else:
            img.show()
        
        return violation_counts