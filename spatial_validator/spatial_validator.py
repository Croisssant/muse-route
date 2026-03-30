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





def segments_intersect(self, p1, p2, p3, p4):
    """
    Check if segment p1→p2 intersects segment p3→p4.
    Uses cross-product orientation test.
    """
    def cross(o, a, b):
        return (a[0]-o[0]) * (b[1]-o[1]) - (a[1]-o[1]) * (b[0]-o[0])

    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True

    # Collinear cases
    if d1 == 0 and self._point_on_segment(p1, p3, p4): return True
    if d2 == 0 and self._point_on_segment(p2, p3, p4): return True
    if d3 == 0 and self._point_on_segment(p3, p1, p2): return True
    if d4 == 0 and self._point_on_segment(p4, p1, p2): return True

    return False

def route_crosses_walls(self, skeleton_points):
    """
    Returns a list of (skeleton_pt, wall_id) pairs where the route
    crosses a wall segment.
    """
    violations = []
    # Build wall segments from all polylines
    wall_segments = []
    for wall in self.walls:
        if wall['shape'] == 'polyline':
            pts = [(p['x'], p['y']) for p in wall['coordinates']['points']]
            for i in range(len(pts) - 1):
                wall_segments.append((wall['id'], pts[i], pts[i+1]))

    # Check each consecutive pair of skeleton points against every wall segment
    for i in range(len(skeleton_points) - 1):
        s1, s2 = skeleton_points[i], skeleton_points[i+1]
        for wall_id, w1, w2 in wall_segments:
            if self.segments_intersect(s1, s2, w1, w2):
                violations.append((s1, wall_id))
                break  # one violation per skeleton segment is enough

    return violations