"""
Convert VIA project JSON to museum layout annotation format
Filters out unlabeled annotations and converts to format compatible with validate_route.py
Supports: walls, galleries, exhibits, entrances, exits, floor_areas, and forbidden_areas
"""

import json
from datetime import datetime

def convert_via_to_museum_layout(via_json_path, output_path):
    """
    Convert VIA project JSON to museum layout format.
    
    Args:
        via_json_path: Path to VIA project JSON file
        output_path: Path to save converted museum layout JSON
    """
    # Load VIA project
    with open(via_json_path, 'r') as f:
        via_data = json.load(f)
    
    # Extract metadata section
    metadata_entries = via_data.get('metadata', {})
    attributes = via_data.get('attribute', {})
    
    # Initialize output structure
    walls = []
    galleries = []
    exhibits = []
    entrances = []
    exits = []
    floor_areas = []
    forbidden_areas = []
    distance_in_mm = []
    other = []
    
    # Track statistics
    total_annotations = len(metadata_entries)
    unlabeled_count = 0
    labeled_count = 0
    
    print(f"Processing {total_annotations} annotations...")
    print("=" * 60)
    
    # Process each annotation
    for annotation_id, annotation in metadata_entries.items():
        av = annotation.get('av', {})
        
        # Skip unlabeled annotations (empty av dict)
        if not av:
            unlabeled_count += 1
            xy = annotation.get('xy', [])
            shape_type = xy[0] if xy else None
            if shape_type == 3:  # Circle
                center_x, center_y, radius = xy[1], xy[2], xy[3]
                print(f"❌ Skipping unlabeled circle at ({center_x}, {center_y})")
            elif shape_type == 2:  # Rectangle
                x, y, width, height = xy[1], xy[2], xy[3], xy[4]
                print(f"❌ Skipping unlabeled rectangle at ({x}, {y})")
            else:
                print(f"❌ Skipping unlabeled annotation: {annotation_id}")
            continue
        
        labeled_count += 1
        xy = annotation.get('xy', [])
        shape_type = xy[0] if xy else None
        
        # Get entity type from attribute 1
        entity_type = av.get('1', '')
        
        # Get gallery name from attribute 3 (if it's a gallery)
        gallery_name = av.get('3', '')
        
        # Process based on shape type
        if shape_type in [6, 7]:  # Polyline (6 = open, 7 = closed polygon)
            # Extract points
            points = []
            for i in range(1, len(xy), 2):
                if i + 1 < len(xy):
                    points.append({
                        'x': xy[i],
                        'y': xy[i + 1]
                    })
            
            polyline_data = {
                'id': annotation_id,
                'annotation_number': len(walls) + len(galleries) + 1,
                'shape': 'polyline',
                'coordinates': {
                    'points': points,
                    'num_points': len(points)
                },
                'area': None
            }
            
            # Check if this is an entrance or exit (entity type 3 or 4)
            # For 4-point polylines, keep as polygon to preserve rotation
            if entity_type in ['3', '4'] and len(points) == 4:
                polygon_data = {
                    'id': annotation_id,
                    'annotation_number': None,  # Will be set based on type
                    'shape': 'polygon',
                    'coordinates': {
                        'points': points,
                        'num_points': len(points)
                    },
                    'area': None  # Could calculate polygon area if needed
                }
                
                if entity_type == '3':  # Entrance
                    polygon_data['annotation_number'] = len(entrances) + 1
                    entrances.append(polygon_data)
                    print(f"✓ Entrance: polygon with {len(points)} points")
                elif entity_type == '4':  # Exit
                    polygon_data['annotation_number'] = len(exits) + 1
                    exits.append(polygon_data)
                    print(f"✓ Exit: polygon with {len(points)} points")
            elif entity_type == '0':  # Wall
                walls.append(polyline_data)
                print(f"✓ Wall: {len(points)} points")
            elif entity_type == '1':  # Gallery
                polyline_data['gallery_name'] = gallery_name if gallery_name else None
                galleries.append(polyline_data)
                gallery_label = f" ({gallery_name})" if gallery_name else ""
                print(f"✓ Gallery: {len(points)} points{gallery_label}")
            else:
                other.append(polyline_data)
                print(f"✓ Other polyline: {len(points)} points")
        
        elif shape_type == 3:  # Circle (Exhibits)
            # Extract circle data
            center_x = xy[1]
            center_y = xy[2]
            radius = xy[3]
            
            # Get exhibit number from attribute 2
            exhibit_number = av.get('2', '')
            
            circle_data = {
                'id': annotation_id,
                'annotation_number': len(exhibits) + 1,
                'shape': 'circle',
                'coordinates': {
                    'center_x': center_x,
                    'center_y': center_y,
                    'radius': round(radius, 2)
                },
                'area': int(3.14159 * radius * radius),
                'exhibit_number': exhibit_number,
                'artifact_info': {
                    'name': None,
                    'description': None,
                    'period': None,
                    'collection': None,
                    'notes': None
                }
            }
            
            exhibits.append(circle_data)
        
        elif shape_type == 2:  # Rectangle (Entrances, Exits, Floor Areas)
            # Extract rectangle data
            x = xy[1]
            y = xy[2]
            width = xy[3]
            height = xy[4]
            
            rectangle_data = {
                'id': annotation_id,
                'annotation_number': None,  # Will be set based on type
                'shape': 'rectangle',
                'coordinates': {
                    'x': round(x, 2),
                    'y': round(y, 2),
                    'width': round(width, 2),
                    'height': round(height, 2)
                },
                'area': int(width * height)
            }
            
            if entity_type == '1':  # Gallery (rectangle)
                rectangle_data['annotation_number'] = len(galleries) + 1
                rectangle_data['gallery_name'] = gallery_name if gallery_name else None
                galleries.append(rectangle_data)
                gallery_label = f" '{gallery_name}'" if gallery_name else ""
                print(f"✓ Gallery: {width:.0f}x{height:.0f} at ({x:.0f}, {y:.0f}){gallery_label}")
            elif entity_type == '3':  # Entrance
                rectangle_data['annotation_number'] = len(entrances) + 1
                entrances.append(rectangle_data)
                print(f"✓ Entrance: {width:.0f}x{height:.0f} at ({x:.0f}, {y:.0f})")
            elif entity_type == '4':  # Exit
                rectangle_data['annotation_number'] = len(exits) + 1
                exits.append(rectangle_data)
                print(f"✓ Exit: {width:.0f}x{height:.0f} at ({x:.0f}, {y:.0f})")
            elif entity_type == '5':  # Floor Area
                rectangle_data['annotation_number'] = len(floor_areas) + 1
                floor_areas.append(rectangle_data)
                print(f"✓ Floor Area: {width:.0f}x{height:.0f} at ({x:.0f}, {y:.0f})")
            elif entity_type == '6':  # Forbidden Area
                rectangle_data['annotation_number'] = len(forbidden_areas) + 1
                forbidden_areas.append(rectangle_data)
                print(f"✓ Forbidden Area: {width:.0f}x{height:.0f} at ({x:.0f}, {y:.0f})")
            elif entity_type == '7':  # Distance Reference (in mm)
                # Get the description which contains the length in mm
                description = av.get('4', '')  # Attribute 4 is the description field
                
                # Calculate pixel length (using width as the reference dimension)
                pixel_length = width
                
                rectangle_data['annotation_number'] = len(distance_in_mm) + 1
                rectangle_data['description'] = description
                rectangle_data['pixel_length'] = round(pixel_length, 2)
                
                # Try to extract numeric value from description
                try:
                    # Remove common units and extract number
                    mm_value = ''.join(filter(lambda x: x.isdigit() or x == '.', description))
                    if mm_value:
                        rectangle_data['length_mm'] = float(mm_value)
                        rectangle_data['px_per_mm'] = round(pixel_length / float(mm_value), 6)
                        rectangle_data['mm_per_px'] = round(float(mm_value)/ pixel_length, 6)
                    else:
                        rectangle_data['length_mm'] = None
                        rectangle_data['px_per_mm'] = None
                        rectangle_data['mm_per_px'] = None
                except:
                    rectangle_data['length_mm'] = None
                    rectangle_data['px_per_mm'] = None
                    rectangle_data['mm_per_px'] = None
                
                distance_in_mm.append(rectangle_data)
                mm_label = f" = {description}" if description else ""
                px_per_mm_label = f" ({rectangle_data['px_per_mm']:.4f} px/mm)" if rectangle_data['px_per_mm'] else ""
                px_per_mm_label = f" ({rectangle_data['mm_per_px']:.4f} mm/px)" if rectangle_data['mm_per_px'] else ""
                print(f"✓ Distance Reference: {pixel_length:.1f} px{mm_label}{px_per_mm_label}")
            else:
                other.append(rectangle_data)
                print(f"✓ Other rectangle: {width:.0f}x{height:.0f} at ({x:.0f}, {y:.0f})")
    
    # Sort exhibits by exhibit number
    exhibits.sort(key=lambda x: int(x['exhibit_number']) if x['exhibit_number'].isdigit() else 999)
    
    # Create output structure
    output = {
        'metadata': {
            'total_annotations': labeled_count,
            'export_date': datetime.now().isoformat(),
            'attributes': attributes,
            'counts': {
                'walls': len(walls),
                'galleries': len(galleries),
                'exhibits': len(exhibits),
                'entrances': len(entrances),
                'exits': len(exits),
                'floor_areas': len(floor_areas),
                'forbidden_areas': len(forbidden_areas),
                'distance_in_mm': len(distance_in_mm),
                'other': len(other)
            }
        },
        'walls': walls,
        'galleries': galleries,
        'exhibits': exhibits,
        'entrances': entrances,
        'exits': exits,
        'floor_areas': floor_areas,
        'forbidden_areas': forbidden_areas,
        'distance_in_mm': distance_in_mm,
        'other': other
    }
    
    # Save to file
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("CONVERSION SUMMARY")
    print("=" * 60)
    print(f"Total annotations in VIA project: {total_annotations}")
    print(f"  ✓ Labeled (kept):     {labeled_count}")
    print(f"  ❌ Unlabeled (removed): {unlabeled_count}")
    print()
    print("Converted annotations:")
    print(f"  Walls:           {len(walls)}")
    print(f"  Galleries:       {len(galleries)}")
    print(f"  Exhibits:        {len(exhibits)}")
    print(f"  Entrances:       {len(entrances)}")
    print(f"  Exits:           {len(exits)}")
    print(f"  Floor Areas:     {len(floor_areas)}")
    print(f"  Forbidden Areas: {len(forbidden_areas)}")
    print(f"  Distance Refs:   {len(distance_in_mm)}")
    print(f"  Other:           {len(other)}")
    print()
    print(f"✅ Saved to: {output_path}")
    print("\nThis file is compatible with validate_route.py --annotations")
    
    return output

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python convert_via_to_museum_layout.py <via_project.json> [output.json]")
        print("\nExample:")
        print("  python convert_via_to_museum_layout.py via_project_24Mar2026_16h58m58s.json museum_layout_complete.json")
        sys.exit(1)
    
    via_json_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else 'museum_layout_complete.json'
    
    convert_via_to_museum_layout(via_json_path, output_path)
