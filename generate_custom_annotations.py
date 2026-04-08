"""
Generate annotated images from original layout with selective annotation types.
Allows specifying which annotation types to display (e.g., entrance, exit, exhibits).
"""

import json
import argparse
from PIL import Image, ImageDraw, ImageFont
import sys

def load_annotations(json_path):
    """Load annotations from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)

def generate_annotated_image(image_path, annotations_data, annotation_types, output_path, show_legend=False, labels_mode=0):
    """
    Generate an annotated image with only specified annotation types.
    
    Args:
        image_path: Path to original image
        annotations_data: Loaded annotation data
        annotation_types: List of annotation types to display 
                         (e.g., ['entrance', 'exit', 'wall', 'gallery', 'exhibit', 'floor_area'])
        output_path: Path to save the output image
        show_legend: Whether to show the legend
        labels_mode: Label display mode (0=no labels, 1=gallery labels only, 2=all labels)
    """
    # Load image and preserve original size
    img = Image.open(image_path)
    original_size = img.size
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    
    # Create overlay for transparency
    overlay = Image.new('RGBA', img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    
    # Create a draw object for opaque elements
    draw_opaque = ImageDraw.Draw(img)
    
    # Try to load font
    try:
        font = ImageFont.truetype("arial.ttf", 16)
        font_small = ImageFont.truetype("arial.ttf", 12)
        font_exhibit = ImageFont.truetype("arial.ttf", 22)
        font_legend = ImageFont.truetype("arial.ttf", 14)
    except:
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()
        font_exhibit = ImageFont.load_default()
        font_legend = ImageFont.load_default()
    
    # Color scheme (same as existing scripts)
    colors = {
        'wall': (0, 0, 255, 255),        # Blue - opaque
        'room': (255, 165, 0, 255),      # Orange - opaque (deprecated)
        'gallery': (128, 0, 128, 100),   # 
        'exhibit': (60, 60, 60, 255),     # 
        'entrance': (0, 255, 0, 200),    # Green - semi-transparent
        'exit': (255, 255, 0, 200),      # Yellow - semi-transparent
        'floor_area': (128, 128, 255, 50), # Light blue - very transparent
        'forbidden_area': (255, 165, 0, 150), # Orange - semi-transparent
    }
    
    print(f"\nGenerating annotated image...")
    print(f"Image: {image_path}")
    print(f"Requested types: {', '.join(annotation_types)}")
    print("=" * 60)
    
    # Track what was actually drawn
    drawn_types = []
    
    # Draw floor areas first (most transparent, in background)
    if 'floor_area' in annotation_types or 'floor_areas' in annotation_types:
        floor_areas = annotations_data.get('floor_areas', [])
        if floor_areas:
            print(f"✓ Drawing {len(floor_areas)} floor area(s)...")
            drawn_types.append('floor_area')
            for floor_area in floor_areas:
                coords = floor_area['coordinates']
                x, y, width, height = coords['x'], coords['y'], coords['width'], coords['height']
                
                # Draw semi-transparent rectangle
                draw.rectangle(
                    [x, y, x + width, y + height],
                    fill=colors['floor_area'],
                    outline=(128, 128, 255, 150),
                    width=2
                )
                
                # Add label
                draw.text((x + 10, y + 10), "Floor Area", fill=(100, 100, 255, 200), font=font)
    
    # Draw walls (polylines)
    if 'wall' in annotation_types or 'walls' in annotation_types:
        walls = annotations_data.get('walls', [])
        if walls:
            print(f"✓ Drawing {len(walls)} wall(s)...")
            drawn_types.append('wall')
            for wall in walls:
                points = [(p['x'], p['y']) for p in wall['coordinates']['points']]
                if len(points) > 1:
                    draw_opaque.line(points, fill=colors['wall'][:3], width=3)
    
    # Draw rooms (polylines) - deprecated, kept for backward compatibility
    if 'room' in annotation_types or 'rooms' in annotation_types:
        rooms = annotations_data.get('rooms', [])
        if rooms:
            print(f"✓ Drawing {len(rooms)} room(s)...")
            drawn_types.append('room')
            for room in rooms:
                points = [(p['x'], p['y']) for p in room['coordinates']['points']]
                if len(points) > 1:
                    draw_opaque.line(points, fill=colors['room'][:3], width=3)
    
    # Draw galleries (rectangles)
    if 'gallery' in annotation_types or 'galleries' in annotation_types:
        galleries = annotations_data.get('galleries', [])
        if galleries:
            print(f"✓ Drawing {len(galleries)} gallery/galleries...")
            drawn_types.append('gallery')
            for gallery in galleries:
                coords = gallery['coordinates']
                x, y, width, height = coords['x'], coords['y'], coords['width'], coords['height']
                
                # Draw semi-transparent rectangle
                draw.rectangle(
                    [x, y, x + width, y + height],
                    fill=colors['gallery'],
                    outline=colors['gallery'],
                    width=3
                )
                
                # Add gallery name label if labels_mode >= 1 (gallery labels or all labels)
                if labels_mode >= 1:
                    gallery_name = gallery.get('gallery_name', 'Gallery')
                    bbox = draw.textbbox((0, 0), gallery_name, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                    
                    label_x = x + 10
                    label_y = y + 10
                    
                    # Draw text background
                    draw.rectangle(
                        [label_x - 3, label_y - 2, label_x + text_width + 3, label_y + text_height + 2],
                        fill=(128, 0, 128, 220)
                    )
                    # Draw text
                    draw.text((label_x, label_y), gallery_name, fill=(255, 255, 255, 255), font=font_small)
    
    # Draw entrances
    if 'entrance' in annotation_types or 'entrances' in annotation_types:
        entrances = annotations_data.get('entrances', [])
        if entrances:
            print(f"✓ Drawing {len(entrances)} entrance(s)...")
            drawn_types.append('entrance')
            for entrance in entrances:
                shape = entrance.get('shape', 'rectangle')
                coords = entrance['coordinates']
                
                if shape == 'polygon':
                    # Handle polygon shape (rotated rectangles)
                    points = [(p['x'], p['y']) for p in coords['points']]
                    draw.polygon(points, fill=colors['entrance'], outline=(0, 200, 0, 255))
                    
                    # Calculate center for label
                    if labels_mode == 2:
                        center_x = sum(p[0] for p in points) / len(points)
                        center_y = sum(p[1] for p in points) / len(points)
                        label = "ENTRANCE"
                        bbox = draw.textbbox((0, 0), label, font=font)
                        text_width = bbox[2] - bbox[0]
                        text_height = bbox[3] - bbox[1]
                        label_x = center_x - text_width // 2
                        label_y = center_y - text_height // 2
                        
                        # Draw text background
                        draw.rectangle(
                            [label_x - 5, label_y - 2, label_x + text_width + 5, label_y + text_height + 2],
                            fill=(0, 150, 0, 255)
                        )
                        # Draw text
                        draw.text((label_x, label_y), label, fill=(255, 255, 255, 255), font=font)
                else:
                    # Handle rectangle shape (default)
                    x, y, width, height = coords['x'], coords['y'], coords['width'], coords['height']
                    
                    # Draw rectangle
                    draw.rectangle(
                        [x, y, x + width, y + height],
                        fill=colors['entrance'],
                        outline=(0, 200, 0, 255),
                        width=4
                    )
                    
                    # Add label with background only if labels_mode == 2 (all labels)
                    if labels_mode == 2:
                        label = "ENTRANCE"
                        bbox = draw.textbbox((0, 0), label, font=font)
                        text_width = bbox[2] - bbox[0]
                        text_height = bbox[3] - bbox[1]
                        
                        label_x = x + (width - text_width) // 2
                        label_y = y + (height - text_height) // 2
                        
                        # Draw text background
                        draw.rectangle(
                            [label_x - 5, label_y - 2, label_x + text_width + 5, label_y + text_height + 2],
                            fill=(0, 150, 0, 255)
                        )
                        # Draw text
                        draw.text((label_x, label_y), label, fill=(255, 255, 255, 255), font=font)
    
    # Draw exits
    if 'exit' in annotation_types or 'exits' in annotation_types:
        exits = annotations_data.get('exits', [])
        if exits:
            print(f"✓ Drawing {len(exits)} exit(s)...")
            drawn_types.append('exit')
            for exit_ann in exits:
                shape = exit_ann.get('shape', 'rectangle')
                coords = exit_ann['coordinates']
                
                if shape == 'polygon':
                    # Handle polygon shape (rotated rectangles)
                    points = [(p['x'], p['y']) for p in coords['points']]
                    draw.polygon(points, fill=colors['exit'], outline=(200, 200, 0, 255))
                    
                    # Calculate center for label
                    if labels_mode == 2:
                        center_x = sum(p[0] for p in points) / len(points)
                        center_y = sum(p[1] for p in points) / len(points)
                        label = "EXIT"
                        bbox = draw.textbbox((0, 0), label, font=font)
                        text_width = bbox[2] - bbox[0]
                        text_height = bbox[3] - bbox[1]
                        label_x = center_x - text_width // 2
                        label_y = center_y - text_height // 2
                        
                        # Draw text background
                        draw.rectangle(
                            [label_x - 5, label_y - 2, label_x + text_width + 5, label_y + text_height + 2],
                            fill=(150, 150, 0, 255)
                        )
                        # Draw text
                        draw.text((label_x, label_y), label, fill=(255, 255, 255, 255), font=font)
                else:
                    # Handle rectangle shape (default)
                    x, y, width, height = coords['x'], coords['y'], coords['width'], coords['height']
                    
                    # Draw rectangle
                    draw.rectangle(
                        [x, y, x + width, y + height],
                        fill=colors['exit'],
                        outline=(200, 200, 0, 255),
                        width=4
                    )
                    
                    # Add label with background only if labels_mode == 2 (all labels)
                    if labels_mode == 2:
                        label = "EXIT"
                        bbox = draw.textbbox((0, 0), label, font=font)
                        text_width = bbox[2] - bbox[0]
                        text_height = bbox[3] - bbox[1]
                        
                        label_x = x + (width - text_width) // 2
                        label_y = y + (height - text_height) // 2
                        
                        # Draw text background
                        draw.rectangle(
                            [label_x - 5, label_y - 2, label_x + text_width + 5, label_y + text_height + 2],
                            fill=(150, 150, 0, 255)
                        )
                        # Draw text
                        draw.text((label_x, label_y), label, fill=(255, 255, 255, 255), font=font)
    
    # Draw exhibits (circles)
    if 'exhibit' in annotation_types or 'exhibits' in annotation_types:
        exhibits = annotations_data.get('exhibits', [])
        if exhibits:
            print(f"✓ Drawing {len(exhibits)} exhibit(s)...")
            drawn_types.append('exhibit')
            for exhibit in exhibits:
                coords = exhibit['coordinates']
                cx, cy, r = coords['center_x'], coords['center_y'], coords['radius']
                
                # Draw circle
                draw.ellipse(
                    [cx - r, cy - r, cx + r, cy + r],
                    fill=colors['exhibit'],
                    outline=(60, 60, 60, 255),
                    width=2
                )
                
                # Add exhibit number (centered)
                exhibit_num = exhibit.get('exhibit_number', '?')
                text = str(exhibit_num)
                bbox = draw.textbbox((0, 0), text, font=font_exhibit)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                # Center the text in the circle
                text_x = cx - text_width / 2
                text_y = cy - text_height / 2
                
                draw.text((text_x, text_y), text, fill=(255, 255, 255, 255), font=font_exhibit)
    
    # Draw forbidden areas
    if 'forbidden_area' in annotation_types or 'forbidden_areas' in annotation_types:
        forbidden_areas = annotations_data.get('forbidden_areas', [])
        if forbidden_areas:
            print(f"✓ Drawing {len(forbidden_areas)} forbidden area(s)...")
            drawn_types.append('forbidden_area')
            for forbidden in forbidden_areas:
                coords = forbidden['coordinates']
                x, y, width, height = coords['x'], coords['y'], coords['width'], coords['height']
                
                # Draw semi-transparent red rectangle with hatching pattern
                draw.rectangle(
                    [x, y, x + width, y + height],
                    fill=colors['forbidden_area'],
                    outline=colors['forbidden_area'],
                    width=3
                )
                         
                # Add label with background only if labels_mode == 2 (all labels)
                if labels_mode == 2:
                    label = "FORBIDDEN"
                    bbox = draw.textbbox((0, 0), label, font=font_small)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                    
                    label_x = x + 10
                    label_y = y + 10
                    
                    # Draw text background
                    draw.rectangle(
                        [label_x - 3, label_y - 2, label_x + text_width + 3, label_y + text_height + 2],
                        fill=(255, 165, 0, 255)
                    )
                    # Draw text
                    draw.text((label_x, label_y), label, fill=(255, 255, 255, 255), font=font_small)
    
    # Composite the overlay onto the image
    img = Image.alpha_composite(img, overlay)
    
    # Add legend if requested and we drew something
    if show_legend and drawn_types:
        draw_final = ImageDraw.Draw(img)
        
        # Legend configuration
        legend_x, legend_y = 20, 20
        legend_padding = 10
        item_height = 20
        
        # Calculate legend size based on items
        legend_items = []
        for dtype in drawn_types:
            if dtype == 'wall':
                legend_items.append(("Walls", colors['wall'][:3]))
            elif dtype == 'room':
                legend_items.append(("Rooms", colors['room'][:3]))
            elif dtype == 'gallery':
                legend_items.append(("Galleries", (255, 165, 0)))
            elif dtype == 'exhibit':
                legend_items.append(("Exhibits", (255, 0, 0)))
            elif dtype == 'entrance':
                legend_items.append(("Entrance", (0, 255, 0)))
            elif dtype == 'exit':
                legend_items.append(("Exit", (255, 255, 0)))
            elif dtype == 'floor_area':
                legend_items.append(("Floor Area", (128, 128, 255)))
            elif dtype == 'forbidden_area':
                legend_items.append(("Forbidden", (255, 0, 0)))
        
        legend_height = 30 + (len(legend_items) * item_height)
        legend_width = 180
        
        # Draw legend background
        draw_final.rectangle(
            [legend_x - legend_padding, legend_y - legend_padding, 
             legend_x + legend_width, legend_y + legend_height],
            fill=(255, 255, 255, 240),
            outline=(0, 0, 0, 255),
            width=2
        )
        
        # Legend title
        draw_final.text((legend_x, legend_y), "Annotations", fill=(0, 0, 0, 255), font=font_legend)
        
        # Legend items
        y_offset = legend_y + 25
        for label, color in legend_items:
            # Draw color box
            draw_final.rectangle(
                [legend_x, y_offset, legend_x + 15, y_offset + 15],
                fill=color,
                outline=(0, 0, 0, 255)
            )
            # Draw label
            draw_final.text((legend_x + 20, y_offset), label, fill=(0, 0, 0, 255), font=font_small)
            y_offset += item_height
    
    # Convert back to RGB for saving - preserve exact dimensions
    # Use RGB mode to match original if it was RGB
    output_img = Image.new('RGB', original_size)
    output_img.paste(img.convert('RGB'), (0, 0))
    
    # Save with maximum quality to preserve all details
    output_img.save(output_path, 'PNG', quality=100, optimize=False)
    
    print("=" * 60)
    print("✅ SUCCESS!")
    print(f"Annotated image saved to: {output_path}")
    print(f"Annotation types displayed: {', '.join(drawn_types)}")
    print("=" * 60)

def main():
    """Main function with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description='Generate annotated museum layout images with selective annotation types',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Draw only entrance and exit (no labels)
  python generate_custom_annotations.py museum_layout_01.png museum_layout_updated.json -t entrance exit -o layout_entrance_exit.png
  
  # Draw galleries with labels (gallery labels only)
  python generate_custom_annotations.py museum_layout_01.png museum_layout_updated.json -t gallery --labels 1 -o layout_galleries.png
  
  # Draw galleries and exhibits with gallery names
  python generate_custom_annotations.py museum_layout_01.png museum_layout_updated.json -t gallery exhibit --labels 1 -o layout_galleries_exhibits.png
  
  # Draw entrance, exit, and galleries with all labels
  python generate_custom_annotations.py museum_layout_01.png museum_layout_updated.json -t entrance exit gallery --labels 2 -o layout_with_all_labels.png
  
  # Draw everything with gallery labels only
  python generate_custom_annotations.py museum_layout_01.png museum_layout_with_forbidden.json -t wall gallery exhibit entrance exit floor_area forbidden_area --labels 1 --legend -o layout_complete.png
  
  # Draw forbidden areas with entrance and exit (all labels)
  python generate_custom_annotations.py museum_layout_01.png museum_layout_with_forbidden.json -t entrance exit forbidden_area --labels 2 --legend -o layout_access_control.png

Label modes:
  --labels 0  No labels on bounding boxes (default)
  --labels 1  Gallery labels only
  --labels 2  All labels (entrance, exit, gallery, forbidden_area)

Available annotation types:
  - entrance / entrances
  - exit / exits
  - wall / walls
  - gallery / galleries (replaces room/rooms)
  - exhibit / exhibits
  - floor_area / floor_areas
  - forbidden_area / forbidden_areas
  
Note: 'room/rooms' is deprecated but still supported for backward compatibility with older JSON files.
        """
    )
    
    parser.add_argument('image', help='Path to original layout image (e.g., museum_layout_01.png)')
    parser.add_argument('annotations', help='Path to annotations JSON file (e.g., museum_layout_annotations.json)')
    parser.add_argument('-t', '--types', nargs='+', required=True,
                       help='Annotation types to display (e.g., entrance exit exhibit gallery forbidden_area)')
    parser.add_argument('-o', '--output', default='custom_annotated_layout.png',
                       help='Output image path (default: custom_annotated_layout.png)')
    parser.add_argument('--legend', action='store_true',
                       help='Enable legend display (disabled by default)')
    parser.add_argument('--labels', type=int, default=0, choices=[0, 1, 2],
                       help='Label display mode: 0=no labels (default), 1=gallery labels only, 2=all labels')
    
    args = parser.parse_args()
    
    # Validate inputs
    try:
        # Load annotations
        annotations_data = load_annotations(args.annotations)
        
        # Normalize annotation types (handle both singular and plural)
        annotation_types = [t.lower() for t in args.types]
        
        # Generate the image
        generate_annotated_image(
            args.image,
            annotations_data,
            annotation_types,
            args.output,
            show_legend=args.legend,
            labels_mode=args.labels
        )
        
    except FileNotFoundError as e:
        print(f"❌ Error: File not found - {e}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON file - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    # If no arguments provided, show help and example usage
    if len(sys.argv) == 1:
        print("=" * 60)
        print("Generate Custom Annotated Museum Layout Images")
        print("=" * 60)
        print("\nQuick Example:")
        print("  python generate_custom_annotations.py museum_layout_01.png floorplan_annotations/museum_layout_updated.json -t gallery exhibit -o layout_galleries_exhibits.png")
        print("\nFor full help and more examples:")
        print("  python generate_custom_annotations.py --help")
        print("=" * 60)
        sys.exit(0)
    
    main()
