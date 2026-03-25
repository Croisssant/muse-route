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

def generate_annotated_image(image_path, annotations_data, annotation_types, output_path, show_legend=False):
    """
    Generate an annotated image with only specified annotation types.
    
    Args:
        image_path: Path to original image
        annotations_data: Loaded annotation data
        annotation_types: List of annotation types to display 
                         (e.g., ['entrance', 'exit', 'wall', 'room', 'exhibit', 'floor_area'])
        output_path: Path to save the output image
        show_legend: Whether to show the legend
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
        font_legend = ImageFont.truetype("arial.ttf", 14)
    except:
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()
        font_legend = ImageFont.load_default()
    
    # Color scheme (same as existing scripts)
    colors = {
        'wall': (0, 0, 255, 255),        # Blue - opaque
        'room': (255, 165, 0, 255),      # Orange - opaque
        'exhibit': (255, 0, 0, 200),     # Red - semi-transparent
        'entrance': (0, 255, 0, 200),    # Green - semi-transparent
        'exit': (255, 255, 0, 200),      # Yellow - semi-transparent
        'floor_area': (128, 128, 255, 50), # Light blue - very transparent
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
    
    # Draw rooms (polylines)
    if 'room' in annotation_types or 'rooms' in annotation_types:
        rooms = annotations_data.get('rooms', [])
        if rooms:
            print(f"✓ Drawing {len(rooms)} room(s)...")
            drawn_types.append('room')
            for room in rooms:
                points = [(p['x'], p['y']) for p in room['coordinates']['points']]
                if len(points) > 1:
                    draw_opaque.line(points, fill=colors['room'][:3], width=3)
    
    # Draw entrances
    if 'entrance' in annotation_types or 'entrances' in annotation_types:
        entrances = annotations_data.get('entrances', [])
        if entrances:
            print(f"✓ Drawing {len(entrances)} entrance(s)...")
            drawn_types.append('entrance')
            for entrance in entrances:
                coords = entrance['coordinates']
                x, y, width, height = coords['x'], coords['y'], coords['width'], coords['height']
                
                # Draw rectangle
                draw.rectangle(
                    [x, y, x + width, y + height],
                    fill=colors['entrance'],
                    outline=(0, 200, 0, 255),
                    width=4
                )
                
                # Add label with background
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
                coords = exit_ann['coordinates']
                x, y, width, height = coords['x'], coords['y'], coords['width'], coords['height']
                
                # Draw rectangle
                draw.rectangle(
                    [x, y, x + width, y + height],
                    fill=colors['exit'],
                    outline=(200, 200, 0, 255),
                    width=4
                )
                
                # Add label with background
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
                    outline=(200, 0, 0, 255),
                    width=2
                )
                
                # Add exhibit number
                exhibit_num = exhibit.get('exhibit_number', '?')
                draw.text((cx - 5, cy - 5), str(exhibit_num), fill=(255, 255, 255, 255), font=font_small)
    
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
            elif dtype == 'exhibit':
                legend_items.append(("Exhibits", (255, 0, 0)))
            elif dtype == 'entrance':
                legend_items.append(("Entrance", (0, 255, 0)))
            elif dtype == 'exit':
                legend_items.append(("Exit", (255, 255, 0)))
            elif dtype == 'floor_area':
                legend_items.append(("Floor Area", (128, 128, 255)))
        
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
  # Draw only entrance and exit
  python generate_custom_annotations.py museum_layout_01.png museum_layout_annotations.json -t entrance exit -o layout_entrance_exit.png
  
  # Draw only exhibits
  python generate_custom_annotations.py museum_layout_01.png museum_layout_annotations.json -t exhibit -o layout_exhibits.png
  
  # Draw walls, entrance, and exit
  python generate_custom_annotations.py museum_layout_01.png museum_layout_annotations.json -t wall entrance exit -o layout_with_navigation.png
  
  # Draw everything
  python generate_custom_annotations.py museum_layout_01.png museum_layout_annotations.json -t wall room exhibit entrance exit floor_area -o layout_complete.png

Available annotation types:
  - entrance / entrances
  - exit / exits
  - wall / walls
  - room / rooms
  - exhibit / exhibits
  - floor_area / floor_areas
        """
    )
    
    parser.add_argument('image', help='Path to original layout image (e.g., museum_layout_01.png)')
    parser.add_argument('annotations', help='Path to annotations JSON file (e.g., museum_layout_annotations.json)')
    parser.add_argument('-t', '--types', nargs='+', required=True,
                       help='Annotation types to display (e.g., entrance exit exhibit)')
    parser.add_argument('-o', '--output', default='custom_annotated_layout.png',
                       help='Output image path (default: custom_annotated_layout.png)')
    parser.add_argument('--legend', action='store_true',
                       help='Enable legend display (disabled by default)')
    
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
            show_legend=args.legend
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
        print("  python generate_custom_annotations.py museum_layout_01.png museum_layout_annotations.json -t entrance exit -o entrance_exit.png")
        print("\nFor full help and more examples:")
        print("  python generate_custom_annotations.py --help")
        print("=" * 60)
        sys.exit(0)
    
    main()
