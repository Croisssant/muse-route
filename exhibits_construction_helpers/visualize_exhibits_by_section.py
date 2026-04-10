"""
Exhibit Visualization by Section
Visualizes exhibits on the museum floorplan, color-coded by section.
Reads exhibit data from CSV file with exhibit_number column.
"""

import json
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid Tkinter issues
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def load_exhibits_from_csv(csv_file):
    """
    Load exhibits from CSV file with exhibit_number column.
    Returns: (exhibits_by_section, exhibit_to_section_mapping)
    """
    df = pd.read_csv(csv_file)
    
    # Verify required columns exist
    required_cols = ['exhibit_number', 'Section']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in CSV: {missing_cols}")
    
    # Create exhibits_by_section dictionary
    exhibits_by_section = {}
    for section in df['Section'].unique():
        section_exhibits = df[df['Section'] == section]['exhibit_number'].tolist()
        exhibits_by_section[section] = sorted(section_exhibits)
    
    # Create exhibit_to_section mapping
    exhibit_to_section = dict(zip(df['exhibit_number'], df['Section']))
    
    return exhibits_by_section, exhibit_to_section


def load_annotations(json_file):
    """Load museum layout annotations."""
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_section_colors(sections):
    """
    Generate distinct colors for each section.
    Returns a dictionary mapping section names to RGB colors.
    """
    # Predefined distinct colors (BGR format for OpenCV)
    color_palette = [
        (255, 0, 0),      # Blue
        (0, 255, 0),      # Green
        (0, 0, 255),      # Red
        (255, 255, 0),    # Cyan
        (255, 0, 255),    # Magenta
        (0, 255, 255),    # Yellow
        (128, 0, 128),    # Purple
        (255, 165, 0),    # Orange
        (0, 128, 128),    # Teal
        (128, 128, 0),    # Olive
    ]
    
    section_colors = {}
    for idx, section in enumerate(sorted(sections)):
        section_colors[section] = color_palette[idx % len(color_palette)]
    
    return section_colors


def visualize_exhibits(floorplan_path, annotations_path, exhibits_csv_path, output_path='exhibit_visualization.png'):
    """
    Visualize exhibits on the floorplan, color-coded by section.
    """
    print("🎨 Exhibit Visualization by Section")
    print("=" * 60)
    
    # Load data
    print(f"📊 Loading exhibits from CSV: {exhibits_csv_path}")
    exhibits_by_section, exhibit_to_section = load_exhibits_from_csv(exhibits_csv_path)
    
    print(f"   Loaded {sum(len(v) for v in exhibits_by_section.values())} exhibits across {len(exhibits_by_section)} sections")
    
    print(f"📄 Loading annotations from: {annotations_path}")
    annotations = load_annotations(annotations_path)
    
    print(f"🖼️  Loading floorplan image from: {floorplan_path}")
    img = cv2.imread(floorplan_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image from {floorplan_path}")
    
    # Create a copy for visualization
    vis_img = img.copy()
    
    # Generate colors for each section
    sections = list(exhibits_by_section.keys())
    section_colors = generate_section_colors(sections)
    
    print(f"\n🎨 Section Colors:")
    for section, color in section_colors.items():
        print(f"   • {section}: RGB{color}")
    
    print(f"\n📍 Processing {len(annotations['exhibits'])} exhibits...")
    
    # Track statistics
    matched_exhibits = 0
    unmatched_exhibits = []
    
    # Draw each exhibit
    for exhibit_data in annotations['exhibits']:
        exhibit_num_str = exhibit_data.get('exhibit_number', '')
        
        # Convert to integer for matching
        try:
            exhibit_num = int(exhibit_num_str)
        except (ValueError, TypeError):
            continue
        
        # Check if this exhibit is in our section mapping
        section = exhibit_to_section.get(exhibit_num)
        
        if section is None:
            # Not in our list, draw in gray
            color = (128, 128, 128)  # Gray
            unmatched_exhibits.append(exhibit_num)
        else:
            # Get section color
            color = section_colors[section]
            matched_exhibits += 1
        
        # Get coordinates
        coords = exhibit_data['coordinates']
        center_x = int(coords['center_x'])
        center_y = int(coords['center_y'])
        radius = int(coords.get('radius', 20))
        
        # Draw filled circle
        cv2.circle(vis_img, (center_x, center_y), radius, color, -1)
        
        # Draw border
        cv2.circle(vis_img, (center_x, center_y), radius, (0, 0, 0), 2)
        
        # Draw exhibit number
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 2
        text = str(exhibit_num)
        
        # Get text size to center it
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = center_x - text_size[0] // 2
        text_y = center_y + text_size[1] // 2
        
        # Draw text with white background for readability
        cv2.putText(vis_img, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness + 1)
        cv2.putText(vis_img, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)
    
    # Print statistics
    print(f"\n📊 Visualization Statistics:")
    print(f"   ✅ Matched exhibits: {matched_exhibits}")
    print(f"   ⚪ Unmatched exhibits: {len(unmatched_exhibits)}")
    if unmatched_exhibits:
        print(f"   Unmatched: {sorted(unmatched_exhibits)}")
    
    # Create visualization with legend using matplotlib (only output)
    print(f"\n📝 Creating visualization with legend...")
    create_legend(section_colors, exhibits_by_section, output_path, vis_img)
    
    return vis_img


def create_legend(section_colors, exhibits_by_section, output_path, img):
    """
    Create a version with a legend using matplotlib.
    """
    # Convert BGR to RGB for matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.imshow(img_rgb)
    ax.axis('off')
    
    # Create legend patches
    legend_patches = []
    for section in sorted(section_colors.keys()):
        color_bgr = section_colors[section]
        color_rgb = (color_bgr[2]/255, color_bgr[1]/255, color_bgr[0]/255)  # Convert BGR to RGB and normalize
        exhibit_count = len(exhibits_by_section[section])
        label = f"{section} ({exhibit_count} exhibits)"
        legend_patches.append(mpatches.Patch(color=color_rgb, label=label))
    
    # Add legend
    plt.legend(handles=legend_patches, loc='upper right', 
               fontsize=8, framealpha=0.9,
               borderpad=0.4, labelspacing=0.5, 
               handlelength=1.5, handletextpad=0.5)
    
    # Save
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"💾 Saved visualization with legend to: {output_path}")


def main():
    """
    Main function with command-line argument parsing.
    """
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Visualize exhibits on museum floorplan, color-coded by section',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default paths
  python visualize_exhibits_by_section.py \\
      -f original_floorplans/museum_layout_01/museum_layout_01(Wordless).png \\
      -a original_floorplans/museum_layout_01/museum_layout_annotations.json \\
      -c selected_exhibits_final.csv
  
  # Specify custom output path
  python visualize_exhibits_by_section.py \\
      -f floorplan.png -a annotations.json -c exhibits.csv \\
      -o custom_output.png
        """
    )
    
    parser.add_argument(
        '-f', '--floorplan',
        type=str,
        required=True,
        help='Path to floorplan image file (PNG format)'
    )
    
    parser.add_argument(
        '-a', '--annotations',
        type=str,
        required=True,
        help='Path to museum layout annotations JSON file'
    )
    
    parser.add_argument(
        '-c', '--csv',
        type=str,
        required=True,
        help='Path to exhibits CSV file with exhibit_number column (generated by generate_exhibit_list.py)'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output path for visualization image (default: same directory as CSV file with name "exhibit_visualization.png")'
    )
    
    args = parser.parse_args()
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        # Default: same directory as CSV file
        csv_path = Path(args.csv)
        output_path = str(csv_path.parent / 'exhibit_visualization.png')
    
    # Verify input files exist
    if not Path(args.floorplan).exists():
        print(f"❌ Floorplan not found: {args.floorplan}")
        return
    
    if not Path(args.annotations).exists():
        print(f"❌ Annotations not found: {args.annotations}")
        return
    
    if not Path(args.csv).exists():
        print(f"❌ Exhibits CSV file not found: {args.csv}")
        print(f"   Please run generate_exhibit_list.py first to create this file.")
        return
    
    # Create visualization
    try:
        visualize_exhibits(args.floorplan, args.annotations, args.csv, output_path)
        print(f"\n✅ Done! Open the visualization file to view exhibits by section.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
