"""
Unified museum route planning script supporting easy, medium, and hard difficulty levels.
"""
import argparse
import base64
import json
import re
import subprocess
import os
import sys
import io
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image, ImageDraw
from build_prompts import build_selection_prompt, build_route_planning_prompt

# -------- Force UTF-8 encoding for stdout on Windows --------
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# -------- Load API key --------
load_dotenv(override=True)
client = OpenAI()


def parse_json_from_text(text):
    """Extract and parse JSON from text that may contain markdown or other formatting."""
    text = text.strip()
    # Remove markdown code fences
    text = re.sub(r"^```json\s*|\s*```$", "", text, flags=re.MULTILINE)
    
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fallback: try to find JSON array in the text
        match = re.search(r"\[\s*\[.*?\]\s*\]", text, re.DOTALL)
        if match:
            return json.loads(match.group())
        # Try to find simple array of numbers
        match = re.search(r"\[\s*\d+(?:\s*,\s*\d+)*\s*\]", text)
        if match:
            return json.loads(match.group())
        raise ValueError(f"Failed to parse JSON from text:\n{text}")


def select_exhibits(difficulty, exhibits_json, configs, user_preference):
    """
    Select exhibits based on difficulty level.
    Returns list of selected exhibit IDs or None for easy level.
    """
    if difficulty == 'easy':
        print("EASY LEVEL: No exhibit selection required - navigating from entrance to exit only")
        return None
    
    print(f"\n{'='*70}")
    print(f"STEP 1: Exhibit Selection ({difficulty.upper()} level)")
    print(f"{'='*70}\n")
    
    system_prompt, user_prompt = build_selection_prompt(
        difficulty, exhibits_json, configs, user_preference
    )
    
    response = client.responses.create(
        model="gpt-5.4",
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    
    selected_ids = parse_json_from_text(response.output_text)
    print(f"Selected exhibit IDs: {selected_ids}")
    
    return selected_ids


def plan_route(difficulty, image_base64, img_width, img_height, visit_distance_text, 
               filtered_exhibits=None):
    """
    Plan the museum route based on difficulty level and selected exhibits.
    Returns list of coordinate tuples representing the route.
    """
    print(f"\n{'='*70}")
    print(f"STEP 2: Route Planning ({difficulty.upper()} level)")
    print(f"{'='*70}\n")
    
    system_prompt, user_prompt = build_route_planning_prompt(
        difficulty, img_width, img_height, visit_distance_text, filtered_exhibits
    )
    
    response = client.responses.create(
        model="gpt-5.4",
        reasoning={"effort": "high"},
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "input_text", "text": user_prompt},
                {"type": "input_image", "image_url": f"data:image/png;base64,{image_base64}"}
            ]}
        ]
    )
    
    route = parse_json_from_text(response.output_text)
    
    if len(route) < 2:
        raise ValueError(f"Route must contain at least 2 coordinate pairs")
    
    return [tuple(point) for point in route]


def validate_route_coordinates(route, img_width, img_height):
    """Validate that all route coordinates are within image bounds."""
    invalid_points = []
    for i, (x, y) in enumerate(route):
        if not (0 <= x < img_width and 0 <= y < img_height):
            invalid_points.append(f"Point {i}: ({x}, {y})")
    
    if invalid_points:
        print("WARNING: Found out-of-bounds coordinates:")
        for point in invalid_points:
            print(f"  {point}")
        print(f"  Valid range: 0 <= x < {img_width}, 0 <= y < {img_height}")


def run_validation_pipeline(args, output_image_path, input_image_path, output_directory):
    """Run the validation pipeline (validation_pipeline.py)."""
    print(f"\n{'='*70}")
    print("STEP 3: Running Validation Pipeline")
    print(f"{'='*70}\n")
    
    output_filename = os.path.basename(output_image_path)
    
    cmd = [
        "python", "validation_pipeline.py",
        "--route-image", str(output_image_path),
        "--original-image", input_image_path,
        "--annotations", args.annotations,
        "--config-file", args.config,
        "--exhibits-csv-file", args.exhibits_csv,
        "--extraction-output-image", str(output_directory / f"extracted_{output_filename}"),
        "--validated-output-image", str(output_directory / f"validated_{output_filename}"),
        "--validated-output-json", str(output_directory / "validation_results.json"),
    ]
    
    if args.tolerance:
        cmd.extend(["--tolerance", str(args.tolerance)])
    if args.marker_size:
        cmd.extend(["--marker-size", str(args.marker_size)])
    if args.debug:
        cmd.append("--debug")
    
    print(f"Running: {' '.join(cmd)}\n")
    
    try:
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8', env=env)
        print(result.stdout)
        
        # Load and display validation results
        validation_json_path = output_directory / "validation_results.json"
        if validation_json_path.exists():
            with open(validation_json_path, "r", encoding="utf-8") as f:
                validation_data = json.load(f)
            
            print(f"\n{'='*70}")
            print("VALIDATION SUMMARY")
            print(f"{'='*70}")
            print(json.dumps(validation_data['validation_summary'], indent=2))
            
            display_validation_metrics(validation_data['validation_summary'], args.difficulty)
            
    except subprocess.CalledProcessError as e:
        print("Error running validation pipeline:")
        print(f"Return code: {e.returncode}")
        if e.stdout:
            print(f"STDOUT:\n{e.stdout}")
        if e.stderr:
            print(f"STDERR:\n{e.stderr}")
    except Exception as e:
        print(f"Unexpected error during validation: {e}")


def display_validation_metrics(summary, difficulty):
    """Display key validation metrics in a formatted way."""
    print(f"\n{'='*70}")
    print(f"KEY METRICS ({difficulty.upper()} LEVEL)")
    print(f"{'='*70}")
    
    # Spatial Validity Results (SVR)
    if 'svr' in summary:
        svr = summary['svr']
        print(f"Connectivity: {svr.get('connectivity', 'N/A')}")
        print(f"No Wall Crossings: {not svr.get('wall_crossings', True)}")
        print(f"No Exhibit Collisions: {not svr.get('exhibit_collision', True)}")
        print(f"Within Floor Area: {not svr.get('out_of_area_violations', True)}")
    
    # Spatial Constraint Satisfaction Results (SCSR)
    if 'scsr' in summary:
        scsr = summary['scsr']
        print(f"Start/End Correct: {scsr.get('start_end_location', 'N/A')}")
        if difficulty != 'easy':
            print(f"Must-Pass Regions: {scsr.get('must_pass_regions', 'N/A')}")
        print(f"No Restricted Area Violations: {not scsr.get('restricted_area_violations', True)}")
        if difficulty != 'easy':
            print(f"Within Distance Budget: {scsr.get('distance_budget', 'N/A')}")
    
    # Semantic Coverage and Requirements (SCAR) - only for medium/hard
    if difficulty != 'easy' and 'scar' in summary:
        scar = summary['scar']
        print("\nSemantic Validation:")
        if 'specific_exhibit_coverage' in scar:
            spec = scar['specific_exhibit_coverage']
            print(f"  Specific Exhibits: {spec.get('valid', False)} ({spec.get('covered', 0)}/{spec.get('required', 0)})")
        if 'at_least_n_exhibits_coverage' in scar:
            min_ex = scar['at_least_n_exhibits_coverage']
            print(f"  Minimum Exhibits: {min_ex.get('valid', False)} ({min_ex.get('covered', 0)}/{min_ex.get('required', 0)})")
        if 'exhibit_category_coverage' in scar:
            print(f"  Category Coverage:")
            for cat, data in scar['exhibit_category_coverage'].items():
                print(f"    {cat}: {data.get('valid', False)}")
    
    print(f"{'='*70}")
    
    if difficulty == 'easy':
        print("Note: Easy level does NOT require gallery visits or exhibit coverage")


def main():
    parser = argparse.ArgumentParser(
        description='Unified museum route planning for easy, medium, and hard difficulty levels',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--difficulty', required=True, choices=['easy', 'medium', 'hard'],
                        help='Difficulty level (easy, medium, or hard)')
    parser.add_argument('--input-image', required=True,
                        help='Path to input museum layout image')
    parser.add_argument('--output-dir', required=True,
                        help='Directory to save output files')
    parser.add_argument('--annotations', required=True,
                        help='Path to museum layout annotations JSON file')
    parser.add_argument('--config', required=True,
                        help='Path to configuration JSON file')
    parser.add_argument('--exhibits-csv', required=True,
                        help='Path to exhibits CSV file')
    
    # Optional arguments for medium/hard levels
    parser.add_argument('--exhibits-json', 
                        help='Path to exhibits JSON file (required for medium/hard)')
    parser.add_argument('--user-preference', default='I want to see interesting exhibits',
                        help='User preference for exhibit selection (for medium/hard)')
    parser.add_argument('--model', default='gpt-5.4',
                        help='OpenAI model to use (default: gpt-5.4)')
    
    # Validation pipeline arguments
    parser.add_argument('--tolerance', type=int, default=3,
                        help='Dilation radius for validation (default: 3)')
    parser.add_argument('--marker-size', type=int, default=15,
                        help='Marker size for visualization (default: 15)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug output')
    
    args = parser.parse_args()
    
    # Validate arguments for medium/hard
    if args.difficulty in ['medium', 'hard'] and not args.exhibits_json:
        parser.error(f"--exhibits-json is required for {args.difficulty} difficulty")
    
    # Create output directory
    output_directory = Path(args.output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    
    # Load image
    print(f"Loading image from: {args.input_image}")
    image = Image.open(args.input_image)
    img_width, img_height = image.size
    print(f"Image dimensions: {img_width}x{img_height}")
    
    # Read image as base64
    with open(args.input_image, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode("utf-8")
    
    # Load configurations
    with open(args.config, "r", encoding="utf-8") as f:
        configs = json.load(f)
    
    with open(args.annotations, "r", encoding="utf-8") as f:
        annotations = json.load(f)
    
    # Calculate visit distance
    visit_distance_mm = configs.get("exhibit_see_distance_in_mm")
    distance_entries = annotations.get("distance_in_mm", [])
    mm_per_px = distance_entries[0].get("mm_per_px") if distance_entries else None
    visit_distance_px = int(round(visit_distance_mm / mm_per_px)) if visit_distance_mm and mm_per_px else None
    visit_distance_text = (
        f"{visit_distance_px} pixels"
        if visit_distance_px is not None
        else f"the configured visit radius derived from {visit_distance_mm} mm"
    )
    
    # Load exhibits for medium/hard
    exhibits_json = None
    if args.difficulty in ['medium', 'hard']:
        with open(args.exhibits_json, "r", encoding="utf-8") as f:
            exhibits_json = json.load(f)
    
    # Step 1: Select exhibits (if not easy)
    selected_ids = select_exhibits(args.difficulty, exhibits_json, configs, args.user_preference)
    
    # Filter exhibits based on selection
    filtered_exhibits = None
    if selected_ids is not None:
        filtered_exhibits = [e for e in exhibits_json if e["exhibit_number"] in selected_ids]
    
    # Step 2: Plan route
    route = plan_route(
        args.difficulty, 
        image_base64, 
        img_width, 
        img_height, 
        visit_distance_text,
        filtered_exhibits
    )
    
    # Validate coordinates
    validate_route_coordinates(route, img_width, img_height)
    
    # Draw route on image
    draw = ImageDraw.Draw(image)
    draw.line(route, fill="red", width=5)
    
    # Save output image
    output_image_path = output_directory / f"{args.difficulty}_route.png"
    image.save(output_image_path)
    print(f"\nRoute drawn and saved to {output_image_path}")
    
    # Step 3: Run validation pipeline
    run_validation_pipeline(args, output_image_path, args.input_image, output_directory)
    
    print("\nPipeline complete!")


if __name__ == '__main__':
    main()
