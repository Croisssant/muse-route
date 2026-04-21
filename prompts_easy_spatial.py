
import json
import subprocess
import os
import sys
import io
import argparse
import logging

from dotenv import load_dotenv
from tqdm import tqdm
from pathlib import Path
 
from build_prompts import build_visit_distance

from prompts_utils import (load_image, prompt_model,
                           parse_route_and_save,
                           discover_complexity_and_layouts, build_paths,
                           extract_scar_validations, extract_validation_fields,
                           build_backend)
 

# -------- Force UTF-8 encoding for stdout on Windows --------
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# -------- Load API key --------
load_dotenv(override=True)


def parse_arguments():
    """Parse command-line arguments for user configurations."""
    parser = argparse.ArgumentParser(
        description='Museum Route Planning - Batch Processor (Easy Spatial) with configurable parameters',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        # Run with OpenAI (default)
        python prompts_easy_spatial.py --backend openai --model gpt-5.4
        
        # Run with a local HuggingFace model
        python prompts_easy_spatial.py --backend huggingface --model google/gemma-4-31B-it
        
        # Run with Ollama
        python prompts_easy_spatial.py --backend ollama --model qwen3.6
        
        # Override model and difficulty
        python prompts_easy_spatial.py --model gpt-4 --difficulty easy_semantic
        
        # Process specific complexity
        python prompts_easy_spatial.py --complexity-mode single --complexity complex
        
        # Process multiple complexities
        python prompts_easy_spatial.py --complexity-mode list --complexity simple complex
        
        # Process specific layouts
        python prompts_easy_spatial.py --layout-mode list --layouts layout_01 layout_02
        
        # Customize validation fields
        python prompts_easy_spatial.py --svr-fields connectivity wall_crossings
        """
    )
    
    parser.add_argument('--backend', type=str, default='openai',
                    choices=['openai', 'huggingface', 'ollama'],
                    help='Model backend to use (default: openai)')

    # Validation fields
    parser.add_argument('--svr-fields', nargs='+', 
                       default=['connectivity', 'wall_crossings', 'exhibit_collision', 'out_of_area_violations'],
                       help='SVR validation fields (default: connectivity wall_crossings exhibit_collision out_of_area_violations)')
    
    parser.add_argument('--scsr-fields', nargs='+',
                       default=['start_end_location', 'must_pass_regions', 'restricted_area_violations'],
                       help='SCSR validation fields (default: start_end_location must_pass_regions restricted_area_violations)')
    
    parser.add_argument('--scar-fields', nargs='+',
                       default=['at_least_n_exhibits_coverage'],
                       help='SCAR validation fields (default: at_least_n_exhibits_coverage)')
    
    # Model configuration
    parser.add_argument('--model', type=str, default='gpt-5.4',
                       help='Model name to use (default: gpt-5.4)')
    
    parser.add_argument('--reasoning', type=str, default=None,
                       help='Reasoning mode (default: None)')
    
    # Task configuration
    parser.add_argument('--difficulty', type=str, default='easy_spatial',
                       help='Difficulty level for output folder structure (default: easy_spatial)')
    
    # Complexity selection
    parser.add_argument('--complexity-mode', type=str, 
                       choices=['all', 'single', 'list'],
                       default='single',
                       help='Complexity selection mode (default: single)')
    
    parser.add_argument('--complexity', nargs='+', type=str,
                       default=['simple'],
                       help='Specific complexity level(s) (default: simple)')
    
    # Layout selection
    parser.add_argument('--layout-mode', type=str,
                       choices=['all', 'single', 'list'],
                       default='all',
                       help='Layout selection mode (default: all)')
    
    parser.add_argument('--layouts', nargs='+', type=str,
                       default=None,
                       help='Specific layout(s) when using single or list mode')
    
    # Directory paths
    parser.add_argument('--floorplan-dir', type=str, default='floorplans',
                       help='Base directory for floorplans (default: floorplans)')
    
    parser.add_argument('--results-dir', type=str, default='results',
                       help='Base directory for results (default: results)')
    
    # Filenames
    parser.add_argument('--image-file', type=str, default='annotated_layout.png',
                       help='Image filename (default: annotated_layout.png)')
    
    parser.add_argument('--config-file', type=str, default='easy_spatial.json',
                       help='Config filename (default: easy_spatial.json)')
    
    parser.add_argument('--exhibits-file', type=str, default='exhibit_list.json',
                       help='Exhibits list filename (default: exhibit_list.json)')
    
    parser.add_argument('--exhibits-csv-file', type=str, default='exhibits.csv',
                       help='Exhibits CSV filename (default: exhibits.csv)')
    
    parser.add_argument('--annotations-file', type=str, default='layout_annotations.json',
                       help='Annotations filename (default: layout_annotations.json)')
    
    # Progress tracking
    parser.add_argument('--no-progress', action='store_true',
                       help='Disable progress bar (useful for automation/logging)')
    
    parser.add_argument('--progress-position', type=int, default=0,
                       help='Progress bar position for concurrent execution (default: 0)')
    
    return parser.parse_args()


# ========== PARSE ARGUMENTS ==========
args = parse_arguments()

# Metric fields to include in final_results.json
SVR_FIELDS = args.svr_fields
SCSR_FIELDS = args.scsr_fields
SCAR_FIELDS = args.scar_fields

# Model configuration
MODEL = args.model
REASONING = args.reasoning

# Build model folder name (same logic as master script)
# If reasoning is provided, folder name is [model]([reasoning]), otherwise just [model]
if REASONING:
    MODEL_FOLDER = f"{MODEL}({REASONING})"
else:
    MODEL_FOLDER = MODEL

# Task configuration
DIFFICULTY = args.difficulty

# Complexity selection mode
COMPLEXITY_MODE = args.complexity_mode
if COMPLEXITY_MODE == 'single':
    SPECIFIC_COMPLEXITIES = args.complexity[0]
elif COMPLEXITY_MODE == 'list':
    SPECIFIC_COMPLEXITIES = args.complexity
else:
    SPECIFIC_COMPLEXITIES = None

# Layout selection mode
LAYOUT_MODE = args.layout_mode
if LAYOUT_MODE == 'single' and args.layouts:
    SPECIFIC_LAYOUTS = args.layouts[0]
elif LAYOUT_MODE == 'list':
    SPECIFIC_LAYOUTS = args.layouts
else:
    SPECIFIC_LAYOUTS = args.layouts

# Directory paths
BASE_FLOORPLAN_DIR = Path(args.floorplan_dir)
BASE_RESULTS_DIR = Path(args.results_dir) / MODEL_FOLDER

# Default filenames
IMAGE_FILENAME = args.image_file
CONFIG_FILENAME = args.config_file
EXHIBITS_FILENAME = args.exhibits_file
EXHIBITS_CSV_FILENAME = args.exhibits_csv_file
ANNOTATIONS_FILENAME = args.annotations_file

# Filename mapping for build_paths function
FILENAMES = {
    'image': IMAGE_FILENAME,
    'config': CONFIG_FILENAME,
    'exhibits': EXHIBITS_FILENAME,
    'exhibits_csv': EXHIBITS_CSV_FILENAME,
    'annotations': ANNOTATIONS_FILENAME
}

BACKEND = build_backend(args.backend, args.model, args.reasoning)

# ========================================


def setup_layout_logger(log_file_path):
    """Setup a logger for a specific layout."""
    logger_name = str(log_file_path)
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    
    logger.handlers = []
    
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', 
                                  datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    
    return logger


def process_layout(layout_folder, model_name, difficulty, complexity, pbar=None):
    """Process a single layout folder.
    
    Args:
        layout_folder: Path to the layout folder
        model_name: Name of the model to use
        difficulty: Difficulty level
        complexity: Complexity level
        pbar: Optional tqdm progress bar to update
    """
    layout_name = layout_folder.name
    
    # Update progress bar if provided
    if pbar:
        pbar.set_description(f"[Spatial] {complexity}/{layout_name}")
    
    # Build paths using the utility function
    input_paths, output_paths = build_paths(
        layout_folder, model_name, difficulty, complexity,
        BASE_RESULTS_DIR, FILENAMES
    )
    
    # Setup per-layout logger
    log_file = output_paths['dir'] / "processing.log"
    logger = setup_layout_logger(log_file)
    logger.info(f"Starting processing for {complexity}/{layout_name}")
    logger.info(f"Model: {model_name}, Difficulty: {difficulty}")
    
    # Validate input files exist
    missing_files = []
    for name, path in input_paths.items():
        if not path.exists():
            missing_files.append(f"{name}: {path}")
    
    if missing_files:
        error_msg = f"ERROR: Missing required files for {layout_folder.name}:"
        logger.error(error_msg)
        for missing in missing_files:
            logger.error(f"  - {missing}")
        print(f"✗ {complexity}/{layout_name}: Missing files")
        return False
    
    try:
        # -------- Load data --------   
        with open(input_paths['config'], "r", encoding="utf-8") as f:
            configs = json.load(f)
        
        with open(input_paths['annotations'], "r", encoding="utf-8") as f:
            annotations_json = json.load(f)
        
        # -------- Build prompts --------
        visit_distance_text = build_visit_distance(configs.get("exhibit_see_distance_in_mm"), annotations_json.get("distance_in_mm", []))
        
        # -------- Image info --------
        image_base64, img_width, img_height = load_image(input_paths['image'])
        
        
        # -------- Route Planning --------
        system_prompt_route = f"""
        You are a museum path planning assistant generating a single drawable polyline on top of the museum image.

        Your ONLY goal is to create a simple, valid path from entrance to exit.

        ### Visual legend
        - BLUE outlines = walls and structural barriers. Never cross them.
        - ORANGE boxes = restricted areas. Never enter them.
        - PURPLE boxes = gallery areas. Passing through this area is mandatory.
        - GREEN box = entrance. The route must start inside it.
        - YELLOW box = exit. The route must end inside it.
        - Numbered circles = exhibits (visit at least ONE of these, DO NOT collide with them).

        ### Non-negotiable constraints
        - Start inside the entrance box (GREEN), with the first point clearly inside rather than on the border.
        - End inside the exit box (YELLOW), with the last point clearly inside rather than on the border.
        - Enter every must-see gallery (PURPLE boxes).
        - Never enter restricted areas (ORANGE boxes).
        - Stay inside the valid museum floor area at all times.
        - Never cross walls (BLUE outlines).
        - Never draw through exhibit markers (NUMBERED circles) or obstacle geometry.
        - Rule priority is: legality first, then correct entrance and exit placement, then required gallery coverage, then selected exhibit coverage, then compactness.
        - Never violate a higher-priority rule to satisfy a lower-priority one.
        - The path must be continuous and physically plausible.

        ### Visit definition
        - A selected exhibit counts as visited when the path comes within {visit_distance_text} of that exhibit's numbered location.

        ### Planning strategy
        - First, silently identify all visible no-go areas: walls, restricted areas, exhibit markers, and dead-end risky spaces.
        - Second, build a safe corridor skeleton from entrance to exit that stays legal from start to finish.
        - Third, convert the final walk into a single continuous polyline.
        - If uncertain, choose the safer route instead of the shorter shortcut.
        - After drafting the route, trim any detour that does not help connect the legal start-to-exit walk.
        - Focus solely on creating a valid, direct path from entrance to exit.
        - The route should be simple and efficient without unnecessary detours.
        - Keep it simple and straightforward.

        ### Path construction rules
        - The path must be one continuous, physically plausible walking route with EXACTLY two endpoints.
        - The path must not branch, split, fork, or intersect at any point.
        - IF the route doubles back, the return path must not overlap the original path; it must be visibly offset so that two separate lines are clearly distinguishable, indicating a U-turn.
        - Use a multi-point polyline with enough waypoints to show the path clearly.
        - Consecutive points should trace a sensible walking path through open floor space.
        - Every straight segment between consecutive points must stay in legal open floor space.
        - If a straight segment would clip a wall, restricted area, or exhibit marker, add waypoints to go around.
        - Prefer orthogonal walking segments (horizontal and vertical) where practical.
        - Favor open corridors and wider spaces over risky shortcuts near hazards.
        - Maintain visible clearance from restricted-area borders and walls.

        ### Coordinate constraints
        - Image width: {img_width} pixels
        - Image height: {img_height} pixels
        - Every coordinate must be an integer pair [x, y].
        - Every coordinate must satisfy 0 <= x < {img_width} and 0 <= y < {img_height}.
        - Do not output floats.
        - Do not output tuples, objects, strings, or nested wrappers.

        ### Output format
        - Output ONLY a JSON array of coordinate pairs.
        - The first item must be the start point (inside GREEN entrance box).
        - The last item must be the exit point (inside YELLOW exit box).
        - Valid example: [[120, 410], [145, 410], [170, 405]]
        - Invalid examples: [120, 410], {{"route": [[120, 410]]}}, [[120.5, 410.2]], [[120, 410]]
        - No commentary, no markdown fences, no explanation.
        """
                
        user_prompt_route = f"""
        Create a simple, direct route from entrance (GREEN box) to exit (YELLOW box).

        Remember:
        - visit at least ONE exhibit as you see fit
        - gallery visits required
        - the route must be a continuous drawable JSON polyline,
        - the first point must be comfortably inside the entrance (GREEN),
        - the last point must be comfortably inside the exit (YELLOW),
        - avoid walls (BLUE outlines),
        - avoid restricted areas (ORANGE boxes),
        - avoid colliding with exhibit markers,
        - stay within the museum floor area,
        - keep the path simple, direct, and efficient.

        Before answering, verify:
        1. every segment avoids walls, restricted areas, and exhibits,
        2. the first point is inside the entrance,
        3. the last point is inside the exit,
        4. the path is simple and direct without unnecessary wandering.

        ### Output format
        - Output ONLY a JSON array of coordinate pairs.
        - The first item must be the start point (inside GREEN entrance box).
        - The last item must be the exit point (inside YELLOW exit box).
        - Valid example: [[120, 410], [145, 410], [170, 405]]
        - Invalid examples: [120, 410], {{"route": [[120, 410]]}}, [[120.5, 410.2]], [[120, 410]]
        - No commentary, no markdown fences, no explanation.

        JSON:
        """
        
        if pbar:
            pbar.set_description(f"[Spatial] {complexity}/{layout_name} - Route planning")
        logger.info("Running route planning...")

        with open(output_paths['dir'] / "prompt_route_planning.txt", "w", encoding="utf-8") as f:
            f.write("="*70 + "\n")
            f.write("SYSTEM PROMPT - ROUTE PLANNING\n")
            f.write("="*70 + "\n\n")
            f.write(system_prompt_route)
            f.write("\n\n" + "="*70 + "\n")
            f.write("USER PROMPT - ROUTE PLANNING\n")
            f.write("="*70 + "\n\n")
            f.write(user_prompt_route)

        text_output_route = prompt_model(BACKEND, system_prompt_route, user_prompt_route, image_base64)
        
        parse_route_and_save(input_paths['image'], output_paths['route_image'], text_output_route)
        logger.info(f"Route saved to: {output_paths['route_image']}")
        
        # -------- Route Validation --------
        if pbar:
            pbar.set_description(f"[Spatial] {complexity}/{layout_name} - Validation")
        logger.info("Running validation pipeline...")
        
        # Construct validation_pipeline.py command
        cmd = [
            "python", "validation_pipeline.py",
            "--route-image", str(output_paths['route_image']),
            "--original-image", str(input_paths['image']),
            "--annotations", str(input_paths['annotations']),
            "--config-file", str(input_paths['config']),
            "--exhibits-csv-file", str(input_paths['exhibits_csv']),
            "--extraction-output-image", str(output_paths['extracted_image']),
            "--validated-output-image", str(output_paths['validated_image']),
            "--validated-output-json", str(output_paths['validation_json']),
        ]
        
        # Run validation_pipeline.py with UTF-8 encoding for child process
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8', env=env)
        
        # Log validation stdout to file only (not console)
        if result.stdout:
            logger.info("Validation pipeline output:")
            for line in result.stdout.split('\n'):
                if line.strip():
                    logger.info(line)
        
        # Load and display validation results
        if output_paths['validation_json'].exists():
            with open(output_paths['validation_json'], "r", encoding="utf-8") as f:
                validation_data = json.load(f)
                
            summary = validation_data['validation_summary']
            
            # Extract and filter validation fields  
            svr = extract_validation_fields(summary["svr"], SVR_FIELDS)
            scsr = extract_validation_fields(summary["scsr"], SCSR_FIELDS)
            scar = extract_scar_validations(summary["scar"], configs, SCAR_FIELDS)
            
            data = {
                "svr": svr,
                "scsr": scsr,
                "scar": scar
            }
            
            with open(output_paths['final_json'], "w") as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Validation complete. Results saved to: {output_paths['final_json']}")
        
        logger.info(f"Successfully processed: {layout_folder.name}")
        logger.info(f"Log file saved to: {log_file}")
        
        # Write success to console (always print for master script capture)
        print(f"✓ {complexity}/{layout_name}: Success")
        
        return True
        
    except subprocess.CalledProcessError as e:
        error_msg = f"Error running validation for {layout_folder.name}"
        logger.error(error_msg)
        logger.error(f"Return code: {e.returncode}")
        if e.stdout:
            logger.error(f"STDOUT:\n{e.stdout}")
        if e.stderr:
            logger.error(f"STDERR:\n{e.stderr}")
        
        print(f"✗ {complexity}/{layout_name}: Validation failed")
        
        return False
    except Exception as e:
        error_msg = f"Error processing {layout_folder.name}: {e}"
        logger.error(error_msg)
        import traceback
        logger.error(traceback.format_exc())
        
        print(f"✗ {complexity}/{layout_name}: {str(e)[:50]}")
        
        return False


def main():
    """Main execution function."""
    print("="*70)
    print("MUSEUM ROUTE PLANNING - BATCH PROCESSOR")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Backend:    {args.backend}")
    print(f"  Model: {MODEL}")
    print(f"  Difficulty: {DIFFICULTY}")
    print(f"  Complexity Mode: {COMPLEXITY_MODE}")
    if COMPLEXITY_MODE != "all":
        print(f"    Specific: {SPECIFIC_COMPLEXITIES}")
    print(f"  Layout Mode: {LAYOUT_MODE}")
    if LAYOUT_MODE != "all":
        print(f"    Specific: {SPECIFIC_LAYOUTS}")
    print(f"  Base directory: {BASE_FLOORPLAN_DIR}")
    print(f"  Config file: {CONFIG_FILENAME}")
    print(f"  Output directory: {BASE_RESULTS_DIR / DIFFICULTY}")
    print("="*70)
    
    # Discover complexity levels and layouts based on configuration
    complexity_layouts = discover_complexity_and_layouts(
        BASE_FLOORPLAN_DIR,
        COMPLEXITY_MODE, SPECIFIC_COMPLEXITIES,
        LAYOUT_MODE, SPECIFIC_LAYOUTS
    )
    
    if not complexity_layouts:
        print(f"\n✗ No layouts found to process")
        return
    
    print(f"\n✓ Found {len(complexity_layouts)} layout(s) to process:")
    for complexity, layout_path in complexity_layouts:
        print(f"  - {complexity}/{layout_path.name}")
    
    print()  # Empty line before progress bar
    
    # Process each layout with progress bar
    results = {}
    
    # Create progress bar (disable if --no-progress flag is set)
    disable_progress = args.no_progress
    with tqdm(complexity_layouts, desc="[Spatial] Processing layouts", 
              unit="layout", disable=disable_progress, position=args.progress_position,
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
              file=sys.stderr, dynamic_ncols=True, leave=False, mininterval=0.5) as pbar:
        
        for complexity, layout_folder in pbar:
            key = f"{complexity}/{layout_folder.name}"
            success = process_layout(layout_folder, MODEL, DIFFICULTY, complexity, pbar=pbar)
            results[key] = "Success" if success else "Failed"
    
    # Summary
    print("\n" + "="*70)
    print("PROCESSING SUMMARY")
    print("="*70)
    successful = sum(1 for status in results.values() if status == "Success")
    print(f"Total: {len(results)} layouts")
    print(f"Successful: {successful}")
    print(f"Failed: {len(results) - successful}")
    print("\nDetails:")
    for key, status in results.items():
        status_symbol = "✓" if status == "Success" else "✗"
        print(f"  {status_symbol} {key}: {status}")
    print("="*70)
    print("\nPipeline complete!")
    
    # Exit with appropriate code
    failed_count = len(results) - successful
    sys.exit(0 if failed_count == 0 else 1)


if __name__ == "__main__":
    main()
