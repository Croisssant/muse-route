
import json
import subprocess
import os
import sys
import io
from dotenv import load_dotenv
from openai import OpenAI

from pathlib import Path

from build_prompts import build_required_categories, build_required_OR_attributes, build_required_AND_attributes, build_visit_distance
from prompts_utils import (load_image, exhibit_selection, prompt_gpt, 
                           parse_route_and_save, 
                           discover_complexity_and_layouts, build_paths, 
                           extract_scar_validations, extract_validation_fields)

# -------- Force UTF-8 encoding for stdout on Windows --------
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# -------- Load API key --------
load_dotenv(override=True)
client = OpenAI()

# ========== USER CONFIGURATION ==========

# Metric fields to include in final_results.json
SVR_FIELDS = ['connectivity', 'wall_crossings', 'exhibit_collision', 'out_of_area_violations']
SCSR_FIELDS = ['start_end_location']
SCAR_FIELDS = ['exhibit_category_coverage', 'attribute_validations']

# Model configuration
MODEL = "gpt-5.4"
REASONING = None

# Task configuration
DIFFICULTY = "easy_semantic"      # Used for output folder structure

# Complexity selection mode
# Options:
#   - "all": Process all complexity levels (simple, complex, medium)
#   - "single": Process one specific complexity level
#   - "list": Process multiple specific complexity levels
COMPLEXITY_MODE = "single"
SPECIFIC_COMPLEXITIES = "simple"  # For single mode
# SPECIFIC_COMPLEXITIES = ["simple", "complex"]  # For list mode

# Layout selection mode
# Options:
#   - "all": Process all layout_* folders in selected complexity levels
#   - "single": Process one specific layout
#   - "list": Process multiple specific layouts
LAYOUT_MODE = "all"
SPECIFIC_LAYOUTS = None  # For single mode: "layout_01"
# SPECIFIC_LAYOUTS = ["layout_01", "layout_03", "layout_05"]  # For list mode

# Directory paths
BASE_FLOORPLAN_DIR = Path("floorplans")
BASE_RESULTS_DIR = Path("results")

# Default filenames (modify if your files have different names)
IMAGE_FILENAME = "annotated_layout.png"
CONFIG_FILENAME = "easy_semantic.json"      # Config file to load from each layout
EXHIBITS_FILENAME = "exhibit_list.json"
EXHIBITS_CSV_FILENAME = "exhibits.csv"      # CSV file for validation
ANNOTATIONS_FILENAME = "layout_annotations.json"

# Filename mapping for build_paths function
FILENAMES = {
    'image': IMAGE_FILENAME,
    'config': CONFIG_FILENAME,
    'exhibits': EXHIBITS_FILENAME,
    'exhibits_csv': EXHIBITS_CSV_FILENAME,
    'annotations': ANNOTATIONS_FILENAME
}

# ========================================


def process_layout(layout_folder, model_name, difficulty, complexity):
    """Process a single layout folder."""
    print(f"\n{'='*70}")
    print(f"Processing: {layout_folder.name}")
    print(f"{'='*70}\n")
    
    # Build paths using the utility function
    input_paths, output_paths = build_paths(
        layout_folder, model_name, difficulty, complexity,
        BASE_RESULTS_DIR, FILENAMES
    )
    
    # Validate input files exist
    missing_files = []
    for name, path in input_paths.items():
        if not path.exists():
            missing_files.append(f"{name}: {path}")
    
    if missing_files:
        print(f"ERROR: Missing required files for {layout_folder.name}:")
        for missing in missing_files:
            print(f"  - {missing}")
        return False
    
    try:
        # -------- Load data --------
        with open(input_paths['exhibits'], "r", encoding="utf-8") as f:
            exhibits_list = json.load(f)
        
        with open(input_paths['config'], "r", encoding="utf-8") as f:
            configs = json.load(f)
        
        with open(input_paths['annotations'], "r", encoding="utf-8") as f:
            annotations_json = json.load(f)
        
        # -------- Build prompts --------
        required_category_requirement, required_category_checklist, required_category_prompt_line = build_required_categories(configs.get("exhibit_categories_to_cover", None))
        required_OR_attribute_requirement, required_OR_attribute_checklist, required_OR_attribute_prompt_line = build_required_OR_attributes(configs.get("exhibit_attribute_constraints", None))
        required_AND_attribute_requirement, required_AND_attribute_checklist, required_AND_attribute_prompt_line = build_required_AND_attributes(configs.get("exhibit_attribute_constraints", None))
        
        visit_distance_text = build_visit_distance(configs.get("exhibit_see_distance_in_mm"), annotations_json.get("distance_in_mm", []))
        
        # -------- Image info --------
        image_base64, img_width, img_height = load_image(input_paths['image'])
        
        # -------- Exhibit Selection --------
        system_prompt_selection = f"""
        You are a museum assistant AI selecting exhibits for a route-planning benchmark.

        Here is the full exhibit list in JSON:
        {exhibits_list}

        Hard benchmark requirements that override user preference when there is any conflict:
        {required_category_requirement}
        {required_OR_attribute_requirement}
        {required_AND_attribute_requirement}

        Selection process:
        - Strongly prefer exhibits that match the user preference.
        - When several exhibits satisfy the preference equally well, prefer the subset that forms a compact visit plan with less backtracking and fewer long jumps.
        - Avoid redundant choices that spread the route across distant regions when a more compact preference-matching option exists.
        - If there is uncertainty, prefer a conservative set that is easier to route legally and compactly rather than a sprawling set.
        - If the user preference conflicts with the hard benchmark requirements, satisfy the hard benchmark requirements first and then maximize preference match.
        - Order the final exhibit numbers in a sensible visiting sequence for a compact walk from entrance to exit.
        - The order should move smoothly through nearby regions instead of jumping back and forth between distant parts of the museum.
        - Prefer an order that reduces backtracking and reduces the need to cross the same corridor multiple times.
        - Prefer optional exhibits that can be covered by one or two compact clusters rather than optional exhibits scattered across many distant regions.

        Validation checklist before responding:
        {required_category_checklist}
        {required_OR_attribute_checklist}
        {required_AND_attribute_checklist}
        - The order should represent a plausible visit order, not a random order.
        - If any required exhibit is missing, replace optional exhibits until all required exhibits are present before responding.

        Output rules:
        1. Output ONLY a JSON array of exhibit numbers.
        2. Do not output markdown, labels, commentary, or any other text.
        3. The response must be valid JSON, for example: [1, 5, 9]
        """
        
        user_prompt_selection = f"""
        Follow the validation checklist before you answer.
        The JSON array order should be the recommended visiting order.
        Avoid orders that bounce between distant exhibit groups.
        Prefer optional exhibits that let the route stay compact instead of visiting many separate clusters.
        {required_category_prompt_line}
        {required_OR_attribute_prompt_line}
        {required_AND_attribute_prompt_line}
        The JSON array should remain valid even if the config values change in a future run.
        """
        
        print("Running exhibit selection...")

        with open(output_paths['dir'] / "prompt_exhibit_selection.txt", "w", encoding="utf-8") as f:
            f.write("="*70 + "\n")
            f.write("SYSTEM PROMPT - ROUTE PLANNING\n")
            f.write("="*70 + "\n\n")
            f.write(system_prompt_selection)
            f.write("\n\n" + "="*70 + "\n")
            f.write("USER PROMPT - ROUTE PLANNING\n")
            f.write("="*70 + "\n\n")
            f.write(user_prompt_selection)

        text_output_selection = prompt_gpt(client, model_name, system_prompt_selection, user_prompt_selection, image_base64, REASONING)
        gpt_selected_exhibits = exhibit_selection(text_output_selection, exhibits_list)
        # print(f"Selected exhibits: {gpt_selected_exhibits}")
        
        # -------- Route Planning --------
        system_prompt_route = f"""
        You are a museum path planning assistant generating a single drawable polyline on top of the museum image.

        Your ONLY goal is to create a simple, valid path from entrance to exit.

        ### Visual legend
        - BLUE outlines = walls and structural barriers. Never cross them.
        - ORANGE boxes = restricted areas. Never enter them.
        - PURPLE boxes = gallery areas. Passing through them is optional.
        - GREEN box = entrance. The route must start inside it.
        - YELLOW box = exit. The route must end inside it.
        - Numbered circles = exhibits (DO NOT collide with them).

        ### Non-negotiable constraints
        - The path must be continuous and physically plausible.
        - Never cross walls (BLUE outlines).
        - Never draw through exhibit markers (NUMBERED circles) or obstacle geometry.
        - Stay inside the valid museum floor area at all times.
        - Start inside the entrance box (GREEN), with the first point clearly inside rather than on the border.
        - End inside the exit box (YELLOW), with the last point clearly inside rather than on the border.
        - Never enter restricted areas (ORANGE boxes).

        ### Visit definition
        - A selected exhibit counts as visited when the path comes within {visit_distance_text} of that exhibit's numbered location.
        - Missing even one selected exhibit is a failure.
        - If a selected exhibit is near a restricted area or obstacle cluster, satisfy the visit from the nearest legal open-floor location instead of entering the risky area.

        ### Planning strategy
        - First, silently identify all visible no-go areas: walls, restricted areas, restricted galleries, exhibit markers, and dead-end risky spaces.
        - Second, build a safe corridor skeleton from entrance to exit that stays legal from start to finish.
        - Third, use the selected exhibit list as the preferred visit order, but reorder when needed to preserve legality and reduce backtracking.
        - Fourth, attach short, legal detours from that skeleton to cover the selected exhibits.
        - Fifth, convert the final walk into a single continuous polyline.
        - If uncertain, choose the safer route instead of the shorter shortcut.
        - After drafting the route, trim any detour that does not help cover a selected exhibit, reach a required gallery, or connect the legal start-to-exit walk.

        ### Path construction rules
        - The path must be one continuous, physically plausible walking route with exactly two endpoints.
        - The path must not branch, split, fork, or intersect at any point.
        - IF the route doubles back, the return path must not overlap the original path; it must be visibly offset so that two separate lines are clearly distinguishable, indicating a U-turn.
        - Use a multi-point polyline with many waypoints, not a single point and not just 2 points.
        - Consecutive points should trace a sensible walking path through open floor space.
        - Every straight segment between consecutive points must be directly drawable through legal open floor. If a straight segment would clip a wall, restricted area, restricted gallery, or exhibit marker, add another waypoint instead of cutting through.
        - Keep the route compact: no loops, no retracing, no sightseeing detours, and no long perimeter sweeps.
        - Prefer orthogonal walking segments where practical, using diagonals only for short local adjustments in open floor space.
        - Favor open corridors and wider spaces over risky shortcuts near hazards.
        - Maintain visible clearance from restricted-region borders and exhibit markers rather than skimming right along them.
        - When satisfying a must-see gallery requirement, make the visit as shallow as possible: enter legally, cover the requirement, and exit without crossing into neighboring risky interiors.
        - After the route reaches the exit, stop immediately. Do not overshoot the exit or hook around it.
        - Avoid accidentally passing near large numbers of unselected exhibits. If many unselected exhibits would also be covered, the route is probably too broad and should be tightened.

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
        Create a simple, direct route from entrance (GREEN box) to exit (YELLOW box) for this exact selected exhibit set:
        {gpt_selected_exhibits}

        Remember:
        - the route must be a continuous drawable JSON polyline,
        - the first point must be inside the entrance,
        - the last point must be inside the exit,
        - keep the first and last points comfortably inside those boxes, not on their borders,
        - the path must include enough waypoints to show the full walk.
        - treat the selected exhibit list as the preferred visiting order, but reorder when needed to stay legal,
        - Keep the route short and deliberate.
        - Avoid sweeping through large parts of the museum just to pass near extra exhibits.
        - Favor a corridor-like Manhattan path made of horizontal and vertical steps.
        - make the first and last coordinates visibly centered inside the green and yellow boxes rather than merely barely inside,
        - if a must-see gallery is close to restricted space, touch the legal portion you need and then leave immediately rather than traversing deeply through nearby gallery interiors,
        - trim any waypoint that does not help legality, selected-exhibit coverage, must-see gallery coverage, or direct progress from entrance to exit,
        - Before answering, silently verify that:
        1. every segment is legal and does not cut through a wall, restricted area, restricted gallery, or exhibit marker,
        2. all selected exhibits are covered from legal open floor,
        3. every must-see gallery is entered,
        4. the first point is inside the entrance,
        5. the last point is inside the exit,
        6. the route is not unnecessarily passing near many unselected exhibits.
        - if an exhibit is near a restricted area, cover it from the nearest legal open-floor position.
        """
        
        print("Running route planning...")

        with open(output_paths['dir'] / "prompt_route_planning.txt", "w", encoding="utf-8") as f:
            f.write("="*70 + "\n")
            f.write("SYSTEM PROMPT - ROUTE PLANNING\n")
            f.write("="*70 + "\n\n")
            f.write(system_prompt_route)
            f.write("\n\n" + "="*70 + "\n")
            f.write("USER PROMPT - ROUTE PLANNING\n")
            f.write("="*70 + "\n\n")
            f.write(user_prompt_route)

        text_output_route = prompt_gpt(client, model_name, system_prompt_route, user_prompt_route, image_base64, REASONING)
        
        parse_route_and_save(input_paths['image'], output_paths['route_image'], text_output_route)
        print(f"Route saved to: {output_paths['route_image']}")
        
        # -------- Route Validation --------
        print("\nRunning validation pipeline...")
        
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
        print(result.stdout)
        
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
            
            print(f"\n✓ Validation complete. Results saved to: {output_paths['final_json']}")
        
        print(f"\n✓ Successfully processed: {layout_folder.name}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error running validation for {layout_folder.name}:")
        print(f"Return code: {e.returncode}")
        if e.stdout:
            print(f"STDOUT:\n{e.stdout}")
        if e.stderr:
            print(f"STDERR:\n{e.stderr}")
        return False
    except Exception as e:
        print(f"\n✗ Error processing {layout_folder.name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main execution function."""
    print("="*70)
    print("MUSEUM ROUTE PLANNING - BATCH PROCESSOR")
    print("="*70)
    print(f"\nConfiguration:")
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
    print(f"  Output directory: {BASE_RESULTS_DIR / MODEL / DIFFICULTY}")
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
    
    # Process each layout
    results = {}
    for i, (complexity, layout_folder) in enumerate(complexity_layouts, 1):
        key = f"{complexity}/{layout_folder.name}"
        print(f"\n[{i}/{len(complexity_layouts)}] " + "="*60)
        success = process_layout(layout_folder, MODEL, DIFFICULTY, complexity)
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


if __name__ == "__main__":
    main()
