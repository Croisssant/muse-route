
import json
import re
import subprocess
import os
import sys
import io
from dotenv import load_dotenv
from openai import OpenAI

from pathlib import Path

from build_prompts import build_required_categories, build_required_OR_attributes, build_required_AND_attributes, build_visit_distance
from utils import load_image, exhibit_selection_gpt, route_planning_gpt, parse_route_and_save, select_fields

# -------- Force UTF-8 encoding for stdout on Windows --------
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# -------- Load API key --------
load_dotenv(override=True)
client = OpenAI()
model = "gpt-5.4"

# -------- File paths --------
input_directory = Path("")
config_path = "./config.json"
input_image_path = "./images/annotated_walls_museum_layout_02/original_images/annotated_walls_museum_layout_02.png"
exhibits_json_path = "./original_floorplans/museum_layout_01/exhibit_list.json"
annotations_json_path = "./original_floorplans/museum_layout_01/museum_layout_annotations.json"

output_directory = Path("./results/easy/chatgpt5.4/layout_02/")
output_image_path = output_directory / "chatgpt5.4_route.png"

# -------- Load exhibit JSON --------
with open(exhibits_json_path, "r", encoding="utf-8") as f:
    exhibits_list = json.load(f)

with open(config_path, "r", encoding="utf-8") as f:
    configs = json.load(f)

with open(annotations_json_path, "r", encoding="utf-8") as f:
    annotations_json = json.load(f)



required_category_requirement, required_category_checklist, required_category_prompt_line = build_required_categories(configs.get("exhibit_categories_to_cover", None))
required_OR_attribute_requirement, required_OR_attribute_checklist, required_OR_attribute_prompt_line = build_required_OR_attributes(configs.get("exhibit_attribute_constraints", None))
required_AND_attribute_requirement, required_AND_attribute_checklist, required_AND_attribute_prompt_line = build_required_AND_attributes(configs.get("exhibit_attribute_constraints", None))


visit_distance_text = build_visit_distance(configs.get("exhibit_see_distance_in_mm"), annotations_json.get("distance_in_mm", []))

# -------- Image info --------
image_base64, img_width, img_height = load_image(input_image_path)

# -------- Exhibit Selection --------

# Build prompt for selecting exhibits
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

gpt_selected_exhibits = exhibit_selection_gpt(client, model, system_prompt_selection, user_prompt_selection, exhibits_list, image_base64)


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

  ### Planning strategy
- First, silently identify all visible no-go areas: walls, restricted areas, restricted galleries, exhibit markers, and dead-end risky spaces.
- Second, build a safe corridor skeleton from entrance to exit that stays legal from start to finish.
- Third, use the selected exhibit list as the preferred visit order, but reorder when needed to preserve legality and reduce backtracking.
- Fourth, attach short, legal detours from that skeleton to cover the selected exhibits.
- Fifth, convert the final walk into a single continuous polyline.
- If uncertain, choose the safer route instead of the shorter shortcut.
- After drafting the route, trim any detour that does not help cover a selected exhibit, reach a required gallery, or connect the legal start-to-exit walk.

 ### Path construction rules
- The path must be one continuous, physically plausible walking route.
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

text_output_route = route_planning_gpt(client, model, system_prompt_route, user_prompt_route, image_base64, reasoning_effort="medium")

parse_route_and_save(input_image_path, output_image_path, text_output_route)


# -------- Route Validation --------
output_filename = os.path.basename(output_image_path)

# Construct main.py command
cmd = [
    "python", "main.py",
    "--route-image", str(output_image_path),
    "--original-image", input_image_path,
    "--annotations", "museum_layout_annotations.json",
    "--extraction-output-image", str(output_directory / f"extracted_{output_filename}"),
    "--validated-output-image", str(output_directory / f"validated_{output_filename}"),
    "--validated-output-json", str(output_directory / "validation_results.json"),
]

print(f"Running: {' '.join(cmd)}\n")

try:
    # Run main.py validation with UTF-8 encoding for child process
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    result = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8', env=env)
    print(result.stdout)

    # Load and display validation results
    if os.path.exists("validation_results.json"):
        with open("validation_results.json", "r", encoding="utf-8") as f:
            validation_data = json.load(f)

        summary = validation_data['validation_summary']
        svr = summary["svr"]
        scsr = summary["scsr"]
        scar = select_fields(summary["scar"], ["start_end_location"])

        data = {
            "svr": svr,
            "scsr": scsr,
            "scar": scar
        }

    with open(str(output_directory / "final_results.json"), "w") as f:
        json.dump(data, f)
      

except subprocess.CalledProcessError as e:
    print("Error running main.py validation:")
    print(f"Return code: {e.returncode}")
    if e.stdout:
        print(f"STDOUT:\n{e.stdout}")
    if e.stderr:
        print(f"STDERR:\n{e.stderr}")
except Exception as e:
    print(f"Unexpected error during validation: {e}")

print("\nPipeline complete!")
