import base64
import json
import re
import subprocess
import os
import sys
import io
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image, ImageDraw
from build_prompts import build_system_prompt, build_user_prompt, build_selection_prompt

# -------- Force UTF-8 encoding for stdout on Windows --------
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# -------- Load API key --------
load_dotenv(override=True)
client = OpenAI()

# -------- File paths --------
input_image_path = "./images/annotated_walls_museum_layout_02/original_images/annotated_walls_museum_layout_02.png"
output_image_path = "./images/annotated_walls_museum_layout_02/route_images/chatgpt_route_2.png"
exhibits_json_path = "./original_floorplans/museum_layout_01/exhibit_list.json"

# -------- Load exhibit JSON --------
with open(exhibits_json_path, "r", encoding="utf-8") as f:
    exhibits_json = json.load(f)

# -------- Image info --------
image = Image.open(input_image_path)
img_width, img_height = image.size

# Read image as base64
with open(input_image_path, "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode("utf-8")

# -------- STEP 1: Exhibit Selection --------
user_preference = "I only want to visit Roman Exhibits"

# Build prompt for selecting exhibits
system_prompt_selection = f"""
    You are a museum assistant AI selecting exhibits for a route-planning benchmark.

    Here is the full exhibit list in JSON:
    {exhibits_json}

    Hard benchmark requirements that override user preference when there is any conflict:
    1. The final route must include exhibit numbers [1, 96, 97, 98, 99, 100].
    2. The final route must cover at least 15 exhibits total.
    3. The final route must include at least one Roman exhibit.

    Selection process:
    1. Start by locking in every hard-required exhibit [1, 96, 97, 98, 99, 100]. These six exhibits are mandatory and cannot be removed.
    2. Add additional exhibits until there are exactly 15 unique exhibit numbers.
    3. Strongly prefer exhibits that match the user preference.
    4. When several exhibits satisfy the preference equally well, prefer the subset that forms a compact visit plan with less backtracking and fewer long jumps.
    5. Avoid redundant choices that spread the route across distant regions when a more compact preference-matching option exists.
    6. If there is uncertainty, prefer a conservative set that is easier to route legally and compactly rather than a sprawling set.
    7. If the user preference conflicts with the hard benchmark requirements, satisfy the hard benchmark requirements first and then maximize preference match.
    8. Order the final exhibit numbers in a sensible visiting sequence for a compact walk from entrance to exit.
    9. The order should move smoothly through nearby regions instead of jumping back and forth between distant parts of the museum.
    10. Prefer an order that reduces backtracking and reduces the need to cross the same corridor multiple times.
    11. Prefer optional exhibits that can be covered by one or two compact clusters rather than optional exhibits scattered across many distant regions.

    Validation checklist before responding:
    1. [1, 96, 97, 98, 99, 100] are all present.
    2. At least one Roman exhibit is present.
    3. The list contains exactly 15 unique integers.
    4. The order should represent a plausible visit order, not a random order.
    5. If any required exhibit is missing, replace optional exhibits until all required exhibits are present before responding.

    Output rules:
    1. Output ONLY a JSON array of exhibit numbers.
    2. Do not output markdown, labels, commentary, or any other text.
    3. The response must be valid JSON, for example: [1, 5, 9]
    """
user_prompt_selection = f"""
User preference: {user_preference}

Return exactly 15 exhibit numbers that satisfy the benchmark requirements above.
Follow the validation checklist before you answer.
The JSON array order should be the recommended visiting order.
Avoid orders that bounce between distant exhibit groups.
Do not submit any answer that omits one of [1, 96, 97, 98, 99, 100].
Prefer optional exhibits that let the route stay compact instead of visiting many separate clusters.
"""

response_selection = client.responses.create(
    model="gpt-5.4",
    input=[
        {"role": "system", "content": system_prompt_selection},
        {"role": "user", "content": user_prompt_selection}
    ]
)

# Parse selected exhibit IDs
text_output_selection = response_selection.output_text.strip()
try:
    selected_ids = json.loads(text_output_selection)
except json.JSONDecodeError:
    # Fallback extraction
    match = re.search(r"\[\s*\d+(?:\s*,\s*\d+)*\s*\]", text_output_selection)
    if match:
        selected_ids = json.loads(match.group())
    else:
        raise ValueError(f"Failed to parse exhibit IDs:\n{text_output_selection}")

print("Selected exhibit IDs:", selected_ids)

# Filter exhibits
filtered_exhibits = [e for e in exhibits_json if e["exhibit_number"] in selected_ids]

# -------- STEP 2: Route Planning --------
system_prompt_route = f"""
        You are a museum path planning assistant generating a single drawable polyline on top of the museum image.

        Produce a route that satisfies the benchmark exactly. Treat the following as HARD requirements.

        ### Visual legend
        - BLUE outlines = walls and structural barriers. Never cross them.
        - ORANGE boxes = restricted areas. Never enter them.
        - PURPLE boxes = gallery areas. Some may be must-see or restricted.
        - GREEN box = entrance. The route must start inside it.
        - YELLOW box = exit. The route must end inside it.
        - Numbered circles = exhibits.

        ### Non-negotiable constraints
        - Start inside the entrance box, with the first point clearly inside rather than on the border.
        - End inside the exit box, with the last point clearly inside rather than on the border.
        - Enter every must-see gallery or region.
        - Never enter any restricted gallery or restricted region.
        - Stay inside the valid museum floor area.
        - Never cross walls.
        - Never draw through exhibit markers or obstacle geometry.
        - Visit all selected exhibits and ignore unselected exhibits.
        - Rule priority is: legality first, then correct entrance and exit placement, then required gallery coverage, then selected exhibit coverage, then compactness.
        - Never violate a higher-priority rule to satisfy a lower-priority one.

        ### Visit definition
        - A selected exhibit counts as visited when the path comes within 183 pixels of that exhibit's numbered location.
        - Missing even one selected exhibit is a failure.
        - Missing a required gallery or region is a failure.
        - If a selected exhibit is near a restricted area or obstacle cluster, satisfy the visit from the nearest legal open-floor location instead of entering the risky area.
        - If a selected exhibit would require entering restricted space or crossing a barrier, approach only as closely as the nearest legal open-floor position allows.
        - A required gallery visit only needs legal entry into that gallery. Once the route has legally entered the required gallery, leave it again by the nearest legal continuation instead of wandering through adjacent interiors.

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
        - Return between 20 and 32 coordinate pairs.
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
        - The first item must be the start point and the last item must be the exit point.
        - Valid example: [[120, 410], [145, 410], [170, 405]]
        - Invalid examples: [120, 410], {{"route": [[120, 410]]}}, [[120.5, 410.2]], [[120, 410]]
        - No commentary, no markdown fences, no explanation.
    """
user_prompt_route = f"""
Plan a valid route for this exact selected exhibit set:
{filtered_exhibits}

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

response_route = client.responses.create(
    model="gpt-5.4",
    input=[
        {"role": "system", "content": system_prompt_route},
        {"role": "user", "content": [
            {"type": "input_text", "text": user_prompt_route},
            {"type": "input_image", "image_url": f"data:image/png;base64,{image_base64}"}
        ]}
    ]
)

# -------- Parse route --------
text_output_route = response_route.output_text.strip()
text_output_route = re.sub(r"^```json\s*|\s*```$", "", text_output_route, flags=re.MULTILINE)

try:
    route = json.loads(text_output_route)
except json.JSONDecodeError:
    match = re.search(r"\[\s*\[.*?\]\s*\]", text_output_route, re.DOTALL)
    if match:
        route = json.loads(match.group())
    else:
        raise ValueError(f"Failed to parse JSON route:\n{text_output_route}")

if len(route) < 2:
    raise ValueError(f"Route must contain at least 2 coordinate pairs:\n{text_output_route}")

route = [tuple(point) for point in route]

# -------- Validate route coordinates --------
invalid_points = []
for i, (x, y) in enumerate(route):
    if not (0 <= x < img_width and 0 <= y < img_height):
        invalid_points.append(f"Point {i}: ({x}, {y})")

if invalid_points:
    print("WARNING: Found out-of-bounds coordinates:")
    for point in invalid_points:
        print(f"  {point}")
    print(f"  Valid range: 0 <= x < {img_width}, 0 <= y < {img_height}")

# -------- Draw route --------
draw = ImageDraw.Draw(image)
draw.line(route, fill="red", width=5)


# -------- Save output image --------
image.save(output_image_path)
print(f"Route drawn and saved to {output_image_path}")

# -------- STEP 3: Route Validation --------
print("\n" + "=" * 70)
print("STEP 3: Running Validation Pipeline (main.py)")
print("=" * 70 + "\n")

# Extract filenames for main.py
output_filename = os.path.basename(output_image_path)
original_filename = os.path.basename(input_image_path)

# Determine the images directory (parent of route_images and original_images)
images_base_dir = os.path.dirname(os.path.dirname(output_image_path))

# Construct main.py command
cmd = [
    "python", "main.py",
    "--route-image", output_filename,
    "--original-image", original_filename,
    "--annotations", "museum_layout_annotations.json",
    "--images-dir", images_base_dir,
    "--extraction-output-image", f"extracted_{output_filename}",
    "--validated-output-image", f"validated_{output_filename}"
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

        print("\n" + "=" * 70)
        print("VALIDATION SUMMARY")
        print("=" * 70)
        print(json.dumps(validation_data['validation_summary'], indent=2))

        # Print key metrics
        summary = validation_data['validation_summary']
        print("\n" + "=" * 70)
        print("KEY METRICS")
        print("=" * 70)
        if 'svr' in summary:
            print(f"Connectivity: {summary['svr'].get('connectivity', 'N/A')}")
            print(f"No Wall Crossings: {not summary['svr'].get('wall_crossings', True)}")
            print(f"No Exhibit Collisions: {not summary['svr'].get('exhibit_collision', True)}")
            print(f"Within Floor Area: {not summary['svr'].get('out_of_area_violations', True)}")

        if 'scsr' in summary:
            print(f"Start/End Correct: {summary['scsr'].get('start_end_location', 'N/A')}")
            print(f"Must-Pass Regions: {summary['scsr'].get('must_pass_regions', 'N/A')}")
            print(f"No Restricted Area Violations: {not summary['scsr'].get('restricted_area_violations', True)}")
            print(f"Within Distance Budget: {summary['scsr'].get('distance_budget', 'N/A')}")

        print("=" * 70)

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
