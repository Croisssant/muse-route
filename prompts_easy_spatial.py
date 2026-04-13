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
from pathlib import Path

# -------- Force UTF-8 encoding for stdout on Windows --------
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# -------- Load API key --------
load_dotenv(override=True)
client = OpenAI()

# -------- File paths --------
input_image_path = "./images/annotated_walls_museum_layout_02/original_images/annotated_walls_museum_layout_02.png"
output_directory = Path("./results/easy/chatgpt5.4/layout_02/")
output_image_path = output_directory / "chatgpt5.4_route.png"

# -------- Image info --------
image = Image.open(input_image_path)
img_width, img_height = image.size

# Read image as base64
with open(input_image_path, "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode("utf-8")



# -------- STEP 1: Route Planning --------
system_prompt_route = f"""
You are a museum path planning assistant generating a single drawable polyline on top of the museum image.

Your ONLY goal is to create a simple, valid path from entrance to exit.

### Visual legend
- BLUE outlines = walls and structural barriers. Never cross them.
- ORANGE boxes = restricted areas. Never enter them.
- PURPLE boxes = gallery areas. Passing through this area is mandatory,
- GREEN box = entrance. The route must start inside it.
- YELLOW box = exit. The route must end inside it.
- Numbered circles = exhibits (visit at least ONE of these, DO NOT collide with them).

### Non-negotiable constraints
- The path must be continuous and physically plausible.
- Never cross walls (BLUE outlines).
- Never draw through exhibit markers (NUMBERED circles) or obstacle geometry.
- Stay inside the valid museum floor area at all times.
- Start inside the entrance box (GREEN), with the first point clearly inside rather than on the border.
- End inside the exit box (YELLOW), with the last point clearly inside rather than on the border.
- Never enter restricted areas (ORANGE boxes).

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
- The path must be one continuous, physically plausible walking route.
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
"""

response_route = client.responses.create(
    model="gpt-5.4",
    reasoning={"effort": "high"},
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

# -------- STEP 2: Route Validation --------
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

        print("\n" + "=" * 70)
        print("VALIDATION SUMMARY")
        print("=" * 70)
        print(json.dumps(validation_data['validation_summary'], indent=2))

        # Print key metrics
        summary = validation_data['validation_summary']
        print("\n" + "=" * 70)
        print("KEY METRICS (EASY LEVEL)")
        print("=" * 70)
        if 'svr' in summary:
            print(f"Connectivity: {summary['svr'].get('connectivity', 'N/A')}")
            print(f"No Wall Crossings: {not summary['svr'].get('wall_crossings', True)}")
            print(f"No Exhibit Collisions: {not summary['svr'].get('exhibit_collision', True)}")
            print(f"Within Floor Area: {not summary['svr'].get('out_of_area_violations', True)}")

        if 'scsr' in summary:
            print(f"Start/End Correct: {summary['scsr'].get('start_end_location', 'N/A')}")
            print(f"No Restricted Area Violations: {not summary['scsr'].get('restricted_area_violations', True)}")

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
