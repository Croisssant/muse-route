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
    You are a museum assistant AI. Your task is to select which exhibits a visitor should see based on their preferences.

        Here is the list of all exhibits in JSON format:
        { exhibits_json }

        Rules:
        1. Only select exhibits that match the user preference.
        2. Output ONLY a JSON array of exhibit_number.
        3. Do not include any text, commentary, or formatting outside the JSON array.
        4. The array should be a valid JSON list of integers, for example: [2, 5, 7]

        Provide your response strictly in the format above.
    """
user_prompt_selection = f"""
Select exhibits based on this user preference: {user_preference}
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
        You are a museum path planning assistant.

        Your task is to generate an optimal navigation path through a museum layout image.

        You MUST strictly follow all constraints.

        ---

        ### Spatial Constraints (MANDATORY)
        - Do NOT pass through walls or restricted areas.
        - Do NOT exit the museum boundaries except at the designated exit.
        - Do NOT collide with exhibits (treat exhibits as obstacles unless visiting).
        - Path must be continuous and physically plausible.
        - Start inside the entrance (green bounding box).
        - End inside the exit (yellow bounding box).

        Key Notes:
        - BLUE outlines repesent the walls.
        - ORANGE bounding box represents restricted area.
        - PURPLE boundig box represents gallary area.

        ---

        You MUST:
        - Ignore all irrelevant exhibits.
        - Visit only selected exhibits.

        ---

        ### Visit Definition
        - A visit is valid if the path comes within { 183 } pixels of the exhibit's approx_location.

        ---

        ### Path Rules
        - Optimize for shortest valid path.
        - Avoid unnecessary detours.
        - Visit exhibits in an efficient order.
        - Strictly follow Spatial Constraints

        ---

        ### Coordinate Constraints (CRITICAL)
        - Image width: { img_width } pixels
        - Image height: { img_height } pixels
        - ALL coordinates MUST satisfy:
        - 0 ≤ x < { img_width }
        - 0 ≤ y < { img_height }

        ### Output Format (STRICT)
        - Output ONLY a JSON array of (x, y) coordinates representing the path.
        - Coordinates must be integers in pixel space.
        - No explanations, no comments, no extra text.

        Example:
        [[10, 20], [30, 40], [50, 80]]
    """
user_prompt_route = f"""
    Plan a path visiting ONLY these exhibits: {filtered_exhibits}"
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

# -------- Validate route coordinates --------
invalid_points = []
for i, (x, y) in enumerate(route):
    if not (0 <= x < img_width and 0 <= y < img_height):
        invalid_points.append(f"Point {i}: ({x}, {y})")

if invalid_points:
    print("⚠️ WARNING: Found out-of-bounds coordinates:")
    for point in invalid_points:
        print(f"  {point}")
    print(f"  Valid range: 0 ≤ x < {img_width}, 0 ≤ y < {img_height}")

# -------- Draw route --------
draw = ImageDraw.Draw(image)
draw.line(route, fill="red", width=5)


# -------- Save output image --------
image.save(output_image_path)
print(f"Route drawn and saved to {output_image_path}")

# -------- STEP 3: Route Validation --------
print("\n" + "="*70)
print("STEP 3: Running Validation Pipeline (main.py)")
print("="*70 + "\n")

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
        
        print("\n" + "="*70)
        print("VALIDATION SUMMARY")
        print("="*70)
        print(json.dumps(validation_data['validation_summary'], indent=2))
        
        # Print key metrics
        summary = validation_data['validation_summary']
        print("\n" + "="*70)
        print("KEY METRICS")
        print("="*70)
        if 'svr' in summary:
            print(f"✓ Connectivity: {summary['svr'].get('connectivity', 'N/A')}")
            print(f"✓ No Wall Crossings: {not summary['svr'].get('wall_crossings', True)}")
            print(f"✓ No Exhibit Collisions: {not summary['svr'].get('exhibit_collision', True)}")
            print(f"✓ Within Floor Area: {not summary['svr'].get('out_of_area_violations', True)}")
        
        if 'scsr' in summary:
            print(f"✓ Start/End Correct: {summary['scsr'].get('start_end_location', 'N/A')}")
            print(f"✓ Must-Pass Regions: {summary['scsr'].get('must_pass_regions', 'N/A')}")
            print(f"✓ No Restricted Area Violations: {not summary['scsr'].get('restricted_area_violations', True)}")
            print(f"✓ Within Distance Budget: {summary['scsr'].get('distance_budget', 'N/A')}")
        
        print("="*70)
    
except subprocess.CalledProcessError as e:
    print(f"❌ Error running main.py validation:")
    print(f"Return code: {e.returncode}")
    if e.stdout:
        print(f"STDOUT:\n{e.stdout}")
    if e.stderr:
        print(f"STDERR:\n{e.stderr}")
except Exception as e:
    print(f"❌ Unexpected error during validation: {e}")

print("\n✅ Pipeline complete!")
