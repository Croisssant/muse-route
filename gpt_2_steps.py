import base64
import json
import re
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image, ImageDraw
from build_prompts import build_system_prompt, build_user_prompt, build_selection_prompt

# -------- Load API key --------
load_dotenv()
client = OpenAI()

# -------- File paths --------
input_image_path = "./images/annotated_walls_museum_layout_02/original_images/annotated_walls_museum_layout_02.png"
output_image_path = "output_2.png"
exhibits_json_path = "./exhibits/exhibit_list.json"

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

# -------- Scale route if needed --------
max_x = max(p[0] for p in route)
max_y = max(p[1] for p in route)

scale_x = img_width / max_x
scale_y = img_height / max_y

scaled_route = [(int(x * scale_x), int(y * scale_y)) for x, y in route]

# -------- Draw route --------
draw = ImageDraw.Draw(image)
draw.line(route, fill="red", width=5)


# -------- Save output image --------
image.save(output_image_path)
print(f"Route drawn and saved to {output_image_path}")