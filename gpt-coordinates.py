import base64
import os
import re
import json
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image, ImageDraw
from build_prompts import build_system_prompt, build_user_prompt

def load_json(file_path):
    """
    Load a JSON file and return its contents as a Python object.
    
    Args:
        file_path (str): Path to the JSON file
        
    Returns:
        dict or list: Parsed JSON data
        
    Raises:
        FileNotFoundError: If the file does not exist
        json.JSONDecodeError: If the file is not valid JSON
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# Load API key
load_dotenv()
client = OpenAI()

input_image_path = "./images/annotated_walls_museum_layout_02/original_images/annotated_walls_museum_layout_02.png"
output_image_path = "output.png"

# Read image as base64
image = Image.open(input_image_path)
width, height = image.size
print(f"Image Width: {width}")
print(f"Image Height: {height}")
with open(input_image_path, "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode("utf-8")

exhibits_json = load_json("./exhibits/exhibit_list.json")
system_prompt = build_system_prompt(width, height, 183)
user_prompt = build_user_prompt("I only want to visit Roman Exhibits", exhibits_json)






# Send request to GPT
response = client.responses.create(
    model="gpt-5.4",  # or GPT-5 if available
    input=[
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": user_prompt},
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{image_base64}"
                }
            ]
        }
    ],
)





# Save a dump of the raw response (optional)
with open("response_dump.json", "w", encoding="utf-8") as f:
    json.dump(response.model_dump(), f, indent=2, ensure_ascii=False, default=str)

# Extract text output from the model
text_output = ""
for output in response.output:
    if hasattr(output, "content"):
        for item in output.content:
            if getattr(item, "type", None) == "output_text":
                text_output += getattr(item, "text", "")

text_output = text_output.strip()

# Remove markdown ```json code block if present
text_output = re.sub(r"^```json\s*|\s*```$", "", text_output, flags=re.MULTILINE).strip()

# Parse JSON
try:
    route = json.loads(text_output)
except json.JSONDecodeError:
    # Optional: try extracting JSON array if extra text exists
    match = re.search(r"\[\s*\[.*?\]\s*\]", text_output, re.DOTALL)
    if match:
        route = json.loads(match.group())
    else:
        raise ValueError(f"Failed to parse JSON from model output:\n{text_output}")

# Draw route on the original image
image = Image.open(input_image_path).convert("RGB")
draw = ImageDraw.Draw(image)

route_tuples = [tuple(point) for point in route]
print(f"Route_tuples: {route_tuples}")
draw.line(route_tuples, fill="red", width=5)

# Save edited image
image.save(output_image_path)
print(f"Route drawn and saved to {output_image_path}")