import base64
import os
import re
import json
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image, ImageDraw

# Load API key
load_dotenv()
client = OpenAI()

input_image_path = "./images/annotated_walls_museum_layout_02/original_images/annotated_walls_museum_layout_02.png"
output_image_path = "output.png"

prompt = """
You are given a museum layout image.

Return a path that traverses the museum.

Output ONLY a JSON array of (x, y) coordinates representing the path.
Coordinates must be in pixel space relative to the image.

Example:
[[10, 20], [30, 40], [50, 80]]

Do not describe anything. Only output JSON.
"""

# Read image as base64
with open(input_image_path, "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode("utf-8")

# Send request to GPT
response = client.responses.create(
    model="gpt-4o",  # or GPT-5 if available
    input=[{
        "role": "user",
        "content": [
            {"type": "input_text", "text": prompt},
            {
                "type": "input_image",
                "image_url": f"data:image/png;base64,{image_base64}"
            }
        ]
    }]
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
draw.line(route_tuples, fill="red", width=5)

# Save edited image
image.save(output_image_path)
print(f"Route drawn and saved to {output_image_path}")