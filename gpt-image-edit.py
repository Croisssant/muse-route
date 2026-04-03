import base64
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load API key
load_dotenv()
client = OpenAI()

input_image_path = "./images/annotated_walls_museum_layout_02/original_images/annotated_walls_museum_layout_02.png"
output_image_path = "output.png"

prompt = """
Draw a route on the image in RED that traverses the museum layout
"""

# Read image as base64
with open(input_image_path, "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode("utf-8")

# ✅ Modern API call
response = client.responses.create(
    model="gpt-4o-mini",  # orchestration model
    input=[{
        "role": "user",
        "content": [
            {"type": "input_text", "text": prompt},
            {
                "type": "input_image",
                "image_url": f"data:image/png;base64,{image_base64}"
            }
        ]
    }],
    tools=[{
        "type": "image_generation",
        "model": "gpt-image-1"
    }]
)

# ✅ Extract generated image
image_base64 = None

for output in response.output:
    if output.type == "image_generation":
        image_base64 = output.result
        break

if image_base64 is None:
    raise ValueError("No image returned")

# Save image
with open(output_image_path, "wb") as f:
    f.write(base64.b64decode(image_base64))

print(f"Edited image saved to {output_image_path}")