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

# Model Output Dump
import json
with open("response_dump.json", "w", encoding="utf-8") as f:
    json.dump(response.model_dump(), f, indent=2, ensure_ascii=False, default=str)

# Check Image
image_base64 = None


from PIL import Image
import base64
import io

# response.output[0] is an ImageGenerationCall object
image_call = response.output[0]  

# The base64 result is in the .result attribute
image_base64 = getattr(image_call, "result", None)

if image_base64 is None:
    raise ValueError("No image returned from model")

image = Image.open(io.BytesIO(base64.b64decode(image_base64)))
image.show()