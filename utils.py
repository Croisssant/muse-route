import base64
import json
import re

from PIL import Image, ImageDraw

def select_fields(source_dict, fields):
    return {
        key: source_dict[key]
        for key in fields
        if key in source_dict
    }

def load_image(input_image_path):
  
    image = Image.open(input_image_path)
    img_width, img_height = image.size

    # Read image as base64
    with open(input_image_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode("utf-8")

    return image_base64, img_width, img_height


def exhibit_selection_gpt(client, model, system_prompt, user_prompt, exhibits_list, image_base64, reasoning_effort="medium"):
    response_selection = client.responses.create(
    model=model,
    reasoning={"effort": reasoning_effort},
    input=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "input_text", "text": user_prompt},
            {"type": "input_image", "image_url": f"data:image/png;base64,{image_base64}"}
        ]}
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
    return  [e for e in exhibits_list if e["exhibit_number"] in selected_ids]


def route_planning_gpt(client, model, system_prompt, user_prompt, image_base64, reasoning_effort="medium"):
    response_route = client.responses.create(
        model=model,
        reasoning={"effort": reasoning_effort},
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "input_text", "text": user_prompt},
                {"type": "input_image", "image_url": f"data:image/png;base64,{image_base64}"}
            ]}
        ]
    )

    text_output_route = response_route.output_text.strip()
    text_output_route = re.sub(r"^```json\s*|\s*```$", "", text_output_route, flags=re.MULTILINE)

    return text_output_route


def parse_route_and_save(input_image_path, output_image_path, text_output_route):

    image = Image.open(input_image_path)
    img_width, img_height = image.size

    # -------- Parse route --------
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