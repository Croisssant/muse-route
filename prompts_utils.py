import base64
import json
import re
from pathlib import Path

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



def prompt_gpt(client, model, system_prompt, user_prompt, image_base64, reasoning_effort=None): 
    if reasoning_effort:
        response = client.responses.create(
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

        return response.output_text.strip()
    
    else:
        response = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "input_text", "text": user_prompt},
                    {"type": "input_image", "image_url": f"data:image/png;base64,{image_base64}"}
                ]}
            ]
        )

        return response.output_text.strip()

def exhibit_selection(text_output_selection, exhibits_list):
    
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


def parse_route_and_save(input_image_path, output_image_path, text_output_route):
     
    text_output_route = re.sub(r"^```json\s*|\s*```$", "", text_output_route, flags=re.MULTILINE)

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


def discover_complexity_and_layouts(base_dir, complexity_mode, specific_complexities,
                                    layout_mode, specific_layouts):
    """
    Discover complexity levels and layouts based on configuration.
    
    Args:
        base_dir: Base floorplans directory (e.g., Path("floorplans"))
        complexity_mode: "all", "single", or "list"
        specific_complexities: str or list of complexity names (e.g., "simple" or ["simple", "complex"])
        layout_mode: "all", "single", or "list"
        specific_layouts: str or list of layout names (e.g., "layout_01" or ["layout_01", "layout_03"])
    
    Returns:
        List of tuples: [(complexity_name, layout_path), ...]
        Example: [("simple", Path("floorplans/simple/layout_01")), ...]
    """
    if not base_dir.exists():
        print(f"ERROR: Base directory does not exist: {base_dir}")
        return []
    
    # Step 1: Determine which complexity levels to process
    if complexity_mode == "all":
        complexity_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
    elif complexity_mode == "single":
        if not specific_complexities:
            print("ERROR: SPECIFIC_COMPLEXITIES required when COMPLEXITY_MODE is 'single'")
            return []
        complexity_dirs = [base_dir / specific_complexities]
    elif complexity_mode == "list":
        if not isinstance(specific_complexities, list):
            print("ERROR: SPECIFIC_COMPLEXITIES must be a list when COMPLEXITY_MODE is 'list'")
            return []
        complexity_dirs = [base_dir / c for c in specific_complexities]
    else:
        print(f"ERROR: Invalid COMPLEXITY_MODE: {complexity_mode}. Use 'all', 'single', or 'list'")
        return []
    
    # Step 2: For each complexity, discover layouts
    results = []
    for complexity_dir in complexity_dirs:
        if not complexity_dir.exists():
            print(f"WARNING: Complexity directory not found, skipping: {complexity_dir}")
            continue
        
        complexity_name = complexity_dir.name
        
        # Discover layouts based on layout mode
        if layout_mode == "all":
            layout_folders = [d for d in complexity_dir.iterdir() 
                            if d.is_dir() and d.name.startswith("layout_")]
        elif layout_mode == "single":
            if not specific_layouts:
                print("ERROR: SPECIFIC_LAYOUTS required when LAYOUT_MODE is 'single'")
                continue
            layout_folders = [complexity_dir / specific_layouts]
        elif layout_mode == "list":
            if not isinstance(specific_layouts, list):
                print("ERROR: SPECIFIC_LAYOUTS must be a list when LAYOUT_MODE is 'list'")
                continue
            layout_folders = [complexity_dir / l for l in specific_layouts]
        else:
            print(f"ERROR: Invalid LAYOUT_MODE: {layout_mode}. Use 'all', 'single', or 'list'")
            continue
        
        # Add valid layouts to results
        for layout_folder in layout_folders:
            if layout_folder.exists():
                results.append((complexity_name, layout_folder))
            else:
                print(f"WARNING: Layout not found, skipping: {layout_folder}")
    
    return sorted(results)


def filter_and_validate_attributes(scar_full, configs):
    """
    Filter attribute validations based on config and return simplified boolean result.
    
    Args:
        scar_full: Full scar results from validation_results.json
        configs: Configuration dict containing exhibit_attribute_constraints
    
    Returns:
        Dict with start_end_location and attribute_validations (boolean)
    """
    
    # Filter attribute_validations based on config constraints
    attribute_constraints = configs.get("exhibit_attribute_constraints", None)
    if attribute_constraints:
        # Get list of non-empty constraint names from config
        filtered_or_constraints = filter_non_empty_fields(
            attribute_constraints, ["_comment", "combined_constraint"]
        )
        
        and_constraints = attribute_constraints.get("combined_constraint", None)
        filtered_and_constraints = []
        if and_constraints:
            filtered_and_constraints = filter_non_empty_fields(
                and_constraints, ["_combined_comment"]
            )
        
        # Get attribute validations from results
        attribute_validations_full = scar_full.get("attribute_validations", {})
        
        # Filter to only include constraints that have values in config
        filtered_attribute_validations = {}
        
        # Add OR constraints that have values
        for constraint_name in filtered_or_constraints:
            if constraint_name in attribute_validations_full:
                filtered_attribute_validations[constraint_name] = \
                    attribute_validations_full[constraint_name]
        
        # Add combined constraint if it has any values
        if filtered_and_constraints and "combined_constraint" in attribute_validations_full:
            filtered_attribute_validations["combined_constraint"] = \
                attribute_validations_full["combined_constraint"]
        
        # Check if ALL filtered constraints are valid
        # True only if all constraints pass, False otherwise
        if filtered_attribute_validations:
            all_valid = all(
                validation.get("valid", False) 
                for validation in filtered_attribute_validations.values()
            )
    
    return all_valid


def build_paths(layout_folder, model_name, difficulty, complexity, 
                base_results_dir, filenames):
    """
    Build all input and output paths for a given layout.
    
    Args:
        layout_folder: Path to the layout directory
        model_name: Model name (e.g., "gpt-5.4")
        difficulty: Difficulty level (e.g., "easy_semantic")
        complexity: Complexity level (e.g., "simple")
        base_results_dir: Base results directory Path
        filenames: Dict with filename mappings (e.g., {'image': 'annotated_layout.png', ...})
    
    Returns:
        Tuple of (input_paths, output_paths) where both are dictionaries
    """
    layout_name = layout_folder.name
    
    # Input paths
    input_paths = {
        key: layout_folder / filename 
        for key, filename in filenames.items()
    }
    
    # Output directory: results/[model]/[difficulty]/[complexity]/[difficulty]_[layout_name]
    output_dir = (base_results_dir / model_name / difficulty / 
                  complexity / f"{difficulty}_{layout_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Output paths
    output_paths = {
        'dir': output_dir,
        'route_image': output_dir / f"{model_name}_route.png",
        'extracted_image': output_dir / f"extracted_{model_name}_route.png",
        'validated_image': output_dir / f"validated_{model_name}_route.png",
        'validation_json': output_dir / "validation_results.json",
        'final_json': output_dir / "final_results.json"
    }
    
    return input_paths, output_paths


def filter_non_empty_fields(dict: dict, exclude_keys:list[str]) -> list[str]:
    """
    GET dictionary keys with non empty values
    """
    return [
        k
        for k, v in dict.items()
        if k not in exclude_keys and any(
            not (isinstance(inner, list) and len(inner) == 0)
            for inner in v.values()
        )
    ]


def extract_validation_fields(validation_dict, fields_to_include):
    """
    Extract specified fields from validation dictionaries (svr, scsr).
    
    Args:
        validation_dict: Dictionary like svr or scsr from validation_results.json
        fields_to_include: List of field names to extract (e.g., ['connectivity', 'wall_crossings'])
    
    Returns:
        Dict with only the specified fields
    """
    return {
        field: validation_dict[field]
        for field in fields_to_include
        if field in validation_dict
    }


def extract_scar_validations(scar_full, configs, fields_to_include):
    """
    Extract specified scar fields and convert to simplified boolean format.
    
    Args:
        scar_full: Full scar results from validation_results.json
        configs: Configuration dict (for attribute validation filtering)
        fields_to_include: List of field names to extract 
            (e.g., ['specific_exhibit_coverage', 'at_least_n_exhibits_coverage', 
                    'exhibit_category_coverage', 'attribute_validations'])
    
    Returns:
        Dict with simplified boolean values for specified fields
    """
    result = {}
    
    for field in fields_to_include:
        if field not in scar_full:
            continue
        
        field_data = scar_full[field]
        
        # Handle exhibit_category_coverage - exclude if no categories in config
        if field == "exhibit_category_coverage":
            # Check if categories are specified in config
            categories_to_cover = configs.get("exhibit_categories_to_cover", [])
            if categories_to_cover and len(categories_to_cover) > 0:
                # Only include if categories exist in config
                if isinstance(field_data, dict):
                    all_categories_valid = all(
                        category_data.get("valid", False)
                        for category_data in field_data.values()
                    )
                    result[field] = all_categories_valid
                else:
                    result[field] = False
            # If no categories in config, skip this field (don't add to result)
        
        # Handle attribute_validations with config filtering
        elif field == "attribute_validations":
            # Use the dedicated function for attribute validation
            result[field] = filter_and_validate_attributes(scar_full, configs)
        
        # Handle standard validation fields with 'valid' key
        elif isinstance(field_data, dict) and "valid" in field_data:
            result[field] = field_data["valid"]
        
        # Handle other fields as-is
        else:
            result[field] = field_data
    
    return result
