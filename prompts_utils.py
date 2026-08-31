import io
import base64
import json
import os
import re
import tempfile

from abc import ABC, abstractmethod
from pathlib import Path
from PIL import Image, ImageDraw

MAX_NEW_TOKENS = 1024

# Signals a validation_pipeline.py subprocess exit when route extraction found
# no route points, so callers can distinguish it from an arbitrary crash and
# record a real zero score instead of leaving the task's results missing.
NO_ROUTE_FOUND_EXIT_CODE = 3


def atomic_write_json(path, data, **json_kwargs):
    """Write JSON to `path` atomically (temp file + os.replace).

    Prevents readers from ever seeing a partially-written file if two
    processes write the same path concurrently, or if a write is
    interrupted partway through on a flaky mount.
    """
    path = Path(path)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, **json_kwargs)
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def atomic_save_image(image, path, **save_kwargs):
    """Save a PIL image to `path` atomically (temp file + os.replace).

    Same rationale as atomic_write_json: image.save() issues many internal
    writes during encoding, and a partial/failed write should never leave a
    corrupted file at the real destination.
    """
    path = Path(path)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}.", suffix=path.suffix)
    os.close(fd)
    try:
        image.save(tmp_path, **save_kwargs)
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise

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

def to_jsonable(obj):
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    return obj
    
def save_response_record(model_name: str, difficulty: str, complexity: str, layout_name: str, response_obj, config_variant=None):
    pass
    # import json
    # from datetime import datetime
    # from pathlib import Path

    # filename = f"openai_response_record.json"
    # filepath = Path.cwd() / filename   

    # record = {
    #     "timestamp": datetime.now().isoformat(),
    #     "model": model_name,
    #     "difficulty": difficulty,
    #     "complexity": complexity,
    #     "layout_name": layout_name,
    #     'response': {
    #         'model': response_obj.model,
    #         'usage': to_jsonable(response_obj.usage),
    #         'output_text': response_obj.output_text.strip(),            
    #     },
    #     "config_variant": config_variant,
    # }

    # if filepath.exists():
    #     with open(filepath, "r", encoding="utf-8") as f:
    #         records = json.load(f)
    # else:
    #     records = []

    # records.append(record)

    # with open(filepath, "w", encoding="utf-8") as f:
    #     json.dump(records, f, indent=4, ensure_ascii=False)


# ===========================================================================
# Backend abstraction
# ===========================================================================
 
class ModelBackend(ABC):
    """Common interface for all model backends."""
 
    @abstractmethod
    def prompt(self, system_prompt: str, user_prompt: str, image_base64: str) -> str:
        """Send a multimodal prompt and return the text response."""
        pass
 
 
class OpenAIBackend(ModelBackend):
    """Backend for OpenAI models (including reasoning models)."""
 
    def __init__(self, client, model: str, reasoning_effort: str | None = None):
        self.client = client
        self.model = model
        self.reasoning_effort = reasoning_effort
 
    def prompt(self, system_prompt: str, user_prompt: str, image_base64: str) -> str:
        kwargs = dict(
            model=self.model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "input_text", "text": user_prompt},
                    {"type": "input_image",
                     "image_url": f"data:image/png;base64,{image_base64}"}
                ]}
            ]
        )
        if self.reasoning_effort:
            kwargs["reasoning"] = {"effort": self.reasoning_effort}
 
        response = self.client.responses.create(**kwargs)
        return response.output_text.strip(), response
        # return response.output_text.strip()
 
 
class HuggingFaceBackend(ModelBackend):
    """Backend for local HuggingFace models via transformers pipeline."""
 
    def __init__(self, model_name: str, **pipeline_kwargs):
        from transformers import pipeline
        self.model_name = model_name
        self.pipe = pipeline(
            "image-text-to-text",
            model=model_name,
            **pipeline_kwargs
        )
 
    def _base64_to_pil(self, image_base64: str) -> Image.Image:
        image_data = base64.b64decode(image_base64)
        return Image.open(io.BytesIO(image_data)).convert("RGB")
 
    def prompt(self, system_prompt: str, user_prompt: str, image_base64: str) -> str:
        image = self._base64_to_pil(image_base64)
        
        # Strategy 1: Try separate system message (better for instruction-tuned models)
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image", "image": image},  # PIL Image, not base64
                    {"type": "text", "text": user_prompt},
                ]
            })
            
            result = self.pipe(text=messages, max_new_tokens=MAX_NEW_TOKENS)
            
            # Parse response
            last = result[0]["generated_text"][-1]
            if isinstance(last, dict):
                response = (last.get("content") or last.get("generated_text", "")).strip()
                if response:  # Only return if we got actual content
                    return response
        
        except (TypeError, KeyError, IndexError, AttributeError):
            # Fall through to Strategy 2
            pass
        
        # Strategy 2: Fallback - combine system and user prompts in chat format
        try:
            combined_prompt = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt
            
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": combined_prompt},
                ]
            }]
            
            result = self.pipe(text=messages, max_new_tokens=MAX_NEW_TOKENS)
            last = result[0]["generated_text"][-1]
            if isinstance(last, dict):
                response = (last.get("content") or last.get("generated_text", "")).strip()
                if response:
                    return response
            return str(last).strip()
        
        except (TypeError, KeyError, IndexError, AttributeError):
            # Fall through to Strategy 3
            pass
        
        # Strategy 3: Simple positional API (for basic vision-language models)
        try:
            combined_prompt = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt
            print("[TEST] pipeline 3")
            result = self.pipe(image, text=combined_prompt, max_new_tokens=MAX_NEW_TOKENS)
            
            # Handle various output formats
            if isinstance(result, list) and len(result) > 0:
                output = result[0]
                if isinstance(output, dict):
                    # Try common keys
                    for key in ["generated_text", "text", "content"]:
                        if key in output:
                            response = str(output[key]).strip()
                            if response:
                                return response
                return str(output).strip()
            return str(result).strip()
        
        except Exception as e:
            # All strategies failed - provide helpful error message
            raise RuntimeError(
                f"All HuggingFace pipeline strategies failed for model {self.model_name}. "
                f"Last error: {type(e).__name__}: {str(e)}. "
                f"Try checking the model's documentation for the correct API usage."
            )
 

class OllamaBackend(ModelBackend):
    """Backend for Ollama models (local or remote)."""
 
    def __init__(self, model: str, base_url: str = "http://127.0.0.1:11434"):
        try:
            from ollama import Client
        except ImportError:
            raise ImportError(
                "The 'ollama' package is required for OllamaBackend. "
                "Install it with: pip install ollama"
            )
        self.model = model
        self.client = Client(host=base_url)
 
    def prompt(self, system_prompt: str, user_prompt: str, image_base64: str) -> str:
        messages = [
            {
                'role': 'system',
                'content': system_prompt
            },
            {
                'role': 'user',
                'content': user_prompt,
                'images': [image_base64]
            }
        ]
        
        chat_kwargs = {
            "model": self.model,
            "messages": messages
        }
        
        # if "qwen" in self.model.lower() or 'gemma3' in self.model.lower():
        chat_kwargs["options"] = {
            'num_ctx': 32768,  
            'num_predict': 8192,   
        }

        response = self.client.chat(**chat_kwargs)

        return response['message']['content'].strip(), None
 
 
# ===========================================================================
# Unified prompt entry-point
# ===========================================================================

def build_backend(backend_type: str, model: str, reasoning_effort: str | None = None, **pipeline_kwargs):
    """
    Factory function to instantiate the correct ModelBackend.
    
    Args:
        backend_type: Either "openai", "huggingface", or "ollama"
        model: Model name/identifier
        reasoning_effort: Optional reasoning effort for OpenAI models
        **pipeline_kwargs: Additional keyword arguments for HuggingFace pipeline or Ollama (e.g., base_url)
    
    Returns:
        ModelBackend instance (OpenAIBackend, HuggingFaceBackend, or OllamaBackend)
    
    Example:
        # OpenAI
        backend = build_backend("openai", "gpt-4-vision", reasoning_effort="medium")
        
        # HuggingFace
        backend = build_backend("huggingface", "google/gemma-4-31B-it")
        
        # Ollama (default)
        backend = build_backend("ollama", "qwen3.6")
        
        # Ollama with custom base_url (for different GPU/port)
        backend = build_backend("ollama", "qwen3.6", base_url="http://127.0.0.1:11435")
    """
    if backend_type == "openai":
        from openai import OpenAI
        client = OpenAI()
        return OpenAIBackend(client, model, reasoning_effort)
    
    elif backend_type == "huggingface":
        return HuggingFaceBackend(model, **pipeline_kwargs)
    
    elif backend_type == "ollama":
        base_url = pipeline_kwargs.pop("base_url", "http://127.0.0.1:11434")
        return OllamaBackend(model, base_url=base_url)
    
    else:
        raise ValueError(f"Unknown backend type: {backend_type!r}. Must be 'openai', 'huggingface', or 'ollama'")
 
def prompt_model(backend: ModelBackend,
                 system_prompt: str,
                 user_prompt: str,
                 image_base64: str) -> str:
    """Single entry-point for all backends. Replaces prompt_gpt()."""
    return backend.prompt(system_prompt, user_prompt, image_base64)


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
    selected_exhibits = [e for e in exhibits_list if e["exhibit_number"] in selected_ids]
    
    # Validate that exhibits were actually selected
    if not selected_exhibits:
        raise ValueError(f"No exhibits found matching selected IDs: {selected_ids}")
    
    return selected_exhibits


def create_failure_final_results(svr_fields, scsr_fields, scar_fields):
    """
    Create a final_results.json structure indicating complete failure.
    All validation checks should fail according to metrics.py expectations.
    
    Args:
        svr_fields: List of SVR field names to include
        scsr_fields: List of SCSR field names to include
        scar_fields: List of SCAR field names to include
    
    Returns:
        Dict with svr, scsr, and scar sections populated with failure values
    """
    # SVR: connectivity should be False (disconnected), others should be True (violations present)
    svr = {field: False if field == 'connectivity' else True for field in svr_fields}
    
    # SCSR: Most should be False (not satisfied), except violation fields which should be True
    scsr = {}
    for field in scsr_fields:
        if 'violation' in field:
            scsr[field] = True  # Violations present = True
        else:
            scsr[field] = False  # Requirements not satisfied = False
    
    # SCAR: All should be False (coverage not satisfied)
    scar = {field: False for field in scar_fields}
    
    return {"svr": svr, "scsr": scsr, "scar": scar}


def parse_route_and_save(input_image_path, output_image_path, text_output_route):

    print(f"text_output_route: \n{text_output_route}")
     
    text_output_route = re.sub(r"^```json\s*|\s*```$", "", text_output_route, flags=re.MULTILINE)

    image = Image.open(input_image_path)
    img_width, img_height = image.size

    # -------- Parse route --------
    try:
        route = json.loads(text_output_route)
    except json.JSONDecodeError:
        # More robust regex: find outermost array brackets with any content between
        match = re.search(r"\[[\s\S]*\]", text_output_route)
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
        error_msg = "Found out-of-bounds coordinates:\n"
        for point in invalid_points:
            error_msg += f"  {point}\n"
        error_msg += f"  Valid range: 0 <= x < {img_width}, 0 <= y < {img_height}"
        raise ValueError(error_msg)

    # -------- Draw route --------
    draw = ImageDraw.Draw(image)
    draw.line(route, fill="red", width=5)

    # -------- Save output image --------
    atomic_save_image(image, output_image_path)
    print(f"Route drawn and saved to {output_image_path}")

    return route


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


def discover_config_files(layout_folder, config_pattern):
    """
    Discover config files matching a pattern in the layout folder.
    
    Args:
        layout_folder: Path to the layout directory
        config_pattern: Filename pattern (e.g., "medium_*.json", "hard.json")
    
    Returns:
        List of tuples: [(config_variant, config_path), ...]
        Example: [("medium_01", Path("medium_01.json")), ("medium_02", Path("medium_02.json"))]
        
        For single files (no wildcard), returns: [("medium", Path("medium.json"))]
    """
    import glob
    
    # Check if pattern contains wildcard
    if '*' in config_pattern:
        # Find all matching files
        pattern_path = str(layout_folder / config_pattern)
        matching_files = sorted(glob.glob(pattern_path))
        
        if not matching_files:
            return []
        
        results = []
        for filepath in matching_files:
            config_path = Path(filepath)
            # Extract variant name from filename (e.g., "medium_01.json" -> "medium_01")
            config_variant = config_path.stem
            results.append((config_variant, config_path))
        
        return sorted(results)
    else:
        # Single file (no wildcard)
        config_path = layout_folder / config_pattern
        if config_path.exists():
            # Extract base name without extension (e.g., "medium.json" -> "medium")
            config_variant = config_path.stem
            return [(config_variant, config_path)]
        else:
            return []


def build_paths(layout_folder, model_name, difficulty, complexity, 
                base_results_dir, filenames, config_variant=None):
    """
    Build all input and output paths for a given layout.
    
    Args:
        layout_folder: Path to the layout directory
        model_name: Model name (e.g., "gpt-5.4")
        difficulty: Difficulty level (e.g., "easy_semantic")
        complexity: Complexity level (e.g., "simple")
        base_results_dir: Base results directory Path
        filenames: Dict with filename mappings (e.g., {'image': 'annotated_layout.png', ...})
        config_variant: Optional config variant name (e.g., "medium_01"). If None, uses difficulty.
    
    Returns:
        Tuple of (input_paths, output_paths) where both are dictionaries
    """
    layout_name = layout_folder.name
    
    # Input paths
    input_paths = {
        key: layout_folder / filename 
        for key, filename in filenames.items()
    }
    
    # Determine output folder prefix
    # If config_variant is provided, use it; otherwise use difficulty
    folder_prefix = config_variant if config_variant else difficulty
    
    # Output directory: base_results_dir already includes model folder, so just add difficulty/complexity/layout
    output_dir = (base_results_dir / difficulty / 
                  complexity / f"{folder_prefix}_{layout_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    sanitized_model_name = model_name.replace('/', '-')
    
    # Output paths
    output_paths = {
        'dir': output_dir,
        'route_image': output_dir / f"{sanitized_model_name}_route.png",
        'extracted_image': output_dir / f"extracted_{sanitized_model_name}_route.png",
        'validated_image': output_dir / f"validated_{sanitized_model_name}_route.png",
        'validation_json': output_dir / "validation_results.json",
        'final_json': output_dir / "final_results.json",
        'route_coordinates': output_dir / "route_coordinates.json"
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
