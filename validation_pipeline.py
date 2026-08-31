import argparse
import json
import sys

from pathlib import Path
from spatial_validator import SpatialValidator
from semantic_validator import SemanticValidator
from route_extractor import RouteExtractor
from route_extractor.route_extractor import NoRouteFoundError
from prompts_utils import atomic_write_json, NO_ROUTE_FOUND_EXIT_CODE

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

def main():

    parser = argparse.ArgumentParser(
    description='MuseRoute pipeline to extract route and process Spatial and Semantics constraints evaluation',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
        Example:
        python main.py \\
        --route-image valid_route.png \\
        --original-image layout_entrance_exit.png \\
        --annotations museum_layout_annotations.json \\
        """
    )
    parser.add_argument('--route-image',          required=True)
    parser.add_argument('--original-image',       required=True)
    parser.add_argument('--annotations',          required=True)
    parser.add_argument('--config-file',          required=True)
    parser.add_argument('--exhibits-csv-file',    required=True)


    parser.add_argument('--extraction-output-image',    default=None)
    parser.add_argument('--validated-output-image',    default=None)
    parser.add_argument('--validated-output-json',    required=True)

    parser.add_argument('--difference-threshold', type=int, default=10)
    parser.add_argument('--tolerance',            type=int, default=3,
                        help='Dilation radius (px) used to forgive residual '
                            'misalignment when subtracting the original image '
                            '(default: 3; increase to 5-8 for larger dimension gaps)')
    parser.add_argument('--marker-size',          type=int, default=15)
    parser.add_argument('--debug', action='store_true', help='Enable debugging displays (disabled by default)')
    parser.add_argument('--skip-alignment', action='store_true',
                        help='Skip homography alignment. Only safe when the route image '
                             'was drawn directly onto an unmodified copy of the original '
                             'image (same dimensions, same coordinate space).')
    parser.add_argument('--route-coordinates-file', default=None,
                        help='Path to a JSON file containing the original source route '
                             'coordinates (e.g. the VLM output route) as a list of [x, y] '
                             'pairs. When provided, computes a geometric_fidelity score '
                             'comparing the extracted skeleton against this route and '
                             'stores it in validation_summary.')
    parser.add_argument('--fidelity-tolerance-px', type=int, default=3,
                        help='Pixel tolerance used for the geometric_fidelity similarity '
                             'percentage (default: 3, roughly half the drawn line width).')

    args = parser.parse_args()

    source_coordinates = None
    if args.route_coordinates_file:
        source_coordinates = load_json(args.route_coordinates_file)

    # Process image directories and filenames
    annotations_path = args.annotations
    route_image_path = args.route_image
    original_image_path = args.original_image
    extraction_output_image_path = args.extraction_output_image
    validated_output_image_path = args.validated_output_image
    config_file = args.config_file
    exhibits_csv_file = args.exhibits_csv_file

    # Load annotations
    annotations_values = load_json(annotations_path)
    configs = load_json(config_file)

    # Value calculations from configs and annotations
    distance_entries = annotations_values.get('distance_in_mm', [])
    mm_per_px = distance_entries[0].get("mm_per_px") if distance_entries else None
    
    # Calculate proximity_threshold from config
    proximity_threshold_px = None
    if mm_per_px is not None and mm_per_px > 0:
        proximity_threshold_px = int(configs['exhibit_see_distance_in_mm'] / mm_per_px)
        print(f"Calculated proximity_threshold: {proximity_threshold_px} px (from {configs['exhibit_see_distance_in_mm']} mm)")

    
    # Use parsed arguments
    re = RouteExtractor(annotations_path)
    try:
        route_extraction_results = re.process_pipeline(
            route_image_path=route_image_path,
            original_image_path=original_image_path,
            output_path=extraction_output_image_path,
            marker_size=args.marker_size,
            difference_threshold=args.difference_threshold,
            tolerance_px=args.tolerance,
            debug=args.debug,
            skip_alignment=args.skip_alignment,
            source_coordinates=source_coordinates,
            fidelity_tolerance_px=args.fidelity_tolerance_px
        )
    except NoRouteFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(NO_ROUTE_FOUND_EXIT_CODE)
    

    print("\n" + "="*70)
    print("Spatial Constraint Validation")
    print("="*70)


    validator = SpatialValidator(
        annotations_file=annotations_path,
        proximity_threshold=proximity_threshold_px if proximity_threshold_px is not None else 25
    )
    
    validator.set_gallery_configurations(configs['gallery_configs'])
    
    validation_result = validator.validate_route(
        route_points=route_extraction_results.points,
        endpoints=route_extraction_results.endpoints
    )
    violation_counts = validator.visualize_validation(
        original_image_path=original_image_path,
        route_points=route_extraction_results.points,
        validation_result=validation_result,
        output_path=validated_output_image_path,
        must_visit_exhibits=configs.get('specific_exhibit_to_cover', []),
        show_all_zones=args.debug
    )

    violation_reasons = validation_result['validation_summary']['violation_reasons']
    svr = {
        "connectivity": route_extraction_results.connectivity["is_connected"],
        "wall_crossings": "wall_crossings" in violation_reasons,
        "exhibit_collision": "exhibit_collisions" in violation_reasons,
        "out_of_area_violations": "floor_area_violations" in violation_reasons,
    }
    scsr = {
        "start_end_location": "entrance_violations" not in violation_reasons and "exit_violations" not in violation_reasons,
        "must_pass_regions": "must_see_gallery_violations" not in violation_reasons,
        "restricted_area_violations": "forbidden_area_violations" in violation_reasons,
        "distance_budget": (route_extraction_results.route_distance * mm_per_px < configs["distance_budget_in_mm"]) if mm_per_px is not None else None
    }

    validation_result['validation_summary']['route_pixels_breakdown'] = violation_counts
    validation_result['validation_summary']['connectivity'] = route_extraction_results.connectivity
    validation_result['validation_summary']['svr'] = svr
    validation_result['validation_summary']['scsr'] = scsr
    validation_result['validation_summary']['geometric_fidelity'] = route_extraction_results.geometric_fidelity
    
    # Semantic Validation
    print("\n" + "="*70)
    print("Semantic Constraint Validation")
    print("="*70)
    
    # Get visited exhibits from spatial validation
    visited_exhibits = validation_result['validation_summary'].get('exhibits_visited', [])
    

    # Create semantic validator
    semantic_validator = SemanticValidator(
        config_file=config_file,
        visited_exhibits=visited_exhibits,
        exhibits_csv_file=exhibits_csv_file
    )

    # Run semantic validation
    scar, semantic_result_detailed = semantic_validator.validate()
    
    # Add semantic validation results to validation_result
    validation_result['validation_summary']['scar'] = scar
    validation_result['semantic_validation_details'] = semantic_result_detailed
    
    atomic_write_json(args.validated_output_json, validation_result, indent=4)
    

if __name__ == '__main__':
    main()