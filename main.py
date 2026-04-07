import argparse
import json

from pathlib import Path
from spatial_validator import SpatialValidator
from route_extractor import RouteExtractor

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
     parser.add_argument('--images-dir', default='./images')
     parser.add_argument('--annotations-dir', default='./original_floorplans/museum_layout_01')
     parser.add_argument('--extraction-output-image',    default=None)
     parser.add_argument('--validated-output-image',    default=None)
     parser.add_argument('--difference-threshold', type=int, default=10)
     parser.add_argument('--tolerance',            type=int, default=3,
                         help='Dilation radius (px) used to forgive residual '
                              'misalignment when subtracting the original image '
                              '(default: 3; increase to 5-8 for larger dimension gaps)')
     parser.add_argument('--marker-size',          type=int, default=15)
     parser.add_argument('--debug', action='store_true', help='Enable debugging displays (disabled by default)')

     args = parser.parse_args()

     # Process annotations path
     annotations = Path(args.annotations_dir) / args.annotations

     # Process image directories and filenames
     images_dir = Path(args.images_dir)
     route_image_path = images_dir / "route_images" / args.route_image
     original_image_path = images_dir / "original_images" / args.original_image
     extraction_output_image_path = args.extraction_output_image
     validated_output_image_path = args.validated_output_image

     if extraction_output_image_path:
          extracted_route_image_path = images_dir / "extracted_route_images"
          extracted_route_image_path.mkdir(parents=True, exist_ok=True)

          extraction_output_image_path = extracted_route_image_path / args.extraction_output_image

     
     if validated_output_image_path:
          validated_output_image_path = images_dir / "validated_images"
          validated_output_image_path.mkdir(parents=True, exist_ok=True)
          
          validated_output_image_path = validated_output_image_path / args.validated_output_image

     # Load annotations
     annotations_values = load_json(annotations)
     configs = load_json("./config.json")

     # Value calculations from configs and annotations
     mm_per_px = None
     if annotations_values != {}:
          mm_per_px = annotations_values['distance_in_mm'][0]['mm_per_px']
     
     # Calculate proximity_threshold from config
     proximity_threshold_px = None
     if mm_per_px is not None and mm_per_px > 0:
          proximity_threshold_px = int(configs['exhibit_see_distance_in_mm'] / mm_per_px)
          print(f"Calculated proximity_threshold: {proximity_threshold_px} px (from {configs['exhibit_see_distance_in_mm']} mm)")

     
     # Use parsed arguments
     re = RouteExtractor(annotations)
     route_extraction_results = re.process_pipeline(
          route_image_path=route_image_path,
          original_image_path=original_image_path,
          output_path=extraction_output_image_path,
          marker_size=args.marker_size,
          difference_threshold=args.difference_threshold,
          tolerance_px=args.tolerance,
          debug=args.debug
     )
     

     print("\n" + "="*70)
     print("Spatial Constraint Validation")
     print("="*70)


     validator = SpatialValidator(
          annotations_file=annotations,
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
          output_path=validated_output_image_path
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
          "restricted_area_violations": "forbidden_area_violations" in violation_reasons or "gallery_violations" in violation_reasons,
          "distance_budget": route_extraction_results.route_distance * mm_per_px < configs["distance_budget_in_mm"]
     }

     validation_result['validation_summary']['route_pixels_breakdown'] = violation_counts
     validation_result['validation_summary']['connectivity'] = route_extraction_results.connectivity
     validation_result['validation_summary']['svr'] = svr
     validation_result['validation_summary']['scsr'] = scsr
     
     with open("validation_results.json", "w") as json_file:
          json.dump(validation_result, json_file, indent=4)
    

if __name__ == '__main__':
    main()