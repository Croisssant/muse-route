import argparse

from pathlib import Path
from spatial_validator import SpatialValidator
from route_extractor import RouteExtractor

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
     parser.add_argument('--annotations-dir', default='./floorplan_annotations')
     parser.add_argument('--extraction-output-image',    default=None)
     parser.add_argument('--validated-output-image',    default=None)
     parser.add_argument('--difference-threshold', type=int, default=10)
     parser.add_argument('--tolerance',            type=int, default=3,
                         help='Dilation radius (px) used to forgive residual '
                              'misalignment when subtracting the original image '
                              '(default: 3; increase to 5-8 for larger dimension gaps)')
     parser.add_argument('--marker-size',          type=int, default=15)
     parser.add_argument('--proximity-threshold',  type=int, default=25,
                         help='Distance for exhibit visit detection in pixels (default: 25)')

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

     
     # Use parsed arguments
     re = RouteExtractor(annotations)
     route_extraction_results = re.process_pipeline(
          route_image_path=route_image_path,
          original_image_path=original_image_path,
          output_path=extraction_output_image_path,
          marker_size=args.marker_size,
          difference_threshold=args.difference_threshold,
          tolerance_px=args.tolerance
     )
     

     print("\n" + "="*70)
     print("Spatial Constraint Validation")
     print("="*70)


     validator = SpatialValidator(
          annotations_file=annotations,
          proximity_threshold=args.proximity_threshold
     )
     validation_result = validator.validate_route(
          route_points=route_extraction_results.points,
          endpoints=route_extraction_results.endpoints
     )
     validator.visualize_validation(
          original_image_path=original_image_path,
          route_points=route_extraction_results.points,
          validation_result=validation_result,
          output_path=validated_output_image_path
     )

    
    

if __name__ == '__main__':
    main()