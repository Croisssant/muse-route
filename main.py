import argparse

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
     parser.add_argument('--extraction-output-image-path',    default=None)
     parser.add_argument('--validated-output-image-path',    default=None)
     parser.add_argument('--difference-threshold', type=int, default=10)
     parser.add_argument('--tolerance',            type=int, default=3,
                         help='Dilation radius (px) used to forgive residual '
                              'misalignment when subtracting the original image '
                              '(default: 3; increase to 5-8 for larger dimension gaps)')
     parser.add_argument('--marker-size',          type=int, default=15)
     parser.add_argument('--proximity-threshold',  type=int, default=25,
                         help='Distance for exhibit visit detection in pixels (default: 25)')

     args = parser.parse_args()
     
     # Use parsed arguments
     re = RouteExtractor(args.annotations)
     route_extraction_results = re.process_pipeline(
          route_image_path=args.route_image,
          original_image_path=args.original_image,
          output_path=args.extraction_output_image_path,
          marker_size=args.marker_size,
          difference_threshold=args.difference_threshold,
          tolerance_px=args.tolerance
     )
     

     print("\n" + "="*70)
     print("Spatial Constraint Validation")
     print("="*70)


     validator = SpatialValidator(
          annotations_file=args.annotations,
          proximity_threshold=args.proximity_threshold
     )
     validation_result = validator.validate_route(
          route_points=route_extraction_results.points,
          endpoints=route_extraction_results.endpoints
     )
     validator.visualize_validation(
          original_image_path=args.original_image,
          route_points=route_extraction_results.points,
          validation_result=validation_result,
          output_path=args.validated_output_image_path
     )

    
    

if __name__ == '__main__':
    main()