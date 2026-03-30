from spatial_validator import SpatialValidator
from route_extractor import RouteExtractor

def main():
   museum_annotations = "./museum_layout_annotations.json"
   route_image_path     = "./route_plan_final.png"
   original_image_path  = "./layout_entrance_exit.png"

   re = RouteExtractor(museum_annotations)
   route_extraction_results = re.process_pipeline(
        route_image_path,
        original_image_path
        # output_path          = args.output,
        # marker_size          = args.marker_size,
        # difference_threshold = args.difference_threshold,
        # tolerance_px         = args.tolerance,
   )
   
   validator = SpatialValidator(museum_annotations)
   validation_result = validator.validate_route(route_extraction_results.points)
   validator.visualize_validation(
        original_image_path=original_image_path,
        route_points=route_extraction_results.points,
        validation_result=validation_result,
        output_path='main-test.png'
    )

    
    

if __name__ == '__main__':
    main()