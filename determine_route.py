"""
Route Determination Script
Determines the START and END points of a route based on entrance/exit intersections
"""

import cv2
import numpy as np
from PIL import Image as PILImage, ImageDraw, ImageFont
import json
import argparse
import math
from pathlib import Path


class RouteDeterminer:
    def __init__(self, annotations_file):
        """Initialize with annotations for entrance/exit detection."""
        self.annotations_file = annotations_file
        
        # Load annotations
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        
        self.entrances = self.annotations.get('entrances', [])
        self.exits = self.annotations.get('exits', [])
        self.walls = self.annotations.get('walls', [])
        
        # Calculate museum bounds
        self.museum_bounds = self._calculate_museum_bounds()
        
        print(f"Loaded annotations:")
        print(f"  Entrances: {len(self.entrances)}")
        print(f"  Exits: {len(self.exits)}")
        if self.museum_bounds:
            print(f"  Museum bounds: X=[{self.museum_bounds['min_x']:.0f}, {self.museum_bounds['max_x']:.0f}], "
                  f"Y=[{self.museum_bounds['min_y']:.0f}, {self.museum_bounds['max_y']:.0f}]")
    
    def _calculate_museum_bounds(self):
        """Calculate bounding box from all annotations."""
        if not self.walls:
            return None
        
        all_x = []
        all_y = []
        
        for wall in self.walls:
            if wall['shape'] == 'polyline':
                for point in wall['coordinates']['points']:
                    all_x.append(point['x'])
                    all_y.append(point['y'])
        
        if not all_x or not all_y:
            return None
        
        return {
            'min_x': min(all_x),
            'max_x': max(all_x),
            'min_y': min(all_y),
            'max_y': max(all_y)
        }
    
    def _distance(self, p1, p2):
        """Calculate Euclidean distance between two points."""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def _skeletonize(self, mask):
        """Apply skeletonization to get route centerline."""
        skeleton = np.zeros(mask.shape, np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        temp_mask = mask.copy()
        
        iteration = 0
        while True:
            eroded = cv2.erode(temp_mask, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(temp_mask, temp)
            skeleton = cv2.bitwise_or(skeleton, temp)
            temp_mask = eroded.copy()
            
            iteration += 1
            if cv2.countNonZero(temp_mask) == 0:
                break
        
        print(f"    Skeletonization completed in {iteration} iterations")
        return skeleton
    
    def _point_in_rectangle(self, point, rect_coords):
        """
        Check if a point is inside a rectangle.
        
        Args:
            point: (x, y) tuple
            rect_coords: Dictionary with x, y, width, height
        
        Returns:
            True if point is inside rectangle
        """
        px, py = point
        x, y = rect_coords['x'], rect_coords['y']
        width, height = rect_coords['width'], rect_coords['height']
        
        return (x <= px <= x + width) and (y <= py <= y + height)
    
    def _find_route_endpoints_by_boxes(self, points):
        """
        Find START and END points based on entrance/exit bounding boxes.
        
        Args:
            points: List of (x, y) route pixels
        
        Returns:
            Dictionary with 'start' and 'end' points, or None if not found
        """
        print(f"\n[Finding Route Endpoints]")
        print(f"  Total route pixels: {len(points)}")
        
        result = {
            'start': None,
            'end': None,
            'start_candidates': [],
            'end_candidates': []
        }
        
        # Find entrance points
        if not self.entrances:
            print("  ⚠️  WARNING: No entrance annotations found!")
            return result
        
        entrance = self.entrances[0]
        if entrance['shape'] != 'rectangle':
            print("  ⚠️  WARNING: Entrance is not a rectangle!")
            return result
        
        entrance_coords = entrance['coordinates']
        entrance_center = (
            entrance_coords['x'] + entrance_coords['width'] / 2,
            entrance_coords['y'] + entrance_coords['height'] / 2
        )
        
        print(f"\n  Entrance box: X=[{entrance_coords['x']:.0f}, {entrance_coords['x'] + entrance_coords['width']:.0f}], "
              f"Y=[{entrance_coords['y']:.0f}, {entrance_coords['y'] + entrance_coords['height']:.0f}]")
        
        # Find all route points within entrance box
        entrance_points = []
        for point in points:
            if self._point_in_rectangle(point, entrance_coords):
                entrance_points.append(point)
        
        print(f"  Route pixels in entrance box: {len(entrance_points)}")
        
        if entrance_points:
            # Choose the point closest to entrance center as START
            start_point = min(entrance_points, key=lambda p: self._distance(p, entrance_center))
            result['start'] = start_point
            result['start_candidates'] = entrance_points
            print(f"  ✅ START point: {start_point} (closest to entrance center)")
        else:
            print(f"  ❌ No route pixels found in entrance box!")
        
        # Find exit points
        if not self.exits:
            print(f"\n  ⚠️  WARNING: No exit annotations found!")
            return result
        
        exit_ann = self.exits[0]
        if exit_ann['shape'] != 'rectangle':
            print("  ⚠️  WARNING: Exit is not a rectangle!")
            return result
        
        exit_coords = exit_ann['coordinates']
        exit_center = (
            exit_coords['x'] + exit_coords['width'] / 2,
            exit_coords['y'] + exit_coords['height'] / 2
        )
        
        print(f"\n  Exit box: X=[{exit_coords['x']:.0f}, {exit_coords['x'] + exit_coords['width']:.0f}], "
              f"Y=[{exit_coords['y']:.0f}, {exit_coords['y'] + exit_coords['height']:.0f}]")
        
        # Find all route points within exit box
        exit_points = []
        for point in points:
            if self._point_in_rectangle(point, exit_coords):
                exit_points.append(point)
        
        print(f"  Route pixels in exit box: {len(exit_points)}")
        
        if exit_points:
            # Choose the point closest to exit center as END
            end_point = min(exit_points, key=lambda p: self._distance(p, exit_center))
            result['end'] = end_point
            result['end_candidates'] = exit_points
            print(f"  ✅ END point: {end_point} (closest to exit center)")
        else:
            print(f"  ❌ No route pixels found in exit box!")
        
        return result
    
    def determine_route(self, route_image_path, original_image_path, 
                       difference_threshold=10, output_path='route_endpoints.png',
                       marker_size=10):
        """
        Determine route START and END points and create visualization.
        
        Args:
            route_image_path: Path to image with drawn route
            original_image_path: Path to original floor plan
            difference_threshold: Pixel difference threshold
            output_path: Path to save visualization
            marker_size: Size of START/END marker circles
        """
        print("\n" + "="*70)
        print("ROUTE DETERMINATION")
        print("="*70)
        
        # Load images
        print("\n[1/6] Loading images...")
        route_pil = PILImage.open(str(route_image_path))
        original_pil = PILImage.open(str(original_image_path))
        
        # Convert to RGB
        if route_pil.mode == 'RGBA':
            route_pil = route_pil.convert('RGB')
        if original_pil.mode == 'RGBA':
            original_pil = original_pil.convert('RGB')
        
        route_img = np.array(route_pil)
        original_img = np.array(original_pil)
        
        # Convert to BGR for OpenCV
        route_img = cv2.cvtColor(route_img, cv2.COLOR_RGB2BGR)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
        
        print(f"    Route image: {route_img.shape[:2]} (H x W)")
        print(f"    Original image: {original_img.shape[:2]} (H x W)")
        
        # Compute difference
        print("\n[2/6] Computing image difference...")
        difference = cv2.absdiff(route_img, original_img)
        gray_diff = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold
        print(f"\n[3/6] Applying threshold (threshold={difference_threshold})...")
        _, mask_threshold = cv2.threshold(gray_diff, difference_threshold, 255, cv2.THRESH_BINARY)
        pixel_count = cv2.countNonZero(mask_threshold)
        print(f"    Detected {pixel_count} changed pixels")
        
        # Apply ROI mask if available
        mask_roi = mask_threshold.copy()
        if self.museum_bounds:
            print(f"\n[4/6] Applying museum ROI mask...")
            roi_mask = np.zeros(mask_threshold.shape, dtype=np.uint8)
            
            left_margin = 200
            right_margin = 0
            top_margin = 200
            bottom_margin = 200
            
            x_min = max(0, int(self.museum_bounds['min_x'] - left_margin))
            x_max = min(mask_threshold.shape[1], int(self.museum_bounds['max_x'] + right_margin))
            y_min = max(0, int(self.museum_bounds['min_y'] - top_margin))
            y_max = min(mask_threshold.shape[0], int(self.museum_bounds['max_y'] + bottom_margin))
            
            roi_mask[y_min:y_max, x_min:x_max] = 255
            mask_roi = cv2.bitwise_and(mask_threshold, roi_mask)
            roi_pixel_count = cv2.countNonZero(mask_roi)
            
            print(f"    After ROI: {roi_pixel_count} pixels")
        else:
            print("\n[4/6] Skipping ROI mask (no museum bounds)")
        
        # Morphological operations
        print("\n[5/6] Applying morphological operations...")
        kernel_small = np.ones((2, 2), np.uint8)
        mask_closed = cv2.morphologyEx(mask_roi, cv2.MORPH_CLOSE, kernel_small, iterations=1)
        mask_opened = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel_small, iterations=1)
        morph_pixel_count = cv2.countNonZero(mask_opened)
        print(f"    After morphology: {morph_pixel_count} pixels")
        
        # Skeletonization
        print("\n[6/6] Skeletonizing route...")
        skeleton = self._skeletonize(mask_opened)
        skeleton_pixel_count = cv2.countNonZero(skeleton)
        print(f"    Skeleton: {skeleton_pixel_count} pixels (centerline)")
        
        # Extract points
        points = np.column_stack(np.where(skeleton > 0))
        if len(points) == 0:
            print("ERROR: No skeleton points found!")
            return
        
        points = [(p[1], p[0]) for p in points]  # Convert to (x, y)
        print(f"    Extracted {len(points)} skeleton points")
        
        # Find endpoints based on entrance/exit boxes
        endpoints = self._find_route_endpoints_by_boxes(points)
        
        # Create visualization
        print("\n" + "="*70)
        print("Creating visualization...")
        print("="*70)
        
        self._create_visualization(
            route_img, skeleton, points,
            endpoints, output_path, marker_size
        )
        
        # Print summary
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        if endpoints['start']:
            print(f"✅ START: {endpoints['start']} (in entrance box)")
        else:
            print(f"❌ START: Not found (no route pixels in entrance box)")
        
        if endpoints['end']:
            print(f"✅ END: {endpoints['end']} (in exit box)")
        else:
            print(f"❌ END: Not found (no route pixels in exit box)")
        
        print("="*70)
    
    def _create_visualization(self, route_img, skeleton, points, 
                             endpoints, output_path, marker_size=10):
        """Create visualization showing detected START and END points."""
        
        # Convert skeleton to RGB for visualization
        def mask_to_rgb(mask, color_false=(30,30,30), color_true=(100,100,100)):
            rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
            rgb[mask > 0] = color_true
            rgb[mask == 0] = color_false
            return rgb
        
        # Prepare image
        h, w = route_img.shape[:2]
        scale = 1.0
        max_dim = 1200
        if h > max_dim or w > max_dim:
            scale = max_dim / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
        else:
            new_h, new_w = h, w
        
        def resize_img(img):
            if scale != 1.0:
                return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            return img
        
        # Use route image as base (convert BGR to RGB for display)
        vis_img = resize_img(route_img.copy())
        
        # Overlay skeleton in dark blue
        skeleton_resized = resize_img(skeleton)
        dark_blue = np.array([139, 0, 0], dtype=np.uint8)  # Dark blue in BGR
        vis_img[skeleton_resized > 0] = dark_blue
        
        # Draw START point in bright green
        if endpoints['start']:
            marker_scaled = int(marker_size * scale)
            start_scaled = (int(endpoints['start'][0] * scale), int(endpoints['start'][1] * scale))
            cv2.circle(vis_img, start_scaled, marker_scaled, (0, 255, 0), -1)
            cv2.circle(vis_img, start_scaled, marker_scaled + 2, (255, 255, 255), 2)
            cv2.putText(vis_img, "START", (start_scaled[0]+marker_scaled+5, start_scaled[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw END point in bright red
        if endpoints['end']:
            marker_scaled = int(marker_size * scale)
            end_scaled = (int(endpoints['end'][0] * scale), int(endpoints['end'][1] * scale))
            cv2.circle(vis_img, end_scaled, marker_scaled, (0, 0, 255), -1)
            cv2.circle(vis_img, end_scaled, marker_scaled + 2, (255, 255, 255), 2)
            cv2.putText(vis_img, "END", (end_scaled[0]+marker_scaled+5, end_scaled[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw entrance box (green)
        if self.entrances:
            for entrance in self.entrances:
                if entrance['shape'] == 'rectangle':
                    coords = entrance['coordinates']
                    x1 = int(coords['x'] * scale)
                    y1 = int(coords['y'] * scale)
                    x2 = int((coords['x'] + coords['width']) * scale)
                    y2 = int((coords['y'] + coords['height']) * scale)
                    cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(vis_img, "ENTRANCE", (x1+5, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw exit box (cyan)
        if self.exits:
            for exit_ann in self.exits:
                if exit_ann['shape'] == 'rectangle':
                    coords = exit_ann['coordinates']
                    x1 = int(coords['x'] * scale)
                    y1 = int(coords['y'] * scale)
                    x2 = int((coords['x'] + coords['width']) * scale)
                    y2 = int((coords['y'] + coords['height']) * scale)
                    cv2.rectangle(vis_img, (x1, y1), (x2, y2), (255, 255, 0), 2)
                    cv2.putText(vis_img, "EXIT", (x1+5, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # Add title and legend
        legend_space = np.ones((150, vis_img.shape[1], 3), dtype=np.uint8) * 40
        final = np.vstack([vis_img, legend_space])
        
        # Convert to PIL for text
        final_pil = PILImage.fromarray(cv2.cvtColor(final, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(final_pil)
        
        try:
            title_font = ImageFont.truetype("arial.ttf", 32)
            text_font = ImageFont.truetype("arial.ttf", 20)
        except:
            title_font = ImageFont.load_default()
            text_font = ImageFont.load_default()
        
        # Title
        title = "Route Endpoint Determination"
        draw.text((20, vis_img.shape[0] + 10), title, fill=(255, 255, 0), font=title_font)
        
        # Legend
        legend_y = vis_img.shape[0] + 60
        legend_x = 20
        
        draw.ellipse([legend_x, legend_y, legend_x+20, legend_y+20], fill=(0, 255, 0))
        draw.text((legend_x+30, legend_y), "= START (route pixel in entrance box)", 
                 fill=(255, 255, 255), font=text_font)
        
        draw.ellipse([legend_x, legend_y+30, legend_x+20, legend_y+50], fill=(255, 0, 0))
        draw.text((legend_x+30, legend_y+30), "= END (route pixel in exit box)", 
                 fill=(255, 255, 255), font=text_font)
        
        draw.rectangle([legend_x, legend_y+60, legend_x+20, legend_y+80], outline=(0, 255, 0), width=2)
        draw.text((legend_x+30, legend_y+60), "= Entrance box", 
                 fill=(255, 255, 255), font=text_font)
        
        draw.rectangle([legend_x+250, legend_y+60, legend_x+270, legend_y+80], outline=(255, 255, 0), width=2)
        draw.text((legend_x+280, legend_y+60), "= Exit box", 
                 fill=(255, 255, 255), font=text_font)
        
        # Save
        final_pil.save(output_path)
        print(f"\n✅ Visualization saved to: {output_path}")
        print(f"   Image size: {final_pil.size[0]} x {final_pil.size[1]} pixels")


def main():
    parser = argparse.ArgumentParser(
        description='Determine route START and END points based on entrance/exit intersections',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python determine_route.py \\
    --route-image valid_route.png \\
    --original-image layout_entrance_exit.png \\
    --annotations museum_layout_annotations.json \\
    --output route_endpoints.png
        """
    )
    
    parser.add_argument('--route-image', required=True, help='Image with drawn route')
    parser.add_argument('--original-image', required=True, help='Original floor plan (no route)')
    parser.add_argument('--annotations', required=True, help='JSON file with annotations')
    parser.add_argument('--difference-threshold', type=int, default=10,
                       help='Pixel difference threshold (default: 10)')
    parser.add_argument('--output', default='route_endpoints.png',
                       help='Output visualization image (default: route_endpoints.png)')
    parser.add_argument('--marker-size', type=int, default=15,
                       help='Size of START/END marker circles in pixels (default: 15)')
    
    args = parser.parse_args()
    
    # Create determiner
    determiner = RouteDeterminer(args.annotations)
    
    # Determine route endpoints
    determiner.determine_route(
        route_image_path=args.route_image,
        original_image_path=args.original_image,
        difference_threshold=args.difference_threshold,
        output_path=args.output,
        marker_size=args.marker_size
    )


if __name__ == '__main__':
    main()
