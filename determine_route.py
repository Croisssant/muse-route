"""
Route Determination Script
Determines the START and END points of a route based on entrance/exit intersections.

Supports route images with different dimensions than the original by aligning them
via homography computed from the detected Entrance (green) and Exit (yellow) bounding boxes.
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

        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)

        self.entrances = self.annotations.get('entrances', [])
        self.exits = self.annotations.get('exits', [])
        self.walls = self.annotations.get('walls', [])

        self.museum_bounds = self._calculate_museum_bounds()

        print(f"Loaded annotations:")
        print(f"  Entrances: {len(self.entrances)}")
        print(f"  Exits: {len(self.exits)}")
        if self.museum_bounds:
            print(f"  Museum bounds: X=[{self.museum_bounds['min_x']:.0f}, {self.museum_bounds['max_x']:.0f}], "
                  f"Y=[{self.museum_bounds['min_y']:.0f}, {self.museum_bounds['max_y']:.0f}]")

    def _calculate_museum_bounds(self):
        """Calculate bounding box from all wall annotations."""
        if not self.walls:
            return None

        all_x, all_y = [], []
        for wall in self.walls:
            if wall['shape'] == 'polyline':
                for point in wall['coordinates']['points']:
                    all_x.append(point['x'])
                    all_y.append(point['y'])

        if not all_x or not all_y:
            return None

        return {
            'min_x': min(all_x), 'max_x': max(all_x),
            'min_y': min(all_y), 'max_y': max(all_y)
        }

    def _distance(self, p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def _rect_corners(self, coords):
        """Return the 4 corners of a rectangle annotation as float32 array (TL, TR, BR, BL)."""
        x, y, w, h = coords['x'], coords['y'], coords['width'], coords['height']
        return np.array([
            [x,     y    ],
            [x + w, y    ],
            [x + w, y + h],
            [x,     y + h],
        ], dtype=np.float32)

    # ------------------------------------------------------------------
    # Alignment helpers
    # ------------------------------------------------------------------

    def _detect_colored_box(self, img_bgr, color: str):
        """
        Detect the bounding rectangle of a solid-colored box in the image.

        Args:
            img_bgr: BGR image (numpy array)
            color: 'green' (entrance) or 'yellow' (exit)

        Returns:
            numpy float32 array of shape (4, 2) with corners [TL, TR, BR, BL],
            or None if not detected.
        """
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

        if color == 'green':
            # Broad green range
            lower = np.array([35, 60, 60])
            upper = np.array([85, 255, 255])
            mask = cv2.inRange(hsv, lower, upper)
        elif color == 'yellow':
            # Broad yellow range
            lower = np.array([20, 80, 80])
            upper = np.array([35, 255, 255])
            mask = cv2.inRange(hsv, lower, upper)
        else:
            raise ValueError(f"Unsupported color: {color}")

        # Clean up small noise
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # Take the largest contour
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 100:   # sanity: ignore tiny blobs
            return None

        bx, by, bw, bh = cv2.boundingRect(largest)
        return np.array([
            [bx,      by     ],
            [bx + bw, by     ],
            [bx + bw, by + bh],
            [bx,      by + bh],
        ], dtype=np.float32)

    def _compute_alignment_homography(self, route_img_bgr):
        """
        Compute a homography H that maps route_img pixel coords →
        annotation/original coordinate space, using the entrance (green)
        and exit (yellow) boxes as anchors.

        Returns:
            H (3×3 numpy array) or None if alignment cannot be computed.
        """
        if not self.entrances or not self.exits:
            print("  ⚠️  Cannot align: missing entrance or exit annotations.")
            return None

        entrance_ann = self.entrances[0]
        exit_ann     = self.exits[0]

        if entrance_ann['shape'] != 'rectangle' or exit_ann['shape'] != 'rectangle':
            print("  ⚠️  Cannot align: entrance/exit must be rectangles.")
            return None

        # Annotation-space corners (destination)
        dst_entrance = self._rect_corners(entrance_ann['coordinates'])
        dst_exit      = self._rect_corners(exit_ann['coordinates'])

        # Detected corners in route image (source)
        src_entrance = self._detect_colored_box(route_img_bgr, 'green')
        src_exit      = self._detect_colored_box(route_img_bgr, 'yellow')

        if src_entrance is None:
            print("  ⚠️  Could not detect green (entrance) box in route image.")
        if src_exit is None:
            print("  ⚠️  Could not detect yellow (exit) box in route image.")

        # Build point correspondences
        src_pts, dst_pts = [], []

        if src_entrance is not None:
            src_pts.append(src_entrance)
            dst_pts.append(dst_entrance)

        if src_exit is not None:
            src_pts.append(src_exit)
            dst_pts.append(dst_exit)

        if not src_pts:
            print("  ⚠️  No landmark boxes detected; falling back to simple resize.")
            return None

        src_pts = np.vstack(src_pts)   # (N*4, 2)
        dst_pts = np.vstack(dst_pts)

        if len(src_pts) >= 4:
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if H is None:
                print("  ⚠️  Homography estimation failed.")
            else:
                inliers = int(mask.sum()) if mask is not None else len(src_pts)
                print(f"  ✅ Homography computed ({inliers}/{len(src_pts)} inliers).")
            return H
        else:
            # Only one box → affine from 3 points (use 3 of the 4 corners)
            H_affine, _ = cv2.estimateAffinePartial2D(src_pts[:3], dst_pts[:3])
            if H_affine is None:
                print("  ⚠️  Affine estimation failed.")
                return None
            # Promote 2×3 affine → 3×3 homography
            H = np.vstack([H_affine, [0, 0, 1]])
            print("  ✅ Affine alignment computed (single landmark).")
            return H

    def _align_route_image(self, route_img_bgr, original_shape):
        """
        Warp route_img into the coordinate space of the original/annotations.

        Strategy:
          1. Try homography from colored landmark boxes.
          2. Fall back to simple resize if landmarks are not found.

        Args:
            route_img_bgr:  The route image in BGR.
            original_shape: (H, W) of the original floor-plan image.

        Returns:
            (aligned_bgr, H_or_None)  – aligned image and the transform used.
        """
        orig_h, orig_w = original_shape[:2]

        print("\n[Alignment] Attempting landmark-based alignment...")
        H = self._compute_alignment_homography(route_img_bgr)

        if H is not None:
            aligned = cv2.warpPerspective(route_img_bgr, H, (orig_w, orig_h),
                                          flags=cv2.INTER_LINEAR,
                                          borderMode=cv2.BORDER_CONSTANT,
                                          borderValue=(255, 255, 255))
            print(f"  Route image warped to {aligned.shape[:2]} (H x W).")
            return aligned, H
        else:
            # Fallback: plain resize
            print(f"  Falling back to resize: {route_img_bgr.shape[:2]} → {original_shape[:2]}")
            aligned = cv2.resize(route_img_bgr, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            return aligned, None

    # ------------------------------------------------------------------
    # Existing helpers (unchanged)
    # ------------------------------------------------------------------

    def _skeletonize(self, mask):
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
        px, py = point
        x, y = rect_coords['x'], rect_coords['y']
        w, h = rect_coords['width'], rect_coords['height']
        return (x <= px <= x + w) and (y <= py <= y + h)

    def _find_route_endpoints_by_boxes(self, points):
        print(f"\n[Finding Route Endpoints]")
        print(f"  Total route pixels: {len(points)}")

        result = {
            'start': None, 'end': None,
            'start_candidates': [], 'end_candidates': []
        }

        if not self.entrances:
            print("  ⚠️  WARNING: No entrance annotations found!")
            return result

        entrance = self.entrances[0]
        if entrance['shape'] != 'rectangle':
            print("  ⚠️  WARNING: Entrance is not a rectangle!")
            return result

        entrance_coords = entrance['coordinates']
        entrance_center = (
            entrance_coords['x'] + entrance_coords['width']  / 2,
            entrance_coords['y'] + entrance_coords['height'] / 2,
        )

        print(f"\n  Entrance box: X=[{entrance_coords['x']:.0f}, "
              f"{entrance_coords['x'] + entrance_coords['width']:.0f}], "
              f"Y=[{entrance_coords['y']:.0f}, "
              f"{entrance_coords['y'] + entrance_coords['height']:.0f}]")

        entrance_points = [p for p in points if self._point_in_rectangle(p, entrance_coords)]
        print(f"  Route pixels in entrance box: {len(entrance_points)}")

        if entrance_points:
            start_point = min(entrance_points, key=lambda p: self._distance(p, entrance_center))
            result['start'] = start_point
            result['start_candidates'] = entrance_points
            print(f"  ✅ START point: {start_point}")
        else:
            print(f"  ❌ No route pixels found in entrance box!")

        if not self.exits:
            print(f"\n  ⚠️  WARNING: No exit annotations found!")
            return result

        exit_ann = self.exits[0]
        if exit_ann['shape'] != 'rectangle':
            print("  ⚠️  WARNING: Exit is not a rectangle!")
            return result

        exit_coords = exit_ann['coordinates']
        exit_center = (
            exit_coords['x'] + exit_coords['width']  / 2,
            exit_coords['y'] + exit_coords['height'] / 2,
        )

        print(f"\n  Exit box: X=[{exit_coords['x']:.0f}, "
              f"{exit_coords['x'] + exit_coords['width']:.0f}], "
              f"Y=[{exit_coords['y']:.0f}, "
              f"{exit_coords['y'] + exit_coords['height']:.0f}]")

        exit_points = [p for p in points if self._point_in_rectangle(p, exit_coords)]
        print(f"  Route pixels in exit box: {len(exit_points)}")

        if exit_points:
            end_point = min(exit_points, key=lambda p: self._distance(p, exit_center))
            result['end'] = end_point
            result['end_candidates'] = exit_points
            print(f"  ✅ END point: {end_point}")
        else:
            print(f"  ❌ No route pixels found in exit box!")

        return result

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def determine_route(self, route_image_path, original_image_path,
                        difference_threshold=10, output_path='route_endpoints.png',
                        marker_size=10):
        """
        Determine route START and END points and create visualization.

        The route image may have different dimensions than the original; alignment
        is handled automatically via the Entrance (green) and Exit (yellow) boxes.
        """
        print("\n" + "="*70)
        print("ROUTE DETERMINATION")
        print("="*70)

        # ── Load images ────────────────────────────────────────────────
        print("\n[1/7] Loading images...")
        route_pil    = PILImage.open(str(route_image_path))
        original_pil = PILImage.open(str(original_image_path))

        for pil_img in (route_pil, original_pil):
            if pil_img.mode == 'RGBA':
                pil_img = pil_img.convert('RGB')

        route_img    = cv2.cvtColor(np.array(route_pil.convert('RGB')),    cv2.COLOR_RGB2BGR)
        original_img = cv2.cvtColor(np.array(original_pil.convert('RGB')), cv2.COLOR_RGB2BGR)

        print(f"    Route image:    {route_img.shape[:2]} (H x W)")
        print(f"    Original image: {original_img.shape[:2]} (H x W)")

        # ── Align route image to original coordinate space ─────────────
        print("\n[2/7] Aligning route image to original coordinate space...")
        same_size = (route_img.shape[:2] == original_img.shape[:2])
        if same_size:
            print("    Images are the same size — skipping alignment.")
            aligned_route = route_img
        else:
            aligned_route, _ = self._align_route_image(route_img, original_img.shape)

        # ── Compute difference ─────────────────────────────────────────
        print("\n[3/7] Computing image difference...")
        difference = cv2.absdiff(aligned_route, original_img)
        gray_diff  = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)

        # ── Threshold ──────────────────────────────────────────────────
        print(f"\n[4/7] Applying threshold (threshold={difference_threshold})...")
        _, mask_threshold = cv2.threshold(gray_diff, difference_threshold, 255, cv2.THRESH_BINARY)
        print(f"    Detected {cv2.countNonZero(mask_threshold)} changed pixels")

        # ── ROI mask ───────────────────────────────────────────────────
        mask_roi = mask_threshold.copy()
        if self.museum_bounds:
            print(f"\n[5/7] Applying museum ROI mask...")
            roi_mask = np.zeros(mask_threshold.shape, dtype=np.uint8)

            left_margin, right_margin = 200, 0
            top_margin,  bottom_margin = 200, 200

            x_min = max(0, int(self.museum_bounds['min_x'] - left_margin))
            x_max = min(mask_threshold.shape[1], int(self.museum_bounds['max_x'] + right_margin))
            y_min = max(0, int(self.museum_bounds['min_y'] - top_margin))
            y_max = min(mask_threshold.shape[0], int(self.museum_bounds['max_y'] + bottom_margin))

            roi_mask[y_min:y_max, x_min:x_max] = 255
            mask_roi = cv2.bitwise_and(mask_threshold, roi_mask)
            print(f"    After ROI: {cv2.countNonZero(mask_roi)} pixels")
        else:
            print("\n[5/7] Skipping ROI mask (no museum bounds)")

        # ── Morphological clean-up ─────────────────────────────────────
        print("\n[6/7] Applying morphological operations...")
        kernel = np.ones((2, 2), np.uint8)
        mask_closed = cv2.morphologyEx(mask_roi,    cv2.MORPH_CLOSE, kernel, iterations=1)
        mask_opened = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN,  kernel, iterations=1)
        print(f"    After morphology: {cv2.countNonZero(mask_opened)} pixels")

        # ── Skeletonize ────────────────────────────────────────────────
        print("\n[7/7] Skeletonizing route...")
        skeleton = self._skeletonize(mask_opened)
        print(f"    Skeleton: {cv2.countNonZero(skeleton)} pixels (centerline)")

        # ── Extract points & find endpoints ───────────────────────────
        raw_pts = np.column_stack(np.where(skeleton > 0))
        if len(raw_pts) == 0:
            print("ERROR: No skeleton points found!")
            return

        points = [(p[1], p[0]) for p in raw_pts]   # → (x, y)
        print(f"    Extracted {len(points)} skeleton points")

        endpoints = self._find_route_endpoints_by_boxes(points)

        # ── Visualize ─────────────────────────────────────────────────
        print("\n" + "="*70)
        print("Creating visualization...")
        print("="*70)

        # Use the *aligned* route image as background so annotation boxes
        # are visually consistent with the drawn route.
        self._create_visualization(aligned_route, skeleton, points, endpoints, output_path, marker_size)

        # ── Summary ───────────────────────────────────────────────────
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        if endpoints['start']:
            print(f"✅ START: {endpoints['start']} (in entrance box)")
        else:
            print(f"❌ START: Not found (no route pixels in entrance box)")
        if endpoints['end']:
            print(f"✅ END:   {endpoints['end']} (in exit box)")
        else:
            print(f"❌ END:   Not found (no route pixels in exit box)")
        print("="*70)

    # ------------------------------------------------------------------
    # Visualization (unchanged from original)
    # ------------------------------------------------------------------

    def _create_visualization(self, route_img, skeleton, points,
                               endpoints, output_path, marker_size=10):
        h, w = route_img.shape[:2]
        scale = 1.0
        max_dim = 1200
        if h > max_dim or w > max_dim:
            scale = max_dim / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)

        def resize_img(img):
            return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA) if scale != 1.0 else img

        vis_img = resize_img(route_img.copy())

        skeleton_resized = resize_img(skeleton)
        dark_blue = np.array([139, 0, 0], dtype=np.uint8)
        vis_img[skeleton_resized > 0] = dark_blue

        if endpoints['start']:
            ms = int(marker_size * scale)
            sp = (int(endpoints['start'][0] * scale), int(endpoints['start'][1] * scale))
            cv2.circle(vis_img, sp, ms, (0, 255, 0), -1)
            cv2.circle(vis_img, sp, ms + 2, (255, 255, 255), 2)
            cv2.putText(vis_img, "START", (sp[0] + ms + 5, sp[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if endpoints['end']:
            ms = int(marker_size * scale)
            ep = (int(endpoints['end'][0] * scale), int(endpoints['end'][1] * scale))
            cv2.circle(vis_img, ep, ms, (0, 0, 255), -1)
            cv2.circle(vis_img, ep, ms + 2, (255, 255, 255), 2)
            cv2.putText(vis_img, "END", (ep[0] + ms + 5, ep[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if self.entrances:
            for entrance in self.entrances:
                if entrance['shape'] == 'rectangle':
                    c = entrance['coordinates']
                    x1, y1 = int(c['x'] * scale), int(c['y'] * scale)
                    x2, y2 = int((c['x'] + c['width']) * scale), int((c['y'] + c['height']) * scale)
                    cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(vis_img, "ENTRANCE", (x1 + 5, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if self.exits:
            for exit_ann in self.exits:
                if exit_ann['shape'] == 'rectangle':
                    c = exit_ann['coordinates']
                    x1, y1 = int(c['x'] * scale), int(c['y'] * scale)
                    x2, y2 = int((c['x'] + c['width']) * scale), int((c['y'] + c['height']) * scale)
                    cv2.rectangle(vis_img, (x1, y1), (x2, y2), (255, 255, 0), 2)
                    cv2.putText(vis_img, "EXIT", (x1 + 5, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        legend_space = np.ones((150, vis_img.shape[1], 3), dtype=np.uint8) * 40
        final = np.vstack([vis_img, legend_space])
        final_pil = PILImage.fromarray(cv2.cvtColor(final, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(final_pil)

        try:
            title_font = ImageFont.truetype("arial.ttf", 32)
            text_font  = ImageFont.truetype("arial.ttf", 20)
        except Exception:
            title_font = ImageFont.load_default()
            text_font  = ImageFont.load_default()

        draw.text((20, vis_img.shape[0] + 10), "Route Endpoint Determination",
                  fill=(255, 255, 0), font=title_font)

        ly, lx = vis_img.shape[0] + 60, 20
        draw.ellipse([lx, ly, lx+20, ly+20], fill=(0, 255, 0))
        draw.text((lx+30, ly), "= START (route pixel in entrance box)", fill=(255,255,255), font=text_font)
        draw.ellipse([lx, ly+30, lx+20, ly+50], fill=(255, 0, 0))
        draw.text((lx+30, ly+30), "= END (route pixel in exit box)", fill=(255,255,255), font=text_font)
        draw.rectangle([lx, ly+60, lx+20, ly+80], outline=(0, 255, 0), width=2)
        draw.text((lx+30, ly+60), "= Entrance box", fill=(255,255,255), font=text_font)
        draw.rectangle([lx+250, ly+60, lx+270, ly+80], outline=(255, 255, 0), width=2)
        draw.text((lx+280, ly+60), "= Exit box", fill=(255,255,255), font=text_font)

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
    parser.add_argument('--route-image',          required=True, help='Image with drawn route')
    parser.add_argument('--original-image',        required=True, help='Original floor plan (no route)')
    parser.add_argument('--annotations',           required=True, help='JSON file with annotations')
    parser.add_argument('--difference-threshold',  type=int, default=10,
                        help='Pixel difference threshold (default: 10)')
    parser.add_argument('--output',                default='route_endpoints.png',
                        help='Output visualization image (default: route_endpoints.png)')
    parser.add_argument('--marker-size',           type=int, default=15,
                        help='Size of START/END marker circles in pixels (default: 15)')

    args = parser.parse_args()

    determiner = RouteDeterminer(args.annotations)
    determiner.determine_route(
        route_image_path=args.route_image,
        original_image_path=args.original_image,
        difference_threshold=args.difference_threshold,
        output_path=args.output,
        marker_size=args.marker_size,
    )


if __name__ == '__main__':
    main()