"""
Route Determination Script
Determines the START and END points of a route based on entrance/exit intersections.

Alignment strategy (best → fallback):
  1. Exhibit circles   – up to 100 annotated circles spread across the full image,
                         detected via HoughCircles; gives dense, well-distributed
                         correspondences for a robust homography.
  2. Entrance / Exit boxes – 2 colored rectangles (green / yellow) used when too
                         few circle matches are found.
  3. Plain resize      – last resort if no landmarks are detected at all.
"""

import cv2
import numpy as np
from PIL import Image as PILImage, ImageDraw, ImageFont
import json
import argparse
import math
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
# Tunables
# ──────────────────────────────────────────────────────────────────────────────
MIN_CIRCLE_MATCHES   = 6    # minimum matched circles to trust the homography
CIRCLE_MATCH_RADIUS  = 30   # px – how close detected ↔ annotated centres must be
HOUGH_PARAM1         = 60   # Canny high threshold for HoughCircles
HOUGH_PARAM2         = 20   # accumulator threshold (lower → more detections)
HOUGH_MIN_DIST_SCALE = 0.8  # min-distance between circles = scale × median radius
# ──────────────────────────────────────────────────────────────────────────────


class RouteDeterminer:
    def __init__(self, annotations_file):
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)

        self.entrances = self.annotations.get('entrances', [])
        self.exits     = self.annotations.get('exits', [])
        self.walls     = self.annotations.get('walls', [])
        self.exhibits  = self.annotations.get('exhibits', [])

        self.museum_bounds = self._calculate_museum_bounds()

        print(f"Loaded annotations:")
        print(f"  Entrances : {len(self.entrances)}")
        print(f"  Exits     : {len(self.exits)}")
        print(f"  Exhibits  : {len(self.exhibits)}")
        if self.museum_bounds:
            print(f"  Museum bounds: X=[{self.museum_bounds['min_x']:.0f}, "
                  f"{self.museum_bounds['max_x']:.0f}], "
                  f"Y=[{self.museum_bounds['min_y']:.0f}, "
                  f"{self.museum_bounds['max_y']:.0f}]")

    # ──────────────────────────────────────────────────────────────────
    # Annotation helpers
    # ──────────────────────────────────────────────────────────────────

    def _calculate_museum_bounds(self):
        if not self.walls:
            return None
        all_x, all_y = [], []
        for wall in self.walls:
            if wall['shape'] == 'polyline':
                for pt in wall['coordinates']['points']:
                    all_x.append(pt['x'])
                    all_y.append(pt['y'])
        if not all_x:
            return None
        return dict(min_x=min(all_x), max_x=max(all_x),
                    min_y=min(all_y), max_y=max(all_y))

    def _distance(self, p1, p2):
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    def _rect_corners(self, coords):
        x, y, w, h = coords['x'], coords['y'], coords['width'], coords['height']
        return np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], dtype=np.float32)

    def _annotation_circles(self):
        """Return list of (cx, cy, r) from exhibit annotations."""
        circles = []
        for ex in self.exhibits:
            if ex['shape'] == 'circle':
                c = ex['coordinates']
                circles.append((c['center_x'], c['center_y'], c['radius']))
        return circles

    # ──────────────────────────────────────────────────────────────────
    # Circle detection in an image
    # ──────────────────────────────────────────────────────────────────

    def _detect_circles_in_image(self, img_bgr, ann_circles):
        """
        Detect circles in img_bgr whose radii are consistent with the
        annotated exhibit circles (scaled to img_bgr dimensions).

        Returns list of (cx, cy, r) detected circles.
        """
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if not ann_circles:
            return []

        radii    = [r for _, _, r in ann_circles]
        median_r = float(np.median(radii))

        # Allow ±60 % around the median to capture scale differences
        min_r    = max(5,  int(median_r * 0.4))
        max_r    = max(30, int(median_r * 1.6))
        min_dist = max(10, int(median_r * HOUGH_MIN_DIST_SCALE))

        detected = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=min_dist,
            param1=HOUGH_PARAM1,
            param2=HOUGH_PARAM2,
            minRadius=min_r,
            maxRadius=max_r,
        )

        if detected is None:
            return []

        return [(x, y, r) for x, y, r in detected[0]]

    # ──────────────────────────────────────────────────────────────────
    # Match detected circles → annotation circles
    # ──────────────────────────────────────────────────────────────────

    def _match_circles(self, detected, ann_circles, route_shape, ann_shape):
        """
        Match detected circles in the route image to annotated circles.

        Because the route image may be a different scale we first estimate a
        rough scale factor from image dimensions, apply it to the annotation
        coordinates, then do nearest-neighbour matching.

        Returns (src_pts, dst_pts) as float32 arrays of matched centres.
        """
        if not detected or not ann_circles:
            return None, None

        rh, rw = route_shape[:2]
        ah, aw = ann_shape[:2]

        sx = rw / aw if aw > 0 else 1.0
        sy = rh / ah if ah > 0 else 1.0

        src_pts, dst_pts = [], []

        for dx, dy, dr in detected:
            best_dist = float('inf')
            best_ann  = None
            for ax, ay, ar in ann_circles:
                scaled_ax = ax * sx
                scaled_ay = ay * sy
                dist = math.sqrt((dx - scaled_ax)**2 + (dy - scaled_ay)**2)
                threshold = max(CIRCLE_MATCH_RADIUS, dr * 1.0)
                if dist < best_dist and dist < threshold:
                    best_dist = dist
                    best_ann  = (ax, ay)

            if best_ann is not None:
                src_pts.append([dx, dy])
                dst_pts.append(list(best_ann))

        if not src_pts:
            return None, None

        return (np.array(src_pts, dtype=np.float32),
                np.array(dst_pts, dtype=np.float32))

    # ──────────────────────────────────────────────────────────────────
    # Colored-box detection (entrance / exit fallback)
    # ──────────────────────────────────────────────────────────────────

    def _detect_colored_box(self, img_bgr, color: str):
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        if color == 'green':
            mask = cv2.inRange(hsv, np.array([35,60,60]), np.array([85,255,255]))
        elif color == 'yellow':
            mask = cv2.inRange(hsv, np.array([20,80,80]), np.array([35,255,255]))
        else:
            raise ValueError(f"Unsupported color: {color}")

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 100:
            return None
        bx, by, bw, bh = cv2.boundingRect(largest)
        return np.array([[bx,bw+bx,bw+bx,bx],[by,by,bh+by,bh+by]], dtype=np.float32).T

    # ──────────────────────────────────────────────────────────────────
    # Master alignment
    # ──────────────────────────────────────────────────────────────────

    def _align_route_image(self, route_img_bgr, original_img_bgr):
        """
        Warp route_img_bgr into the coordinate space of original_img_bgr.

        Strategy
        --------
        1. Detect exhibit circles in the route image, match to annotation
           circles, compute homography.  Requires MIN_CIRCLE_MATCHES matches.
        2. Fall back to entrance (green) + exit (yellow) box corners.
        3. Last resort: plain resize.

        Returns (aligned_bgr, method_name).
        """
        orig_h, orig_w = original_img_bgr.shape[:2]
        ann_circles = self._annotation_circles()

        # ── Strategy 1: exhibit circles ───────────────────────────────
        if ann_circles:
            print("  [Alignment] Detecting exhibit circles in route image…")
            detected = self._detect_circles_in_image(route_img_bgr, ann_circles)
            print(f"             {len(detected)} circles detected in route image")

            ann_shape = original_img_bgr.shape
            src_pts, dst_pts = self._match_circles(
                detected, ann_circles, route_img_bgr.shape, ann_shape)

            if src_pts is not None and len(src_pts) >= MIN_CIRCLE_MATCHES:
                print(f"             {len(src_pts)} circles matched → computing homography")
                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if H is not None:
                    inliers = int(mask.sum()) if mask is not None else len(src_pts)
                    print(f"  ✅ Circle-based homography ({inliers}/{len(src_pts)} inliers)")
                    aligned = cv2.warpPerspective(
                        route_img_bgr, H, (orig_w, orig_h),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=(255, 255, 255))
                    return aligned, 'circles'
                else:
                    print("  ⚠️  Homography from circles failed; trying box fallback…")
            else:
                n = len(src_pts) if src_pts is not None else 0
                print(f"  ⚠️  Only {n} circle matches (need {MIN_CIRCLE_MATCHES}); "
                      f"trying box fallback…")

        # ── Strategy 2: entrance / exit colored boxes ─────────────────
        if self.entrances and self.exits:
            entrance_ann = self.entrances[0]
            exit_ann     = self.exits[0]
            if (entrance_ann['shape'] == 'rectangle' and
                    exit_ann['shape'] == 'rectangle'):

                src_entrance = self._detect_colored_box(route_img_bgr, 'green')
                src_exit     = self._detect_colored_box(route_img_bgr, 'yellow')
                dst_entrance = self._rect_corners(entrance_ann['coordinates'])
                dst_exit     = self._rect_corners(exit_ann['coordinates'])

                src_pts_b, dst_pts_b = [], []
                if src_entrance is not None:
                    src_pts_b.append(src_entrance)
                    dst_pts_b.append(dst_entrance)
                if src_exit is not None:
                    src_pts_b.append(src_exit)
                    dst_pts_b.append(dst_exit)

                if src_pts_b:
                    src_all = np.vstack(src_pts_b)
                    dst_all = np.vstack(dst_pts_b)
                    H, mask = cv2.findHomography(src_all, dst_all, cv2.RANSAC, 5.0)
                    if H is not None:
                        inliers = int(mask.sum()) if mask is not None else len(src_all)
                        print(f"  ✅ Box-based homography ({inliers}/{len(src_all)} inliers)")
                        aligned = cv2.warpPerspective(
                            route_img_bgr, H, (orig_w, orig_h),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(255, 255, 255))
                        return aligned, 'boxes'

        # ── Strategy 3: plain resize ───────────────────────────────────
        print("  ⚠️  Falling back to plain resize.")
        aligned = cv2.resize(route_img_bgr, (orig_w, orig_h),
                             interpolation=cv2.INTER_LINEAR)
        return aligned, 'resize'

    # ──────────────────────────────────────────────────────────────────
    # Skeletonization
    # ──────────────────────────────────────────────────────────────────

    def _skeletonize(self, mask):
        skeleton  = np.zeros(mask.shape, np.uint8)
        element   = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        temp_mask = mask.copy()
        iteration = 0
        while True:
            eroded   = cv2.erode(temp_mask, element)
            temp     = cv2.dilate(eroded, element)
            temp     = cv2.subtract(temp_mask, temp)
            skeleton = cv2.bitwise_or(skeleton, temp)
            temp_mask = eroded.copy()
            iteration += 1
            if cv2.countNonZero(temp_mask) == 0:
                break
        print(f"    Skeletonization completed in {iteration} iterations")
        return skeleton

    # ──────────────────────────────────────────────────────────────────
    # Endpoint detection
    # ──────────────────────────────────────────────────────────────────

    def _point_in_rectangle(self, point, rect_coords):
        px, py = point
        x, y   = rect_coords['x'], rect_coords['y']
        w, h   = rect_coords['width'], rect_coords['height']
        return (x <= px <= x + w) and (y <= py <= y + h)

    def _find_route_endpoints_by_boxes(self, points):
        print(f"\n[Finding Route Endpoints]")
        print(f"  Total route pixels: {len(points)}")

        result = {'start': None, 'end': None,
                  'start_candidates': [], 'end_candidates': []}

        if not self.entrances:
            print("  ⚠️  WARNING: No entrance annotations found!")
            return result

        entrance = self.entrances[0]
        if entrance['shape'] != 'rectangle':
            return result

        ec = entrance['coordinates']
        e_center = (ec['x'] + ec['width']/2, ec['y'] + ec['height']/2)
        e_pts = [p for p in points if self._point_in_rectangle(p, ec)]
        print(f"\n  Entrance box: X=[{ec['x']:.0f}, {ec['x']+ec['width']:.0f}], "
              f"Y=[{ec['y']:.0f}, {ec['y']+ec['height']:.0f}]")
        print(f"  Route pixels in entrance box: {len(e_pts)}")

        if e_pts:
            result['start'] = min(e_pts, key=lambda p: self._distance(p, e_center))
            result['start_candidates'] = e_pts
            print(f"  ✅ START: {result['start']}")
        else:
            print(f"  ❌ No route pixels in entrance box")

        if not self.exits:
            print("  ⚠️  WARNING: No exit annotations found!")
            return result

        exit_ann = self.exits[0]
        if exit_ann['shape'] != 'rectangle':
            return result

        xc = exit_ann['coordinates']
        x_center = (xc['x'] + xc['width']/2, xc['y'] + xc['height']/2)
        x_pts = [p for p in points if self._point_in_rectangle(p, xc)]
        print(f"\n  Exit box: X=[{xc['x']:.0f}, {xc['x']+xc['width']:.0f}], "
              f"Y=[{xc['y']:.0f}, {xc['y']+xc['height']:.0f}]")
        print(f"  Route pixels in exit box: {len(x_pts)}")

        if x_pts:
            result['end'] = min(x_pts, key=lambda p: self._distance(p, x_center))
            result['end_candidates'] = x_pts
            print(f"  ✅ END: {result['end']}")
        else:
            print(f"  ❌ No route pixels in exit box")

        return result

    # ──────────────────────────────────────────────────────────────────
    # Main entry point
    # ──────────────────────────────────────────────────────────────────

    def determine_route(self, route_image_path, original_image_path,
                        difference_threshold=10, output_path='route_endpoints.png',
                        marker_size=10):
        print("\n" + "="*70)
        print("ROUTE DETERMINATION")
        print("="*70)

        # ── Load ──────────────────────────────────────────────────────
        print("\n[1/7] Loading images…")
        route_img    = cv2.cvtColor(
            np.array(PILImage.open(str(route_image_path)).convert('RGB')),
            cv2.COLOR_RGB2BGR)
        original_img = cv2.cvtColor(
            np.array(PILImage.open(str(original_image_path)).convert('RGB')),
            cv2.COLOR_RGB2BGR)

        print(f"    Route image:    {route_img.shape[:2]} (H x W)")
        print(f"    Original image: {original_img.shape[:2]} (H x W)")

        # ── Align ─────────────────────────────────────────────────────
        print("\n[2/7] Aligning route image to original coordinate space…")
        same_size = (route_img.shape[:2] == original_img.shape[:2])
        if same_size:
            print("    Images are the same size — skipping alignment.")
            aligned_route    = route_img
            alignment_method = 'none'
        else:
            aligned_route, alignment_method = self._align_route_image(
                route_img, original_img)
            print(f"    Alignment method used: {alignment_method}")

        # ── Difference ────────────────────────────────────────────────
        print("\n[3/7] Computing image difference…")
        difference = cv2.absdiff(aligned_route, original_img)
        gray_diff  = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)

        # ── Threshold ─────────────────────────────────────────────────
        print(f"\n[4/7] Applying threshold (threshold={difference_threshold})…")
        _, mask_threshold = cv2.threshold(
            gray_diff, difference_threshold, 255, cv2.THRESH_BINARY)
        print(f"    Detected {cv2.countNonZero(mask_threshold)} changed pixels")

        # ── ROI ───────────────────────────────────────────────────────
        mask_roi = mask_threshold.copy()
        if self.museum_bounds:
            print(f"\n[5/7] Applying museum ROI mask…")
            roi_mask = np.zeros(mask_threshold.shape, dtype=np.uint8)
            x_min = max(0, int(self.museum_bounds['min_x'] - 200))
            x_max = min(mask_threshold.shape[1], int(self.museum_bounds['max_x']))
            y_min = max(0, int(self.museum_bounds['min_y'] - 200))
            y_max = min(mask_threshold.shape[0], int(self.museum_bounds['max_y'] + 200))
            roi_mask[y_min:y_max, x_min:x_max] = 255
            mask_roi = cv2.bitwise_and(mask_threshold, roi_mask)
            print(f"    After ROI: {cv2.countNonZero(mask_roi)} pixels")
        else:
            print("\n[5/7] Skipping ROI mask (no museum bounds)")

        # ── Morphology ────────────────────────────────────────────────
        print("\n[6/7] Applying morphological operations…")
        kernel      = np.ones((2, 2), np.uint8)
        mask_closed = cv2.morphologyEx(mask_roi,    cv2.MORPH_CLOSE, kernel, iterations=1)
        mask_opened = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN,  kernel, iterations=1)
        print(f"    After morphology: {cv2.countNonZero(mask_opened)} pixels")

        # ── Skeletonize ───────────────────────────────────────────────
        print("\n[7/7] Skeletonizing route…")
        skeleton = self._skeletonize(mask_opened)
        print(f"    Skeleton: {cv2.countNonZero(skeleton)} pixels")

        # ── Endpoints ─────────────────────────────────────────────────
        raw_pts = np.column_stack(np.where(skeleton > 0))
        if len(raw_pts) == 0:
            print("ERROR: No skeleton points found!")
            return
        points    = [(int(p[1]), int(p[0])) for p in raw_pts]
        endpoints = self._find_route_endpoints_by_boxes(points)

        # ── Visualise ─────────────────────────────────────────────────
        print("\n" + "="*70)
        print("Creating visualization…")
        self._create_visualization(aligned_route, skeleton, points,
                                   endpoints, output_path, marker_size,
                                   alignment_method)

        # ── Summary ───────────────────────────────────────────────────
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"Alignment method : {alignment_method}")
        if endpoints['start']:
            print(f"✅ START: {endpoints['start']}")
        else:
            print(f"❌ START: Not found")
        if endpoints['end']:
            print(f"✅ END:   {endpoints['end']}")
        else:
            print(f"❌ END:   Not found")
        print("="*70)

    # ──────────────────────────────────────────────────────────────────
    # Visualization
    # ──────────────────────────────────────────────────────────────────

    def _create_visualization(self, route_img, skeleton, points,
                               endpoints, output_path, marker_size=10,
                               alignment_method='unknown'):
        h, w  = route_img.shape[:2]
        scale = min(1.0, 1200 / max(h, w))
        new_h, new_w = int(h * scale), int(w * scale)

        def rsz(img):
            return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        vis_img = rsz(route_img.copy())
        vis_img[rsz(skeleton) > 0] = np.array([139, 0, 0], dtype=np.uint8)

        ms = int(marker_size * scale)
        if endpoints['start']:
            sp = (int(endpoints['start'][0]*scale), int(endpoints['start'][1]*scale))
            cv2.circle(vis_img, sp, ms, (0,255,0), -1)
            cv2.circle(vis_img, sp, ms+2, (255,255,255), 2)
            cv2.putText(vis_img, "START", (sp[0]+ms+5, sp[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        if endpoints['end']:
            ep = (int(endpoints['end'][0]*scale), int(endpoints['end'][1]*scale))
            cv2.circle(vis_img, ep, ms, (0,0,255), -1)
            cv2.circle(vis_img, ep, ms+2, (255,255,255), 2)
            cv2.putText(vis_img, "END", (ep[0]+ms+5, ep[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        for entrance in self.entrances:
            if entrance['shape'] == 'rectangle':
                c = entrance['coordinates']
                cv2.rectangle(vis_img,
                              (int(c['x']*scale), int(c['y']*scale)),
                              (int((c['x']+c['width'])*scale),
                               int((c['y']+c['height'])*scale)),
                              (0,255,0), 2)
                cv2.putText(vis_img, "ENTRANCE",
                            (int(c['x']*scale)+5, int(c['y']*scale)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        for exit_ann in self.exits:
            if exit_ann['shape'] == 'rectangle':
                c = exit_ann['coordinates']
                cv2.rectangle(vis_img,
                              (int(c['x']*scale), int(c['y']*scale)),
                              (int((c['x']+c['width'])*scale),
                               int((c['y']+c['height'])*scale)),
                              (255,255,0), 2)
                cv2.putText(vis_img, "EXIT",
                            (int(c['x']*scale)+5, int(c['y']*scale)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)

        legend = np.ones((170, vis_img.shape[1], 3), dtype=np.uint8) * 40
        final  = np.vstack([vis_img, legend])
        pil    = PILImage.fromarray(cv2.cvtColor(final, cv2.COLOR_BGR2RGB))
        draw   = ImageDraw.Draw(pil)

        try:
            tf = ImageFont.truetype("arial.ttf", 28)
            lf = ImageFont.truetype("arial.ttf", 18)
        except Exception:
            tf = lf = ImageFont.load_default()

        base = vis_img.shape[0]
        draw.text((20, base+8),
                  f"Route Endpoint Determination  [alignment: {alignment_method}]",
                  fill=(255,255,0), font=tf)

        items = [
            ((0,255,0),   False, "= START (route pixel in entrance box)"),
            ((255,0,0),   False, "= END (route pixel in exit box)"),
            ((0,255,0),   True,  "= Entrance box"),
            ((255,255,0), True,  "= Exit box"),
        ]
        for i, (color, is_rect, label) in enumerate(items):
            lx = 20 + (i % 2) * 440
            ly = base + 50 + (i // 2) * 30
            if is_rect:
                draw.rectangle([lx, ly, lx+20, ly+18], outline=color, width=2)
            else:
                draw.ellipse([lx, ly, lx+18, ly+18], fill=color)
            draw.text((lx+28, ly), label, fill=(255,255,255), font=lf)

        pil.save(output_path)
        print(f"\n✅ Visualization saved to: {output_path} "
              f"({pil.size[0]}×{pil.size[1]} px)")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Determine route START and END points',
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
    parser.add_argument('--route-image',          required=True)
    parser.add_argument('--original-image',       required=True)
    parser.add_argument('--annotations',          required=True)
    parser.add_argument('--difference-threshold', type=int, default=10)
    parser.add_argument('--output',               default='route_endpoints.png')
    parser.add_argument('--marker-size',          type=int, default=15)

    args = parser.parse_args()
    det  = RouteDeterminer(args.annotations)
    det.determine_route(
        route_image_path     = args.route_image,
        original_image_path  = args.original_image,
        difference_threshold = args.difference_threshold,
        output_path          = args.output,
        marker_size          = args.marker_size,
    )


if __name__ == '__main__':
    main()