import cv2
import numpy as np
from PIL import Image as PILImage, ImageDraw, ImageFont
import json
import argparse
import math
from pathlib import Path
from dataclasses import dataclass
from skimage.morphology import skeletonize

from prompts_utils import atomic_save_image


# ──────────────────────────────────────────────────────────────────────────────
# Custom Exceptions
# ──────────────────────────────────────────────────────────────────────────────

class RouteAlignmentError(Exception):
    """Raised when route image cannot be aligned to the original coordinate space."""
    pass


class NoRouteFoundError(RuntimeError):
    """Raised when the extracted route mask has zero pixels (no route detected)."""
    pass


# ──────────────────────────────────────────────────────────────────────────────
# Tunables
# ──────────────────────────────────────────────────────────────────────────────
MIN_CIRCLE_MATCHES   = 6    # minimum matched circles to trust the homography
CIRCLE_MATCH_RADIUS  = 30   # px – how close detected ↔ annotated centres must be
HOUGH_PARAM1         = 60   # Canny high threshold for HoughCircles
HOUGH_PARAM2         = 20   # accumulator threshold (lower → more detections)
HOUGH_MIN_DIST_SCALE = 0.8  # min-distance between circles = scale × median radius
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class RouteExtractionResult:
    aligned_route: any
    points: any
    endpoints: any
    connectivity: any
    final_mask: any
    skeleton_mask: any
    route_distance: float
    alignment_method: str
    geometric_fidelity: any = None


class RouteExtractor:
    def __init__(self, annotations_file):
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)

        self.entrances = self.annotations.get('entrances', [])
        self.exits     = self.annotations.get('exits', [])
        self.exhibits  = self.annotations.get('exhibits', [])

        print(f"Loaded annotations:")
        print(f"  Entrances : {len(self.entrances)}")
        print(f"  Exits     : {len(self.exits)}")
        print(f"  Exhibits  : {len(self.exhibits)}")

    # ──────────────────────────────────────────────────────────────────
    # Annotation helpers
    # ──────────────────────────────────────────────────────────────────

    def _distance(self, p1, p2):
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    def _rect_corners(self, coords):
        x, y, w, h = coords['x'], coords['y'], coords['width'], coords['height']
        return np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], dtype=np.float32)
    
    def _polygon_corners(self, coords):
        """Extract corners from polygon coordinates."""
        points = coords['points']
        return np.array([[p['x'], p['y']] for p in points], dtype=np.float32)
    
    def _shape_corners(self, shape_data):
        """Extract corners from either rectangle or polygon shape."""
        shape = shape_data.get('shape', 'rectangle')
        coords = shape_data['coordinates']
        if shape == 'rectangle':
            return self._rect_corners(coords)
        elif shape == 'polygon':
            return self._polygon_corners(coords)
        else:
            raise ValueError(f"Unsupported shape: {shape}")
    
    def _shape_center(self, shape_data):
        """Calculate center point for either rectangle or polygon shape."""
        shape = shape_data.get('shape', 'rectangle')
        coords = shape_data['coordinates']
        if shape == 'rectangle':
            return (coords['x'] + coords['width']/2, coords['y'] + coords['height']/2)
        elif shape == 'polygon':
            points = coords['points']
            cx = sum(p['x'] for p in points) / len(points)
            cy = sum(p['y'] for p in points) / len(points)
            return (cx, cy)
        else:
            raise ValueError(f"Unsupported shape: {shape}")

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
    # Colored-box detection (entrance / exit)
    # ──────────────────────────────────────────────────────────────────

    def _detect_colored_box(self, img_bgr, color: str):
        """Detect the bounding rectangle of a solid green or yellow box."""
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        if color == 'green':
            mask = cv2.inRange(hsv, np.array([35, 60, 60]), np.array([85, 255, 255]))
        elif color == 'yellow':
            mask = cv2.inRange(hsv, np.array([20, 80, 80]), np.array([35, 255, 255]))
        else:
            raise ValueError(f"Unsupported color: {color}")

        kernel = np.ones((5, 5), np.uint8)
    
        # MORPH_CLOSE: Dilate -> Erosion (Fills small holes inside the object)
        # MORPH_OPEN: Erosion -> Dilate (Removes small noises blobs)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)

        # Find outer contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 100:
            return None
        bx, by, bw, bh = cv2.boundingRect(largest)
        return np.array([[bx, bw+bx, bw+bx, bx],
                         [by, by,    bh+by, bh+by]], dtype=np.float32).T

    def _box_anchors(self, route_img_bgr):
        """
        Return (detected_corners, annotated_corners) from entrance+exit box detection, or (None, None).
        detected_corners are corners in the route image; annotated_corners are annotation-space corners.
        
        Handles mixed scenarios: if one is polygon and one is rectangle, uses only the rectangle for alignment.
        """
        if not (self.entrances and self.exits):
            return None, None
        entrance_ann = self.entrances[0]
        exit_ann     = self.exits[0]
        
        entrance_shape = entrance_ann.get('shape', 'rectangle')
        exit_shape = exit_ann.get('shape', 'rectangle')

        detected_corners, annotated_corners = [], []
        
        # Try entrance box (only if rectangle)
        if entrance_shape == 'rectangle':
            detected_entrance_corners = self._detect_colored_box(route_img_bgr, 'green')
            if detected_entrance_corners is not None:
                annotated_entrance_corners = self._rect_corners(entrance_ann['coordinates'])
                detected_corners.append(detected_entrance_corners)
                annotated_corners.append(annotated_entrance_corners)
        else:
            print("  [Alignment] Skipping entrance box (polygon shape - would cause mismatch)")
        
        # Try exit box (only if rectangle)
        if exit_shape == 'rectangle':
            detected_exit_corners = self._detect_colored_box(route_img_bgr, 'yellow')
            if detected_exit_corners is not None:
                annotated_exit_corners = self._rect_corners(exit_ann['coordinates'])
                detected_corners.append(detected_exit_corners)
                annotated_corners.append(annotated_exit_corners)
        else:
            print("  [Alignment] Skipping exit box (polygon shape - would cause mismatch)")
        
        if not detected_corners:
            return None, None

        return np.vstack(detected_corners), np.vstack(annotated_corners)

    # ──────────────────────────────────────────────────────────────────
    # Circle matching — two-pass with coarse-homography seeding
    # ──────────────────────────────────────────────────────────────────

    def _match_circles(self, detected, ann_circles, H_coarse=None):
        """
        Match detected circles (route-image space) to annotation circles
        (annotation space).

        Pass 1  – if H_coarse is provided, project each detected centre through
                  H_coarse to get a good initial estimate of its annotation-space
                  position, then do nearest-neighbour within a tight radius.
        Pass 2  – fall back to a loose linear-scale nearest-neighbour for any
                  circles not matched in pass 1 (or when H_coarse is None).

        Returns (src_pts, dst_pts) as float32 arrays, or (None, None).
        """
        if not detected or not ann_circles:
            return None, None

        ann_xy = np.array([[ax, ay] for ax, ay, _ in ann_circles], dtype=np.float32)
        src_pts, dst_pts = [], []
        used_ann = set()   # prevent duplicate annotation assignments

        # ── Pass 1: H_coarse-guided tight matching ─────────────────────
        if H_coarse is not None:
            det_xy  = np.array([[dx, dy] for dx, dy, _ in detected], dtype=np.float32)
            # perspectiveTransform needs shape (N,1,2)
            # projects the detected points into annotation space using H_coarse
            proj = cv2.perspectiveTransform(det_xy.reshape(-1, 1, 2), H_coarse)
            proj = proj.reshape(-1, 2)

            tight_r = CIRCLE_MATCH_RADIUS * 0.6   # tighter because projection is accurate
            
            # For each projected point:
            # 1: Compute Euclidean distance to all annotation points
            # 2: Find the closest one
            # 3: If it’s within the tight radius and not already used -> assign as match
            for i, (px, py) in enumerate(proj):
                dists = np.linalg.norm(ann_xy - np.array([px, py]), axis=1)
                j = int(np.argmin(dists))
                if dists[j] < tight_r and j not in used_ann:
                    src_pts.append(list(detected[i][:2]))
                    dst_pts.append(ann_xy[j].tolist())
                    used_ann.add(j)

        # ── Pass 2: loose linear-scale matching for remaining circles ──
        # Estimate rough scale from already-matched pairs if available,
        # otherwise fall back to a 1:1 assumption (same-size images).
        if src_pts:
            # Derive sx, sy from the matched pairs so far
            # sx, sy are rough scale factors between source and destination
            s  = np.array(src_pts, dtype=np.float32)
            d  = np.array(dst_pts, dtype=np.float32)
            sx = float(np.median(d[:, 0] / np.maximum(s[:, 0], 1)))
            sy = float(np.median(d[:, 1] / np.maximum(s[:, 1], 1)))
        else:
            sx, sy = 1.0, 1.0

        already_src = {tuple(p) for p in src_pts}

        for dx, dy, dr in detected:
            if (dx, dy) in already_src:
                continue
            # scaled = ann_xy * np.array([sx, sy])   # rough annotation-space position
            # dists  = np.linalg.norm(scaled - np.array([dx * sx, dy * sy]), axis=1)
            # recompute without scaling confusion — use direct distance in route space
            # computing a distance between the detected point and scaled annotation points, but in route image space, not annotation space.
            # ann_xy / np.array(sx, sy): approximates where that annotation point would be in route image coordinates
            dists2 = np.linalg.norm(
                ann_xy / np.array([sx if sx > 0 else 1, sy if sy > 0 else 1])
                - np.array([dx, dy]), axis=1)
            j = int(np.argmin(dists2))
            threshold = max(CIRCLE_MATCH_RADIUS, dr)
            if dists2[j] < threshold and j not in used_ann:
                src_pts.append([dx, dy])
                dst_pts.append(ann_xy[j].tolist())
                used_ann.add(j)

        if not src_pts:
            return None, None
        return (np.array(src_pts, dtype=np.float32),
                np.array(dst_pts, dtype=np.float32))

    # ──────────────────────────────────────────────────────────────────
    # Master alignment  — fused boxes + circles
    # ──────────────────────────────────────────────────────────────────

    def _align_route_image(self, route_img_bgr, original_img_bgr):
        """
        Warp route_img_bgr into the coordinate space of original_img_bgr.

        Strategy
        --------
        1. Detect entrance/exit boxes → coarse homography H_coarse.
           This gives a reliable seed even when circle detection is noisy.
        2. Use H_coarse to guide a two-pass circle match over all 100 exhibits.
           Fuse box corners + matched circle centres into a single point set.
        3. Compute final homography from the fused set (RANSAC).
           Fall back through: circles-only → boxes-only → plain resize.

        Returns (aligned_bgr, method_name).

        ** Note: 
            src = corners detected from route image
            dst = corners from annotation (original image)
        """
        orig_h, orig_w = original_img_bgr.shape[:2]
        ann_circles    = self._annotation_circles()

        # ── Step 1: coarse homography from boxes ──────────────────────
        src_box, dst_box = self._box_anchors(route_img_bgr)
        H_coarse = None
        if src_box is not None and len(src_box) >= 4:
            """
            https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html
            https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga4abc2ece9fab9398f2e560d53c8c9780

            RANSAC: Try many different random subsets of the corresponding point pairs (of four pairs each, collinear pairs are discarded), 
            estimate the homography matrix using this subset and a simple least-squares algorithm, and then compute the quality/goodness 
            of the computed homography (which is the number of inliers for RANSAC or the least median re-projection error for LMeDS). 
            The best subset is then used to produce the initial estimate of the homography matrix and the mask of inliers/outliers.
            
            5.0: Reprojection Threshold. After transforming a source point, 
            if it lands within 5 pixels of its expected destination point, consider it correct.
            """
         
            H_coarse, _ = cv2.findHomography(src_box, dst_box, cv2.RANSAC, 5.0)
            if H_coarse is not None:
                print(f"  [Alignment] Coarse box homography computed "
                      f"({len(src_box)} pts) — seeding circle search")
            else:
                print("  [Alignment] Coarse box homography failed; "
                      "circle matching will use linear-scale seed")
        else:
            print("  [Alignment] Box detection incomplete; "
                  "circle matching will use linear-scale seed")

        # ── Step 2: detect & match exhibit circles ────────────────────
        src_circ = dst_circ = None
        if ann_circles:
            print("  [Alignment] Detecting exhibit circles in route image…")
            detected = self._detect_circles_in_image(route_img_bgr, ann_circles)
            print(f"             {len(detected)} circles detected")

            src_circ, dst_circ = self._match_circles(
                detected, ann_circles, H_coarse=H_coarse)
            n_matched = len(src_circ) if src_circ is not None else 0
            print(f"             {n_matched} circles matched")

        # ── Step 3: fuse boxes + circles → final homography ───────────
        # Always include box corners so the entrance/exit region is tightly
        # anchored, regardless of how many circles are found nearby.
        fuse_src, fuse_dst = [], []

        if src_circ is not None and len(src_circ) >= MIN_CIRCLE_MATCHES:
            fuse_src.append(src_circ)
            fuse_dst.append(dst_circ)

        if src_box is not None:
            fuse_src.append(src_box)
            fuse_dst.append(dst_box)

        if fuse_src:
            all_src = np.vstack(fuse_src)
            all_dst = np.vstack(fuse_dst)
            H, mask = cv2.findHomography(all_src, all_dst, cv2.RANSAC, 5.0)
            if H is not None:
                inliers = int(mask.sum()) if mask is not None else len(all_src)
                n_circ  = len(src_circ) if src_circ is not None else 0
                n_box   = len(src_box)  if src_box  is not None else 0
                
                # Determine alignment method based on what was actually used
                if n_circ >= MIN_CIRCLE_MATCHES and n_box > 0:
                    method = 'circles+boxes'
                elif n_box > 0:
                    method = 'boxes-only'
                else:
                    method = 'circles-only'
                
                print(f"  ✅ Final homography: {inliers}/{len(all_src)} inliers "
                      f"({n_circ} circles + {n_box} box pts) [{method}]")
                aligned = cv2.warpPerspective(
                    route_img_bgr, H, (orig_w, orig_h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(255, 255, 255))
                return aligned, method

        # ── Fallback: abort with descriptive error ─────────────────────
        print("  ❌ All alignment strategies failed!")
        print("     • No entrance/exit boxes detected, OR")
        print("     • Insufficient exhibit circles matched")
        print("     Aborting - cannot proceed with misaligned route.")
        
        raise RouteAlignmentError(
            "Failed to align route image to original coordinate space. "
            "Ensure the route image contains:\n"
            "  1. Clearly visible green (entrance) box, AND/OR\n"
            "  2. Clearly visible yellow (exit) box, AND/OR\n"
            "  3. At least 6 detectable exhibit circles matching the annotations"
        )

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
    # Geometric fidelity (extracted skeleton vs. source coordinates)
    # ──────────────────────────────────────────────────────────────────

    def _point_to_segment_distances(self, points, seg_starts, seg_ends):
        """
        Vectorized minimum distance from each point to any of the given segments.

        points     : (N, 2) array of (x, y)
        seg_starts : (M, 2) array of segment start points
        seg_ends   : (M, 2) array of segment end points

        Returns (N,) array of distances.
        """
        p  = points[:, None, :]      # (N, 1, 2)
        a  = seg_starts[None, :, :]  # (1, M, 2)
        b  = seg_ends[None, :, :]    # (1, M, 2)
        ab = b - a                   # (1, M, 2)
        ap = p - a                   # (N, M, 2)

        ab_len_sq = np.sum(ab * ab, axis=2)                    # (1, M)
        ab_len_sq_safe = np.where(ab_len_sq == 0, 1, ab_len_sq)
        t = np.sum(ap * ab, axis=2) / ab_len_sq_safe           # (N, M)
        t = np.clip(t, 0, 1)

        proj = a + t[..., None] * ab   # (N, M, 2)
        diff = p - proj                # (N, M, 2)
        dist = np.sqrt(np.sum(diff * diff, axis=2))  # (N, M)
        return dist.min(axis=1)        # (N,)

    def _calculate_geometric_fidelity(self, skeleton_mask, source_coordinates, tolerance_px=3):
        """
        Compare the extracted route skeleton against the original source
        coordinates (e.g. the VLM's raw output route) to measure how much
        shape was preserved through the draw -> image -> extract round trip.

        For every skeleton pixel, computes the distance to the nearest point
        on the polyline formed by connecting consecutive source_coordinates.
        `similarity_percent` is the fraction of skeleton pixels within
        `tolerance_px` of that ideal polyline.
        """
        ys, xs = np.where(skeleton_mask > 0)
        skeleton_points = np.column_stack([xs, ys]).astype(np.float64)

        coords = np.array(source_coordinates, dtype=np.float64) if source_coordinates else np.empty((0, 2))

        if len(coords) < 2 or len(skeleton_points) == 0:
            return {
                'similarity_percent': 0.0,
                'tolerance_px': tolerance_px,
                'mean_deviation_px': None,
                'max_deviation_px': None,
                'skeleton_point_count': int(len(skeleton_points)),
                'points_within_tolerance': 0,
            }

        seg_starts = coords[:-1]
        seg_ends = coords[1:]

        distances = self._point_to_segment_distances(skeleton_points, seg_starts, seg_ends)
        within = distances <= tolerance_px
        similarity_percent = 100.0 * np.count_nonzero(within) / len(distances)

        return {
            'similarity_percent': round(float(similarity_percent), 2),
            'tolerance_px': tolerance_px,
            'mean_deviation_px': round(float(distances.mean()), 3),
            'max_deviation_px': round(float(distances.max()), 3),
            'skeleton_point_count': int(len(distances)),
            'points_within_tolerance': int(np.count_nonzero(within)),
        }

    # ──────────────────────────────────────────────────────────────────
    # Endpoint detection
    # ──────────────────────────────────────────────────────────────────

    def calculate_distance_between_points(self, final_mask, point_a, point_b):
        """
        Calculate the shortest distance between two points along route pixels only.

        Uses BFS so the path is guaranteed shortest, and only steps through
        pixels that exist in final_mask — no straight-line shortcuts.

        Parameters
        ----------
        final_mask : uint8 binary mask (route pixels = 255 / 1)
        point_a    : (x, y) start point — must lie on a route pixel
        point_b    : (x, y) end point   — must lie on a route pixel

        Returns
        -------
        float  arc distance in pixels, or None if no route path exists
        """
        from collections import deque

        ys, xs = np.where(final_mask > 0)
        pixel_set = set(zip(xs.tolist(), ys.tolist()))

        if point_a not in pixel_set:
            print(f"  ⚠️ point_a {point_a} is not on a route pixel")
            return None
        if point_b not in pixel_set:
            print(f"  ⚠️ point_b {point_b} is not on a route pixel")
            return None
        if point_a == point_b:
            return 0.0

        dist  = {point_a: 0.0}
        queue = deque([point_a])

        while queue:
            cx, cy = queue.popleft()

            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nb = (cx + dx, cy + dy)
                    if nb not in pixel_set or nb in dist:
                        continue
                    dist[nb] = dist[(cx, cy)] + math.sqrt(dx*dx + dy*dy)
                    if nb == point_b:
                        print(f"  ✅ Route distance {point_a} → {point_b}: {dist[nb]:.2f} px")
                        return dist[nb]
                    queue.append(nb)

        # point_b was never reached — on a different fragment
        print(f"  ❌ No route path between {point_a} and {point_b} (disconnected fragments)")
        return None

    def skeleton_endpoints(self, skel):
        # Make our input nice, possibly necessary.
        skel = skel.copy()
        skel[skel!=0] = 1
        skel = np.uint8(skel)

        # Apply the convolution.
        kernel = np.uint8([[1,  1, 1],
                        [1, 10, 1],
                        [1,  1, 1]])
        src_depth = -1
        filtered = cv2.filter2D(skel,src_depth,kernel)

        # Look through to find the value of 11.
        # This returns a mask of the endpoints, but if you
        # just want the coordinates, you could simply
        # return np.where(filtered==11)
        out = np.zeros_like(skel)
        out[np.where(filtered==11)] = 1
        return out

    
    def _prune_skeleton(self, skeleton, n_iter=10):
        """
        Remove short branch spurs from a skeleton by iteratively deleting
        endpoint pixels (pixels with exactly one neighbour in the 8-connected
        sense).  Any branch shorter than n_iter pixels is fully removed, which
        eliminates the tiny artefact branches produced by skeletonization
        without disturbing the true endpoints of the main path.

        Parameters
        ----------
        skeleton : uint8 ndarray  (non-zero = skeleton pixel)
        n_iter   : int            number of pruning passes (≈ max spur length
                                  to remove, in pixels)

        Returns
        -------
        pruned skeleton as uint8 ndarray
        """
        skel = skeleton.copy().astype(np.uint8)
        skel[skel != 0] = 1                         # normalise to 0/1

        kernel = np.uint8([[1, 1, 1],
                           [1, 10, 1],
                           [1, 1, 1]])

        for _ in range(n_iter):
            filtered      = cv2.filter2D(skel, -1, kernel)
            endpoint_mask = filtered == 11           # centre=10 + exactly 1 neighbour
            if not endpoint_mask.any():
                break
            skel[endpoint_mask] = 0

        return skel

    def _find_route_endpoints(self, final_mask, connectivity, debug=False, prune_iter=5):
        result = {'start': None, 'end': None}

        if debug:
            vis_pil = PILImage.fromarray(final_mask)
            draw = ImageDraw.Draw(vis_pil)
    
            # Add text at top of image
            try:
                font = ImageFont.truetype("arial.ttf", 24)
            except:
                font = ImageFont.load_default()
            
            draw.text((10, 10), "Original Final Mask", fill=255, font=font)
            vis_pil.show()     

        # MORPH_CLOSE = Dilation → Erosion (fills gaps without breaking connections)
        kernel = np.ones((3, 3), np.uint8)
        final_mask_closed = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Optional: gentle dilation to ensure connectivity (Keep for now)
        kernel_dilate = np.ones((3, 3), np.uint8)  # Smaller kernel!
        final_mask_dilated = cv2.dilate(final_mask_closed, kernel_dilate, iterations=1) 


        if debug:
            vis_dilated = PILImage.fromarray(final_mask_dilated)
            draw = ImageDraw.Draw(vis_dilated)
    
            # Add text at top of image
            try:
                font = ImageFont.truetype("arial.ttf", 24)
            except:
                font = ImageFont.load_default()
            
            draw.text((10, 10), "Final Mask Dilated", fill=255, font=font)

            vis_dilated.show()

        skeleton_mask = skeletonize(final_mask_dilated > 0)  # skimage expects boolean

        # == Pruning — remove tiny spur branches left by skeletonization.
        # Iteratively deletes endpoint pixels for `prune_iter` passes, which
        # trims every branch shorter than prune_iter pixels and prevents those
        # spurious tips from being misidentified as route endpoints.
        skeleton_pruned = self._prune_skeleton(np.uint8(skeleton_mask), n_iter=prune_iter)
        print(f"    Skeleton pruned ({prune_iter} iterations); "
              f"pixels before={skeleton_mask.sum()}, after={skeleton_pruned.sum()}")

        if debug:
            # vis_pil = PILImage.fromarray(skeleton_mask)
            vis_pil = PILImage.fromarray(skeleton_pruned * 255)
            draw = ImageDraw.Draw(vis_pil)
    
            # Add text at top of image
            try:
                font = ImageFont.truetype("arial.ttf", 24)
            except:
                font = ImageFont.load_default()
            
            draw.text((10, 10), "Skeleton Mask", fill=255, font=font)

            vis_pil.show()
        
        # Detect skeleton endpoints using convolution method
        endpoint_mask = self.skeleton_endpoints(skeleton_pruned)
        # endpoint_mask = self.skeleton_endpoints(np.uint8(skeleton_mask))
        
        # Extract endpoint coordinates
        endpoint_coords_yx = np.column_stack(np.where(endpoint_mask > 0))
        endpoint_list = [(int(p[1]), int(p[0])) for p in endpoint_coords_yx]  # Convert to (x, y)
        
        print(f"\n[Skeleton Endpoint Analysis]")
        print(f"  Endpoints detected: {len(endpoint_list)}")
        
        # Select START: endpoint closest to entrance center
        if endpoint_list and self.entrances:
            e_center = self._shape_center(self.entrances[0])
            
            # Find closest endpoint to entrance
            start_endpoint = min(endpoint_list, key=lambda ep: self._distance(ep, e_center))
            result['start'] = start_endpoint
            dist_to_entrance = self._distance(start_endpoint, e_center)
            print(f"  ✅ START: {start_endpoint} (distance to entrance: {dist_to_entrance:.1f} px)")
        else:
            print(f"  ❌ No START: No endpoints or entrance not defined")
        
        # (exclude START if already assigned)
        remaining_endpoints = [ep for ep in endpoint_list if ep != result['start']]
        route_distance = None
        if len(remaining_endpoints) == 1:
            result['end'] = remaining_endpoints[0]
            route_distance = self.calculate_distance_between_points(skeleton_pruned, result['start'], result['end'])
            print(f"  ✅ END: {result['end']} (only remaining endpoint)")
            print(f"  Distance from Start to End: {route_distance}px")
        
        elif len(remaining_endpoints) > 1:
            print(f"  Using furthest distance method")
            if connectivity['is_connected'] and result['start']:
                distances = {}
                for xp in remaining_endpoints:
                    dist = self.calculate_distance_between_points(skeleton_pruned, result['start'], xp)
                    if dist is not None:
                        distances[xp] = dist

                if distances:
                    result['end'] = max(distances, key=distances.get)
                    route_distance = distances[result['end']]
                    print(f"  ✅ END: {result['end']} (furthest from START: {route_distance} px)")

                else:
                    result['end'] = None
            
            else:
                print(f"  Using closest to Exit Box method")
                # Select END: endpoint closest to exit center (excluding START)
                if endpoint_list and self.exits:
                    x_center = self._shape_center(self.exits[0])
                    
                    # Find closest endpoint to exit 
                    if remaining_endpoints:
                        # If only 1 endpoint left then its exit, else if connected then find the furthest point, else use the point closest to exit
                        end_endpoint = min(remaining_endpoints, key=lambda ep: self._distance(ep, x_center))
                        result['end'] = end_endpoint
                        dist_to_exit = self._distance(end_endpoint, x_center)
                        print(f"  ✅ END: {end_endpoint} (distance to exit: {dist_to_exit:.1f} px)")

                    else:
                        print(f"  ❌ No END: No remaining endpoints")
                else:
                    print(f"  ❌ No END: No endpoints or exit not defined")
        else:
            result['end'] = None

       
        if debug:
            # Create visualization: skeleton with endpoints marked in red
            # Convert boolean skeleton to proper uint8 grayscale image
            skeleton_uint8 = skeleton_pruned * 255
            # skeleton_uint8 = (skeleton_mask.astype(np.uint8)) * 255
            vis_skeleton = cv2.cvtColor(skeleton_uint8, cv2.COLOR_GRAY2BGR)
            
            # Draw large circles at each endpoint for visibility
            for endpoint_coord in endpoint_list:
                x, y = endpoint_coord
                # Draw filled red circle
                cv2.circle(vis_skeleton, (x, y), radius=10, color=(0, 0, 255), thickness=-1)
                # Draw white outline for contrast
                cv2.circle(vis_skeleton, (x, y), radius=12, color=(255, 255, 255), thickness=2)
            
            # Convert BGR to RGB for PIL
            vis_skeleton_rgb = cv2.cvtColor(vis_skeleton, cv2.COLOR_BGR2RGB)
            vis_pil = PILImage.fromarray(vis_skeleton_rgb)
            vis_pil.show()


        return result, skeleton_pruned, route_distance

    # ──────────────────────────────────────────────────────────────────
    # Connectivity check
    # ──────────────────────────────────────────────────────────────────

    def _check_route_connectivity(self, final_mask, noise_threshold_ratio=0.01):
        """
        Check whether the route pixels form a single connected line.

        Uses cv2.connectedComponentsWithStats to label every contiguous island
        of non-zero pixels in final_mask.  The largest island is assumed to be
        the true route; all others are considered disconnected fragments.
        Fragments whose area is below noise_threshold_ratio × (largest island
        area) are classified as noise rather than genuine disconnections.

        Parameters
        ----------
        final_mask            : uint8 binary mask (route pixels = 255)
        noise_threshold_ratio : fragments smaller than this fraction of the
                                main route are treated as noise (default 1 %)

        Returns
        -------
        dict with keys:
          is_connected   – True if only one meaningful component was found
          num_components – total labelled components (excluding background)
          num_fragments  – components that are large enough to be non-noise
          main_size      – pixel count of the largest component
          fragments      – list of dicts {label, size, centroid} for every
                           non-main component above the noise threshold,
                           sorted largest → smallest
        """
        print("\n[Connectivity Check]")
        total_px = cv2.countNonZero(final_mask)
        if total_px == 0:
            print("  ⚠️  Mask is empty — cannot check connectivity.")
            return {'is_connected': False, 'num_components': 0,
                    'num_fragments': 0, 'main_size': 0, 'fragments': []}

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            final_mask, connectivity=8)

        # Label 0 is background — skip it.
        # stats columns: CC_STAT_LEFT, CC_STAT_TOP, CC_STAT_WIDTH,
        #                CC_STAT_HEIGHT, CC_STAT_AREA
        component_areas = [
            (label, int(stats[label, cv2.CC_STAT_AREA]),
             (float(centroids[label][0]), float(centroids[label][1])))
            for label in range(1, num_labels)   # skip background (0)
        ]

        # Sort largest → smallest
        component_areas.sort(key=lambda x: x[1], reverse=True)
        num_components = len(component_areas)

        if num_components == 0:
            print("  ⚠️  No components found.")
            return {'is_connected': False, 'num_components': 0,
                    'num_fragments': 0, 'main_size': 0, 'fragments': []}

        main_label, main_size, main_centroid = component_areas[0]
        noise_cutoff = max(1, int(main_size * noise_threshold_ratio))

        fragments = [
            {'label': lbl, 'size': sz,
             'centroid': (round(cx, 1), round(cy, 1))}
            for lbl, sz, (cx, cy) in component_areas[1:]
            if sz >= noise_cutoff
        ]

        is_connected  = (len(fragments) == 0)
        num_fragments = len(fragments)

        print(f"  Total route pixels   : {total_px}")
        print(f"  Connected components : {num_components}  "
              f"(noise cutoff < {noise_cutoff} px)")
        print(f"  Main route size      : {main_size} px  "
              f"@ centroid {(round(main_centroid[0],1), round(main_centroid[1],1))}")

        if is_connected:
            print("  ✅ Route is fully connected — single continuous line.")
        else:
            print(f"  ❌ Route has {num_fragments} disconnected fragment(s):")
            for i, frag in enumerate(fragments, 1):
                pct = frag['size'] / main_size * 100
                print(f"     Fragment {i}: {frag['size']} px "
                      f"({pct:.1f}% of main)  @ centroid {frag['centroid']}")

        return {
            'is_connected'  : is_connected,
            'num_components': num_components,
            'num_fragments' : num_fragments,
            'main_size'     : main_size,
            'fragments'     : fragments,
        }
    

    # ──────────────────────────────────────────────────────────────────
    # Route distance calculation
    # ──────────────────────────────────────────────────────────────────

    def calculate_route_distance(self, skeleton_mask, connectivity):
        """
        Compute the total distance travelled along the drawn route.

        For fragmented routes every connected component is measured independently
        (only skeleton pixels count — no straight-line shortcuts between fragments).
        Per-component distances are summed to give the overall route length.

        Parameters
        ----------
        skeleton_mask   : uint8 binary skeleton mask (route pixels = 255 / 1)
        connectivity    : dict returned by _check_route_connectivity()
        px_per_unit     : scale factor — pixels per real-world unit (e.g. px/metre).
                          Pass 1.0 to keep distances in pixels.
        unit_label      : string used in printed output ("px", "m", "cm", …)

        Returns
        -------
        dict with keys:
        total_distance        – sum of all component distances (real-world units)
        total_distance_px     – same, always in pixels
        component_distances   – list of dicts, one per component:
                                {label, distance_px, distance, num_pixels, centroid}
        """
        # ── Build pixel adjacency map ──────────────────────────────────────
        # Collect every lit skeleton pixel; neighbours are the 8-connected set.
        ys, xs = np.where(skeleton_mask > 0)
        pixel_set = set(zip(xs.tolist(), ys.tolist()))   # {(x, y), …}

        def neighbours(x, y):
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if (nx, ny) in pixel_set:
                        yield nx, ny, math.sqrt(dx*dx + dy*dy)   # 1.0 or √2

        # ── Measure each connected component separately ────────────────────
        num_labels, labels_img, _, centroids = cv2.connectedComponentsWithStats(
            skeleton_mask, connectivity=8)

        component_distances = []

        for lbl in range(1, num_labels):                   # 0 = background
            # Pixels belonging to this component
            comp_ys, comp_xs = np.where(labels_img == lbl)
            comp_pixels = list(zip(comp_xs.tolist(), comp_ys.tolist()))

            if not comp_pixels:
                continue

            # BFS / DFS: walk every edge once and accumulate its Euclidean length.
            # Because the skeleton is 1-px wide this equals the arc length exactly.
            visited  = set()
            stack    = [comp_pixels[0]]
            dist_px  = 0.0

            while stack:
                cx, cy = stack.pop()
                if (cx, cy) in visited:
                    continue
                visited.add((cx, cy))
                for nx, ny, step in neighbours(cx, cy):
                    if (nx, ny) not in visited:
                        dist_px += step          # count each edge once (DFS tree edge)
                        stack.append((nx, ny))

            cx_c, cy_c = float(centroids[lbl][0]), float(centroids[lbl][1])
            component_distances.append({
                'label'      : lbl,
                'num_pixels' : len(comp_pixels),
                'centroid'   : (round(cx_c, 1), round(cy_c, 1)),
                'distance_px': round(dist_px, 2),
            })

        # Sort largest → smallest component
        component_distances.sort(key=lambda d: d['distance_px'], reverse=True)
        total_px   = sum(d['distance_px'] for d in component_distances)

        # ── Pretty-print ──────────────────────────────────────────────────
        print("\n[Route Distance Calculation]")
        main_frag  = connectivity.get('num_fragments', 0)
        frag_label = "fully connected" if main_frag == 0 else f"{main_frag} fragment(s)"
        print(f"  Route connectivity : {frag_label}")
        print(f"  Components measured: {len(component_distances)}")
        for i, c in enumerate(component_distances, 1):
            tag = "MAIN" if i == 1 else f"FRAG {i-1}"
            print(f"    [{tag}]  {c['distance_px']:.1f} px"
                f"  |  {c['num_pixels']} px  @  {c['centroid']}")
        print(f"  ──────────────────────────────────────────────")
        print(f"  TOTAL DISTANCE : {total_px} px")

        return {
            'total_distance_px'  : total_px,
            'component_distances': component_distances,
        }

    # ──────────────────────────────────────────────────────────────────
    # Route colour sampling
    # ──────────────────────────────────────────────────────────────────

    def _sample_route_color(self, aligned_route, candidate_mask, original_img,
                            min_saturation=50, presence_threshold=0.01):
        """
        Sample colors at candidate route pixels, identify the most dominant
        color bin, and confirm it is absent (or rare) in the original image.

        This is used as a post-processing step after the diff-based candidate
        mask is built.  When the route image has different dimensions from the
        original, warping / clipping introduces disconnected fragments and
        residual artefacts; rebuilding the mask from the *colour* of the route
        avoids those problems entirely.

        Strategy
        --------
        1.  Convert the aligned route and the original to HSV.
        2.  Collect HSV values at every candidate pixel (non-zero in
            candidate_mask).
        3.  Quantize into (H × S × V) bins and rank by frequency.
        4.  For each bin (most frequent first):
              a.  Skip achromatic / near-black bins (low saturation) — they are
                  background artefacts, not a deliberately drawn colour.
              b.  Validate that the bin represents a significant portion of
                  candidate pixels (at least 10%).
              c.  Build a tolerant HSV range around the bin centre, handling
                  red-hue wrap-around (H ≈ 0° / 180°).
              d.  Count how many pixels in the *original* image fall in that
                  range.  If the fraction exceeds presence_threshold, the colour
                  is too common in the map background → try the next bin.
              e.  Accept the first bin that passes all validation tests.

        Parameters
        ----------
        aligned_route      : BGR image warped to the original's coordinate space
        candidate_mask     : uint8 binary mask of candidate route pixels
        original_img       : BGR reference image
        min_saturation     : HSV-S floor; bins below this are skipped as
                             achromatic artefacts (default 50, increased from 30)
        presence_threshold : maximum fraction of original-image pixels that may
                             share the dominant colour before it is rejected
                             (default 0.01 = 1%, reduced from 2% for stricter matching)

        Returns
        -------
        dict with keys lo1, hi1 (uint8 arrays, HSV lower/upper bounds) and
        lo2, hi2 (same but for the wrap-around band, or None if not needed),
        or None if no distinctive route colour could be confirmed.
        """
        ys, xs = np.where(candidate_mask > 0)
        if len(ys) < 10:
            print("    [Color sampling] Too few candidate pixels — skipping.")
            return None

        hsv_route    = cv2.cvtColor(aligned_route, cv2.COLOR_BGR2HSV)
        hsv_original = cv2.cvtColor(original_img,  cv2.COLOR_BGR2HSV)
        hsv_pixels   = hsv_route[ys, xs]   # shape (N, 3): H in [0,179], S,V in [0,255]

        # ── Quantize ──────────────────────────────────────────────────
        # H → 36 bins (5° each), S → 8 bins (32 levels), V → 4 bins (64 levels)
        # Finer hue binning (5° instead of 10°) for better color discrimination
        H_bin = (hsv_pixels[:, 0].astype(np.int32) // 5)    # 0–35
        S_bin = (hsv_pixels[:, 1].astype(np.int32) // 32)   # 0–7
        V_bin = (hsv_pixels[:, 2].astype(np.int32) // 64)   # 0–3
        keys  = H_bin * 32 + S_bin * 4 + V_bin              # unique per (H,S,V) triple

        unique_keys, counts = np.unique(keys, return_counts=True)
        order = np.argsort(-counts)   # most-frequent first

        total_orig_px = float(hsv_original.shape[0] * hsv_original.shape[1])
        total_candidate_px = len(ys)
        print(f"    [Color sampling] {total_candidate_px} candidate pixels → "
              f"{len(unique_keys)} quantized bins")

        # ── Evaluate bins, most frequent first ────────────────────────
        for rank, idx in enumerate(order):
            key   = int(unique_keys[idx])
            count = int(counts[idx])
            
            # Skip bins with too few pixels
            if count < 5:
                break   # all remaining bins are negligible
            
            # Validate that dominant color represents a significant portion
            # (at least 10% of candidate pixels) to avoid picking up noise
            color_fraction = count / total_candidate_px
            if rank == 0 and color_fraction < 0.10:
                print(f"    [Color sampling] Dominant bin only {color_fraction:.1%} "
                      f"of candidates — likely noisy mask, aborting color-based detection")
                return None

            h_bin = key  // 32
            s_bin = (key %  32) // 4
            v_bin = key  %   4

            h_c = h_bin * 5 + 2.5   # bin centre values (adjusted for 5° bins)
            s_c = s_bin * 32 + 16
            v_c = v_bin * 64 + 32

            # Skip achromatic / near-black pixels (not a coloured route)
            if s_c < min_saturation:
                print(f"    [Color sampling] Rank {rank+1}: H≈{h_c:.1f}° S={s_c} V={v_c} "
                      f"count={count} ({color_fraction:.1%}) — skipped (low saturation < {min_saturation})")
                continue

            # ── Tighter tolerant range around bin centre ──────────────
            # Reduced from (15, 50, 70) to (10, 35, 50) for stricter matching
            h_tol, s_tol, v_tol = 10, 35, 50

            s_lo = int(max(0,   s_c - s_tol))
            s_hi = int(min(255, s_c + s_tol))
            v_lo = int(max(0,   v_c - v_tol))
            v_hi = int(min(255, v_c + v_tol))

            h_lo_f = h_c - h_tol
            h_hi_f = h_c + h_tol

            # Handle hue wrap-around (red straddles 0°/180° in OpenCV)
            wraps = (h_lo_f < 0) or (h_hi_f > 179)
            if wraps:
                lo1 = np.array([max(0, int(h_lo_f) % 180), s_lo, v_lo], dtype=np.uint8)
                hi1 = np.array([179,                        s_hi, v_hi], dtype=np.uint8)
                lo2 = np.array([0,                          s_lo, v_lo], dtype=np.uint8)
                hi2 = np.array([min(179, int(h_hi_f) % 180), s_hi, v_hi], dtype=np.uint8)
                orig_mask = cv2.bitwise_or(
                    cv2.inRange(hsv_original, lo1, hi1),
                    cv2.inRange(hsv_original, lo2, hi2))
            else:
                lo1 = np.array([int(h_lo_f), s_lo, v_lo], dtype=np.uint8)
                hi1 = np.array([int(h_hi_f), s_hi, v_hi], dtype=np.uint8)
                lo2 = hi2 = None
                orig_mask = cv2.inRange(hsv_original, lo1, hi1)

            orig_ratio = cv2.countNonZero(orig_mask) / total_orig_px

            print(f"    [Color sampling] Rank {rank+1}: H≈{h_c:.1f}°±{h_tol} "
                  f"S={s_c}±{s_tol} V={v_c}±{v_tol} — "
                  f"count={count} ({color_fraction:.1%}), orig presence={orig_ratio:.3%}")

            if orig_ratio <= presence_threshold:
                print(f"    ✅ Route colour confirmed: HSV ≈ ({h_c:.1f}, {s_c}, {v_c}) "
                      f"[{color_fraction:.1%} of candidates, {orig_ratio:.3%} in original]")
                return {'lo1': lo1, 'hi1': hi1, 'lo2': lo2, 'hi2': hi2}

            print(f"       ↳ Too common in original ({orig_ratio:.3%} > "
                  f"{presence_threshold:.3%}); trying next bin…")

        print("    [Color sampling] No distinctive route colour found; "
              "will use difference-based mask.")
        return None

    # ──────────────────────────────────────────────────────────────────
    # Route mask extraction
    # ──────────────────────────────────────────────────────────────────

    def _extract_route_mask(self, aligned_route, original_img,
                            difference_threshold=10, tolerance_px=3):
        """
        Isolate the drawn route by removing pixels that also exist in the
        original image, with a spatial tolerance to absorb residual
        misalignment after warping.

        Strategy
        --------
        1.  Convert both images to greyscale and apply a small Gaussian blur
            (radius ≈ tolerance_px) so that sub-pixel shift noise is smoothed
            before any comparison is made.

        2.  Threshold both blurred images to binary "ink" masks.
            Pixels darker than (255 − difference_threshold) are considered ink.

        3.  Dilate the *original* ink mask by tolerance_px pixels.
            This creates a "forgiveness zone": any route pixel that falls
            within tolerance_px of an original ink pixel is considered
            background and discarded.

        4.  Subtract the dilated original mask from the aligned route mask.
            Only genuinely new pixels (the drawn route) survive.

        5.  As a safety net, also compute the classic absdiff in colour space
            and OR it with step 4's result — this catches coloured routes that
            are lighter than the original background but still visually distinct.

        Parameters
        ----------
        aligned_route        : BGR image warped to original's coordinate space
        original_img         : BGR reference image
        difference_threshold : ink darkness cutoff (0–255); lower = stricter
        tolerance_px         : dilation radius to forgive residual misalignment

        Returns
        -------
        gray_diff : uint8 greyscale difference map (for threshold step)
        """
        # ── Step 1: blur both images ──────────────────────────────────
        k = max(1, tolerance_px * 2 + 1)           # must be odd
        blur_ksize = (k | 1, k | 1)                # ensure odd
        aligned_blur  = cv2.GaussianBlur(aligned_route, blur_ksize, 0)
        original_blur = cv2.GaussianBlur(original_img,  blur_ksize, 0)

        # ── Step 2: binary ink masks ──────────────────────────────────
        gray_aligned  = cv2.cvtColor(aligned_blur,  cv2.COLOR_BGR2GRAY)
        gray_original = cv2.cvtColor(original_blur, cv2.COLOR_BGR2GRAY)

        ink_thresh = max(0, 255 - difference_threshold)
        _, ink_aligned  = cv2.threshold(
            gray_aligned,  ink_thresh, 255, cv2.THRESH_BINARY_INV)
        _, ink_original = cv2.threshold(
            gray_original, ink_thresh, 255, cv2.THRESH_BINARY_INV)

        # ── Step 3: dilate original ink → forgiveness zone ───────────
        tol_kernel       = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (tolerance_px * 2 + 1, tolerance_px * 2 + 1))
        ink_original_fat = cv2.dilate(ink_original, tol_kernel)

        # ── Step 4: new-ink mask (route pixels not in original) ───────
        new_ink_mask = cv2.subtract(ink_aligned, ink_original_fat)

        # ── Step 5: colour absdiff fallback (catches coloured routes) ─
        color_diff = cv2.absdiff(aligned_blur, original_blur)
        gray_color_diff = cv2.cvtColor(color_diff, cv2.COLOR_BGR2GRAY)
        # Suppress color-diff where the original already has ink
        # (avoid re-introducing background artefacts)
        gray_color_diff = cv2.subtract(gray_color_diff,
                                       cv2.dilate(ink_original, tol_kernel))

        # Merge: take the maximum signal from both strategies
        gray_diff = cv2.max(new_ink_mask, gray_color_diff)

        n_ink   = cv2.countNonZero(new_ink_mask)
        n_col   = cv2.countNonZero(gray_color_diff)
        n_merge = cv2.countNonZero(gray_diff)
        print(f"    Ink-subtraction pixels : {n_ink}")
        print(f"    Colour-diff pixels     : {n_col}")
        print(f"    Merged diff pixels     : {n_merge}")

        # ── Step 6: colour-sampling refinement ────────────────────────
        # Use the merged diff as a cheap candidate set, then ask
        # _sample_route_color to identify the single dominant route colour
        # that is absent from the original image.  If confirmed, rebuild
        # the mask purely by colour matching against aligned_route — this
        # is far more robust than the diff approach when dimension
        # differences cause warping clipping or residual misalignment.
        print("\n    [Step 6] Colour-sampling refinement…")
        _, candidate_mask = cv2.threshold(
            gray_diff, difference_threshold, 255, cv2.THRESH_BINARY)

        color_result = self._sample_route_color(
            aligned_route, candidate_mask, original_img)

        if color_result is not None:
            hsv_aligned = cv2.cvtColor(aligned_route, cv2.COLOR_BGR2HSV)
            color_mask  = cv2.inRange(hsv_aligned,
                                      color_result['lo1'], color_result['hi1'])
            if color_result['lo2'] is not None:
                # Merge the wrap-around band (e.g. red hue near 0°/180°)
                color_mask = cv2.bitwise_or(
                    color_mask,
                    cv2.inRange(hsv_aligned,
                                color_result['lo2'], color_result['hi2']))

            # No subtraction needed: _sample_route_color already verified
            # that this colour is rare/absent in the original image, so every
            # pixel that matches the colour is genuine route.
            n_color = cv2.countNonZero(color_mask)
            print(f"    Color-based mask pixels: {n_color}")
            return color_mask, True

        # No distinctive colour found — fall back to the diff-based mask.
        return gray_diff, False
    

    # ──────────────────────────────────────────────────────────────────
    # Main entry point
    # ──────────────────────────────────────────────────────────────────

    def determine_route(self, route_image_path, original_image_path, difference_threshold=10, tolerance_px=3, debug=False, skip_alignment=False, source_coordinates=None, fidelity_tolerance_px=3):
        print("\n" + "="*70)
        print("ROUTE DETERMINATION")
        print("="*70)

        # ── Load ──────────────────────────────────────────────────────
        print("\n[1/5] Loading images…")
        route_img    = cv2.cvtColor(
            np.array(PILImage.open(str(route_image_path)).convert('RGB')),
            cv2.COLOR_RGB2BGR)
        original_img = cv2.cvtColor(
            np.array(PILImage.open(str(original_image_path)).convert('RGB')),
            cv2.COLOR_RGB2BGR)

        print(f"    Route image:    {route_img.shape[:2]} (H x W)")
        print(f"    Original image: {original_img.shape[:2]} (H x W)")
        print(f"    tolerance_px:   {tolerance_px}")

        # ── Align ─────────────────────────────────────────────────────
        # Always align, even when dimensions match — same size does not
        # guarantee the same coordinate space (e.g. the route image may have
        # been re-exported at the same resolution but with a different crop or
        # padding). Alignment ensures annotations land at the correct pixels.
        # skip_alignment bypasses this entirely for callers that drew the
        # route directly onto an unmodified copy of the original image, where
        # the coordinate spaces are already guaranteed identical.
        if skip_alignment:
            if route_img.shape[:2] != original_img.shape[:2]:
                raise RouteAlignmentError(
                    "skip_alignment=True requires the route image and original "
                    f"image to have identical dimensions, got {route_img.shape[:2]} "
                    f"vs {original_img.shape[:2]}."
                )
            print("\n[2/5] Skipping alignment (route drawn directly on original image)…")
            aligned_route, alignment_method = route_img, 'skipped (drawn on original)'
        else:
            print("\n[2/5] Aligning route image to original coordinate space…")
            aligned_route, alignment_method = self._align_route_image(
                route_img, original_img)
        print(f"    Alignment method used: {alignment_method}")

        # ── Difference ────────────────────────────────────────────────
        print("\n[3/5] Computing image difference…")
        print(f"    tolerance_px={tolerance_px}  threshold={difference_threshold}")
        route_mask, is_color_based = self._extract_route_mask(
            aligned_route, original_img,
            difference_threshold=difference_threshold,
            tolerance_px=tolerance_px,
        )

        if is_color_based:
            # Color sampling found a distinctive route colour — the returned
            # mask is already a clean binary mask covering all route pixels.
            # Threshold, morphology, and skeletonization are not needed;
            # use the mask directly to preserve the route in its original form.
            print("\n[4/5] Skipping threshold  (color-based mask is already binary)")
            print("[5/5] Skipping morphology (color-based mask is already clean)")
            print("\nSkipping skeletonization  (preserving original route shape)")
            final_mask = route_mask
        else:
            # ── Threshold ─────────────────────────────────────────────
            # Converts the greyscale difference into a binary route mask.
            # All downstream steps (morphology, skeletonization, endpoint
            # search) operate on this binary mask.
            print(f"\n[4/5] Applying threshold (threshold={difference_threshold})…")
            _, mask_threshold = cv2.threshold(
                route_mask, difference_threshold, 255, cv2.THRESH_BINARY)
            print(f"    Detected {cv2.countNonZero(mask_threshold)} changed pixels")

            # ── Morphology ────────────────────────────────────────────
            # CLOSE fills small gaps in the drawn route line so the
            # skeleton stays connected through the entrance/exit boxes.
            # OPEN removes isolated noise specks that would otherwise
            # produce spurious skeleton branches and slow down skeletonization.
            print("\n[5/5] Applying morphological operations…")
            kernel      = np.ones((2, 2), np.uint8)
            mask_closed = cv2.morphologyEx(mask_threshold, cv2.MORPH_CLOSE, kernel, iterations=1)
            mask_opened = cv2.morphologyEx(mask_closed,    cv2.MORPH_OPEN,  kernel, iterations=1)
            print(f"    After morphology: {cv2.countNonZero(mask_opened)} pixels")

            # ── Skeletonize ───────────────────────────────────────────
            print("\nSkeletonizing route…")
            final_mask = self._skeletonize(mask_opened)
            print(f"    Skeleton: {cv2.countNonZero(final_mask)} pixels")

        # ── Connectivity check ────────────────────────────────────────
        connectivity = self._check_route_connectivity(final_mask)

        # ── Endpoints ─────────────────────────────────────────────────
        raw_pts = np.column_stack(np.where(final_mask > 0))
        if len(raw_pts) == 0:
            print("ERROR: No route points found!")
            return
        points    = [(int(p[1]), int(p[0])) for p in raw_pts]
        endpoints, skeleton_mask, route_distance = self._find_route_endpoints(final_mask, connectivity, debug)

        # ── Geometric fidelity ───────────────────────────────────────────
        # Only computed when the caller supplies the original source
        # coordinates (e.g. the VLM's raw output route) to compare against.
        geometric_fidelity = None
        if source_coordinates is not None:
            print("\nComputing geometric fidelity vs. source coordinates…")
            geometric_fidelity = self._calculate_geometric_fidelity(
                skeleton_mask, source_coordinates, tolerance_px=fidelity_tolerance_px)
            print(f"    Similarity: {geometric_fidelity['similarity_percent']}% "
                  f"(tolerance={fidelity_tolerance_px}px, "
                  f"mean_dev={geometric_fidelity['mean_deviation_px']}px, "
                  f"max_dev={geometric_fidelity['max_deviation_px']}px)")

        return RouteExtractionResult(
            aligned_route=aligned_route,
            points=points,
            endpoints=endpoints,
            connectivity=connectivity,
            final_mask=final_mask,
            skeleton_mask=skeleton_mask,
            route_distance=route_distance,
            alignment_method=alignment_method,
            geometric_fidelity=geometric_fidelity,
        )
    
    def summary(self, alignment_method, connectivity, endpoints):
        # ── Summary ───────────────────────────────────────────────────
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"Alignment method : {alignment_method}")
        if connectivity['is_connected']:
            print(f"✅ ROUTE: Fully connected  ({connectivity['main_size']} px)")
        else:
            print(f"❌ ROUTE: {connectivity['num_fragments']} disconnected fragment(s)  "
                  f"(main = {connectivity['main_size']} px)")
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
                               endpoints, output_path=None, marker_size=10,
                               alignment_method='unknown'):
        h, w  = route_img.shape[:2]
        scale = min(1.0, 1200 / max(h, w))
        new_h, new_w = int(h * scale), int(w * scale)

        print("\n" + "="*70)
        print("Creating visualization…")

        def rsz(img):
            return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        def sx(v): return int(v * scale)   # scale a single coordinate value

        vis_img = rsz(route_img.copy())

        # ── Route mask overlay (dark red) ─────────────────────────────
        vis_img[rsz(skeleton) > 0] = np.array([139, 0, 0], dtype=np.uint8)

        # ── Floor area (magenta outline) ──────────────────────────────
        # Drawn first so all other annotations render on top of it.
        for fa in self.annotations.get('floor_areas', []):
            if fa['shape'] == 'rectangle':
                c = fa['coordinates']
                pt1 = (sx(c['x']),             sx(c['y']))
                pt2 = (sx(c['x'] + c['width']), sx(c['y'] + c['height']))
                # Simulate a dashed outline by alternating draw / skip in
                # fixed-length segments along each edge.
                def _draw_dashed_rect(img, p1, p2, color, dash=12, gap=6):
                    x1, y1 = p1;  x2, y2 = p2
                    for edge in [((x1,y1),(x2,y1)), ((x2,y1),(x2,y2)),
                                 ((x2,y2),(x1,y2)), ((x1,y2),(x1,y1))]:
                        (ex1,ey1),(ex2,ey2) = edge
                        length = math.hypot(ex2-ex1, ey2-ey1)
                        if length == 0:
                            continue
                        dx, dy = (ex2-ex1)/length, (ey2-ey1)/length
                        pos = 0.0
                        drawing = True
                        while pos < length:
                            seg_len = min(dash if drawing else gap, length - pos)
                            if drawing:
                                p_start = (int(ex1 + dx*pos),        int(ey1 + dy*pos))
                                p_end   = (int(ex1 + dx*(pos+seg_len)), int(ey1 + dy*(pos+seg_len)))
                                cv2.line(img, p_start, p_end, color, 1)
                            pos    += seg_len
                            drawing = not drawing
                _draw_dashed_rect(vis_img, pt1, pt2, (255, 0, 255))

        # ── Walls (cyan polylines) ────────────────────────────────────
        for wall in self.annotations.get('walls', []):
            if wall['shape'] == 'polyline':
                pts = wall['coordinates']['points']
                arr = np.array([[sx(p['x']), sx(p['y'])] for p in pts],
                               dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(vis_img, [arr], isClosed=False,
                              color=(255, 255, 0), thickness=1)

        # ── Exhibits (orange circle outlines + exhibit number) ────────
        for ex in self.exhibits:
            if ex['shape'] == 'circle':
                c  = ex['coordinates']
                cx = sx(c['center_x'])
                cy = sx(c['center_y'])
                r  = max(3, sx(c['radius']))
                cv2.circle(vis_img, (cx, cy), r, (0, 165, 255), 1)
                label = str(ex.get('exhibit_number', ''))
                if label:
                    font_scale = max(0.25, scale * 0.55)
                    (tw, th), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
                    cv2.putText(vis_img, label,
                                (cx - tw // 2, cy + th // 2),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                font_scale, (0, 165, 255), 1, cv2.LINE_AA)

        # ── Entrance boxes (green) ────────────────────────────────────
        for entrance in self.entrances:
            shape = entrance.get('shape', 'rectangle')
            if shape == 'rectangle':
                c = entrance['coordinates']
                cv2.rectangle(vis_img,
                              (sx(c['x']),              sx(c['y'])),
                              (sx(c['x'] + c['width']), sx(c['y'] + c['height'])),
                              (0, 255, 0), 2)
                cv2.putText(vis_img, "ENTRANCE",
                            (sx(c['x']) + 5, sx(c['y']) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            elif shape == 'polygon':
                pts = entrance['coordinates']['points']
                arr = np.array([[sx(p['x']), sx(p['y'])] for p in pts],
                               dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(vis_img, [arr], isClosed=True, color=(0, 255, 0), thickness=2)
                # Calculate center for label
                center_x = sum(p['x'] for p in pts) / len(pts)
                center_y = sum(p['y'] for p in pts) / len(pts)
                cv2.putText(vis_img, "ENTRANCE",
                            (sx(center_x) - 30, sx(center_y) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # ── Exit boxes (yellow) ───────────────────────────────────────
        for exit_ann in self.exits:
            shape = exit_ann.get('shape', 'rectangle')
            if shape == 'rectangle':
                c = exit_ann['coordinates']
                cv2.rectangle(vis_img,
                              (sx(c['x']),              sx(c['y'])),
                              (sx(c['x'] + c['width']), sx(c['y'] + c['height'])),
                              (255, 255, 0), 2)
                cv2.putText(vis_img, "EXIT",
                            (sx(c['x']) + 5, sx(c['y']) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            elif shape == 'polygon':
                pts = exit_ann['coordinates']['points']
                arr = np.array([[sx(p['x']), sx(p['y'])] for p in pts],
                               dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(vis_img, [arr], isClosed=True, color=(255, 255, 0), thickness=2)
                # Calculate center for label
                center_x = sum(p['x'] for p in pts) / len(pts)
                center_y = sum(p['y'] for p in pts) / len(pts)
                cv2.putText(vis_img, "EXIT",
                            (sx(center_x) - 15, sx(center_y) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        # ── START / END endpoint markers ──────────────────────────────
        ms = int(marker_size * scale)
        if endpoints['start']:
            sp = (sx(endpoints['start'][0]), sx(endpoints['start'][1]))
            cv2.circle(vis_img, sp, ms, (0, 255, 0), -1)
            cv2.circle(vis_img, sp, ms + 2, (255, 255, 255), 2)
            cv2.putText(vis_img, "START", (sp[0] + ms + 5, sp[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if endpoints['end']:
            ep = (sx(endpoints['end'][0]), sx(endpoints['end'][1]))
            cv2.circle(vis_img, ep, ms, (0, 0, 255), -1)
            cv2.circle(vis_img, ep, ms + 2, (255, 255, 255), 2)
            cv2.putText(vis_img, "END", (ep[0] + ms + 5, ep[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # ── Legend ────────────────────────────────────────────────────
        # 8 items at 2 per row = 4 rows; 30 px row height + 55 px header.
        legend_h = 55 + 4 * 30 + 15          # ≈ 220 px
        legend   = np.ones((legend_h, vis_img.shape[1], 3), dtype=np.uint8) * 40
        final    = np.vstack([vis_img, legend])
        pil      = PILImage.fromarray(cv2.cvtColor(final, cv2.COLOR_BGR2RGB))
        draw     = ImageDraw.Draw(pil)

        try:
            tf = ImageFont.truetype("arial.ttf", 28)
            lf = ImageFont.truetype("arial.ttf", 18)
        except Exception:
            tf = lf = ImageFont.load_default()

        base = vis_img.shape[0]
        draw.text((20, base + 8),
                  f"Route Endpoint Determination  [alignment: {alignment_method}]",
                  fill=(255, 255, 0), font=tf)

        # Each tuple: (RGB color, shape, label)
        # shape: 'circle_fill' | 'circle_outline' | 'rect' | 'line'
        items = [
            ((139,   0,   0), 'circle_fill',    "= Detected route"),
            ((  0, 255,   0), 'circle_fill',    "= START point"),
            ((255,   0,   0), 'circle_fill',    "= END point"),
            ((  0, 255,   0), 'rect',           "= Entrance box"),
            ((255, 255,   0), 'rect',           "= Exit box"),
            ((255, 255,   0), 'line',           "= Wall"),
            ((255, 165,   0), 'circle_outline', "= Exhibit"),
            ((255,   0, 255), 'line',           "= Floor area"),
        ]
        for i, (color, shape, label) in enumerate(items):
            col = i % 2
            row = i // 2
            lx  = 20  + col * 440
            ly  = base + 55 + row * 30
            mx  = lx + 10   # midpoint of the 20-px icon slot

            if shape == 'circle_fill':
                draw.ellipse([lx, ly, lx + 18, ly + 18], fill=color)
            elif shape == 'circle_outline':
                draw.ellipse([lx, ly, lx + 18, ly + 18], outline=color, width=2)
            elif shape == 'rect':
                draw.rectangle([lx, ly, lx + 20, ly + 18], outline=color, width=2)
            elif shape == 'line':
                draw.line([(lx, ly + 9), (lx + 20, ly + 9)], fill=color, width=2)

            draw.text((lx + 28, ly), label, fill=(255, 255, 255), font=lf)

        if output_path:
            atomic_save_image(pil, output_path)
            print(f"\n✅ Visualization saved to: {output_path} "
                f"({pil.size[0]}×{pil.size[1]} px)")
        else:
            pil.show()

    def process_pipeline(self, route_image_path, original_image_path, output_path=None, marker_size=10, difference_threshold=10, tolerance_px=3, debug=False, skip_alignment=False, source_coordinates=None, fidelity_tolerance_px=3):

        results = self.determine_route(
            route_image_path,
            original_image_path,
            difference_threshold,
            tolerance_px,
            debug,
            skip_alignment,
            source_coordinates,
            fidelity_tolerance_px
        )

        if results is None:
            raise NoRouteFoundError("No route points found in the extracted mask")

        if results.route_distance is None:
            distance_info = self.calculate_route_distance(
                results.skeleton_mask,
                results.connectivity
            )

            print(distance_info)
            results.route_distance = distance_info['total_distance_px']

        # self.summary(results.alignment_method, results.connectivity, results.endpoints)

        self._create_visualization(results.aligned_route, 
                                   results.final_mask, 
                                   results.points,
                                   results.endpoints, 
                                   output_path, 
                                   marker_size,
                                   results.alignment_method)
        
        return results

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
    parser.add_argument('--tolerance',            type=int, default=3,
                        help='Dilation radius (px) used to forgive residual '
                             'misalignment when subtracting the original image '
                             '(default: 3; increase to 5-8 for larger dimension gaps)')
    parser.add_argument('--output',               default='route_endpoints.png')
    parser.add_argument('--marker-size',          type=int, default=15)
    parser.add_argument('--debug', action='store_true', help='Enable debugging displays (disabled by default)')
    parser.add_argument('--skip-alignment', action='store_true',
                        help='Skip homography alignment. Only safe when the route image '
                             'was drawn directly onto an unmodified copy of the original '
                             'image (same dimensions, same coordinate space).')
    parser.add_argument('--source-coordinates-file', default=None,
                        help='Path to a JSON file containing the original source route '
                             'coordinates (e.g. the VLM output route) as a list of [x, y] '
                             'pairs. When provided, computes a geometric_fidelity score '
                             'comparing the extracted skeleton against this route.')
    parser.add_argument('--fidelity-tolerance-px', type=int, default=3,
                        help='Pixel tolerance used for the geometric_fidelity similarity '
                             'percentage (default: 3, roughly half the drawn line width).')

    args = parser.parse_args()

    source_coordinates = None
    if args.source_coordinates_file:
        with open(args.source_coordinates_file, 'r', encoding='utf-8') as f:
            source_coordinates = json.load(f)

    route_extractor  = RouteExtractor(args.annotations)
    route_extractor.process_pipeline(
        route_image_path      = args.route_image,
        original_image_path   = args.original_image,
        output_path           = args.output,
        marker_size           = args.marker_size,
        difference_threshold  = args.difference_threshold,
        tolerance_px          = args.tolerance,
        debug                 = args.debug,
        skip_alignment        = args.skip_alignment,
        source_coordinates    = source_coordinates,
        fidelity_tolerance_px = args.fidelity_tolerance_px
    )
  

if __name__ == '__main__':
    main()