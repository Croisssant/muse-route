# Museum Route Validation System

A comprehensive Python-based system for detecting, analyzing, and validating visitor routes on museum floor plans. The system uses computer vision techniques to automatically extract routes from images and validate them against museum layout constraints.

---

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Core Components](#core-components)
4. [Installation & Setup](#installation--setup)
5. [Workflows](#workflows)
6. [Technical Details](#technical-details)
7. [File Formats](#file-formats)
8. [Troubleshooting](#troubleshooting)

---

## Overview

### What Does This System Do?

The Museum Route Validation System provides tools to:

1. **Manage Annotations**: Convert and visualize museum floor plan annotations (walls, exhibits, entrances, exits)
2. **Detect Routes**: Automatically extract visitor routes from hand-drawn images using image differencing
3. **Determine Endpoints**: Identify route start (entrance) and end (exit) points
4. **Validate Routes**: Check routes for violations (wall crossings, exhibit collisions, invalid entry/exit)
5. **Visualize Results**: Generate annotated visualizations showing detected routes and validation results

### Key Capabilities

- ✅ Automatic route detection from image comparisons (no manual coordinate entry)
- ✅ Skeleton extraction for precise centerline representation
- ✅ Entrance/exit-based endpoint determination
- ✅ Comprehensive validation rules (walls, exhibits, floor areas)
- ✅ Visual debugging tools with step-by-step processing
- ✅ Visit zone visualization for exhibit proximity detection

### Use Cases

- **Museum Planning**: Validate proposed visitor routes during floor plan design
- **Visitor Analytics**: Analyze actual visitor paths from tracking data
- **Accessibility**: Verify routes meet accessibility requirements
- **Research**: Study visitor behavior and traffic patterns
- **Education**: Demonstrate computer vision algorithms in practical applications

---

## System Architecture

### Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    MUSEUM ROUTE VALIDATION SYSTEM                │
└─────────────────────────────────────────────────────────────────┘

PHASE 1: Annotation Setup
┌─────────────────────┐
│  VIA Annotation     │ ──────┐
│  Tool (via_xxx.json)│       │
└─────────────────────┘       │
                              ▼
                   ┌────────────────────────────┐
                   │ convert_via_to_museum_     │
                   │ layout.py                  │
                   └────────────────────────────┘
                              │
                              ▼
                   ┌────────────────────────────┐
                   │ museum_layout_             │
                   │ annotations.json           │
                   └────────────────────────────┘
                              │
                              ▼
                   ┌────────────────────────────┐
                   │ generate_custom_           │
                   │ annotations.py             │
                   └────────────────────────────┘
                              │
                              ▼
                   ┌────────────────────────────┐
                   │ layout_entrance_exit.png   │
                   │ (Base floor plan for route │
                   │  drawing)                  │
                   └────────────────────────────┘

PHASE 2: Route Drawing
                   ┌────────────────────────────┐
                   │ User draws route on image  │
                   │ (MS Paint, etc.)           │
                   └────────────────────────────┘
                              │
                              ▼
                   ┌────────────────────────────┐
                   │ valid_route.png            │
                   │ invalid_route.png          │
                   └────────────────────────────┘

PHASE 3: Route Analysis (Choose one or both)
                              │
                 ┌────────────┴────────────┐
                 ▼                         ▼
    ┌─────────────────────┐   ┌─────────────────────┐
    │ determine_route.py  │   │ validate_route.py   │
    │ (Endpoint finding)  │   │ (Full validation)   │
    └─────────────────────┘   └─────────────────────┘
                 │                         │
                 ▼                         ▼
    ┌─────────────────────┐   ┌─────────────────────┐
    │ route_endpoints.png │   │ validation_report.  │
    │                     │   │ json + visual.png   │
    └─────────────────────┘   └─────────────────────┘

PHASE 4: Debugging (Optional)
                   ┌────────────────────────────┐
                   │ visualize_route_detection_ │
                   │ process.py                 │
                   │ (8-panel debug view)       │
                   └────────────────────────────┘
                              │
                              ▼
                   ┌────────────────────────────┐
                   │ route_detection_debug.png  │
                   │ (Step-by-step analysis)    │
                   └────────────────────────────┘
```

### Component Relationships

- **Annotation scripts** provide the ground truth (museum layout structure)
- **Route analysis scripts** use annotations to detect and validate routes
- **Visualization scripts** help debug and understand the detection process
- All scripts share common image processing pipeline (differencing → skeletonization → analysis)

---

## Core Components

### 1. convert_via_to_museum_layout.py

#### What It Does

Converts annotations from VIA (VGG Image Annotator) format to the museum layout JSON format used by validation scripts.

#### How It Works

1. **Loads VIA project JSON** with annotations created in the VIA annotation tool
2. **Filters annotations**: Removes unlabeled items (VIA creates entries for all clicks, even incomplete ones)
3. **Categorizes by entity type**:
   - Walls (polylines)
   - Exhibits (circles with exhibit numbers)
   - Entrances (rectangles)
   - Exits (rectangles)
   - Floor areas (rectangles)
4. **Exports clean JSON** in standardized format for downstream tools

#### Why It's Needed

- VIA produces verbose JSON with many internal fields
- Need clean, simplified format for route validation
- Automatic filtering saves manual cleanup time
- Standardized schema ensures compatibility across tools

#### Usage

```bash
python convert_via_to_museum_layout.py via_project_24Mar2026.json museum_layout_annotations.json
```

#### Input Format

- VIA project JSON exported from VGG Image Annotator
- Must have attribute "Entity" with options: wall, room, exhibit, entrance, exit, floor_area
- Exhibits should have attribute "Exhibit" for numbering

#### Output Format

- Clean JSON with categorized annotations
- Statistics: total counts, filtered items
- Ready for use with validation scripts

---

### 2. generate_custom_annotations.py

#### What It Does

Creates annotated versions of floor plan images with selective display of annotation types (e.g., show only entrance and exit).

#### How It Works

1. **Loads museum layout annotations** from JSON
2. **Loads original floor plan image**
3. **Selectively draws** only requested annotation types:
   - Walls: Blue lines
   - Exhibits: Red circles with numbers
   - Entrances: Green rectangles with labels
   - Exits: Yellow rectangles with labels
   - Floor areas: Light blue semi-transparent boxes
4. **Preserves original image dimensions** exactly (critical for image differencing)
5. **Saves as PNG** with maximum quality

#### Why It's Needed

- **Base Image Creation**: Creates clean floor plans for users to draw routes on
- **Custom Views**: Generate different views for different purposes (navigation vs. exhibits)
- **Exact Dimensions**: Ensures pixel-perfect alignment for route detection
- **User-Friendly**: Provides clear visual reference for route drawing

#### Usage

```bash
# Create image with only entrance and exit for route drawing
python generate_custom_annotations.py \
  museum_layout_01.png \
  museum_layout_annotations.json \
  -t entrance exit \
  -o layout_entrance_exit.png

# Create image with all exhibits
python generate_custom_annotations.py \
  museum_layout_01.png \
  museum_layout_annotations.json \
  -t exhibit \
  -o layout_exhibits.png

# Create complete annotated layout
python generate_custom_annotations.py \
  museum_layout_01.png \
  museum_layout_annotations.json \
  -t wall entrance exit exhibit \
  -o layout_complete.png --legend
```

#### Annotation Types

- `entrance` / `entrances`
- `exit` / `exits`
- `wall` / `walls`
- `room` / `rooms`
- `exhibit` / `exhibits`
- `floor_area` / `floor_areas`

---

### 3. determine_route.py

#### What It Does

Determines the START and END points of a route by finding route pixels within entrance and exit bounding boxes.

#### How It Works

**Algorithm:**

1. **Image Differencing**
   - Loads route image (with drawn route) and original floor plan
   - Computes absolute pixel difference: `difference = |route_img - original_img|`
   - Applies threshold to create binary mask of changed pixels

2. **Region of Interest (ROI) Filtering**
   - Uses museum bounds from annotations
   - Filters out pixels outside museum area (removes edge artifacts)

3. **Morphological Operations**
   - Closing: Fills small gaps in route
   - Opening: Removes small noise
   - Preserves main route structure

4. **Skeletonization**
   - Reduces thick drawn line to 1-pixel-wide centerline
   - Uses iterative erosion algorithm
   - Results in clean, continuous path

5. **Endpoint Detection** (Simplified Approach)
   - Finds all skeleton pixels within **entrance bounding box** → START candidates
   - Finds all skeleton pixels within **exit bounding box** → END candidates
   - Selects points closest to box centers as START and END

6. **Visualization**
   - Shows original route image as base
   - Overlays detected skeleton in **dark blue**
   - Marks START (green circle) and END (red circle)
   - Draws entrance/exit boxes for reference

#### Why It's Needed

- **Quick Route Analysis**: Fast determination of route endpoints
- **Visual Verification**: See exactly what was detected vs. what was drawn
- **Simplified Logic**: No complex graph traversal needed
- **Debugging Tool**: Helps verify route detection is working correctly

#### Usage

```bash
python determine_route.py \
  --route-image valid_route.png \
  --original-image layout_entrance_exit.png \
  --annotations museum_layout_annotations.json \
  --output route_endpoints.png \
  --marker-size 15
```

#### Parameters

- `--route-image`: Image with drawn route
- `--original-image`: Original floor plan (no route)
- `--annotations`: Museum layout annotations JSON
- `--difference-threshold`: Pixel difference threshold (default: 10)
- `--output`: Output visualization path (default: route_endpoints.png)
- `--marker-size`: Size of START/END circles in pixels (default: 15)

#### Output

- **Visualization**: Shows route image with dark blue skeleton overlay + START/END markers
- **Console**: Prints START and END coordinates with statistics

---

### 4. validate_route.py

#### What It Does

Performs comprehensive validation of visitor routes against museum layout constraints, checking for violations and tracking exhibit visits.

#### How It Works

**Detection Phase** (Same as determine_route.py):

1. Image differencing to detect route
2. ROI filtering to focus on museum area
3. Morphological operations to clean route
4. Skeletonization to extract centerline
5. Endpoint determination using entrance/exit boxes

**Validation Phase**:

1. **Entrance Validation**
   - Checks if any route pixels exist in entrance bounding box
   - If yes: Route properly starts from entrance ✅
   - If no: Entrance violation ❌

2. **Exit Validation**
   - Checks if any route pixels exist in exit bounding box
   - If yes: Route properly exits at designated exit ✅
   - If no: Exit violation ❌

3. **Wall Crossing Detection**
   - For each route pixel, checks if it's inside any wall polygon
   - Uses ray-casting algorithm for point-in-polygon testing
   - Wall polygons represent solid wall structures (routes should be OUTSIDE)

4. **Exhibit Collision Detection**
   - For each route pixel, calculates distance to each exhibit center
   - If distance < exhibit_radius: **Collision** (route passes through exhibit) ❌
   - If distance < exhibit_radius + proximity_threshold: **Visit** (route passes near exhibit) ✅

5. **Floor Area Validation**
   - Checks if route stays within designated walkable floor areas
   - Reports violations for points outside floor boundaries

**Visualization Phase**:

1. Loads original floor plan as base
2. Draws actual skeleton pixels (all 6012+ pixels) in green (valid) or orange (invalid)
3. Draws visit zone spheres around ALL exhibits:
   - **Yellow circles**: Exhibits that were visited (route came close enough)
   - **Gray circles**: Exhibits not visited (route too far away)
4. Marks violations with specific symbols
5. Highlights visited exhibits with checkmarks
6. Adds legend explaining all symbols

#### Why It's Needed

- **Compliance Checking**: Ensures routes follow museum rules and safety guidelines
- **Quality Assurance**: Verifies routes don't damage exhibits or cross walls
- **Analytics**: Tracks which exhibits visitors see
- **Optimization**: Helps design better routes by showing visit patterns
- **Documentation**: Provides evidence of route validity for approvals

#### Usage

```bash
python validate_route.py \
  --route-image my_route.png \
  --original-image layout_entrance_exit.png \
  --annotations museum_layout_annotations.json \
  --proximity-threshold 25 \
  --output-json validation_report.json \
  --output-image validation_visual.png
```

#### Parameters

- `--route-image`: Image with drawn route
- `--original-image`: Original floor plan (no route)
- `--annotations`: Museum layout annotations JSON
- `--proximity-threshold`: Distance for exhibit visit detection in pixels (default: 25)
- `--difference-threshold`: Pixel difference threshold (default: 10)
- `--min-route-pixels`: Minimum route pixels to consider valid (default: 50)
- `--output-json`: Validation report JSON path (default: route_validation_report.json)
- `--output-image`: Visualization image path (default: route_validation_visual.png)

#### Output

**JSON Report** (validation_report.json):

```json
{
  "validation_summary": {
    "is_valid": true,
    "total_violations": 0,
    "route_length_pixels": 6012,
    "exhibits_visited": ["1", "5", "12"]
  },
  "violations": {
    "wall_crossings": [],
    "exhibit_collisions": [],
    "entrance_violations": [],
    "exit_violations": [],
    "floor_area_violations": []
  },
  "exhibit_visits": {
    "1": { "visited": true, "closest_distance": 18.5 },
    "2": { "visited": false, "closest_distance": 85.2 }
  }
}
```

**Visualization** (validation_visual.png):

- Original floor plan as base
- Green skeleton showing detected route (all pixels)
- Yellow circles around visited exhibits (visit zones)
- Gray circles around unvisited exhibits
- Violation markers (if any)
- Summary legend with validation status

---

### 5. visualize_route_detection_process.py

#### What It Does

Debug tool that shows step-by-step how routes are detected, displaying all intermediate processing stages in a comprehensive 8-panel grid.

#### How It Works

**8-Panel Visualization** (4 rows × 2 columns):

1. **Panel 1**: Route Image (original with drawn route)
2. **Panel 2**: Original Image (clean floor plan)
3. **Panel 3**: Difference (grayscale absolute difference)
4. **Panel 4**: Threshold (binary mask after thresholding)
5. **Panel 5**: ROI Mask (after museum boundary filtering)
6. **Panel 6**: Morphology (after closing and opening operations)
7. **Panel 7**: Skeleton (final 1-pixel centerline)
8. **Panel 8**: Endpoints (START in green, END in red, with entrance/exit boxes)

**Endpoint Detection Methods**:

- Uses advanced **brute-force longest-path walk** algorithm
- Builds adjacency graph from skeleton pixels
- Finds degree-1 nodes (endpoints with only 1 neighbor)
- Walks longest path from entrance-closest endpoint
- More sophisticated than determine_route.py (for debugging purposes)

**Optional Modes**:

- `--endpoints-only`: Shows only Panel 8 for quick endpoint verification
- `--marker-size`: Adjusts START/END marker sizes

#### Why It's Needed

- **Debugging**: When route detection fails, see exactly where the problem occurs
- **Understanding**: Educational tool to visualize image processing pipeline
- **Tuning**: Helps adjust thresholds and parameters for difficult cases
- **Documentation**: Creates visual explanations of the algorithm
- **Quality Control**: Verify detection working correctly on new image types

#### Usage

```bash
# Full 8-panel debug view
python visualize_route_detection_process.py \
  --route-image valid_route.png \
  --original-image layout_entrance_exit.png \
  --annotations museum_layout_annotations.json \
  --output route_detection_debug.png

# Quick endpoint-only view
python visualize_route_detection_process.py \
  --route-image valid_route.png \
  --original-image layout_entrance_exit.png \
  --annotations museum_layout_annotations.json \
  --output endpoints_only.png \
  --endpoints-only \
  --marker-size 20
```

#### Parameters

- `--route-image`: Image with drawn route
- `--original-image`: Original floor plan
- `--annotations`: Annotations JSON
- `--difference-threshold`: Threshold for difference detection (default: 10)
- `--output`: Output path (default: route_detection_debug.png)
- `--endpoints-only`: Generate only Panel 8 instead of full 8-panel grid
- `--marker-size`: Marker circle size in pixels (default: 10)

#### When To Use This Tool

**Use this when:**

- Route detection is failing (no endpoints found)
- Routes detected in wrong location
- Endpoints detected incorrectly
- Need to adjust difference-threshold parameter
- Want to understand why certain pixels are/aren't detected

**The 8-panel view shows:**

- If difference is being detected (Panel 3)
- If threshold is appropriate (Panel 4)
- If ROI mask is correct (Panel 5)
- If morphology is helping or hurting (Panel 6)
- If skeleton is clean (Panel 7)
- If endpoints are logical (Panel 8)

---

## Installation & Setup

### Prerequisites

**Python 3.7+** with the following packages:

```bash
pip install opencv-python numpy pillow
```

### Package Details

- **opencv-python (cv2)**: Image processing, morphological operations, skeletonization
- **numpy**: Array operations, efficient pixel manipulation
- **Pillow (PIL)**: Image loading/saving, drawing text and shapes

### File Structure

```
muse/
├── README.md (this file)
├── museum_layout_annotations.json (annotations)
├── layout_entrance_exit.png (base floor plan)
│
├── Annotation Management Scripts
├── convert_via_to_museum_layout.py
├── generate_custom_annotations.py
│
├── Route Analysis Scripts
├── determine_route.py
├── validate_route.py
├── visualize_route_detection_process.py
│
└── Sample Data
    ├── valid_route.png
    ├── valid_route_2.png
    ├── invalid_route.png
    └── validation_with_zones.png
```

---

## Workflows

### Workflow 1: Initial Setup (One-Time)

**Goal**: Set up museum floor plan with annotations

```bash
# Step 1: Annotate museum layout in VIA tool
# - Mark walls (polylines)
# - Mark exhibits (circles with numbers)
# - Mark entrance (rectangle)
# - Mark exit (rectangle)
# - Export as via_project_xxx.json

# Step 2: Convert VIA annotations to museum format
python convert_via_to_museum_layout.py \
  via_project_24Mar2026.json \
  museum_layout_annotations.json

# Step 3: Generate base floor plan with entrance/exit
python generate_custom_annotations.py \
  museum_layout_01.png \
  museum_layout_annotations.json \
  -t entrance exit \
  -o layout_entrance_exit.png
```

**Result**: `layout_entrance_exit.png` ready for route drawing

---

### Workflow 2: Route Validation (Repeated)

**Goal**: Draw, detect, and validate visitor routes

```bash
# Step 1: Draw route on base floor plan
# - Open layout_entrance_exit.png in MS Paint or image editor
# - Draw route from entrance to exit
# - Save as valid_route.png (DO NOT change image dimensions)

# Step 2: Quick endpoint check (optional)
python determine_route.py \
  --route-image valid_route.png \
  --original-image layout_entrance_exit.png \
  --annotations museum_layout_annotations.json \
  --output route_endpoints.png

# Step 3: Full validation
python validate_route.py \
  --route-image valid_route.png \
  --original-image layout_entrance_exit.png \
  --annotations museum_layout_annotations.json \
  --proximity-threshold 25 \
  --output-json validation_report.json \
  --output-image validation_visual.png

# Step 4: Review results
# - Open validation_visual.png to see detected route
# - Check validation_report.json for detailed violations
# - Green route = valid, Orange route = invalid
# - Yellow circles = visited exhibits, Gray circles = not visited
```

---

### Workflow 3: Debugging Route Detection

**Goal**: Troubleshoot route detection issues

```bash
# Generate full debug visualization
python visualize_route_detection_process.py \
  --route-image problem_route.png \
  --original-image layout_entrance_exit.png \
  --annotations museum_layout_annotations.json \
  --output debug_full.png

# Check the 8 panels:
# - Panel 3: Is route visible in difference image?
# - Panel 4: Is threshold detecting the route?
# - Panel 5: Is ROI mask correct?
# - Panel 7: Is skeleton clean and continuous?
# - Panel 8: Are endpoints in correct locations?

# Try adjusting parameters based on debug output:
python visualize_route_detection_process.py \
  --route-image problem_route.png \
  --original-image layout_entrance_exit.png \
  --annotations museum_layout_annotations.json \
  --difference-threshold 15 \
  --output debug_adjusted.png
```

**Common Issues & Solutions:**

| Issue             | Panel   | Solution                                                  |
| ----------------- | ------- | --------------------------------------------------------- |
| No route detected | Panel 3 | Images may be identical; check you're using correct files |
| Route too faint   | Panel 4 | Lower `--difference-threshold` (try 5)                    |
| Extra artifacts   | Panel 5 | Check ROI bounds include full museum                      |
| Broken skeleton   | Panel 7 | Route may have gaps; draw more carefully                  |
| Wrong endpoints   | Panel 8 | Entrance/exit boxes may be misaligned                     |

---

## Technical Details

### Image Differencing Algorithm

**Purpose**: Detect drawn route by comparing two images

**Process**:

```python
# 1. Load images in RGB format (ignore alpha)
route_img = load_and_convert_to_RGB(route_image)
original_img = load_and_convert_to_RGB(original_image)

# 2. Compute absolute difference
difference = |route_img - original_img|

# 3. Convert to grayscale
gray_diff = rgb_to_gray(difference)

# 4. Apply threshold
mask = (gray_diff > threshold) ? 255 : 0
```

**Why This Works**:

- Only pixels that changed between images are detected
- Drawn route has different pixel values than floor plan
- Threshold filters out compression artifacts and noise
- Result: Binary mask showing exactly where route was drawn

**Critical Requirements**:

- ✅ Both images must have **identical dimensions** (pixel-perfect)
- ✅ Same file format (PNG recommended)
- ✅ Same base image (original must be unmodified copy)
- ❌ Don't resize, crop, or compress after drawing route

---

### Skeletonization Algorithm

**Purpose**: Extract 1-pixel-wide centerline from thick drawn route

**Method**: Iterative morphological thinning

```python
skeleton = empty_image
while mask has pixels:
    eroded = erode(mask)
    temp = dilate(eroded)
    temp = mask - temp
    skeleton = skeleton | temp
    mask = eroded
```

**Why It's Needed**:

- Hand-drawn routes have varying thickness (5-50 pixels wide)
- Need consistent centerline for distance calculations
- Removes ambiguity about "where" the route actually is
- Preserves topology (connectivity, endpoints)

**Properties**:

- Produces 1-pixel-wide lines
- Maintains route connectivity
- Preserves endpoint locations
- May have small branches (handled by endpoint detection)

---

### Endpoint Detection: Entrance/Exit Box Intersection

**Method** (Used in determine_route.py and validate_route.py):

```python
# 1. Find all skeleton pixels
skeleton_points = extract_nonzero_pixels(skeleton)

# 2. Check which pixels fall in entrance box
entrance_points = [p for p in skeleton_points
                   if point_in_rectangle(p, entrance_box)]

# 3. Choose START = point closest to entrance center
START = min(entrance_points, key=lambda p: distance(p, entrance_center))

# 4. Repeat for exit box
exit_points = [p for p in skeleton_points
               if point_in_rectangle(p, exit_box)]
END = min(exit_points, key=lambda p: distance(p, exit_center))
```

**Advantages**:

- ✅ Simple and fast
- ✅ No complex graph algorithms needed
- ✅ Works even if route has branches or noise
- ✅ Directly enforces entrance/exit constraints

**Alternative Method** (visualize_route_detection_process.py uses this):

- Degree-1 node detection (nodes with only 1 neighbor)
- Brute-force longest-path walk through skeleton graph
- More sophisticated but also more complex
- Used for debugging and comparison purposes

---

### Validation Rules

#### Rule 1: Entrance Constraint

- **Rule**: Route MUST have at least one pixel within entrance bounding box
- **Why**: Routes must start from designated entrance for safety/logistics
- **Detection**: Check if any skeleton pixel intersects entrance rectangle

#### Rule 2: Exit Constraint

- **Rule**: Route MUST have at least one pixel within exit bounding box
- **Why**: Routes must end at designated exit point
- **Detection**: Check if any skeleton pixel intersects exit rectangle

#### Rule 3: Wall Avoidance

- **Rule**: Route must NOT pass through walls
- **Why**: Physical impossibility; walls are solid barriers
- **Detection**: Point-in-polygon test for each route pixel against all wall polygons
- **Note**: Wall polygons represent wall structure itself, not rooms

#### Rule 4: Exhibit Collision

- **Rule**: Route must NOT pass through exhibits (distance < radius)
- **Why**: Exhibits are physical objects that block movement
- **Detection**: Calculate distance from route pixel to exhibit center
- **Violation**: `distance < exhibit_radius`

#### Rule 5: Exhibit Visit

- **Rule**: Route within proximity threshold counts as "visiting" exhibit
- **Why**: Visitors can see/interact with exhibits from nearby
- **Detection**: `exhibit_radius < distance < exhibit_radius + proximity_threshold`
- **Default**: proximity_threshold = 25 pixels

#### Rule 6: Floor Area

- **Rule**: Route should stay within designated floor areas
- **Why**: Some areas may be off-limits (storage, restricted zones)
- **Detection**: Point-in-rectangle test for each route pixel

---

### Visit Zone Calculation

**Exhibit Visit Logic**:

```
Exhibit center (cx, cy) with radius R
Proximity threshold = T (default: 25px)

For each route point P:
  distance = sqrt((Px - cx)² + (Py - cy)²)

  if distance < R:
      → COLLISION (route passes through exhibit) ❌

  elif distance < R + T:
      → VISIT (route passes near exhibit) ✅
      → Exhibit marked as "visited"

  else:
      → NO VISIT (route too far away)
```

**Visualization**:

- Inner circle (solid): Exhibit boundary (radius R)
- Outer circle (dashed): Visit zone (radius R + T)
- Yellow outer circle: Visit zone was entered (exhibit visited)
- Gray outer circle: Visit zone not entered (exhibit not visited)

**Tuning Proximity Threshold**:

- **Smaller values** (10-15px): Require route to pass very close
- **Default** (25px): Reasonable viewing distance
- **Larger values** (40-50px): More lenient, count distant passes

---

## File Formats

### Museum Layout Annotations JSON

**Schema**:

```json
{
  "metadata": {
    "total_annotations": 109,
    "export_date": "2026-03-25T17:40:16.163129",
    "counts": {
      "walls": 6,
      "exhibits": 100,
      "entrances": 1,
      "exits": 1,
      "floor_areas": 1
    }
  },
  "walls": [
    {
      "id": "unique_id",
      "shape": "polyline",
      "coordinates": {
        "points": [
          {"x": 562, "y": 2311},
          {"x": 553, "y": 2295}
        ],
        "num_points": 82
      }
    }
  ],
  "exhibits": [
    {
      "id": "unique_id",
      "shape": "circle",
      "coordinates": {
        "center_x": 1416,
        "center_y": 1320,
        "radius": 22.67
      },
      "exhibit_number": "1",
      "area": 1614
    }
  ],
  "entrances": [
    {
      "id": "unique_id",
      "shape": "rectangle",
      "coordinates": {
        "x": 548.17,
        "y": 2638.83,
        "width": 240.08,
        "height": 302.09
      },
      "area": 72525
    }
  ],
  "exits": [...],
  "floor_areas": [...]
}
```

### Validation Report JSON

**Schema**:

```json
{
  "validation_summary": {
    "is_valid": boolean,
    "total_violations": integer,
    "route_length_pixels": integer,
    "exhibits_visited": [string array of exhibit IDs]
  },
  "violations": {
    "wall_crossings": [
      {
        "point": [x, y],
        "index": pixel_index,
        "wall_number": wall_id,
        "reason": "description"
      }
    ],
    "exhibit_collisions": [
      {
        "exhibit_id": "exhibit_number",
        "point": [x, y],
        "index": pixel_index,
        "distance": float
      }
    ],
    "entrance_violations": [...],
    "exit_violations": [...],
    "floor_area_violations": [...]
  },
  "exhibit_visits": {
    "exhibit_id": {
      "visited": boolean,
      "closest_distance": float
    }
  }
}
```

---

## How Scripts Connect Together

### Information Flow

1. **Annotations are the foundation**:
   - Created in VIA tool → converted by `convert_via_to_museum_layout.py`
   - Used by ALL other scripts as ground truth

2. **Base image generation**:
   - `generate_custom_annotations.py` creates clean floor plans
   - User draws routes on these images
   - Pixel-perfect alignment is critical

3. **Route detection** (shared pipeline):
   - Both `determine_route.py` and `validate_route.py` use same detection:
     - Image differencing
     - Skeletonization
     - Endpoint determination via entrance/exit boxes

4. **Different purposes**:
   - `determine_route.py`: Quick endpoint finding + visualization
   - `validate_route.py`: Full validation + comprehensive reporting
   - `visualize_route_detection_process.py`: Debugging + education

### Shared Code Patterns

All route analysis scripts share:

```python
# Common initialization
class RouteAnalyzer:
    def __init__(self, annotations_file):
        self.annotations = load_json(annotations_file)
        self.entrances = self.annotations['entrances']
        self.exits = self.annotations['exits']
        self.walls = self.annotations['walls']
        self.exhibits = self.annotations['exhibits']

# Common detection pipeline
def extract_route(route_img, original_img):
    difference = compute_difference(route_img, original_img)
    mask = apply_threshold(difference)
    mask = apply_roi_mask(mask, museum_bounds)
    mask = morphological_operations(mask)
    skeleton = skeletonize(mask)
    points = extract_points(skeleton)
    return points

# Common endpoint detection
def find_endpoints_by_boxes(points, entrance_box, exit_box):
    entrance_points = [p for p in points if in_box(p, entrance_box)]
    START = closest_to_center(entrance_points, entrance_box)

    exit_points = [p for p in points if in_box(p, exit_box)]
    END = closest_to_center(exit_points, exit_box)

    return START, END
```

---

## Troubleshooting

### Common Issues

#### Issue 1: "Image dimensions don't match"

**Error Message**:

```
Image dimensions don't match!
  Original: (3195, 1984) (H x W)
  Route: (3200, 1990) (H x W)
```

**Cause**: Route image was resized, cropped, or had canvas size changed

**Solution**:

1. Always use "Save" not "Save As" when drawing routes
2. Don't crop or resize after drawing
3. Check image properties: both should have identical dimensions
4. If using Paint: Don't use "Resize" or "Canvas Size"

---

#### Issue 2: "No route pixels detected"

**Error Message**:

```
Too few route pixels detected (0 < 50).
```

**Cause**: Route not visible in difference image

**Solutions**:

- **Problem**: Images are identical
  - Check you're comparing route image with ORIGINAL floor plan
  - Verify you actually drew a route on the image
- **Problem**: Route too faint
  - Use `--difference-threshold 5` (lower threshold)
  - Draw route with more opaque color
- **Problem**: Wrong color mode
  - Save both images as PNG (not JPEG which has compression)

**Debug**:

```bash
python visualize_route_detection_process.py \
  --route-image your_route.png \
  --original-image layout_entrance_exit.png \
  --annotations museum_layout_annotations.json \
  --output debug.png

# Check Panel 3 (Difference): Should show route clearly
# Check Panel 4 (Threshold): Should show route as white pixels
```

---

#### Issue 3: "No route pixels found in entrance bounding box"

**Error Message**:

```
❌ No route pixels found in entrance bounding box
```

**Cause**: Route doesn't intersect entrance box, or endpoints detected incorrectly

**Solutions**:

1. **Check entrance box alignment**: Run determine_route.py to see entrance box location
2. **Extend route**: Make sure drawn route enters the entrance area
3. **Check annotations**: Verify entrance box is positioned correctly
4. **View debug**: Use visualize_route_detection_process.py Panel 8

---

#### Issue 4: Route detected in wrong location

**Symptoms**: Route appears outside museum, endpoints far from entrance/exit

**Cause**: Image differences in unexpected areas (watermarks, labels, artifacts)

**Solutions**:

1. **Verify base images**: Use EXACT same image as base for both original and route
2. **Check for watermarks**: Remove any signatures, dates, or labels
3. **Use clean images**: Generate base with `generate_custom_annotations.py`
4. **Check format**: Save both as PNG (avoid JPEG compression differences)

**Debug**:

```bash
# See where differences are detected
python visualize_route_detection_process.py \
  --route-image your_route.png \
  --original-image layout_entrance_exit.png \
  --annotations museum_layout_annotations.json \
  --output debug.png

# Check Panel 3: Should show ONLY the route, nothing else
```

---

#### Issue 5: Wrong exhibits marked as visited

**Symptoms**: Exhibits far from route marked as visited, or nearby exhibits not marked

**Cause**: Proximity threshold too large/small, or route detection inaccurate

**Solutions**:

1. **Adjust proximity threshold**:

   ```bash
   # Smaller threshold (strict)
   python validate_route.py ... --proximity-threshold 15

   # Larger threshold (lenient)
   python validate_route.py ... --proximity-threshold 40
   ```

2. **Check route detection**: Use determine_route.py to verify skeleton looks correct

3. **Verify exhibit positions**: Generate image with exhibits to check coordinates

---

## Best Practices

### For Accurate Route Detection

1. **Image Quality**:
   - Use PNG format (lossless)
   - Don't compress or resize
   - Keep original dimensions exactly

2. **Route Drawing**:
   - Use opaque, solid color (avoid semi-transparent)
   - Draw continuous lines (no gaps)
   - Draw within museum bounds
   - Start in entrance, end in exit

3. **Testing Workflow**:
   - First: Run `determine_route.py` to check endpoints
   - Then: Run `validate_route.py` for full analysis
   - If issues: Use `visualize_route_detection_process.py` to debug

4. **Parameter Tuning**:
   - Start with defaults (difference-threshold=10, proximity-threshold=25)
   - Adjust only if detection fails
   - Use debug visualization to guide adjustments

### For Creating Base Floor Plans

1. **Use generate_custom_annotations.py**:
   - Ensures exact dimensions preserved
   - Clean, uncluttered images
   - Only show needed annotations

2. **Don't manually edit**:
   - Avoid adding text, labels, or watermarks
   - Don't crop or resize in image editors
   - Use PNG export from generate script

3. **Version control**:
   - Keep original floor plan separate
   - Name base images clearly (e.g., `layout_entrance_exit.png`)
   - Track which base was used for each route

---

## Advanced Usage

### Batch Validation

Validate multiple routes programmatically:

```python
import glob
from validate_route import RouteValidator

validator = RouteValidator('museum_layout_annotations.json', proximity_threshold=25)

for route_image in glob.glob('routes/*.png'):
    print(f"\nValidating {route_image}...")
    try:
        route_points = validator.extract_route_by_difference(
            route_image,
            'layout_entrance_exit.png',
            difference_threshold=10
        )
        result = validator.validate_route(route_points)
        print(f"Valid: {result['validation_summary']['is_valid']}")
        print(f"Exhibits visited: {len(result['validation_summary']['exhibits_visited'])}")
    except Exception as e:
        print(f"Error: {e}")
```

### Custom Proximity Thresholds Per Exhibit

Currently, all exhibits use the same proximity threshold. To customize per exhibit, modify `validate_route.py`:

```python
# In validate_route method, replace:
elif dist < radius + self.proximity_threshold:

# With:
custom_threshold = exhibit.get('custom_proximity', self.proximity_threshold)
elif dist < radius + custom_threshold:
```

Then add `custom_proximity` field to exhibits in JSON.

---

## Example Validation Results

### Valid Route Example

**Input**: `valid_route.png` drawn from entrance to exit

**Detection Results**:

- 6012 skeleton pixels extracted
- START: (664, 2710) ✅ in entrance box
- END: (918, 2710) ✅ in exit box

**Validation Results**:

- ✅ VALID
- 0 violations
- 0 exhibits visited (route doesn't pass close enough)

**Visualization**: Shows green skeleton on floor plan with gray visit zones (no exhibits visited)

---

### Invalid Route Example

**Input**: `invalid_route.png` crossing through walls

**Detection Results**:

- 4523 skeleton pixels extracted
- START: (665, 2710) ✅ in entrance box
- END: (920, 2710) ✅ in exit box

**Validation Results**:

- ❌ INVALID
- 234 violations (wall crossings)
- Route passes through wall structure at 234 points

**Visualization**: Shows orange skeleton with red markers at wall crossing points

---

## Script Comparison

| Feature         | determine_route.py   | validate_route.py     | visualize_route_detection_process.py |
| --------------- | -------------------- | --------------------- | ------------------------------------ |
| **Purpose**     | Quick endpoint check | Full validation       | Debugging/education                  |
| **Detection**   | Entrance/exit box    | Entrance/exit box     | Longest-path walk                    |
| **Validation**  | No                   | Yes (complete)        | No                                   |
| **Output**      | Visualization only   | JSON + Visualization  | 8-panel grid                         |
| **Speed**       | Fast                 | Medium                | Medium                               |
| **Complexity**  | Simple               | Medium                | Complex                              |
| **When to Use** | Quick checks         | Production validation | Troubleshooting                      |

---

## API Reference (For Programmatic Use)

### RouteDeterminer Class

```python
from determine_route import RouteDeterminer

# Initialize
determiner = RouteDeterminer('museum_layout_annotations.json')

# Determine route
determiner.determine_route(
    route_image_path='valid_route.png',
    original_image_path='layout_entrance_exit.png',
    difference_threshold=10,
    output_path='endpoints.png',
    marker_size=15
)
```

### RouteValidator Class

```python
from validate_route import RouteValidator

# Initialize
validator = RouteValidator(
    'museum_layout_annotations.json',
    proximity_threshold=25
)

# Extract route
route_points = validator.extract_route_by_difference(
    'valid_route.png',
    'layout_entrance_exit.png',
    difference_threshold=10
)

# Validate
result = validator.validate_route(route_points)

# Check results
if result['validation_summary']['is_valid']:
    print(f"Valid route! Visited {len(result['validation_summary']['exhibits_visited'])} exhibits")
else:
    print(f"Invalid: {result['validation_summary']['total_violations']} violations")
```

---

## Performance Considerations

### Processing Time

Typical processing times on standard hardware:

- **Image Differencing**: <0.1s
- **Skeletonization**: 0.5-2s (depends on route complexity)
- **Endpoint Detection**: <0.1s (entrance/exit box method)
- **Validation**: 1-5s (depends on number of exhibits and route pixels)
- **Visualization**: 0.5-1s

**Total**: ~2-8 seconds per route validation

### Memory Usage

- **Small floor plans** (2000×2000px): ~50MB RAM
- **Large floor plans** (4000×4000px): ~200MB RAM
- **Batch processing**: Linear scaling with number of routes

### Optimization Tips

1. **For many routes on same floor plan**:
   - Initialize validator once
   - Reuse for multiple routes (annotations loaded only once)

2. **For large images**:
   - Consider downscaling for visualization (detection still uses full resolution)
   - Script auto-scales if dimension > 1200px

3. **For many exhibits** (>500):
   - Visit detection is O(n) per route pixel
   - Consider spatial indexing for very large museums

---

## Future Enhancements

### Possible Improvements

1. **Path Planning**: Generate optimal routes automatically
2. **Multi-floor Support**: Handle multiple floor plans
3. **Interactive Visualization**: Web-based route drawing tool
4. **Machine Learning**: Learn common route patterns
5. **Real-time Tracking**: Integrate with visitor tracking systems
6. **3D Visualization**: Show routes in 3D museum models
7. **Accessibility Analysis**: Check wheelchair accessibility, slope calculations
8. **Crowd Simulation**: Predict congestion from route patterns

---

## Credits & License

**Museum Route Validation System**

- Version: 1.0
- Last Updated: March 26, 2026
- Repository: https://github.com/Croisssant/muse-route.git

**Dependencies**:

- OpenCV (cv2) - Image processing
- NumPy - Numerical operations
- Pillow (PIL) - Image I/O and drawing

---

## Quick Reference

### Most Common Commands

```bash
# 1. Setup: Convert VIA annotations
python convert_via_to_museum_layout.py via_project.json annotations.json

# 2. Setup: Create base floor plan
python generate_custom_annotations.py layout.png annotations.json -t entrance exit -o base.png

# 3. Draw route on base.png (MS Paint, etc.)

# 4. Validate the route
python validate_route.py \
  --route-image my_route.png \
  --original-image base.png \
  --annotations annotations.json

# 5. If issues, debug with:
python visualize_route_detection_process.py \
  --route-image my_route.png \
  --original-image base.png \
  --annotations annotations.json \
  --output debug.png
```

---

## Support

For issues, questions, or contributions:

- Check the [Troubleshooting](#troubleshooting) section
- Review the [8-panel debug output](#workflow-3-debugging-route-detection)
- Examine sample images in the repository

**Common support scenarios answered**:

- ✅ Route not detected → See "Issue 2: No route pixels detected"
- ✅ Wrong endpoints → See "Issue 3: No route pixels in entrance box"
- ✅ Route looks different → See "Visualization" sections in script docs
- ✅ Exhibits not visited → See "Visit Zone Calculation" section
