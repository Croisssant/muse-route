# Annotation Setup Pipeline

Automates the VIA annotation conversion and annotated image generation workflow.

## Overview

The `setup_annotation_pipeline.py` script chains together two annotation processing steps:

1. **Convert VIA to Museum Layout** - Converts VIA project JSON to museum-compatible format
2. **Generate Annotated Image** - Creates color-coded visualization with conditional forbidden areas

## Quick Start

```bash
# Setup annotations for a simple layout
python setup_annotation_pipeline.py -c simple -l layout_01

# Setup annotations for a complex layout
python setup_annotation_pipeline.py -c complex -l layout_03
```

## Prerequisites

Before running the pipeline, ensure your layout directory contains:

### Required Files

1. **via*project*\*.json** (exactly one)
   - Exported from VIA (VGG Image Annotator) tool
   - Contains all annotations: walls, galleries, exhibits, entrances, exits, etc.
   - Pattern: `via_project_[timestamp].json`
   - **Important**: Directory must contain exactly ONE VIA JSON file

2. **Layout Image** (auto-detected)
   - Either `{layout}.jpg` or `{layout}.png`
   - Base floorplan image without annotations

## Pipeline Steps

### Step 1: Convert VIA to Museum Layout

**Command:**

```bash
python via_musuem_helpers/convert_via_to_museum_layout.py \
  {layout_dir}/via_project_*.json \
  {layout_dir}/layout_annotations.json
```

**Output:**

- `layout_annotations.json` - Museum-compatible annotation format

**What it does:**

- Parses VIA project JSON structure
- Filters out unlabeled annotations
- Converts to standardized museum layout format
- Counts annotation types (walls, galleries, exhibits, forbidden_areas, etc.)
- Reports conversion statistics

### Step 2: Generate Annotated Image

**Command:**

```bash
python via_musuem_helpers/generate_custom_annotations.py \
  {layout_dir}/{layout}.jpg \
  {layout_dir}/layout_annotations.json \
  -t entrance exit wall gallery exhibit [forbidden_area] \
  --labels 1 \
  -o {layout_dir}/annotated_layout.png
```

**Output:**

- `annotated_layout.png` - Color-coded annotated floorplan

**What it does:**

- Overlays annotations on base layout image
- Colors each annotation type distinctly
- Adds gallery name labels
- **Conditionally includes forbidden_area**:
  - If `forbidden_areas > 0` in layout_annotations.json → Include `forbidden_area` type
  - If `forbidden_areas = 0` → Exclude `forbidden_area` type

**Annotation Types & Colors:**

- **Wall** - Blue (opaque)
- **Gallery** - Purple (semi-transparent)
- **Entrance** - Green (semi-transparent)
- **Exit** - Yellow (semi-transparent)
- **Exhibit** - Dark gray circles
- **Forbidden Area** - Orange (semi-transparent) - _conditional_

## Command-Line Arguments

| Argument       | Short | Required | Default | Description                                  |
| -------------- | ----- | -------- | ------- | -------------------------------------------- |
| `--complexity` | `-c`  | Yes      | -       | Complexity level (e.g., "simple", "complex") |
| `--layout`     | `-l`  | Yes      | -       | Layout name (e.g., "layout_01", "layout_02") |
| `--types`      | `-t`  | No       | Auto    | Override annotation types to display         |

## Example Usage

### Basic Usage (Auto-detect forbidden areas)

```bash
# Simple layouts
python setup_annotation_pipeline.py -c simple -l layout_01
python setup_annotation_pipeline.py -c simple -l layout_02

# Complex layouts
python setup_annotation_pipeline.py -c complex -l layout_03
```

### Override Annotation Types

```bash
# Only show walls and galleries
python setup_annotation_pipeline.py -c simple -l layout_01 \
  -t wall gallery

# Show everything except exhibits
python setup_annotation_pipeline.py -c simple -l layout_01 \
  -t entrance exit wall gallery forbidden_area
```

## Pipeline Output

### Success Output (With Forbidden Areas)

```
======================================================================
🏛️  ANNOTATION SETUP PIPELINE
======================================================================
Complexity: simple
Layout: layout_01
Directory: floorplans\simple\layout_01

======================================================================
🔍 Validating Prerequisites...
======================================================================
✅ Found: VIA project JSON (via_project_02Apr2026_17h16m56s.json)
✅ Found: layout image (layout_01.png)

======================================================================
📋 Step 1: Convert VIA to Museum Layout
======================================================================
[... conversion output ...]
✅ Step 1 completed successfully!

======================================================================
🔍 Forbidden Area Detection
======================================================================
✅ Detected 2 forbidden area(s)
   → Will include 'forbidden_area' in annotated image

======================================================================
📋 Step 2: Generate Annotated Image
======================================================================
[... image generation output ...]
✅ Step 2 completed successfully!

======================================================================
📊 PIPELINE SUMMARY
======================================================================
✨ All steps completed successfully!

📁 Layout Directory: floorplans\simple\layout_01

📄 Generated Files (2):
   ✅ layout_annotations.json
   ✅ annotated_layout.png

📊 Forbidden Areas: 2
   → Annotated image includes forbidden_area annotations
======================================================================
```

### Success Output (Without Forbidden Areas)

```
======================================================================
🔍 Forbidden Area Detection
======================================================================
ℹ️  No forbidden areas detected
   → Will exclude 'forbidden_area' from annotated image

[... rest similar but excludes forbidden_area type ...]

📊 Forbidden Areas: 0
   → Annotated image excludes forbidden_area annotations
======================================================================
```

## Validation Checks

The pipeline validates:

1. ✅ Exactly ONE `via_project_*.json` file exists (error if 0 or >1)
2. ✅ Layout image exists (.jpg or .png - auto-detected)
3. ✅ Each script completes successfully
4. ✅ Forbidden areas are detected from generated JSON

## Forbidden Area Detection Logic

The pipeline automatically detects forbidden areas:

```python
# After Step 1 completes
1. Read layout_annotations.json
2. Check metadata.counts.forbidden_areas
3. If count > 0:
   → Include 'forbidden_area' in annotation types
4. Else:
   → Exclude 'forbidden_area' from annotation types
```

## Integration with Exhibit Pipeline

### Complete Workflow

```bash
# 1. Create and convert VIA annotations
python setup_annotation_pipeline.py -c simple -l layout_02

# 2. Create exhibit_input.txt manually
# (Define exhibit number assignments to sections)

# 3. Setup exhibits and generate configs
python setup_layout_pipeline.py -c simple -l layout_02
```

**Result**: Fully configured layout ready for route validation!

## Directory Structure

### Before Pipeline

```
floorplans/simple/layout_02/
├── via_project_15Apr2026_11h31m10s.json  [INPUT]
└── layout_02.jpg                          [INPUT]
```

### After Pipeline

```
floorplans/simple/layout_02/
├── via_project_15Apr2026_11h31m10s.json  [INPUT]
├── layout_02.jpg                          [INPUT]
├── layout_annotations.json                [OUTPUT]
└── annotated_layout.png                   [OUTPUT]
```

## Troubleshooting

### "Missing: VIA project JSON file"

**Cause:** No `via_project_*.json` file found

**Solution:**

- Ensure you've exported VIA project to the layout directory
- File must match pattern `via_project_*.json`
- Check the layout path is correct

### "Multiple VIA project JSON files found"

**Cause:** More than one `via_project_*.json` file in directory

**Solution:**

- Keep only the most recent VIA export
- Delete or move old VIA JSON files
- Ensure only ONE VIA file per layout

### "Missing: layout image"

**Cause:** No .jpg or .png file matching layout name

**Solution:**

- Ensure image is named exactly `{layout}.jpg` or `{layout}.png`
- Check the layout name matches the directory name
- Verify image file extension

### "Step 1 failed"

**Cause:** VIA JSON format error or conversion issue

**Solution:**

- Verify VIA JSON is valid and not corrupted
- Check VIA export was complete
- Ensure annotations are properly labeled in VIA

### "Step 2 failed"

**Cause:** Image processing error

**Solution:**

- Verify layout_annotations.json is valid
- Check image file is readable
- Ensure required Python packages are installed (PIL/Pillow)

## Exit Codes

- **0** - Success: All steps completed
- **1** - Failure: Validation error or step failed

## Generated File Details

### layout_annotations.json

Museum-compatible annotation format containing:

```json
{
  "metadata": {
    "total_annotations": 114,
    "export_date": "2026-04-02T17:28:41",
    "counts": {
      "walls": 6,
      "galleries": 2,
      "exhibits": 100,
      "entrances": 1,
      "exits": 1,
      "floor_areas": 1,
      "forbidden_areas": 2,
      "distance_in_mm": 1
    }
  },
  "walls": [...],
  "galleries": [...],
  "exhibits": [...],
  "entrances": [...],
  "exits": [...],
  "floor_areas": [...],
  "forbidden_areas": [...]
}
```

### annotated_layout.png

Visual representation with:

- Color-coded annotations by type
- Gallery name labels
- Semi-transparent overlays
- Conditional forbidden area display

## Advanced Usage

### Custom Annotation Types

Override which annotations to display:

```bash
# Only walls and galleries
python setup_annotation_pipeline.py -c simple -l layout_01 \
  -t wall gallery

# Everything except forbidden areas
python setup_annotation_pipeline.py -c simple -l layout_01 \
  -t entrance exit wall gallery exhibit

# Only exhibits (useful for checking placement)
python setup_annotation_pipeline.py -c simple -l layout_01 \
  -t exhibit
```

## Tips

1. **Verify VIA Export**: Check that your VIA JSON file is complete before running
2. **Check Counts**: Review metadata.counts in layout_annotations.json to verify all annotations converted
3. **Visual Verification**: Open annotated_layout.png to verify annotation accuracy
4. **Forbidden Areas**: Pipeline automatically handles conditional display
5. **Idempotent**: Safe to run multiple times (overwrites previous output)

## Related Scripts

- `convert_via_to_museum_layout.py` - Step 1 (can run standalone)
- `generate_custom_annotations.py` - Step 2 (can run standalone)
- `setup_layout_pipeline.py` - Exhibit setup pipeline (run after this)
- `validation_pipeline.py` - Route validation (uses generated files)

## Complete Workflow Example

```bash
# Step 1: Setup annotations
python setup_annotation_pipeline.py -c simple -l layout_02

# Step 2: Create exhibit_input.txt
# (Edit manually: assign exhibit numbers to sections)

# Step 3: Setup exhibits and configs
python setup_layout_pipeline.py -c simple -l layout_02

# Step 4: Validate a route
python validation_pipeline.py \
  --layout floorplans/simple/layout_02/layout_annotations.json \
  --config floorplans/simple/layout_02/easy_spatial.json \
  --exhibits floorplans/simple/layout_02/exhibit_list.json \
  --route your_route.json
```

## Version History

- **v1.0** - Initial annotation pipeline implementation
  - Automated VIA conversion workflow
  - Conditional forbidden area detection
  - Auto-detect VIA JSON and layout images
  - Configurable annotation types
  - Comprehensive error handling
