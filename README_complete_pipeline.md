# Complete Layout Setup Pipeline

**Master pipeline that automates the entire museum layout setup from VIA annotations to validation-ready configs.**

## Overview

The `setup_complete_pipeline.py` script is a master orchestrator that runs both sub-pipelines sequentially:

1. **Annotation Pipeline** (2 steps) → Converts VIA annotations and generates annotated image
2. **Exhibit Pipeline** (3 steps) → Generates exhibits, visualization, and config files

**Result**: Fully configured layout with 9 files ready for route validation!

## Quick Start

```bash
# Complete setup in one command
python setup_complete_pipeline.py -c simple -l layout_02
python setup_complete_pipeline.py -c complex -l layout_03
```

## Prerequisites

Your layout directory must contain these 3 input files:

1. **via*project*\*.json** (exactly one)
   - VIA annotation export from VGG Image Annotator
   - Contains walls, galleries, exhibits, entrances, exits, forbidden areas

2. **{layout}.jpg or .png**
   - Base floorplan image (auto-detected)

3. **exhibit_input.txt**
   - Exhibit number assignments to sections
   - Format: `1-10: Greek`, `11-20: Roman`, etc.

**Plus global file:**

- `exhibits_construction_helpers/data/selected_exhibits_preprocessed.csv`

## Complete Pipeline Flow

```
┌─────────────────────────────────────────────────────────┐
│  VALIDATE ALL PREREQUISITES                             │
│  ✓ VIA JSON (exactly 1)                                 │
│  ✓ Layout image (.jpg/.png)                             │
│  ✓ exhibit_input.txt                                    │
│  ✓ selected_exhibits_preprocessed.csv                   │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  ANNOTATION PIPELINE                                    │
│  Step 1: Convert VIA → layout_annotations.json          │
│  Step 2: Generate annotated_layout.png                  │
│          (conditional forbidden_area)                   │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  EXHIBIT PIPELINE                                       │
│  Step 3: Generate exhibits.csv + exhibit_list.json      │
│  Step 4: Generate exhibit_visualization.png             │
│  Step 5: Generate 4 config files                        │
│          (easy_spatial, easy_semantic, medium, hard)    │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  COMPLETE! 9 files generated                            │
│  Layout ready for route validation                      │
└─────────────────────────────────────────────────────────┘
```

## Generated Files (9 total)

### Annotation Files (2)

1. **layout_annotations.json**
   - Museum-compatible annotation format
   - Contains walls, galleries, exhibits, entrances, exits, forbidden areas
   - Used by validation pipeline

2. **annotated_layout.png**
   - Color-coded visualization of all annotations
   - Includes gallery labels
   - Conditional forbidden area display

### Exhibit Files (3)

3. **exhibits.csv**
   - Full exhibit data with exhibit numbers
   - All artifact attributes (materials, dates, techniques, etc.)

4. **exhibit_list.json**
   - Exhibit descriptions for display
   - Used by validation pipeline

5. **exhibit_visualization.png**
   - Color-coded exhibit map by section
   - Helps verify exhibit distribution

### Config Files (4)

6. **easy_spatial.json**
   - Spatial constraints only
   - Gallery must_see regions
   - Minimal exhibit coverage (1)

7. **easy_semantic.json**
   - Semantic constraints only
   - Attribute/category validation
   - Uses least common values

8. **medium.json**
   - Mixed spatial + semantic constraints
   - ~10 exhibits required
   - Uses most common values

9. **hard.json**
   - All constraint types
   - ~25 exhibits + specific exhibits
   - Full SVR + Full SCSR validation

## Command-Line Arguments

| Argument       | Short | Required | Description                                  |
| -------------- | ----- | -------- | -------------------------------------------- |
| `--complexity` | `-c`  | Yes      | Complexity level (e.g., "simple", "complex") |
| `--layout`     | `-l`  | Yes      | Layout name (e.g., "layout_01", "layout_02") |

## Example Usage

```bash
# Setup simple layouts
python setup_complete_pipeline.py -c simple -l layout_01
python setup_complete_pipeline.py -c simple -l layout_02
python setup_complete_pipeline.py -c simple -l layout_05

# Setup complex layouts
python setup_complete_pipeline.py -c complex -l layout_03
python setup_complete_pipeline.py -c complex -l layout_07
```

## Directory Structure

### Before Pipeline

```
floorplans/simple/layout_02/
├── via_project_15Apr2026_11h31m10s.json  [INPUT]
├── layout_02.jpg                          [INPUT]
└── exhibit_input.txt                      [INPUT]
```

### After Pipeline

```
floorplans/simple/layout_02/
├── via_project_15Apr2026_11h31m10s.json  [INPUT]
├── layout_02.jpg                          [INPUT]
├── exhibit_input.txt                      [INPUT]
│
├── layout_annotations.json                [OUTPUT - Annotations]
├── annotated_layout.png                   [OUTPUT - Annotations]
│
├── exhibits.csv                           [OUTPUT - Exhibits]
├── exhibit_list.json                      [OUTPUT - Exhibits]
├── exhibit_visualization.png              [OUTPUT - Exhibits]
│
├── easy_spatial.json                      [OUTPUT - Configs]
├── easy_semantic.json                     [OUTPUT - Configs]
├── medium.json                            [OUTPUT - Configs]
└── hard.json                              [OUTPUT - Configs]
```

## Pipeline Output Example

```
======================================================================
🏛️  COMPLETE LAYOUT SETUP PIPELINE
======================================================================
Complexity: simple
Layout: layout_01
Directory: floorplans\simple\layout_01

This will run:
  • Annotation Pipeline (2 steps)
  • Exhibit Pipeline (3 steps)
  • Total: 5 automated steps

======================================================================
🔍 Validating All Prerequisites...
======================================================================
✅ Found: VIA project JSON (via_project_02Apr2026_17h16m56s.json)
✅ Found: layout image (layout_01.png)
✅ Found: exhibit_input.txt
✅ Found: selected_exhibits_preprocessed.csv

✅ All prerequisites validated!

======================================================================
🚀 Running Annotation Pipeline
======================================================================
[... annotation pipeline output ...]
✅ Annotation Pipeline completed successfully!

======================================================================
🚀 Running Exhibit Pipeline
======================================================================
[... exhibit pipeline output ...]
✅ Exhibit Pipeline completed successfully!

======================================================================
📊 COMPLETE PIPELINE SUMMARY
======================================================================
✨ Complete layout setup finished successfully!

📁 Layout Directory: floorplans\simple\layout_01

📄 All Generated Files (9 total):

   Annotation Files:
   ✅ layout_annotations.json
   ✅ annotated_layout.png

   Exhibit Files:
   ✅ exhibits.csv
   ✅ exhibit_list.json
   ✅ exhibit_visualization.png

   Config Files:
   ✅ easy_spatial.json
   ✅ easy_semantic.json
   ✅ medium.json
   ✅ hard.json

🎉 Layout is now ready for route validation!

Next steps:
  1. Review exhibit_visualization.png for exhibit distribution
  2. Review annotated_layout.png for annotation accuracy
  3. Use config files with validation_pipeline.py
======================================================================
```

## Validation Checks

The pipeline validates ALL prerequisites before starting:

1. ✅ Exactly ONE `via_project_*.json` file
2. ✅ Layout image exists (.jpg or .png)
3. ✅ `exhibit_input.txt` exists
4. ✅ `selected_exhibits_preprocessed.csv` exists
5. ✅ Each sub-pipeline completes successfully

## Error Handling

### Pipeline Stops on First Error

If any validation or step fails, the entire pipeline stops:

```
❌ Prerequisite validation failed!
   Please ensure all required files exist before running.
```

Or:

```
⚠️  Annotation pipeline failed. Stopping complete pipeline.
```

### Common Errors

**"Multiple VIA project JSON files found"**

- Keep only the most recent VIA export
- Delete old VIA JSON files

**"Missing: exhibit_input.txt"**

- Create exhibit_input.txt with section assignments
- Example format:
  ```
  1-27: Greek
  28-53: Ancient Egyptian
  54-66: Japanese
  67-82: Mesopotamian
  83-100: Tang dynasty
  ```

**"Annotation Pipeline failed"**

- Check VIA JSON is valid
- Verify layout image exists
- See annotation pipeline logs for details

**"Exhibit Pipeline failed"**

- Verify exhibit_input.txt format is correct
- Check section names match database
- Ensure sufficient artifacts available
- See exhibit pipeline logs for details

## Integration with Validation

After running the complete pipeline, validate routes using the generated files:

```bash
# Test with easy spatial config
python validation_pipeline.py \
  --layout floorplans/simple/layout_02/layout_annotations.json \
  --config floorplans/simple/layout_02/easy_spatial.json \
  --exhibits floorplans/simple/layout_02/exhibit_list.json \
  --route your_route.json

# Test with hard config
python validation_pipeline.py \
  --layout floorplans/simple/layout_02/layout_annotations.json \
  --config floorplans/simple/layout_02/hard.json \
  --exhibits floorplans/simple/layout_02/exhibit_list.json \
  --route your_route.json
```

## When to Use Each Pipeline

### Use `setup_complete_pipeline.py` when:

✅ Starting a completely new layout from scratch  
✅ You have all 3 input files ready  
✅ You want full automation (annotations + exhibits + configs)

### Use `setup_annotation_pipeline.py` when:

✅ Only need to convert VIA annotations  
✅ Already have exhibits setup  
✅ Just need annotated image

### Use `setup_layout_pipeline.py` when:

✅ Already have layout_annotations.json  
✅ Only need to setup exhibits and configs  
✅ VIA conversion already done

## Comparison of Pipelines

| Feature                    | Complete Pipeline | Annotation Pipeline | Exhibit Pipeline |
| -------------------------- | ----------------- | ------------------- | ---------------- |
| **Steps**                  | 5                 | 2                   | 3                |
| **Files Generated**        | 9                 | 2                   | 7                |
| **Requires VIA JSON**      | ✅                | ✅                  | ❌               |
| **Requires exhibit_input** | ✅                | ❌                  | ✅               |
| **Requires annotations**   | ❌ (generates)    | ❌ (generates)      | ✅               |
| **Use Case**               | New layout        | Annotations only    | Exhibits only    |

## Tips

1. **Prepare All Inputs First**: Ensure all 3 input files are ready before running
2. **Review Visualizations**: Check both PNG outputs to verify accuracy
3. **Validate Incrementally**: Test each config file with validation_pipeline.py
4. **Rerun if Needed**: Pipeline is idempotent - safe to run multiple times
5. **Check Logs**: If errors occur, review individual pipeline outputs for details

## Exit Codes

- **0** - Success: All 5 steps completed, 9 files generated
- **1** - Failure: Validation error or pipeline failed

## Related Scripts

### Individual Scripts (Standalone)

- `convert_via_to_museum_layout.py` - VIA conversion only
- `generate_exhibit_list.py` - Exhibit list generation only
- `visualize_exhibits_by_section.py` - Exhibit visualization only
- `generate_config_files.py` - Config generation only
- `generate_custom_annotations.py` - Annotated image only

### Pipeline Scripts (Orchestrators)

- `setup_annotation_pipeline.py` - Annotations only (2 steps)
- `setup_layout_pipeline.py` - Exhibits only (3 steps)
- `setup_complete_pipeline.py` - **THIS SCRIPT** - Everything (5 steps)

### Validation

- `validation_pipeline.py` - Validate routes using generated files

## Workflow Diagram

```
VIA Annotations (.json)     Floorplan Image (.jpg/.png)     exhibit_input.txt
        │                              │                            │
        └──────────────┬───────────────┴────────────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────┐
        │  setup_complete_pipeline.py      │
        │  (Master Orchestrator)           │
        └──────────────────────────────────┘
                       │
        ┌──────────────┴──────────────┐
        │                             │
        ▼                             ▼
┌───────────────────┐      ┌──────────────────────┐
│ Annotation        │      │ Exhibit              │
│ Pipeline          │      │ Pipeline             │
│ (2 steps)         │      │ (3 steps)            │
└───────────────────┘      └──────────────────────┘
        │                             │
        ▼                             ▼
┌───────────────────┐      ┌──────────────────────┐
│ • annotations.json│      │ • exhibits.csv       │
│ • annotated.png   │      │ • exhibit_list.json  │
└───────────────────┘      │ • visualization.png  │
                           │ • 4 config files     │
                           └──────────────────────┘
                                     │
                                     ▼
                           ┌──────────────────────┐
                           │ READY FOR VALIDATION │
                           │ (9 files total)      │
                           └──────────────────────┘
```

## What Gets Automated

### ✅ Fully Automated (No Manual Intervention)

- VIA JSON parsing and conversion
- Annotation format standardization
- Forbidden area detection
- Annotated image generation with conditional forbidden areas
- Exhibit list generation from assignments
- Exhibit-section visualization
- Attribute analysis (materials, techniques, locations, dates)
- Config file generation for all 4 difficulty levels
- Gallery must_see/restricted configuration

### 🔧 Manual Steps Required (Before Pipeline)

1. **Create VIA annotations** using VIA tool
2. **Export VIA project** to layout directory
3. **Create exhibit_input.txt** with section assignments

### 📋 Manual Steps Recommended (After Pipeline)

1. **Review visualizations** (annotated_layout.png, exhibit_visualization.png)
2. **Verify configs** match your difficulty requirements
3. **Test with validation_pipeline.py**

## Complete Workflow Example

### Step-by-Step Setup

```bash
# Step 1: Prepare your layout directory
mkdir floorplans/simple/layout_02
# Add: via_project_*.json, layout_02.jpg, exhibit_input.txt

# Step 2: Run complete pipeline
python setup_complete_pipeline.py -c simple -l layout_02

# Step 3: Review outputs
# - Check annotated_layout.png
# - Check exhibit_visualization.png
# - Verify config files

# Step 4: Test validation
python validation_pipeline.py \
  --layout floorplans/simple/layout_02/layout_annotations.json \
  --config floorplans/simple/layout_02/easy_spatial.json \
  --exhibits floorplans/simple/layout_02/exhibit_list.json \
  --route test_route.json
```

## Troubleshooting

### Pipeline Stops at Validation

**Symptom**: Pipeline stops before Step 1

**Common Causes**:

1. Missing input files
2. Multiple VIA JSON files
3. Wrong file names

**Solution**: Check validation output and ensure all files exist with correct names

### Annotation Pipeline Fails

**Symptom**: Pipeline fails after Step 1 or 2

**Common Causes**:

1. Invalid VIA JSON format
2. Image file unreadable
3. Missing Python packages (PIL/Pillow)

**Solution**: Run annotation pipeline standalone to see detailed error:

```bash
python setup_annotation_pipeline.py -c simple -l layout_02
```

### Exhibit Pipeline Fails

**Symptom**: Pipeline fails after Step 3, 4, or 5

**Common Causes**:

1. Invalid exhibit_input.txt format
2. Section names don't match database
3. Insufficient artifacts for requested sections
4. Missing matplotlib or pandas packages

**Solution**: Run exhibit pipeline standalone to see detailed error:

```bash
python setup_layout_pipeline.py -c simple -l layout_02
```

## Exit Codes

- **0** - Success: All 5 steps completed, 9 files generated
- **1** - Failure: Validation failed or any pipeline step failed

## Performance

Typical execution time per layout:

- **Simple layouts** (~100 exhibits): 10-20 seconds
- **Complex layouts** (~200+ exhibits): 20-40 seconds

_Times vary based on system performance and exhibit count_

## Advantages of Complete Pipeline

### vs. Individual Scripts

✅ **One command** instead of 5 separate commands  
✅ **Upfront validation** of all prerequisites  
✅ **Stops on first error** - no partial results  
✅ **Comprehensive summary** of all generated files  
✅ **Consistent execution** - same order every time

### vs. Manual Setup

✅ **Eliminates errors** from manual command entry  
✅ **Enforces workflow** order  
✅ **Validates dependencies** between steps  
✅ **Saves time** - full automation  
✅ **Reproducible** - same process for every layout

## Best Practices

1. **Organize Input Files**: Keep all 3 input files in layout directory before starting

2. **One VIA Export**: Ensure only ONE via*project*\*.json file exists

3. **Name Consistently**: Layout directory name must match layout image name

4. **Test Early**: Run pipeline on simple layouts first to catch issues

5. **Review Outputs**: Always check both visualization images for accuracy

6. **Version Control**: Commit input files but consider .gitignore for generated files

## Comparison Table

| Script Name                    | Pipelines Run | Steps | Files | Use When                    |
| ------------------------------ | ------------- | ----- | ----- | --------------------------- |
| `setup_complete_pipeline.py`   | Both          | 5     | 9     | **New layout from scratch** |
| `setup_annotation_pipeline.py` | Annotations   | 2     | 2     | Annotations only            |
| `setup_layout_pipeline.py`     | Exhibits      | 3     | 7     | Exhibits only               |

## Version History

- **v1.0** - Initial complete pipeline implementation
  - Combines annotation and exhibit pipelines
  - Comprehensive prerequisite validation
  - Sequential execution with error handling
  - Complete summary of all generated files
  - Exit code support for script integration
