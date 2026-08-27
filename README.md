# Muse-Route: Museum Route Planning Benchmark

A benchmark dataset for evaluating AI systems on museum route planning tasks that require simultaneous spatial navigation and semantic reasoning.

## Overview

Museum route planning presents a challenging multi-constraint satisfaction problem where AI systems must generate valid tour routes while simultaneously satisfying:

- **Spatial constraints**: Navigation feasibility, connectivity, collision avoidance
- **Semantic constraints**: Exhibit selection based on cultural attributes, materials, dates, and techniques
- **User preferences**: Start/end locations, must-visit regions, exhibit coverage requirements

Muse-Route provides a standardized benchmark with museum layouts and tasks across varying difficulty levels to evaluate AI route planning capabilities.

## Dataset

### Dataset Structure

```
dataset/
├── croissant_rai_HistoricalArtifactsCollection.json  # Croissant metadata schema
└── selected_exhibits_preprocessed.csv

floorplans/
├── simple/          # (lower complexity)
│   ├── layout_01/
│   ├── layout_02/
│   └── ...
└── complex/         # layouts (higher complexity)
    ├── layout_03/
    ├── layout_04/
    └── ...

Each layout directory contains:
├── layout_XX.png                    # Floor plan image
├── layout_annotations.json          # Spatial annotations (walls, galleries, exhibits)
├── easy_spatial.json                # Easy spatial task config
├── easy_semantic.json               # Easy semantic task config
├── medium_01.json ... _11.json      # 11 medium task configs
├── hard_01.json ... _11.json        # 11 hard task configs
├── exhibits.csv                      # Layout-specific exhibit metadata
├── exhibit_list.json                # Exhibit descriptions
├── exhibit_visualization.png         # Visual exhibit distribution map
└── annotated_layout.png             # Annotated floor plan visualization
```

## Quick Start

### Validate a Route

Evaluate an AI-generated route against spatial and semantic constraints:

```bash
python validation_pipeline.py \
  --route-image your_route.png \
  --original-image floorplans/simple/layout_01/layout_01.png \
  --annotations floorplans/simple/layout_01/layout_annotations.json \
  --config-file floorplans/simple/layout_01/easy_spatial.json \
  --exhibits-csv-file floorplans/simple/layout_01/exhibits.csv \
  --validated-output-json results.json
```

**Optional Parameters:**

- `--extraction-output-image` - Visualize extracted route
- `--validated-output-image` - Visualize validation results
- `--debug` - Enable debugging displays
- `--tolerance` - Dilation radius for alignment (default: 3)
- `--marker-size` - Route point marker size (default: 15)

### Setup New Layout

Automate the complete setup of a new museum layout:

```bash
# Complete pipeline: VIA annotations → exhibits → task configs
python setup_complete_pipeline.py -c simple -l layout_02

# Or run individual pipelines
python setup_annotation_pipeline.py -c simple -l layout_02  # Annotations only
python setup_layout_pipeline.py -c simple -l layout_02       # Exhibits only
```

See detailed documentation: [Complete Pipeline Guide](README/README_complete_pipeline.md)

## Validation Metrics

### SVR (Spatial Validity Rate)

Evaluates basic spatial feasibility of the route:

- **connectivity**: Route forms a connected path
- **wall_crossings**: No route segments cross walls
- **exhibit_collision**: No route points collide with exhibits
- **out_of_area_violations**: Route stays within navigable floor areas

### SCSR (Spatial Constraint Satisfaction Rate)

Evaluates adherence to user-specified spatial constraints:

- **start_end_location**: Route starts/ends at designated entrance/exit
- **must_pass_regions**: Route passes through all required gallery regions
- **restricted_area_violations**: Route avoids forbidden areas
- **distance_budget**: Total route distance within specified budget

### SCAR (Semantic Constraint Adherence Rate)

Evaluates exhibit selection based on semantic attributes:

- **specific_exhibit_coverage**: Visits all specifically required exhibits
- **at_least_n_exhibits_coverage**: Visits minimum number of exhibits
- **exhibit_category_coverage**: Visits all exhibits in specified categories (e.g., all Greek artifacts)
- **attribute_validations**: Visits exhibits matching attribute criteria
  - **OR logic**: Visit exhibits matching ANY of the specified attributes
    - Date ranges (e.g., 500-400 BCE)
    - Materials (e.g., pottery, bronze)
    - Find spots (e.g., Athens, Vulci)
    - Techniques (e.g., painted, glazed)
  - **AND logic**: Visit exhibits matching ALL specified attributes simultaneously

### Validation Metrics by Difficulty

#### Easy-Spatial

1. **Full SVR** (connectivity, wall_crossings, exhibit_collision, out_of_area_violations)
2. **SCSR**
   - start_end_location
   - must_pass_regions
   - restricted_area_violations
3. **SCAR**
   - at_least_n_exhibits_coverage (≥1)

#### Easy-Semantic

1. **Full SVR** (connectivity, wall_crossings, exhibit_collision, out_of_area_violations)
2. **SCSR**
   - start_end_location
3. **SCAR**
   - exhibit_category_coverage / attribute_validations (OR / AND)

#### Medium Task

1. **Full SVR** (connectivity, wall_crossings, exhibit_collision, out_of_area_violations)
2. **SCSR**
   - start_end_location
   - must_pass_regions
   - restricted_area_violations
3. **SCAR**
   - at_least_n_exhibits_coverage (~10 exhibits)
   - exhibit_category_coverage / attribute_validations (OR / AND)

#### Hard Task

1. **Full SVR** (connectivity, wall_crossings, exhibit_collision, out_of_area_violations)
2. **Full SCSR** (start_end_location, must_pass_regions, restricted_area_violations, distance_budget)
3. **SCAR**
   - specific_exhibit_coverage
   - at_least_n_exhibits_coverage (~25 exhibits)
   - exhibit_category_coverage
   - attribute_validations (OR / AND)

## Pipeline Documentation

### Validation Pipeline

**Script**: `validation_pipeline.py`

Evaluates AI-generated museum routes through a comprehensive validation pipeline:

1. **Route Extraction**: Extracts route points and endpoints from route image
2. **Spatial Validation**: Validates spatial feasibility and constraint satisfaction (SVR + SCSR)
3. **Semantic Validation**: Validates exhibit selection against semantic criteria (SCAR)

**Output**: JSON file containing detailed validation results, violation breakdowns, and metrics scores.

### Setup Pipelines

Automate the creation of museum layout configurations:

| Pipeline       | Script                         | Purpose             | Steps | Output Files |
| -------------- | ------------------------------ | ------------------- | ----- | ------------ |
| **Complete**   | `setup_complete_pipeline.py`   | Full layout setup   | 5     | 9 files      |
| **Annotation** | `setup_annotation_pipeline.py` | VIA conversion only | 2     | 2 files      |
| **Layout**     | `setup_layout_pipeline.py`     | Exhibits & configs  | 3     | 7 files      |

**Detailed Guides:**

- [Complete Pipeline Documentation](README/README_complete_pipeline.md)
- [Annotation Pipeline Documentation](README/README_annotation_pipeline.md)
- [Layout Pipeline Documentation](README/README_setup_pipeline.md)

## Project Structure

```
muse-route/
├── dataset/                          # Artifact metadata and preprocessed data
│   ├── croissant_rai_HistoricalArtifactsCollection.json
│   └── selected_exhibits_preprocessed.csv
├── floorplans/                       # Museum layouts and task configurations
│   ├── simple/
│   └── complex/
├── route_extractor/                  # Route extraction module
│   ├── __init__.py
│   └── route_extractor.py
├── spatial_validator/                # Spatial constraint validation module
│   ├── __init__.py
│   └── spatial_validator.py
├── semantic_validator/               # Semantic constraint validation module
│   ├── __init__.py
│   └── semantic_validator.py
├── validation_pipeline.py            # Main validation pipeline
├── metrics.py                        # Evaluation metrics computation
├── setup_complete_pipeline.py        # Complete layout setup automation
├── setup_annotation_pipeline.py      # Annotation conversion automation
├── setup_layout_pipeline.py          # Exhibit and config generation
├── requirements.txt                  # Python dependencies
└── README/                           # Detailed documentation
    ├── README_complete_pipeline.md
    ├── README_annotation_pipeline.md
    └── README_setup_pipeline.md
```

## Dataset Metadata

Complete dataset metadata is available in Croissant format at:

- `dataset/croissant_rai_HistoricalArtifactsCollection.json`

This metadata includes:

- Detailed field descriptions for all artifact attributes
- Data quality and completeness information
- Preprocessing steps and transformations
- RAI (Responsible AI) documentation: limitations, biases, use cases, social impact
- Provenance and derivation information

## Additional Resources

- **Detailed Pipeline Documentation**: See `README/` directory for comprehensive guides
- **Dataset Limitations**: See RAI fields in Croissant metadata for cultural and geographic limitations
- **Data Biases**: Dataset exhibits selection, preservation, and class biases (see Croissant metadata)
- **Intended Use Cases**: Museum route planning, spatial navigation systems, multi-constraint satisfaction benchmarking
- **Dataset Source**: [MBZUAI TimeTravel Dataset](https://huggingface.co/datasets/MBZUAI/TimeTravel)
