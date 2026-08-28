#!/usr/bin/env python3
"""
Script to check for missing results in the results directory.
Checks both:
1. Missing result directories (layout x task combinations that should exist but don't)
2. Missing final_results.json files in existing directories
"""

import os
import glob
import json
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime


def discover_expected_tasks(floorplans_dir='floorplans'):
    """
    Discover all expected layout x task combinations from the floorplans directory.
    Returns a dictionary mapping (complexity, layout) -> [task_files]
    """
    expected_tasks = defaultdict(list)

    # Fixed single-file tasks, plus glob patterns for however many
    # medium/hard variations exist per layout (count is not hardcoded here -
    # generate_config_files.py controls how many get generated).
    static_task_files = {
        'easy_semantic.json': 'easy_semantic',
        'easy_spatial.json': 'easy_spatial',
    }
    dynamic_patterns = [
        ('medium_*.json', 'medium'),
        ('hard_*.json', 'hard'),
    ]

    # Walk through the floorplans directory
    for complexity in ['simple', 'complex']:
        complexity_path = os.path.join(floorplans_dir, complexity)
        if not os.path.exists(complexity_path):
            continue

        for layout in sorted(os.listdir(complexity_path)):
            layout_path = os.path.join(complexity_path, layout)
            if not os.path.isdir(layout_path):
                continue

            # Check which static task files exist for this layout
            for task_file, difficulty in static_task_files.items():
                task_path = os.path.join(layout_path, task_file)
                if os.path.exists(task_path):
                    key = (complexity, layout, difficulty, task_file)
                    expected_tasks[key] = task_path

            # Discover however many medium/hard variations actually exist
            for pattern, difficulty in dynamic_patterns:
                for task_path in sorted(glob.glob(os.path.join(layout_path, pattern))):
                    task_file = os.path.basename(task_path)
                    key = (complexity, layout, difficulty, task_file)
                    expected_tasks[key] = task_path

    return expected_tasks


def get_expected_result_folder_name(task_file, layout):
    """
    Get the expected result folder name based on task file and layout.
    Examples:
    - easy_semantic.json, layout_01 -> easy_semantic_layout_01
    - hard_01.json, layout_03 -> hard_01_layout_03
    """
    # Remove .json extension
    task_name = task_file.replace('.json', '')
    return f"{task_name}_{layout}"


def check_missing_directories(expected_tasks, results_dir='results'):
    """
    Check for missing result directories (layout x task combinations).
    Returns a dictionary grouped by model and difficulty.
    """
    missing_dirs = defaultdict(lambda: defaultdict(list))
    
    # Get all models in results directory
    if not os.path.exists(results_dir):
        return missing_dirs
        
    models = [d for d in os.listdir(results_dir) 
              if os.path.isdir(os.path.join(results_dir, d))]
    
    # For each model, check if all expected tasks exist
    for model in models:
        for (complexity, layout, difficulty, task_file), _ in expected_tasks.items():
            # Build expected path
            expected_folder = get_expected_result_folder_name(task_file, layout)
            expected_path = os.path.join(results_dir, model, difficulty, complexity, expected_folder)
            
            # Check if directory exists
            if not os.path.exists(expected_path):
                relative_path = os.path.relpath(expected_path, results_dir)
                missing_dirs[model][difficulty].append({
                    'path': relative_path,
                    'complexity': complexity,
                    'layout': layout,
                    'task': task_file
                })
    
    return missing_dirs


def check_missing_final_results(results_dir='results'):
    """
    Check for missing final_results.json files in existing directories.
    Returns a dictionary grouped by model and difficulty.
    """
    missing_files = defaultdict(lambda: defaultdict(list))
    total_folders = 0
    
    # Walk through the results directory
    for root, dirs, files in os.walk(results_dir):
        # Check if this directory contains files (indicating it's a leaf directory)
        if files and not dirs:
            total_folders += 1
            
            # Check if final_results.json exists
            if 'final_results.json' not in files:
                # Parse the path to extract model, difficulty, complexity
                path_parts = Path(root).parts
                
                if len(path_parts) >= 4:  # results/model/difficulty/complexity/layout
                    model = path_parts[1]
                    difficulty = path_parts[2]
                    relative_path = os.path.relpath(root, results_dir)
                    
                    missing_files[model][difficulty].append(relative_path)
    
    return missing_files, total_folders


def print_missing_results(missing_dirs, missing_files, total_folders, expected_count):
    """
    Print the missing directories and files grouped by model and difficulty.
    """
    total_missing_dirs = sum(len(tasks) for model_data in missing_dirs.values() 
                             for tasks in model_data.values())
    total_missing_files = sum(len(folders) for model_data in missing_files.values() 
                              for folders in model_data.values())
    
    print("=" * 80)
    print("MISSING RESULTS REPORT")
    print("=" * 80)
    
    # Get all models from both dictionaries
    all_models = set(missing_dirs.keys()) | set(missing_files.keys())
    
    if total_missing_dirs == 0 and total_missing_files == 0:
        print(f"\n✓ All {total_folders} folders exist and have final_results.json!")
        return
    
    # Print for each model
    for model in sorted(all_models):
        print(f"\n{'='*80}")
        print(f"📁 Model: {model}")
        print(f"{'='*80}")
        
        # Get all difficulties for this model
        all_difficulties = set()
        if model in missing_dirs:
            all_difficulties.update(missing_dirs[model].keys())
        if model in missing_files:
            all_difficulties.update(missing_files[model].keys())
        
        model_missing_dirs = 0
        model_missing_files = 0
        
        for difficulty in sorted(all_difficulties):
            print(f"\n  📂 Category: {difficulty}")
            print(f"  {'-'*76}")
            
            # Print missing directories
            if model in missing_dirs and difficulty in missing_dirs[model]:
                dirs_list = missing_dirs[model][difficulty]
                if dirs_list:
                    print(f"\n    ❌ Missing Result Directories ({len(dirs_list)}):")
                    for item in sorted(dirs_list, key=lambda x: x['path']):
                        print(f"      • {item['path']}")
                        print(f"        └─ Missing: {item['complexity']}/{item['layout']} × {item['task']}")
                    model_missing_dirs += len(dirs_list)
            
            # Print missing final_results.json
            if model in missing_files and difficulty in missing_files[model]:
                files_list = missing_files[model][difficulty]
                if files_list:
                    print(f"\n    ⚠️  Missing final_results.json ({len(files_list)}):")
                    for path in sorted(files_list):
                        print(f"      • {path}")
                    model_missing_files += len(files_list)
        
        # Model summary
        print(f"\n  {'-'*76}")
        print(f"  Subtotal for {model}:")
        print(f"    • Missing directories: {model_missing_dirs}")
        print(f"    • Missing final_results.json: {model_missing_files}")
        print(f"    • Total issues: {model_missing_dirs + model_missing_files}")
    
    # Overall summary
    print(f"\n{'='*80}")
    print(f"OVERALL SUMMARY")
    print(f"{'='*80}")
    print(f"  Expected total result folders per model: {expected_count}")
    print(f"  Actual existing folders checked: {total_folders}")
    print(f"  Missing result directories: {total_missing_dirs}")
    print(f"  Missing final_results.json: {total_missing_files}")
    print(f"  Total issues: {total_missing_dirs + total_missing_files}")
    if total_folders > 0:
        print(f"  Completion rate (for existing folders): {((total_folders - total_missing_files) / total_folders * 100):.1f}%")


def print_model_summary(missing_dirs, missing_files):
    """
    Print a summary of missing results count per model with breakdown by difficulty.
    """
    # Get all models
    all_models = set(missing_dirs.keys()) | set(missing_files.keys())
    
    if not all_models:
        print("\n" + "=" * 80)
        print("MISSING RESULTS SUMMARY BY MODEL")
        print("=" * 80)
        print("\n✓ No missing results found!")
        return
    
    print("\n" + "=" * 80)
    print("MISSING RESULTS SUMMARY BY MODEL")
    print("=" * 80)
    
    # Calculate and display for each model
    for model in sorted(all_models):
        # Get all difficulties for this model
        all_difficulties = set()
        if model in missing_dirs:
            all_difficulties.update(missing_dirs[model].keys())
        if model in missing_files:
            all_difficulties.update(missing_files[model].keys())
        
        print(f"\nModel: {model}")
        
        model_total = 0
        for difficulty in sorted(all_difficulties):
            count = 0
            
            # Count missing directories
            if model in missing_dirs and difficulty in missing_dirs[model]:
                count += len(missing_dirs[model][difficulty])
            
            # Count missing files
            if model in missing_files and difficulty in missing_files[model]:
                count += len(missing_files[model][difficulty])
            
            print(f"  • {difficulty}: {count}")
            model_total += count
        
        print(f"  Total: {model_total}")
    
    print("=" * 80)


def build_missing_by_model(missing_dirs, missing_files):
    """
    Merge missing directories and missing final_results.json entries into a
    single {model: {difficulty: [paths]}} structure - the 'missing_by_model'
    format rerun_missing_tasks.py consumes, either from a saved JSON file or
    computed live in-process.
    """
    missing_by_model = {}
    all_models = set(missing_dirs.keys()) | set(missing_files.keys())

    for model in sorted(all_models):
        missing_by_model[model] = {}

        # Get all difficulties for this model
        all_difficulties = set()
        if model in missing_dirs:
            all_difficulties.update(missing_dirs[model].keys())
        if model in missing_files:
            all_difficulties.update(missing_files[model].keys())

        # Merge missing directories and missing files into a single list per difficulty
        for difficulty in sorted(all_difficulties):
            combined_paths = []

            # Add missing directories (extract just the path string)
            if model in missing_dirs and difficulty in missing_dirs[model]:
                combined_paths.extend([item['path'] for item in missing_dirs[model][difficulty]])

            # Add missing files (already just path strings)
            if model in missing_files and difficulty in missing_files[model]:
                combined_paths.extend(missing_files[model][difficulty])

            # Sort and store
            missing_by_model[model][difficulty] = sorted(combined_paths)

    return missing_by_model


def save_to_json(missing_dirs, missing_files, total_folders, expected_count, output_file='missing_final_results.json'):
    """
    Save the missing results data to a JSON file.
    Maintains compatibility with rerun_missing_tasks.py by using 'missing_by_model' format.
    """
    # Calculate totals
    total_missing_dirs = sum(len(tasks) for model_data in missing_dirs.values()
                             for tasks in model_data.values())
    total_missing_files = sum(len(folders) for model_data in missing_files.values()
                              for folders in model_data.values())

    # Build output structure (compatible with rerun_missing_tasks.py)
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "expected_folders_per_model": expected_count,
            "actual_existing_folders_checked": total_folders,
            "total_missing_directories": total_missing_dirs,
            "total_missing_final_results": total_missing_files,
            "total_missing": total_missing_dirs + total_missing_files,
            "success_rate_percent": round((total_folders - total_missing_files) / total_folders * 100, 1) if total_folders > 0 else 0
        },
        "missing_by_model": build_missing_by_model(missing_dirs, missing_files)
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    return output_file


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Check for missing results (directories and final_results.json files)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python check_missing_results.py
  python check_missing_results.py --results-dir ./results_second
        """
    )

    parser.add_argument('--floorplans-dir', type=str, default='floorplans',
                       help='Path to the floorplans directory (default: floorplans)')

    parser.add_argument('--results-dir', type=str, default='results',
                       help='Path to the results directory to check (default: results)')

    parser.add_argument('--output-file', type=str, default='missing_final_results.json',
                       help='Path to write the missing-results JSON report to (default: missing_final_results.json)')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    print("Checking for missing results...\n")

    # Check if directories exist
    if not os.path.exists(args.floorplans_dir):
        print(f"Error: '{args.floorplans_dir}' directory not found!")
        exit(1)

    if not os.path.exists(args.results_dir):
        print(f"Error: '{args.results_dir}' directory not found!")
        exit(1)

    # Discover expected tasks
    print("📋 Discovering expected layout × task combinations...")
    expected_tasks = discover_expected_tasks(args.floorplans_dir)
    expected_count = len(expected_tasks)
    print(f"   Found {expected_count} expected combinations\n")

    # Check for missing directories
    print("🔍 Checking for missing result directories...")
    missing_dirs = check_missing_directories(expected_tasks, args.results_dir)

    # Check for missing final_results.json in existing directories
    print("🔍 Checking for missing final_results.json files...")
    missing_files, total_folders = check_missing_final_results(args.results_dir)

    print()

    # Save to JSON file
    output_file = save_to_json(missing_dirs, missing_files, total_folders, expected_count, args.output_file)
    print(f"✓ Results saved to: {output_file}")

    # Print model summary
    print_model_summary(missing_dirs, missing_files)
