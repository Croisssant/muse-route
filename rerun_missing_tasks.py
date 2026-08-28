#!/usr/bin/env python3
"""
Script to rerun missing tasks based on missing_final_results.json.
Automatically generates and executes the appropriate prompt script commands.
"""

import json
import argparse
import subprocess
import sys
from pathlib import Path
from collections import defaultdict
import re

from check_missing_results import (
    discover_expected_tasks,
    check_missing_directories,
    check_missing_final_results,
    build_missing_by_model,
)


def parse_model_and_reasoning(model_folder):
    """
    Parse model folder name into model and reasoning components.
    Examples:
        "gpt-5.4(high)" -> ("gpt-5.4", "high")
        "gpt-5-pro" -> ("gpt-5-pro", None)
    """
    match = re.match(r'^(.+?)\((.+?)\)$', model_folder)
    if match:
        return match.group(1), match.group(2)
    return model_folder, None


def parse_missing_path(path):
    """
    Parse a missing path into its components.
    Example: "gpt-5.4(high)\\hard\\simple\\hard_01_layout_15"
    Returns: {
        'model_folder': 'gpt-5.4(high)',
        'difficulty': 'hard',
        'complexity': 'simple',
        'task': 'hard_01',
        'layout': 'layout_15'
    }
    """
    parts = path.replace('\\', '/').split('/')
    
    if len(parts) != 4:
        return None
    
    model_folder, difficulty, complexity, full_name = parts
    
    # Extract task and layout from the full name
    # Patterns: "hard_01_layout_15", "medium_02_layout_03", "easy_semantic_layout_05"
    if difficulty in ['hard', 'medium']:
        # Format: task_layout (e.g., "hard_01_layout_15")
        match = re.match(r'^(.+?)_(layout_\d+)$', full_name)
        if match:
            task = match.group(1)
            layout = match.group(2)
        else:
            return None
    else:
        # For easy_semantic and easy_spatial, there's no task variant
        # Format: "easy_semantic_layout_05"
        match = re.match(r'^.+?_(layout_\d+)$', full_name)
        if match:
            task = None
            layout = match.group(1)
        else:
            return None
    
    return {
        'model_folder': model_folder,
        'difficulty': difficulty,
        'complexity': complexity,
        'task': task,
        'layout': layout
    }


def get_script_for_difficulty(difficulty):
    """Map difficulty level to the appropriate prompt script."""
    script_map = {
        'easy_semantic': 'prompts_easy_semantic.py',
        'easy_spatial': 'prompts_easy_spatial.py',
        'medium': 'prompts_medium.py',
        'hard': 'prompts_hard.py'
    }
    return script_map.get(difficulty)


def group_missing_tasks(missing_data, model_filter=None, difficulty_filter=None):
    """
    Group missing tasks for efficient command generation.
    Groups by: (model_folder, difficulty, complexity, task) -> [layouts]
    """
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    
    for model_folder, difficulties in missing_data.items():
        # Apply model filter
        if model_filter and model_folder != model_filter:
            continue
        
        for difficulty, paths in difficulties.items():
            # Apply difficulty filter
            if difficulty_filter and difficulty not in difficulty_filter:
                continue
            
            for path in paths:
                parsed = parse_missing_path(path)
                if not parsed:
                    print(f"⚠️  Warning: Could not parse path: {path}")
                    continue
                
                complexity = parsed['complexity']
                task = parsed['task']
                layout = parsed['layout']
                
                # Group by model_folder, difficulty, complexity, and task
                grouped[model_folder][difficulty][f"{complexity}|{task}"].add(layout)
    
    return grouped


def generate_commands(grouped_tasks, backend='openai', timeout=None, base_url=None, results_dir='results'):
    """
    Generate command list from grouped tasks.
    Returns list of tuples: (description, command_args)
    """
    commands = []

    for model_folder, difficulties in sorted(grouped_tasks.items()):
        model, reasoning = parse_model_and_reasoning(model_folder)

        for difficulty, complexity_tasks in sorted(difficulties.items()):
            script = get_script_for_difficulty(difficulty)
            if not script:
                print(f"⚠️  Warning: Unknown difficulty '{difficulty}' for model {model_folder}")
                continue

            for complexity_task_key, layouts in sorted(complexity_tasks.items()):
                complexity, task = complexity_task_key.split('|')

                # Build command
                cmd = ['python', script, '--backend', backend, '--model', model]

                # Add reasoning if present
                if reasoning:
                    cmd.extend(['--reasoning', reasoning])

                # Add complexity
                cmd.extend(['--complexity-mode', 'single', '--complexity', complexity])

                # Add layouts
                cmd.extend(['--layout-mode', 'list', '--layouts'] + sorted(layouts))

                # Add tasks if applicable (hard/medium have task variants)
                if task != 'None':
                    cmd.extend(['--tasks', task])

                # Add base_url if specified
                if base_url:
                    cmd.extend(['--base-url', base_url])

                # Add timeout if specified
                if timeout is not None:
                    cmd.extend(['--timeout', str(timeout)])

                # Results dir - so reruns land back in the same directory the
                # missing tasks were discovered from, not the script's default
                cmd.extend(['--results-dir', results_dir])

                # Description for display
                task_display = f"[{task}]" if task != 'None' else ""
                desc = f"{model_folder} | {difficulty} | {complexity} {task_display} | {len(layouts)} layouts"

                commands.append((desc, cmd))

    return commands


def execute_commands(commands, dry_run=False):
    """Execute or display the generated commands."""
    if dry_run:
        print("DRY RUN MODE - Commands that would be executed:\n")
        print("=" * 80)
        for i, (desc, cmd) in enumerate(commands, 1):
            print(f"\n[{i}/{len(commands)}] {desc}")
            print(f"Command: {' '.join(cmd)}")
        print("\n" + "=" * 80)
        print(f"\nTotal commands: {len(commands)}")
        return
    
    print(f"Executing {len(commands)} command(s)...\n")
    print("=" * 80)
    
    results = []
    for i, (desc, cmd) in enumerate(commands, 1):
        print(f"\n[{i}/{len(commands)}] {desc}")
        print(f"Running: {' '.join(cmd)}")
        print("-" * 80)
        
        try:
            result = subprocess.run(cmd, check=False, capture_output=False, text=True)
            success = result.returncode == 0
            results.append((desc, success))
            
            if success:
                print(f"✓ Command completed successfully")
            else:
                print(f"✗ Command failed with exit code: {result.returncode}")
                
        except Exception as e:
            print(f"✗ Error executing command: {e}")
            results.append((desc, False))
    
    # Summary
    print("\n" + "=" * 80)
    print("EXECUTION SUMMARY")
    print("=" * 80)
    successful = sum(1 for _, success in results if success)
    print(f"Total commands: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {len(results) - successful}")
    
    if len(results) - successful > 0:
        print("\nFailed commands:")
        for desc, success in results:
            if not success:
                print(f"  ✗ {desc}")


def main():
    parser = argparse.ArgumentParser(
        description='Rerun missing tasks based on missing_final_results.json',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Rerun all missing tasks for all models
  python rerun_missing_tasks.py

  # Rerun only for specific model
  python rerun_missing_tasks.py --model "gpt-5.4(high)"

  # Dry-run to see what would be executed
  python rerun_missing_tasks.py --dry-run

  # Rerun only hard tasks with HuggingFace backend
  python rerun_missing_tasks.py --difficulty hard --backend huggingface
  
  # Rerun with Ollama on custom port
  python rerun_missing_tasks.py --backend ollama --base-url http://127.0.0.1:11435

  # Rerun for multiple difficulties
  python rerun_missing_tasks.py --difficulty hard medium
        """
    )
    
    parser.add_argument('--input-file', type=str, default='missing_final_results.json',
                       help='Path to missing results JSON file (default: missing_final_results.json). '
                            'Ignored when --results-dir is given.')

    parser.add_argument('--results-dir', type=str, default=None,
                       help='Results directory to check and rerun into (e.g. ./results_second). '
                            'When given, missing tasks are discovered live from this directory '
                            'instead of reading --input-file, and generated commands write back '
                            'into this same directory. Default: read --input-file, write into "results".')

    parser.add_argument('--model', type=str, default=None,
                       help='Filter by specific model folder name (e.g., "gpt-5.4(high)"). Default: process all models')
    
    parser.add_argument('--difficulty', nargs='+', type=str, default=None,
                       help='Filter by specific difficulty level(s) (e.g., hard medium). Default: process all')
    
    parser.add_argument('--backend', type=str, default='openai',
                       choices=['openai', 'huggingface', 'ollama'],
                       help='Backend to use for model execution (default: openai)')
    
    parser.add_argument('--base-url', type=str, default='http://127.0.0.1:11434',
                       help='Base URL for Ollama backend (default: http://127.0.0.1:11434)')
    
    parser.add_argument('--timeout', type=int, default=None,
                       help='Timeout in seconds for each layout processing (default: None, no timeout)')
    
    parser.add_argument('--dry-run', action='store_true',
                       help='Show commands without executing them')
    
    args = parser.parse_args()

    if args.results_dir:
        # Live mode: discover missing tasks directly from the results
        # directory instead of a pre-generated missing_final_results.json -
        # always fresh, and scoped to exactly this results-dir.
        if not Path(args.results_dir).exists():
            print(f"✗ Error: Results directory '{args.results_dir}' not found!")
            sys.exit(1)

        results_dir = args.results_dir
        print(f"📋 Discovering missing tasks live from: {results_dir}")
        expected_tasks = discover_expected_tasks()
        missing_dirs = check_missing_directories(expected_tasks, results_dir)
        missing_files, _ = check_missing_final_results(results_dir)
        missing_by_model = build_missing_by_model(missing_dirs, missing_files)
    else:
        results_dir = 'results'
        if not Path(args.input_file).exists():
            print(f"✗ Error: Input file '{args.input_file}' not found!")
            sys.exit(1)

        with open(args.input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        missing_by_model = data.get('missing_by_model', {})

    if not missing_by_model:
        print("✓ No missing results found!")
        return

    # Display summary
    print("=" * 80)
    print("RERUN MISSING TASKS")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Results dir: {results_dir}")
    if not args.results_dir:
        print(f"  Input file: {args.input_file}")
    print(f"  Backend: {args.backend}")
    print(f"  Model filter: {args.model if args.model else 'All models'}")
    print(f"  Difficulty filter: {args.difficulty if args.difficulty else 'All difficulties'}")
    if args.timeout:
        print(f"  Timeout: {args.timeout}s per layout")
    print(f"  Mode: {'DRY RUN' if args.dry_run else 'EXECUTE'}")
    print("=" * 80)
    
    # Group tasks
    grouped = group_missing_tasks(missing_by_model, args.model, args.difficulty)
    
    if not grouped:
        print("\n✗ No tasks match the specified filters!")
        return
    
    # Generate commands
    commands = generate_commands(grouped, args.backend, args.timeout, args.base_url, results_dir)
    
    if not commands:
        print("\n✗ No commands generated!")
        return
    
    # Execute or display commands
    execute_commands(commands, args.dry_run)
    
    if args.dry_run:
        print("\nTo execute, run without --dry-run flag")


if __name__ == '__main__':
    main()
