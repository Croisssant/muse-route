#!/usr/bin/env python3
"""
Calculate average geometric_fidelity.similarity_percent per model from
final_results.json files under a results directory.

Tasks whose final_results.json has no "geometric_fidelity" key are skipped -
that field is only written for a successfully-parsed, validated route, so its
absence means the task failed (e.g. an out-of-bounds or unparseable route).

After the per-model breakdown, an overall section reports, for each difficulty,
the average of every model's average (each model weighted equally).
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict


# Results folder name -> display label, in the order difficulties are printed
DIFFICULTY_LABELS = {
    'easy_spatial': 'Easy Spatial',
    'easy_semantic': 'Easy Semantic',
    'medium': 'Medium',
    'hard': 'Hard',
}
DIFFICULTY_ORDER = list(DIFFICULTY_LABELS.values())


def scan_model_results(results_dir, model_filter=None):
    """Walk results_dir/{model}/{difficulty}/{complexity}/{task}/final_results.json
    and collect geometric_fidelity.similarity_percent, grouped by model and
    difficulty (Easy Spatial, Easy Semantic, Medium, Hard).

    Returns:
        dict: {model_name: {
            'scores': {'Easy Spatial': [...], 'Easy Semantic': [...], 'Medium': [...],
                       'Hard': [...], 'All': [...]},
            'found': int,     # final_results.json files found
            'skipped': int,   # found but missing geometric_fidelity (failed tasks)
        }}
    """
    results_dir = Path(results_dir)
    model_stats = {}

    model_dirs = [d for d in results_dir.iterdir() if d.is_dir()]
    if model_filter:
        model_dirs = [d for d in model_dirs if d.name == model_filter]

    for model_dir in sorted(model_dirs):
        model_name = model_dir.name
        scores = defaultdict(list)
        found = 0
        skipped = 0

        for difficulty_dir in model_dir.iterdir():
            if not difficulty_dir.is_dir():
                continue

            grouped_difficulty = DIFFICULTY_LABELS.get(difficulty_dir.name)
            if grouped_difficulty is None:
                continue

            for complexity_dir in difficulty_dir.iterdir():
                if not complexity_dir.is_dir():
                    continue

                for task_dir in complexity_dir.iterdir():
                    if not task_dir.is_dir():
                        continue

                    final_json = task_dir / 'final_results.json'
                    if not final_json.exists():
                        continue
                    found += 1

                    try:
                        with open(final_json, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                    except (json.JSONDecodeError, OSError) as e:
                        print(f"⚠️  Could not read {final_json}: {e}")
                        continue

                    similarity = data.get('geometric_fidelity', {}).get('similarity_percent')
                    if similarity is None:
                        skipped += 1
                        continue

                    scores[grouped_difficulty].append(similarity)
                    scores['All'].append(similarity)

        model_stats[model_name] = {'scores': scores, 'found': found, 'skipped': skipped}

    return model_stats


def print_results(model_stats):
    print("\n" + "=" * 80)
    print("AVERAGE GEOMETRIC FIDELITY (similarity_percent) BY MODEL")
    print("(tasks without geometric_fidelity - i.e. failed - are skipped)")
    print("=" * 80 + "\n")

    for model_name in sorted(model_stats.keys()):
        stats = model_stats[model_name]
        scores = stats['scores']
        found = stats['found']
        skipped = stats['skipped']

        print(f"Model: {model_name}")
        print("-" * 80)

        if 'All' in scores:
            avg = sum(scores['All']) / len(scores['All'])
            print(f"  {'All':13} avg={avg:6.2f}%  (n={len(scores['All']):4}/{found:4} tasks, {skipped} skipped/failed)")
        else:
            print(f"  {'All':13} No successful geometric_fidelity results ({found} tasks found, {skipped} skipped/failed)")

        for difficulty in DIFFICULTY_ORDER:
            if difficulty in scores:
                avg = sum(scores[difficulty]) / len(scores[difficulty])
                print(f"  {difficulty:13} avg={avg:6.2f}%  (n={len(scores[difficulty]):4})")

        print()

    print("=" * 80)


def print_overall_summary(model_stats):
    """Print, per difficulty, the mean of each model's average (models weighted equally)."""
    print("\n" + "=" * 80)
    print("OVERALL - AVERAGE OF MODEL AVERAGES BY DIFFICULTY")
    print("(each model counts equally, regardless of how many tasks it completed)")
    print("=" * 80 + "\n")

    for difficulty in DIFFICULTY_ORDER + ['All']:
        model_avgs = [
            sum(stats['scores'][difficulty]) / len(stats['scores'][difficulty])
            for stats in model_stats.values()
            if stats['scores'].get(difficulty)
        ]
        if model_avgs:
            avg = sum(model_avgs) / len(model_avgs)
            print(f"  {difficulty:13} avg={avg:6.2f}%  (across {len(model_avgs)} model{'s' if len(model_avgs) != 1 else ''})")
        else:
            print(f"  {difficulty:13} No results")

    print("\n" + "=" * 80)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Calculate average geometric_fidelity.similarity_percent per model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python calculate_geometric_fidelity.py
  python calculate_geometric_fidelity.py --results-dir ./results_second
  python calculate_geometric_fidelity.py --results-dir ./results_second --model gemma3:4b
        """
    )

    parser.add_argument('--results-dir', type=str, default='results',
                       help='Path to the results directory to scan (default: results)')

    parser.add_argument('--model', type=str, default=None,
                       help='Restrict to a single model folder name (default: all models)')

    return parser.parse_args()


def main():
    args = parse_arguments()
    results_dir = Path(args.results_dir)

    if not results_dir.exists():
        print(f"Error: Results directory not found at {results_dir}")
        return

    print(f"Scanning {results_dir} and calculating geometric fidelity...")
    model_stats = scan_model_results(results_dir, args.model)

    if not model_stats:
        print("No model results found!")
        return

    print_results(model_stats)
    print_overall_summary(model_stats)


if __name__ == '__main__':
    main()
