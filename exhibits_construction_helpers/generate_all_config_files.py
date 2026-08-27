"""
Batch Config File Generator for Museum Route Validation
Regenerates difficulty-specific config files for every floorplan layout at once.

Usage:
    python generate_all_config_files.py
    python generate_all_config_files.py --medium-variations 11 --hard-variations 11
"""

import argparse
from pathlib import Path

from generate_config_files import (
    DEFAULT_MEDIUM_VARIATIONS,
    DEFAULT_HARD_VARIATIONS,
    generate_configs_for_layout,
)


def find_all_layout_dirs(floorplans_dir='floorplans'):
    """Sorted list of every layout_XX dir under floorplans/{simple,complex}/*."""
    dirs = []
    for complexity in ('simple', 'complex'):
        base = Path(floorplans_dir) / complexity
        if base.exists():
            dirs.extend(sorted(p for p in base.iterdir() if p.is_dir()))
    return dirs


def main():
    parser = argparse.ArgumentParser(
        description='Regenerate difficulty-specific config files for all floorplan layouts',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '-f', '--floorplans-dir',
        type=str,
        default='floorplans',
        help='Path to the floorplans directory containing simple/ and complex/ (default: floorplans)'
    )

    parser.add_argument(
        '-c', '--csv',
        type=str,
        default='exhibits.csv',
        help='Name of the exhibits CSV file within each layout (default: exhibits.csv)'
    )

    parser.add_argument(
        '--medium-variations',
        type=int,
        default=DEFAULT_MEDIUM_VARIATIONS,
        help=f'Number of medium task variations to generate per layout (default: {DEFAULT_MEDIUM_VARIATIONS})'
    )

    parser.add_argument(
        '--hard-variations',
        type=int,
        default=DEFAULT_HARD_VARIATIONS,
        help=f'Number of hard task variations to generate per layout (default: {DEFAULT_HARD_VARIATIONS})'
    )

    args = parser.parse_args()

    layout_dirs = find_all_layout_dirs(args.floorplans_dir)
    if not layout_dirs:
        print(f"❌ Error: No layout directories found under {args.floorplans_dir}/{{simple,complex}}")
        return

    print("=" * 60)
    print("🏛️  Batch Config File Generator for Museum Route Validation")
    print("=" * 60)
    print(f"Found {len(layout_dirs)} layout directories")
    print(f"Medium variations/layout: {args.medium_variations}")
    print(f"Hard variations/layout: {args.hard_variations}")
    print("=" * 60)

    summaries = []
    failures = []
    total_duplicate_warnings = 0

    for layout_path in layout_dirs:
        try:
            summary = generate_configs_for_layout(
                layout_path,
                csv_filename=args.csv,
                medium_variations=args.medium_variations,
                hard_variations=args.hard_variations,
                verbose=False,
            )
            summaries.append(summary)
            n_warnings = len(summary['duplicate_warnings'])
            total_duplicate_warnings += n_warnings
            suffix = f" — {n_warnings} duplicate warning(s)" if n_warnings else ""
            print(f"✅ {layout_path}: {summary['total_count']} files "
                  f"({summary['medium_count']} medium, {summary['hard_count']} hard){suffix}")
        except FileNotFoundError as e:
            failures.append((layout_path, e))
            print(f"❌ {layout_path}: FAILED — {e}")

    grand_total = sum(s['total_count'] for s in summaries)

    print("\n" + "=" * 60)
    print("✨ Batch Generation Complete!")
    print("=" * 60)
    print(f"Processed {len(summaries)}/{len(layout_dirs)} layouts")
    if failures:
        print(f"⚠️  {len(failures)} layout(s) failed:")
        for layout_path, e in failures:
            print(f"   • {layout_path}: {e}")
    if total_duplicate_warnings:
        print(f"⚠️  {total_duplicate_warnings} duplicate/near-duplicate variation warning(s) "
              f"printed above (re-run with fewer variations on low-diversity layouts if this matters)")
    print(f"Grand total task config files: {grand_total}")
    print(f"Target >= 1000: {'MET ✅' if grand_total >= 1000 else 'NOT MET ❌'}")


if __name__ == '__main__':
    main()
