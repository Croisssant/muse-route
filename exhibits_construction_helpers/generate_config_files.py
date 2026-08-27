"""
Config File Generator for Museum Route Validation
Generates difficulty-specific config files based on exhibit list CSV and layout annotations.

Usage:
    python generate_config_files.py -l floorplans/simple/layout_01
    python generate_config_files.py -l floorplans/complex/layout_03
"""

import json
import argparse
import random
from pathlib import Path
from collections import Counter
import pandas as pd


DEFAULT_MEDIUM_VARIATIONS = 11
DEFAULT_HARD_VARIATIONS = 11

MEDIUM_SECTION_WINDOW = 1
MEDIUM_MATERIAL_WINDOW = 2
MEDIUM_COUNTRY_WINDOW = 2

HARD_SECTION_WINDOW = 2
HARD_MATERIAL_WINDOW = 3
HARD_TECHNIQUE_WINDOW = 2
HARD_COUNTRY_WINDOW = 3
HARD_COMBINED_MATERIAL_WINDOW = 1
HARD_COMBINED_COUNTRY_WINDOW = 1

# Per-attribute phase offsets so different constraint fields rotate out of
# phase with each other instead of repeating in lockstep when pool sizes
# happen to coincide.
_ATTR_PHASE = {
    'sections': 0,
    'materials': 2,
    'countries': 5,
    'techniques': 9,
    'combined_materials': 13,
    'combined_countries': 17,
}


def _rotate_window(items, variation_index, window_size, phase=0):
    """Pick `window_size` items from `items`, shifting the start position by
    1 per variation_index (offset by `phase`), wrapping circularly around the
    end of the list.

    A step of 1 (rather than window_size) maximizes the number of distinct
    windows achievable before the rotation cycles back to a repeat. If the
    pool is too small to fill a window, everything available is returned -
    that's unavoidable exhaustion on low-diversity floorplans (e.g. a
    2-section layout), not a bug in the rotation itself.
    """
    n = len(items)
    if n == 0:
        return []
    if n <= window_size:
        return list(items)
    start = (variation_index + phase) % n
    end = start + window_size
    return items[start:end] if end <= n else items[start:] + items[:end - n]


class ExhibitAnalyzer:
    """Analyzes exhibit CSV to extract statistics and available attributes."""

    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        self.stats = self._analyze()

    def _analyze(self):
        """Extract all statistics from the exhibit CSV."""
        stats = {
            'total_exhibits': len(self.df),
            'sections': self._count_attribute('Section'),
            'materials': self._count_materials(),
            'techniques': self._count_attribute('Technique'),
            'find_spot_cities': self._count_attribute('find_spot_city'),
            'find_spot_countries': self._count_attribute('find_spot_country'),
            'date_ranges': self._extract_date_ranges(),
        }
        return stats

    def _count_attribute(self, column):
        """Count occurrences of each value in a column."""
        if column not in self.df.columns:
            return Counter()

        # Filter out NaN values
        values = self.df[column].dropna()
        return Counter(values)

    def _count_materials(self):
        """Count materials, handling semicolon-separated values."""
        if 'Materials' not in self.df.columns:
            return Counter()

        materials = []
        for val in self.df['Materials'].dropna():
            # Split by semicolon and clean up
            parts = [m.strip() for m in str(val).split(';')]
            materials.extend(parts)

        return Counter(materials)

    def _extract_date_ranges(self):
        """Extract available date ranges from the dataset."""
        if 'production_year_min' not in self.df.columns or 'production_year_max' not in self.df.columns:
            return []

        # Get all date ranges, filter out NaN
        ranges = []
        for _, row in self.df.iterrows():
            if pd.notna(row['production_year_min']) and pd.notna(row['production_year_max']):
                ranges.append((row['production_year_min'], row['production_year_max']))

        return ranges

    def get_least_common(self, attribute_type, n=2):
        """Get the n least common values for an attribute type."""
        counter = self._get_counter(attribute_type)
        if not counter:
            return []

        # Get least common items
        least_common = counter.most_common()[-n:] if len(counter) >= n else list(counter.items())
        return [item[0] for item in least_common]

    def get_most_common(self, attribute_type, n=2):
        """Get the n most common values for an attribute type.

        Pass n=None to get every distinct value (sorted by frequency) -
        useful when a caller wants to rotate through the full pool rather
        than a truncated top-N slice.
        """
        counter = self._get_counter(attribute_type)
        if not counter:
            return []

        most_common = counter.most_common(n)
        return [item[0] for item in most_common]

    def _get_counter(self, attribute_type):
        """Get the appropriate counter for an attribute type."""
        mapping = {
            'sections': 'sections',
            'materials': 'materials',
            'techniques': 'techniques',
            'find_spot_cities': 'find_spot_cities',
            'find_spot_countries': 'find_spot_countries',
        }

        key = mapping.get(attribute_type)
        return self.stats.get(key, Counter())

    def validate_constraint(self, constraint_type, values):
        """Validate that a constraint matches at least some exhibits."""
        if not values:
            return 0

        count = 0
        if constraint_type == 'sections':
            for section in values:
                count += len(self.df[self.df['Section'] == section])

        elif constraint_type == 'materials':
            for material in values:
                # Check if material appears in Materials column (may be semicolon-separated)
                count += len(self.df[self.df['Materials'].str.contains(material, na=False, case=False)])

        elif constraint_type == 'techniques':
            for technique in values:
                count += len(self.df[self.df['Technique'].str.contains(technique, na=False, case=False)])

        elif constraint_type == 'find_spot_cities':
            for city in values:
                count += len(self.df[self.df['find_spot_city'] == city])

        elif constraint_type == 'find_spot_countries':
            for country in values:
                count += len(self.df[self.df['find_spot_country'] == country])

        return count

    def get_random_date_range(self, prefer_common=True):
        """Get a random date range that covers multiple exhibits."""
        date_ranges = self.stats['date_ranges']
        if not date_ranges:
            return []

        # Sort by how many exhibits fall in the range
        range_counts = []
        for date_min, date_max in date_ranges:
            count = len(self.df[
                (self.df['production_year_min'] >= date_min) &
                (self.df['production_year_max'] <= date_max)
            ])
            range_counts.append((date_min, date_max, count))

        # Sort by count
        range_counts.sort(key=lambda x: x[2], reverse=prefer_common)

        # Pick from top candidates
        if range_counts:
            top_ranges = range_counts[:5] if prefer_common else range_counts[-5:]
            selected = random.choice(top_ranges)
            return [self._format_date(selected[0]), self._format_date(selected[1])]

        return []

    def _format_date(self, year):
        """Format year as BC or AD string."""
        if year < 0:
            return f"{abs(int(year))} BC"
        else:
            return f"{int(year)}"


class ConfigGenerator:
    """Generates configuration files for different difficulty levels."""

    def __init__(self, analyzer, gallery_names):
        self.analyzer = analyzer
        self.gallery_names = gallery_names

    def _base_config(self):
        """Create base configuration structure."""
        return {
            "_gallery_config_options": {
                "must_see": "Must vist route",
                "restricted": "Route cannot enter",
                "normal": "No restrictions"
            },
            "gallery_configs": {},
            "distance_budget_in_mm": 1000000000,
            "exhibit_see_distance_in_mm": 1500,
            "exhibit_categories_to_cover": [],
            "at_least_n_exhibits_to_cover": 1,
            "specific_exhibit_to_cover": [],
            "exhibit_attribute_constraints": {
                "_comment": "Individual constraints work independently (OR logic between them)",
                "date_constraint": {
                    "in_range": []
                },
                "material_constraint": {
                    "materials": []
                },
                "find_spot_constraint": {
                    "locations": []
                },
                "technique_constraint": {
                    "techniques": []
                },
                "combined_constraint": {
                    "_combined_comment": "Combined constraint uses AND logic - exhibit must match ALL specified conditions",
                    "date_constraint": {
                        "in_range": []
                    },
                    "material_constraint": {
                        "materials": []
                    },
                    "find_spot_constraint": {
                        "locations": []
                    },
                    "technique_constraint": {
                        "techniques": []
                    }
                }
            }
        }

    def generate_easy_spatial(self):
        """Generate easy-spatial config with gallery regions and minimal exhibits."""
        config = self._base_config()

        # Set all galleries to must_see
        for gallery in self.gallery_names:
            config["gallery_configs"][gallery] = "must_see"

        # Minimal exhibit coverage
        config["at_least_n_exhibits_to_cover"] = 1

        return config

    def generate_easy_semantic(self):
        """Generate easy-semantic config with attribute constraints."""
        config = self._base_config()

        # No gallery constraints
        config["gallery_configs"] = {}

        # Minimal exhibit coverage
        config["at_least_n_exhibits_to_cover"] = 1

        # Choose ONE constraint type with least common values
        constraint_types = ['materials', 'techniques', 'find_spot_countries']
        chosen_type = random.choice(constraint_types)

        if chosen_type == 'materials':
            materials = self.analyzer.get_least_common('materials', n=1)
            if materials:
                config["exhibit_attribute_constraints"]["material_constraint"]["materials"] = materials

        elif chosen_type == 'techniques':
            techniques = self.analyzer.get_least_common('techniques', n=1)
            if techniques:
                config["exhibit_attribute_constraints"]["technique_constraint"]["techniques"] = techniques

        elif chosen_type == 'find_spot_countries':
            countries = self.analyzer.get_least_common('find_spot_countries', n=1)
            if countries:
                config["exhibit_attribute_constraints"]["find_spot_constraint"]["locations"] = countries

        # Optionally add a date range (least common)
        date_range = self.analyzer.get_random_date_range(prefer_common=False)
        if date_range and random.random() < 0.5:
            config["exhibit_attribute_constraints"]["date_constraint"]["in_range"] = date_range

        return config

    def generate_medium(self, variation_index=0):
        """Generate medium config with mixed spatial and semantic constraints.

        Args:
            variation_index: creates different combinations of constraints by
                cyclically rotating through each attribute's full value pool
        """
        config = self._base_config()

        # Set all galleries to must_see
        for gallery in self.gallery_names:
            config["gallery_configs"][gallery] = "must_see"

        # Moderate exhibit coverage - vary slightly, capped at total_exhibits
        total = self.analyzer.stats['total_exhibits']
        base_coverage = min(10, max(5, total // 10))
        config["at_least_n_exhibits_to_cover"] = min(base_coverage + variation_index, total)

        # Add sections - rotate through the full pool
        sections = self.analyzer.get_most_common('sections', n=None)
        if sections:
            config["exhibit_categories_to_cover"] = _rotate_window(
                sections, variation_index, MEDIUM_SECTION_WINDOW, _ATTR_PHASE['sections'])

        # Add materials - rotate through the full pool
        all_materials = self.analyzer.get_most_common('materials', n=None)
        if all_materials:
            materials = _rotate_window(
                all_materials, variation_index, MEDIUM_MATERIAL_WINDOW, _ATTR_PHASE['materials'])
            if materials:
                config["exhibit_attribute_constraints"]["material_constraint"]["materials"] = materials

        # Add countries - rotate through the full pool
        all_countries = self.analyzer.get_most_common('find_spot_countries', n=None)
        if all_countries:
            countries = _rotate_window(
                all_countries, variation_index, MEDIUM_COUNTRY_WINDOW, _ATTR_PHASE['countries'])
            if countries:
                config["exhibit_attribute_constraints"]["find_spot_constraint"]["locations"] = countries

        # Add date range if available - use seed for reproducibility
        random.seed(variation_index * 100)
        date_range = self.analyzer.get_random_date_range(prefer_common=True)
        if date_range:
            config["exhibit_attribute_constraints"]["date_constraint"]["in_range"] = date_range
        random.seed()  # Reset seed

        return config

    def generate_hard(self, variation_index=0):
        """Generate hard config with all constraint types.

        Args:
            variation_index: creates different combinations of constraints by
                cyclically rotating through each attribute's full value pool
        """
        config = self._base_config()

        # Set all galleries to must_see
        for gallery in self.gallery_names:
            config["gallery_configs"][gallery] = "must_see"

        # High exhibit coverage - vary slightly, capped at total_exhibits
        total = self.analyzer.stats['total_exhibits']
        base_coverage = min(25, max(15, int(total * 0.25)))
        config["at_least_n_exhibits_to_cover"] = min(base_coverage + variation_index, total)

        # Set distance budget for hard difficulty (2.5km = 2,500,000mm)
        config["distance_budget_in_mm"] = 2500000

        # Add multiple common categories - rotate through the full pool
        all_sections = self.analyzer.get_most_common('sections', n=None)
        if all_sections:
            sections = _rotate_window(
                all_sections, variation_index, HARD_SECTION_WINDOW, _ATTR_PHASE['sections'])
            if sections:
                config["exhibit_categories_to_cover"] = sections

        # Add specific exhibits - use seed for reproducible randomness
        random.seed(variation_index * 1000)
        exhibit_numbers = list(range(1, total + 1))
        num_specific = min(10, max(5, total // 20))
        specific_exhibits = random.sample(exhibit_numbers, num_specific)
        config["specific_exhibit_to_cover"] = specific_exhibits
        random.seed()  # Reset seed

        # Fill ALL OR constraints with most common values - rotate through the full pool
        all_materials = self.analyzer.get_most_common('materials', n=None)
        if all_materials:
            materials = _rotate_window(
                all_materials, variation_index, HARD_MATERIAL_WINDOW, _ATTR_PHASE['materials'])
            if materials:
                config["exhibit_attribute_constraints"]["material_constraint"]["materials"] = materials

        all_techniques = self.analyzer.get_most_common('techniques', n=None)
        if all_techniques:
            techniques = _rotate_window(
                all_techniques, variation_index, HARD_TECHNIQUE_WINDOW, _ATTR_PHASE['techniques'])
            if techniques:
                config["exhibit_attribute_constraints"]["technique_constraint"]["techniques"] = techniques

        all_countries = self.analyzer.get_most_common('find_spot_countries', n=None)
        if all_countries:
            countries = _rotate_window(
                all_countries, variation_index, HARD_COUNTRY_WINDOW, _ATTR_PHASE['countries'])
            if countries:
                config["exhibit_attribute_constraints"]["find_spot_constraint"]["locations"] = countries

        # Add date range - use seed for reproducibility
        random.seed(variation_index * 500)
        date_range = self.analyzer.get_random_date_range(prefer_common=True)
        if date_range:
            config["exhibit_attribute_constraints"]["date_constraint"]["in_range"] = date_range
        random.seed()  # Reset seed

        # Add combined AND constraint - vary which attributes are combined,
        # using a different phase so it doesn't just mirror the OR constraint above
        if all_materials:
            combined_materials = _rotate_window(
                all_materials, variation_index, HARD_COMBINED_MATERIAL_WINDOW, _ATTR_PHASE['combined_materials'])
            if combined_materials:
                config["exhibit_attribute_constraints"]["combined_constraint"]["material_constraint"]["materials"] = combined_materials

        if all_countries:
            combined_countries = _rotate_window(
                all_countries, variation_index, HARD_COMBINED_COUNTRY_WINDOW, _ATTR_PHASE['combined_countries'])
            if combined_countries:
                config["exhibit_attribute_constraints"]["combined_constraint"]["find_spot_constraint"]["locations"] = combined_countries

        return config


def load_gallery_names(annotations_path):
    """Load gallery names from layout_annotations.json."""
    try:
        with open(annotations_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        galleries = data.get('galleries', [])
        gallery_names = [g.get('gallery_name') for g in galleries if g.get('gallery_name')]

        return gallery_names

    except Exception as e:
        print(f"⚠️  Warning: Could not load galleries from {annotations_path}: {e}")
        return []


def save_config(config, output_path, verbose=True):
    """Save configuration to JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    if verbose:
        print(f"✅ Saved: {output_path}")


def _fingerprint(config):
    """Content-bearing fields of a task config, order-independent lists
    sorted, used to detect exact-duplicate variations."""
    ac = config["exhibit_attribute_constraints"]
    cc = ac["combined_constraint"]
    return (
        tuple(sorted(config.get("exhibit_categories_to_cover", []))),
        tuple(sorted(ac["material_constraint"]["materials"])),
        tuple(sorted(ac["find_spot_constraint"]["locations"])),
        tuple(sorted(ac["technique_constraint"]["techniques"])),
        tuple(ac["date_constraint"]["in_range"]),
        tuple(sorted(config.get("specific_exhibit_to_cover", []))),
        config.get("at_least_n_exhibits_to_cover"),
        tuple(sorted(cc["material_constraint"]["materials"])),
        tuple(sorted(cc["find_spot_constraint"]["locations"])),
    )


def _semantic_fingerprint(config):
    """Same as `_fingerprint` but excludes specific_exhibit_to_cover, date
    range, and coverage count - catches "same constraints, different
    coverage target" near-duplicates that an exact fingerprint would miss."""
    ac = config["exhibit_attribute_constraints"]
    cc = ac["combined_constraint"]
    return (
        tuple(sorted(config.get("exhibit_categories_to_cover", []))),
        tuple(sorted(ac["material_constraint"]["materials"])),
        tuple(sorted(ac["find_spot_constraint"]["locations"])),
        tuple(sorted(ac["technique_constraint"]["techniques"])),
        tuple(sorted(cc["material_constraint"]["materials"])),
        tuple(sorted(cc["find_spot_constraint"]["locations"])),
    )


def check_duplicate_variations(variations, layout_label, difficulty_label):
    """Warn (without raising) about exact or near-duplicate variations within
    one layout+difficulty batch. Some repetition is unavoidable on
    low-diversity floorplans (e.g. only 2 distinct sections); this makes it
    visible instead of silent.

    Args:
        variations: list of (filename, config) tuples

    Returns:
        list of warning message strings
    """
    warnings = []
    seen_exact, seen_semantic = {}, {}

    for filename, config in variations:
        exact_fp = _fingerprint(config)
        semantic_fp = _semantic_fingerprint(config)

        if exact_fp in seen_exact:
            msg = (f"⚠️  {layout_label} {difficulty_label}: '{filename}' is an exact "
                   f"duplicate of '{seen_exact[exact_fp]}'")
            print(msg)
            warnings.append(msg)
        elif semantic_fp in seen_semantic:
            msg = (f"⚠️  {layout_label} {difficulty_label}: '{filename}' has the same "
                   f"categories/materials/countries/techniques as "
                   f"'{seen_semantic[semantic_fp]}' (near-duplicate, differs only in "
                   f"coverage/specific exhibits/date range)")
            print(msg)
            warnings.append(msg)

        seen_exact.setdefault(exact_fp, filename)
        seen_semantic.setdefault(semantic_fp, filename)

    return warnings


def generate_configs_for_layout(
    layout_path,
    csv_filename='exhibits.csv',
    medium_variations=DEFAULT_MEDIUM_VARIATIONS,
    hard_variations=DEFAULT_HARD_VARIATIONS,
    verbose=True,
):
    """Generate easy_spatial, easy_semantic, medium_01..NN and hard_01..NN
    config files for one layout directory.

    Raises:
        FileNotFoundError: if the exhibits CSV is missing for this layout.

    Returns:
        dict with keys: layout_path, files_written, medium_count, hard_count,
        total_count, duplicate_warnings.
    """
    layout_path = Path(layout_path)
    csv_path = layout_path / csv_filename
    annotations_path = layout_path / 'layout_annotations.json'

    if verbose:
        print(f"📁 Layout: {layout_path}")

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    if not annotations_path.exists():
        if verbose:
            print(f"⚠️  Warning: Annotations file not found: {annotations_path}")
            print("   Gallery configs will be empty.")
        gallery_names = []
    else:
        gallery_names = load_gallery_names(annotations_path)
        if verbose:
            print(f"📊 Found {len(gallery_names)} galleries: {', '.join(gallery_names)}")

    if verbose:
        print(f"\n📊 Analyzing exhibits from: {csv_path}")
    analyzer = ExhibitAnalyzer(csv_path)

    if verbose:
        print(f"   Total exhibits: {analyzer.stats['total_exhibits']}")
        print(f"   Sections: {len(analyzer.stats['sections'])}")
        print(f"   Materials: {len(analyzer.stats['materials'])}")
        print(f"   Techniques: {len(analyzer.stats['techniques'])}")
        print(f"   Find spot countries: {len(analyzer.stats['find_spot_countries'])}")

        print("\n" + "=" * 60)
        print("🎮 Generating Config Files...")
        print("=" * 60)

    generator = ConfigGenerator(analyzer, gallery_names)
    files_written = []

    # Easy Spatial
    if verbose:
        print("\n1️⃣  Easy Spatial")
    easy_spatial = generator.generate_easy_spatial()
    save_config(easy_spatial, layout_path / 'easy_spatial.json', verbose=verbose)
    files_written.append(layout_path / 'easy_spatial.json')

    # Easy Semantic
    if verbose:
        print("\n2️⃣  Easy Semantic")
    easy_semantic = generator.generate_easy_semantic()
    save_config(easy_semantic, layout_path / 'easy_semantic.json', verbose=verbose)
    files_written.append(layout_path / 'easy_semantic.json')

    # Medium
    if verbose:
        print(f"\n3️⃣  Medium ({medium_variations} variations)")
    medium_variants = []
    for i in range(medium_variations):
        if verbose:
            print(f"   Variation {i+1}/{medium_variations}")
        medium = generator.generate_medium(variation_index=i)
        filename = f'medium_{i+1:02d}.json'
        save_config(medium, layout_path / filename, verbose=verbose)
        files_written.append(layout_path / filename)
        medium_variants.append((filename, medium))

    # Hard
    if verbose:
        print(f"\n4️⃣  Hard ({hard_variations} variations)")
    hard_variants = []
    for i in range(hard_variations):
        if verbose:
            print(f"   Variation {i+1}/{hard_variations}")
        hard = generator.generate_hard(variation_index=i)
        filename = f'hard_{i+1:02d}.json'
        save_config(hard, layout_path / filename, verbose=verbose)
        files_written.append(layout_path / filename)
        hard_variants.append((filename, hard))

    layout_label = str(layout_path)
    duplicate_warnings = []
    duplicate_warnings += check_duplicate_variations(medium_variants, layout_label, 'medium')
    duplicate_warnings += check_duplicate_variations(hard_variants, layout_label, 'hard')

    if verbose:
        print("\n" + "=" * 60)
        print("✨ Config Generation Complete!")
        print("=" * 60)
        print(f"📁 Output directory: {layout_path}")
        print("   • easy_spatial.json")
        print("   • easy_semantic.json")
        print(f"   • medium_01.json ... medium_{medium_variations:02d}.json")
        print(f"   • hard_01.json ... hard_{hard_variations:02d}.json")

    return {
        'layout_path': layout_path,
        'files_written': files_written,
        'medium_count': medium_variations,
        'hard_count': hard_variations,
        'total_count': len(files_written),
        'duplicate_warnings': duplicate_warnings,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Generate difficulty-specific config files for museum route validation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_config_files.py -l floorplans/simple/layout_01
  python generate_config_files.py -l floorplans/complex/layout_03
        """
    )

    parser.add_argument(
        '-l', '--layout',
        type=str,
        required=True,
        help='Path to layout directory (e.g., floorplans/simple/layout_01)'
    )

    parser.add_argument(
        '-c', '--csv',
        type=str,
        default='exhibits.csv',
        help='Name of the exhibits CSV file (default: exhibits.csv)'
    )

    parser.add_argument(
        '--medium-variations',
        type=int,
        default=DEFAULT_MEDIUM_VARIATIONS,
        help=f'Number of medium task variations to generate (default: {DEFAULT_MEDIUM_VARIATIONS})'
    )

    parser.add_argument(
        '--hard-variations',
        type=int,
        default=DEFAULT_HARD_VARIATIONS,
        help=f'Number of hard task variations to generate (default: {DEFAULT_HARD_VARIATIONS})'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("🏛️  Config File Generator for Museum Route Validation")
    print("=" * 60)

    try:
        generate_configs_for_layout(
            Path(args.layout),
            csv_filename=args.csv,
            medium_variations=args.medium_variations,
            hard_variations=args.hard_variations,
            verbose=True,
        )
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return


if __name__ == '__main__':
    main()
