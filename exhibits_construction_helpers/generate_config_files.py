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
        """Get the n most common values for an attribute type."""
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
            variation_index: 0-3, creates different combinations of constraints
        """
        config = self._base_config()
        
        # Set all galleries to must_see
        for gallery in self.gallery_names:
            config["gallery_configs"][gallery] = "must_see"
        
        # Moderate exhibit coverage - vary slightly
        total = self.analyzer.stats['total_exhibits']
        base_coverage = min(10, max(5, total // 10))
        config["at_least_n_exhibits_to_cover"] = base_coverage + variation_index
        
        # Add sections - vary which sections are chosen
        sections = self.analyzer.get_most_common('sections', n=4)
        if sections:
            # Different variations pick different sections
            if variation_index == 0:
                config["exhibit_categories_to_cover"] = sections[:1]
            elif variation_index == 1:
                config["exhibit_categories_to_cover"] = sections[1:2] if len(sections) > 1 else sections[:1]
            elif variation_index == 2:
                config["exhibit_categories_to_cover"] = sections[2:3] if len(sections) > 2 else sections[:1]
            else:
                config["exhibit_categories_to_cover"] = sections[3:4] if len(sections) > 3 else sections[:1]
        
        # Add materials - vary selection
        all_materials = self.analyzer.get_most_common('materials', n=6)
        if all_materials:
            if variation_index == 0:
                materials = all_materials[:2]
            elif variation_index == 1:
                materials = all_materials[1:3] if len(all_materials) > 2 else all_materials[:2]
            elif variation_index == 2:
                materials = all_materials[2:4] if len(all_materials) > 3 else all_materials[:2]
            else:
                materials = all_materials[3:5] if len(all_materials) > 4 else all_materials[:2]
            
            if materials:
                config["exhibit_attribute_constraints"]["material_constraint"]["materials"] = materials
        
        # Add countries - vary selection
        all_countries = self.analyzer.get_most_common('find_spot_countries', n=6)
        if all_countries:
            if variation_index == 0:
                countries = all_countries[:2]
            elif variation_index == 1:
                countries = all_countries[1:3] if len(all_countries) > 2 else all_countries[:2]
            elif variation_index == 2:
                countries = all_countries[2:4] if len(all_countries) > 3 else all_countries[:2]
            else:
                countries = all_countries[3:5] if len(all_countries) > 4 else all_countries[:2]
            
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
            variation_index: 0-3, creates different combinations of constraints
        """
        config = self._base_config()
        
        # Set all galleries to must_see
        for gallery in self.gallery_names:
            config["gallery_configs"][gallery] = "must_see"
        
        # High exhibit coverage - vary slightly
        total = self.analyzer.stats['total_exhibits']
        base_coverage = min(25, max(15, int(total * 0.25)))
        config["at_least_n_exhibits_to_cover"] = base_coverage + variation_index

        # Set distance budget for hard difficulty (2.5km = 2,500,000mm)
        config["distance_budget_in_mm"] = 2500000
        
        # Add multiple common categories - vary selection
        all_sections = self.analyzer.get_most_common('sections', n=6)
        if all_sections:
            if variation_index == 0:
                sections = all_sections[:2]
            elif variation_index == 1:
                sections = all_sections[1:3] if len(all_sections) > 2 else all_sections[:2]
            elif variation_index == 2:
                sections = all_sections[2:4] if len(all_sections) > 3 else all_sections[:2]
            else:
                sections = all_sections[3:5] if len(all_sections) > 4 else all_sections[:2]
            
            if sections:
                config["exhibit_categories_to_cover"] = sections
        
        # Add specific exhibits - use seed for reproducible randomness
        random.seed(variation_index * 1000)
        exhibit_numbers = list(range(1, total + 1))
        num_specific = min(10, max(5, total // 20))
        specific_exhibits = random.sample(exhibit_numbers, num_specific)
        config["specific_exhibit_to_cover"] = specific_exhibits
        random.seed()  # Reset seed
        
        # Fill ALL OR constraints with most common values - vary selection
        all_materials = self.analyzer.get_most_common('materials', n=9)
        if all_materials:
            if variation_index == 0:
                materials = all_materials[:3]
            elif variation_index == 1:
                materials = all_materials[2:5] if len(all_materials) > 4 else all_materials[:3]
            elif variation_index == 2:
                materials = all_materials[4:7] if len(all_materials) > 6 else all_materials[:3]
            else:
                materials = all_materials[6:9] if len(all_materials) > 8 else all_materials[:3]
            
            if materials:
                config["exhibit_attribute_constraints"]["material_constraint"]["materials"] = materials
        
        all_techniques = self.analyzer.get_most_common('techniques', n=6)
        if all_techniques:
            if variation_index == 0:
                techniques = all_techniques[:2]
            elif variation_index == 1:
                techniques = all_techniques[1:3] if len(all_techniques) > 2 else all_techniques[:2]
            elif variation_index == 2:
                techniques = all_techniques[2:4] if len(all_techniques) > 3 else all_techniques[:2]
            else:
                techniques = all_techniques[3:5] if len(all_techniques) > 4 else all_techniques[:2]
            
            if techniques:
                config["exhibit_attribute_constraints"]["technique_constraint"]["techniques"] = techniques
        
        all_countries = self.analyzer.get_most_common('find_spot_countries', n=9)
        if all_countries:
            if variation_index == 0:
                countries = all_countries[:3]
            elif variation_index == 1:
                countries = all_countries[2:5] if len(all_countries) > 4 else all_countries[:3]
            elif variation_index == 2:
                countries = all_countries[4:7] if len(all_countries) > 6 else all_countries[:3]
            else:
                countries = all_countries[6:9] if len(all_countries) > 8 else all_countries[:3]
            
            if countries:
                config["exhibit_attribute_constraints"]["find_spot_constraint"]["locations"] = countries
        
        # Add date range - use seed for reproducibility
        random.seed(variation_index * 500)
        date_range = self.analyzer.get_random_date_range(prefer_common=True)
        if date_range:
            config["exhibit_attribute_constraints"]["date_constraint"]["in_range"] = date_range
        random.seed()  # Reset seed
        
        # Add combined AND constraint - vary which attributes are combined
        if all_materials:
            if variation_index == 0:
                combined_materials = all_materials[:1]
            elif variation_index == 1:
                combined_materials = all_materials[1:2] if len(all_materials) > 1 else all_materials[:1]
            elif variation_index == 2:
                combined_materials = all_materials[2:3] if len(all_materials) > 2 else all_materials[:1]
            else:
                combined_materials = all_materials[3:4] if len(all_materials) > 3 else all_materials[:1]
            
            if combined_materials:
                config["exhibit_attribute_constraints"]["combined_constraint"]["material_constraint"]["materials"] = combined_materials
        
        if all_countries:
            if variation_index == 0:
                combined_countries = all_countries[:1]
            elif variation_index == 1:
                combined_countries = all_countries[1:2] if len(all_countries) > 1 else all_countries[:1]
            elif variation_index == 2:
                combined_countries = all_countries[2:3] if len(all_countries) > 2 else all_countries[:1]
            else:
                combined_countries = all_countries[3:4] if len(all_countries) > 3 else all_countries[:1]
            
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


def save_config(config, output_path):
    """Save configuration to JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved: {output_path}")


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
    
    args = parser.parse_args()
    
    # Set up paths
    layout_path = Path(args.layout)
    csv_path = layout_path / args.csv
    annotations_path = layout_path / 'layout_annotations.json'
    
    print("=" * 60)
    print("🏛️  Config File Generator for Museum Route Validation")
    print("=" * 60)
    print(f"📁 Layout: {layout_path}")
    
    # Check if files exist
    if not csv_path.exists():
        print(f"❌ Error: CSV file not found: {csv_path}")
        return
    
    if not annotations_path.exists():
        print(f"⚠️  Warning: Annotations file not found: {annotations_path}")
        print("   Gallery configs will be empty.")
        gallery_names = []
    else:
        gallery_names = load_gallery_names(annotations_path)
        print(f"📊 Found {len(gallery_names)} galleries: {', '.join(gallery_names)}")
    
    # Analyze exhibits
    print(f"\n📊 Analyzing exhibits from: {csv_path}")
    analyzer = ExhibitAnalyzer(csv_path)
    
    print(f"   Total exhibits: {analyzer.stats['total_exhibits']}")
    print(f"   Sections: {len(analyzer.stats['sections'])}")
    print(f"   Materials: {len(analyzer.stats['materials'])}")
    print(f"   Techniques: {len(analyzer.stats['techniques'])}")
    print(f"   Find spot countries: {len(analyzer.stats['find_spot_countries'])}")
    
    # Generate configs
    print("\n" + "=" * 60)
    print("🎮 Generating Config Files...")
    print("=" * 60)
    
    generator = ConfigGenerator(analyzer, gallery_names)
    
    # Easy Spatial
    print("\n1️⃣  Easy Spatial")
    easy_spatial = generator.generate_easy_spatial()
    save_config(easy_spatial, layout_path / 'easy_spatial.json')
    
    # Easy Semantic
    print("\n2️⃣  Easy Semantic")
    easy_semantic = generator.generate_easy_semantic()
    save_config(easy_semantic, layout_path / 'easy_semantic.json')
    
    # Medium - Generate 4 variations
    print("\n3️⃣  Medium (4 variations)")
    for i in range(4):
        print(f"   Variation {i+1}/4")
        medium = generator.generate_medium(variation_index=i)
        save_config(medium, layout_path / f'medium_{i+1:02d}.json')
    
    # Hard - Generate 4 variations
    print("\n4️⃣  Hard (4 variations)")
    for i in range(4):
        print(f"   Variation {i+1}/4")
        hard = generator.generate_hard(variation_index=i)
        save_config(hard, layout_path / f'hard_{i+1:02d}.json')
    
    print("\n" + "=" * 60)
    print("✨ Config Generation Complete!")
    print("=" * 60)
    print(f"📁 Output directory: {layout_path}")
    print("   • easy_spatial.json")
    print("   • easy_semantic.json")
    print("   • medium_01.json, medium_02.json, medium_03.json, medium_04.json")
    print("   • hard_01.json, hard_02.json, hard_03.json, hard_04.json")


if __name__ == '__main__':
    main()
