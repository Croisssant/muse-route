import cv2
import numpy as np
import json
import argparse
import re
import pandas as pd
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import math

class SemanticValidator:
    def __init__(self, config_file, visited_exhibits, exhibits_csv_file=None):
        """
        Initialize the semantic validator.
        
        Args:
            config_file: Path to JSON file with exhibit configurations
            visited_exhibits: List of visited exhibit_numbers (can be strings or integers)
            exhibits_csv_file: Path to CSV file with exhibit_number, Section, and preprocessed columns
        """

        with open(config_file, 'r') as f:
            self.config = json.load(f)
        
        # Convert visited exhibits to integers for consistent comparison
        self.visited_exhibits = [int(e) if isinstance(e, str) else e for e in visited_exhibits]
        
        self.exhibit_categories_to_cover = self.config.get("exhibit_categories_to_cover", [])
        self.at_least_n_exhibits_to_cover = self.config.get("at_least_n_exhibits_to_cover", 0)
        self.specific_exhibit_to_cover = self.config.get("specific_exhibit_to_cover", [])
        
        # Load exhibit metadata from CSV
        self.exhibits_df = None
        self.exhibits_by_section = {}
        
        if exhibits_csv_file and Path(exhibits_csv_file).exists():
            self.exhibits_df = pd.read_csv(exhibits_csv_file)
            print(f"✓ Loaded {len(self.exhibits_df)} exhibits from {exhibits_csv_file}")
            
            # Verify required columns exist
            required_cols = ['exhibit_number', 'Section']
            missing_cols = [col for col in required_cols if col not in self.exhibits_df.columns]
            if missing_cols:
                print(f"⚠️  Warning: Missing required columns: {missing_cols}")
            else:
                # Build exhibits_by_section mapping from CSV
                for section in self.exhibits_df['Section'].unique():
                    section_exhibits = self.exhibits_df[self.exhibits_df['Section'] == section]['exhibit_number'].tolist()
                    self.exhibits_by_section[section] = section_exhibits
                print(f"✓ Built section mappings for {len(self.exhibits_by_section)} sections")
            
            # Validate that CSV has preprocessed columns for attribute validation
            if self.config.get("exhibit_attribute_constraints"):
                preprocessed_cols = ['production_year_min', 'production_year_max', 'find_spot_city', 'find_spot_country']
                missing_preprocessed = [col for col in preprocessed_cols if col not in self.exhibits_df.columns]
                
                if missing_preprocessed:
                    print(f"⚠️  Warning: Missing preprocessed columns: {missing_preprocessed}")
                    print(f"   Run preprocess_exhibits.py first to generate these columns")
                else:
                    print(f"✓ Preprocessed columns detected")
        
        # Load attribute constraints
        self.attribute_constraints = self.config.get("exhibit_attribute_constraints", {})
    
    def _parse_year_value(self, year_value):
        """
        Parse year value from config - supports both numeric and string formats.
        
        Args:
            year_value: Can be int (e.g., 600, -400) or string (e.g., "600 AD", "400 BC")
        
        Returns:
            int: Numeric year (positive for AD, negative for BC) or None if invalid
        """
        if year_value is None:
            return None
        
        # If already numeric, return as-is
        if isinstance(year_value, (int, float)):
            return int(year_value)
        
        # Parse string format
        if isinstance(year_value, str):
            year_str = year_value.strip()
            
            # Try to match "NUMBER BC" or "NUMBER AD"
            import re
            match = re.match(r'^(\d+)\s*(BC|AD|bc|ad)?$', year_str, re.IGNORECASE)
            if match:
                number = int(match.group(1))
                era = match.group(2)
                
                if era and era.upper() == 'BC':
                    return -number  # BC years are negative
                else:
                    return number  # AD years are positive (or no era specified = AD)
            
            # Try just a number string
            try:
                return int(year_str)
            except ValueError:
                pass
        
        return None
    
    def _check_date_match(self, row, date_constraint):
        """
        Check if a single exhibit row matches date constraint.
        
        Args:
            row: DataFrame row
            date_constraint: Date constraint dict
        
        Returns:
            bool: True if matches, False otherwise
        """
        if not date_constraint:
            return True
        
        # Use preprocessed columns
        if pd.notna(row['production_year_min']):
            start_year = int(row['production_year_min'])
            end_year = int(row['production_year_max'])
        else:
            return False
        
        before_year = self._parse_year_value(date_constraint.get('before_year'))
        after_year = self._parse_year_value(date_constraint.get('after_year'))
        in_range = date_constraint.get('in_range')
        
        if in_range and isinstance(in_range, list) and len(in_range) == 2:
            in_range = [self._parse_year_value(in_range[0]), self._parse_year_value(in_range[1])]
        else:
            in_range = None
        
        if before_year is not None:
            if end_year >= before_year:
                return False
        
        if after_year is not None:
            if start_year <= after_year:
                return False
        
        if in_range is not None:
            range_start, range_end = in_range
            if start_year < range_start or end_year > range_end:
                return False
        
        return True
    
    def _check_material_match(self, row, material_constraint):
        """
        Check if a single exhibit row matches material constraint.
        
        Args:
            row: DataFrame row
            material_constraint: Material constraint dict
        
        Returns:
            bool: True if matches, False otherwise
        """
        if not material_constraint:
            return True
        
        required_materials = material_constraint.get('materials', [])
        if not required_materials:
            return True
        
        material_str = str(row['Materials']).lower()
        
        for req_material in required_materials:
            if req_material.lower() in material_str:
                return True
        
        return False
    
    def _check_location_match(self, row, location_constraint):
        """
        Check if a single exhibit row matches find spot constraint.
        
        Args:
            row: DataFrame row
            location_constraint: Find spot constraint dict
        
        Returns:
            bool: True if matches, False otherwise
        """
        if not location_constraint:
            return True
        
        locations = location_constraint.get('locations', [])
        if not locations:
            return True
        
        # Use preprocessed columns
        city = str(row['find_spot_city']).lower() if pd.notna(row['find_spot_city']) else ''
        country = str(row['find_spot_country']).lower() if pd.notna(row['find_spot_country']) else ''
        
        for location in locations:
            location_lower = location.lower()
            if location_lower in city or location_lower in country:
                return True
        
        return False
    
    def _check_technique_match(self, row, technique_constraint):
        """
        Check if a single exhibit row matches technique constraint.
        
        Args:
            row: DataFrame row
            technique_constraint: Technique constraint dict
        
        Returns:
            bool: True if matches, False otherwise
        """
        if not technique_constraint:
            return True
        
        required_techniques = technique_constraint.get('techniques', [])
        if not required_techniques:
            return True
        
        technique_str = str(row['Technique']).lower()
        
        for req_technique in required_techniques:
            if req_technique.lower() in technique_str:
                return True
        
        return False
    
    def validate_combined_constraint(self):
        """
        Validate that ALL exhibits matching ALL combined constraints were visited.
        Uses AND logic: exhibit must match ALL specified constraints simultaneously.
        
        Returns:
            dict: Validation result with 'passed' boolean and details
        """
        if not self.attribute_constraints or 'combined_constraint' not in self.attribute_constraints:
            return {
                'passed': True,
                'message': "✓ No combined constraints specified",
                'required_exhibits': [],
                'visited_exhibits': [],
                'missing_exhibits': []
            }
        
        if self.exhibits_df is None:
            return {
                'passed': False,
                'message': "✗ Cannot validate combined constraints: exhibits CSV not loaded",
                'required_exhibits': [],
                'visited_exhibits': [],
                'missing_exhibits': []
            }
        
        combined = self.attribute_constraints['combined_constraint']
        
        # Extract individual constraints
        date_constraint = combined.get('date_constraint')
        material_constraint = combined.get('material_constraint')
        location_constraint = combined.get('find_spot_constraint')
        technique_constraint = combined.get('technique_constraint')
        
        # Build list of active constraints for display
        active_constraints = []
        if date_constraint:
            active_constraints.append("date")
        if material_constraint:
            active_constraints.append("material")
        if location_constraint:
            active_constraints.append("location")
        if technique_constraint:
            active_constraints.append("technique")
        
        if not active_constraints:
            return {
                'passed': True,
                'message': "✓ Combined constraint has no conditions specified",
                'required_exhibits': [],
                'visited_exhibits': [],
                'missing_exhibits': []
            }
        
        # Find exhibits matching ALL constraints
        matching_exhibit_ids = []
        
        for _, row in self.exhibits_df.iterrows():
            # Check ALL constraints - must pass all to match
            if date_constraint and not self._check_date_match(row, date_constraint):
                continue
            if material_constraint and not self._check_material_match(row, material_constraint):
                continue
            if location_constraint and not self._check_location_match(row, location_constraint):
                continue
            if technique_constraint and not self._check_technique_match(row, technique_constraint):
                continue
            
            # Passed all checks!
            matching_exhibit_ids.append(int(row['exhibit_number']))
        
        # Check which matching exhibits were visited
        visited_matching = [eid for eid in matching_exhibit_ids if eid in self.visited_exhibits]
        missing = [eid for eid in matching_exhibit_ids if eid not in self.visited_exhibits]
        
        passed = len(missing) == 0
        
        if len(matching_exhibit_ids) > 0:
            percentage = (len(visited_matching) / len(matching_exhibit_ids)) * 100
        else:
            percentage = 100.0
        
        constraint_str = " AND ".join(active_constraints)
        
        return {
            'passed': passed,
            'required_exhibits': matching_exhibit_ids,
            'visited_exhibits': visited_matching,
            'missing_exhibits': missing,
            'percentage': round(percentage, 1),
            'message': f"{'✓' if passed else '✗'} Combined constraint ({constraint_str}): {len(visited_matching)}/{len(matching_exhibit_ids)} exhibits visited ({percentage:.1f}%){'' if passed else f' (missing: {missing})'}"
        }
    
    def validate_date_constraint(self):
        """
        Validate that ALL exhibits matching date constraints were visited.
        Requires preprocessed CSV with production_year_min and production_year_max columns.
        
        Returns:
            dict: Validation result with 'passed' boolean and details
        """
        if not self.attribute_constraints or 'date_constraint' not in self.attribute_constraints:
            return {
                'passed': True,
                'message': "✓ No date constraints specified",
                'required_exhibits': [],
                'visited_exhibits': [],
                'missing_exhibits': []
            }
        
        if self.exhibits_df is None:
            return {
                'passed': False,
                'message': "✗ Cannot validate date constraints: exhibits CSV not loaded",
                'required_exhibits': [],
                'visited_exhibits': [],
                'missing_exhibits': []
            }
        
        # Verify preprocessed columns exist
        if 'production_year_min' not in self.exhibits_df.columns:
            return {
                'passed': False,
                'message': "✗ Cannot validate date constraints: CSV not preprocessed (run preprocess_exhibits.py first)",
                'required_exhibits': [],
                'visited_exhibits': [],
                'missing_exhibits': []
            }
        
        date_constraint = self.attribute_constraints['date_constraint']
        
        # Parse year values (supports both numeric and string formats)
        before_year = self._parse_year_value(date_constraint.get('before_year'))
        after_year = self._parse_year_value(date_constraint.get('after_year'))
        
        # Parse in_range if provided
        in_range = date_constraint.get('in_range')
        if in_range:
            if isinstance(in_range, list) and len(in_range) == 2:
                in_range = [self._parse_year_value(in_range[0]), self._parse_year_value(in_range[1])]
            else:
                in_range = None
        
        # Find all exhibits matching the date constraint
        matching_exhibit_ids = []
        
        for _, row in self.exhibits_df.iterrows():
            # Use preprocessed columns directly
            if pd.notna(row['production_year_min']):
                start_year = int(row['production_year_min'])
                end_year = int(row['production_year_max'])
            else:
                # Skip rows where date parsing failed
                continue
            
            matches = True
            
            if before_year is not None:
                # Both start and end must be before the threshold
                if end_year >= before_year:
                    matches = False
            
            if after_year is not None:
                # Both start and end must be after the threshold
                if start_year <= after_year:
                    matches = False
            
            if in_range is not None and len(in_range) == 2:
                range_start, range_end = in_range
                # Exhibit must be entirely within range
                if start_year < range_start or end_year > range_end:
                    matches = False
            
            if matches:
                matching_exhibit_ids.append(int(row['exhibit_number']))
        
        # Check which matching exhibits were visited
        visited_matching = [eid for eid in matching_exhibit_ids if eid in self.visited_exhibits]
        missing = [eid for eid in matching_exhibit_ids if eid not in self.visited_exhibits]
        
        passed = len(missing) == 0
        
        if len(matching_exhibit_ids) > 0:
            percentage = (len(visited_matching) / len(matching_exhibit_ids)) * 100
        else:
            percentage = 100.0
        
        # Build constraint description
        constraint_desc = []
        if before_year is not None:
            constraint_desc.append(f"before {before_year} AD")
        if after_year is not None:
            constraint_desc.append(f"after {after_year} AD" if after_year > 0 else f"after {-after_year} BC")
        if in_range is not None and len(in_range) == 2:
            constraint_desc.append(f"between {in_range[0]}-{in_range[1]}")
        
        constraint_str = " AND ".join(constraint_desc) if constraint_desc else "specified dates"
        
        return {
            'passed': passed,
            'required_exhibits': matching_exhibit_ids,
            'visited_exhibits': visited_matching,
            'missing_exhibits': missing,
            'percentage': round(percentage, 1),
            'message': f"{'✓' if passed else '✗'} Date constraint ({constraint_str}): {len(visited_matching)}/{len(matching_exhibit_ids)} exhibits visited ({percentage:.1f}%){'' if passed else f' (missing: {missing})'}"
        }
    
    def validate_material_constraint(self):
        """
        Validate that ALL exhibits matching material constraints were visited.
        
        Returns:
            dict: Validation result with 'passed' boolean and details
        """
        if not self.attribute_constraints or 'material_constraint' not in self.attribute_constraints:
            return {
                'passed': True,
                'message': "✓ No material constraints specified",
                'required_exhibits': [],
                'visited_exhibits': [],
                'missing_exhibits': []
            }
        
        if self.exhibits_df is None:
            return {
                'passed': False,
                'message': "✗ Cannot validate material constraints: exhibits CSV not loaded"
            }
        
        material_constraint = self.attribute_constraints['material_constraint']
        required_materials = material_constraint.get('materials', [])
        
        if not required_materials:
            return {
                'passed': True,
                'message': "✓ No specific materials specified",
                'required_exhibits': [],
                'visited_exhibits': [],
                'missing_exhibits': []
            }
        
        # Find all exhibits with any of the required materials
        matching_exhibit_ids = []
        
        for _, row in self.exhibits_df.iterrows():
            material_str = str(row['Materials']).lower()
            
            # Check if any required material is in this exhibit's materials
            for req_material in required_materials:
                if req_material.lower() in material_str:
                    matching_exhibit_ids.append(int(row['exhibit_number']))
                    break
        
        # Check which matching exhibits were visited
        visited_matching = [eid for eid in matching_exhibit_ids if eid in self.visited_exhibits]
        missing = [eid for eid in matching_exhibit_ids if eid not in self.visited_exhibits]
        
        passed = len(missing) == 0
        
        if len(matching_exhibit_ids) > 0:
            percentage = (len(visited_matching) / len(matching_exhibit_ids)) * 100
        else:
            percentage = 100.0
        
        material_str = ", ".join(required_materials)
        
        return {
            'passed': passed,
            'required_exhibits': matching_exhibit_ids,
            'visited_exhibits': visited_matching,
            'missing_exhibits': missing,
            'percentage': round(percentage, 1),
            'message': f"{'✓' if passed else '✗'} Material constraint ({material_str}): {len(visited_matching)}/{len(matching_exhibit_ids)} exhibits visited ({percentage:.1f}%){'' if passed else f' (missing: {missing})'}"
        }
    
    def validate_find_spot_constraint(self):
        """
        Validate that ALL exhibits matching find spot constraints were visited.
        Requires preprocessed CSV with find_spot_city and find_spot_country columns.
        
        Returns:
            dict: Validation result with 'passed' boolean and details
        """
        if not self.attribute_constraints or 'find_spot_constraint' not in self.attribute_constraints:
            return {
                'passed': True,
                'message': "✓ No find spot constraints specified",
                'required_exhibits': [],
                'visited_exhibits': [],
                'missing_exhibits': []
            }
        
        if self.exhibits_df is None:
            return {
                'passed': False,
                'message': "✗ Cannot validate find spot constraints: exhibits CSV not loaded",
                'required_exhibits': [],
                'visited_exhibits': [],
                'missing_exhibits': []
            }
        
        # Verify preprocessed columns exist
        if 'find_spot_city' not in self.exhibits_df.columns or 'find_spot_country' not in self.exhibits_df.columns:
            return {
                'passed': False,
                'message': "✗ Cannot validate find spot constraints: CSV not preprocessed (run preprocess_exhibits.py first)",
                'required_exhibits': [],
                'visited_exhibits': [],
                'missing_exhibits': []
            }
        
        find_spot_constraint = self.attribute_constraints['find_spot_constraint']
        locations = find_spot_constraint.get('locations', [])
        
        if not locations:
            return {
                'passed': True,
                'message': "✓ No specific locations specified",
                'required_exhibits': [],
                'visited_exhibits': [],
                'missing_exhibits': []
            }
        
        # Find all exhibits from any of the specified locations using preprocessed columns
        matching_exhibit_ids = []
        
        for _, row in self.exhibits_df.iterrows():
            # Use preprocessed city and country columns
            city = str(row['find_spot_city']).lower() if pd.notna(row['find_spot_city']) else ''
            country = str(row['find_spot_country']).lower() if pd.notna(row['find_spot_country']) else ''
            
            # Check if any required location matches the city or country
            for location in locations:
                location_lower = location.lower()
                if location_lower in city or location_lower in country:
                    matching_exhibit_ids.append(int(row['exhibit_number']))
                    break
        
        # Check which matching exhibits were visited
        visited_matching = [eid for eid in matching_exhibit_ids if eid in self.visited_exhibits]
        missing = [eid for eid in matching_exhibit_ids if eid not in self.visited_exhibits]
        
        passed = len(missing) == 0
        
        if len(matching_exhibit_ids) > 0:
            percentage = (len(visited_matching) / len(matching_exhibit_ids)) * 100
        else:
            percentage = 100.0
        
        location_str = ", ".join(locations)
        
        return {
            'passed': passed,
            'required_exhibits': matching_exhibit_ids,
            'visited_exhibits': visited_matching,
            'missing_exhibits': missing,
            'percentage': round(percentage, 1),
            'message': f"{'✓' if passed else '✗'} Find spot constraint ({location_str}): {len(visited_matching)}/{len(matching_exhibit_ids)} exhibits visited ({percentage:.1f}%){'' if passed else f' (missing: {missing})'}"
        }
    
    def validate_technique_constraint(self):
        """
        Validate that ALL exhibits matching technique constraints were visited.
        
        Returns:
            dict: Validation result with 'passed' boolean and details
        """
        if not self.attribute_constraints or 'technique_constraint' not in self.attribute_constraints:
            return {
                'passed': True,
                'message': "✓ No technique constraints specified",
                'required_exhibits': [],
                'visited_exhibits': [],
                'missing_exhibits': []
            }
        
        if self.exhibits_df is None:
            return {
                'passed': False,
                'message': "✗ Cannot validate technique constraints: exhibits CSV not loaded"
            }
        
        technique_constraint = self.attribute_constraints['technique_constraint']
        required_techniques = technique_constraint.get('techniques', [])
        
        if not required_techniques:
            return {
                'passed': True,
                'message': "✓ No specific techniques specified",
                'required_exhibits': [],
                'visited_exhibits': [],
                'missing_exhibits': []
            }
        
        # Find all exhibits with any of the required techniques
        matching_exhibit_ids = []
        
        for _, row in self.exhibits_df.iterrows():
            technique_str = str(row['Technique']).lower()
            
            # Check if any required technique is in this exhibit's techniques
            for req_technique in required_techniques:
                if req_technique.lower() in technique_str:
                    matching_exhibit_ids.append(int(row['exhibit_number']))
                    break
        
        # Check which matching exhibits were visited
        visited_matching = [eid for eid in matching_exhibit_ids if eid in self.visited_exhibits]
        missing = [eid for eid in matching_exhibit_ids if eid not in self.visited_exhibits]
        
        passed = len(missing) == 0
        
        if len(matching_exhibit_ids) > 0:
            percentage = (len(visited_matching) / len(matching_exhibit_ids)) * 100
        else:
            percentage = 100.0
        
        technique_str = ", ".join(required_techniques)
        
        return {
            'passed': passed,
            'required_exhibits': matching_exhibit_ids,
            'visited_exhibits': visited_matching,
            'missing_exhibits': missing,
            'percentage': round(percentage, 1),
            'message': f"{'✓' if passed else '✗'} Technique constraint ({technique_str}): {len(visited_matching)}/{len(matching_exhibit_ids)} exhibits visited ({percentage:.1f}%){'' if passed else f' (missing: {missing})'}"
        }

    
    def validate_at_least_n_exhibit_visits(self):
        """
        Validate that at least N exhibits were visited.
        
        Returns:
            dict: Validation result with 'passed' boolean and details
        """
        num_visited = len(self.visited_exhibits)
        passed = num_visited >= self.at_least_n_exhibits_to_cover
        
        # Calculate percentage
        if self.at_least_n_exhibits_to_cover > 0:
            percentage = (num_visited / self.at_least_n_exhibits_to_cover) * 100
        else:
            percentage = 100.0
        
        return {
            'passed': passed,
            'required': self.at_least_n_exhibits_to_cover,
            'actual': num_visited,
            'percentage': round(percentage, 1),
            'message': f"{'✓' if passed else '✗'} Visited {num_visited} exhibits (required: {self.at_least_n_exhibits_to_cover}) - {percentage:.1f}%"
        }
    
    def validate_category_coverage(self):
        """
        Validate that all exhibits from each required category was visited.
        
        Returns:
            dict: Validation result with 'passed' boolean and details per category
        """
        if not self.exhibit_categories_to_cover:
            return {
                'passed': True,
                'required_categories': [],
                'category_results': {},
                'message': "✓ No category requirements specified"
            }
        
        if not self.exhibits_by_section:
            return {
                'passed': False,
                'required_categories': self.exhibit_categories_to_cover,
                'category_results': {},
                'message': "✗ Cannot validate categories: exhibits_by_section file not provided"
            }
        
        category_results = {}
        all_passed = True
        
        for category in self.exhibit_categories_to_cover:
            if category not in self.exhibits_by_section:
                category_results[category] = {
                    'passed': False,
                    'exhibits_in_category': [],
                    'visited_from_category': [],
                    'message': f"✗ Category '{category}' not found in exhibits_by_section"
                }
                all_passed = False
                continue
            
            exhibits_in_category = self.exhibits_by_section[category]
            visited_from_category = [e for e in self.visited_exhibits if e in exhibits_in_category]
            passed = len(visited_from_category) == len(exhibits_in_category)
            
            # Calculate percentage for this category
            total_in_category = len(exhibits_in_category)
            if total_in_category > 0:
                category_percentage = (len(visited_from_category) / total_in_category) * 100
            else:
                category_percentage = 0.0
            
            category_results[category] = {
                'passed': passed,
                'exhibits_in_category': exhibits_in_category,
                'visited_from_category': visited_from_category,
                'count': len(visited_from_category),
                'total_in_category': total_in_category,
                'percentage': round(category_percentage, 1),
                'message': f"{'✓' if passed else '✗'} Category '{category}': {len(visited_from_category)}/{total_in_category} exhibit(s) visited ({category_percentage:.1f}%)"
            }
            
            if not passed:
                all_passed = False
        
        return {
            'passed': all_passed,
            'required_categories': self.exhibit_categories_to_cover,
            'category_results': category_results,
            'message': f"{'✓' if all_passed else '✗'} Category coverage: {sum(1 for r in category_results.values() if r['passed'])}/{len(self.exhibit_categories_to_cover)} categories covered"
        }
    
    def validate_specific_exhibits(self):
        """
        Validate that all specific required exhibits were visited.
        
        Returns:
            dict: Validation result with 'passed' boolean and details
        """
        if not self.specific_exhibit_to_cover:
            return {
                'passed': True,
                'required_exhibits': [],
                'visited_exhibits': [],
                'missing_exhibits': [],
                'percentage': 100.0,
                'message': "✓ No specific exhibit requirements specified"
            }
        
        required_set = set(self.specific_exhibit_to_cover)
        visited_set = set(self.visited_exhibits)
        
        visited_required = required_set.intersection(visited_set)
        missing_exhibits = list(required_set - visited_set)
        
        passed = len(missing_exhibits) == 0
        
        # Calculate percentage
        total_required = len(self.specific_exhibit_to_cover)
        if total_required > 0:
            percentage = (len(visited_required) / total_required) * 100
        else:
            percentage = 100.0
        
        return {
            'passed': passed,
            'required_exhibits': self.specific_exhibit_to_cover,
            'visited_exhibits': list(visited_required),
            'missing_exhibits': missing_exhibits,
            'percentage': round(percentage, 1),
            'message': f"{'✓' if passed else '✗'} Specific exhibits: {len(visited_required)}/{len(self.specific_exhibit_to_cover)} required exhibits visited ({percentage:.1f}%){'' if passed else f' (missing: {missing_exhibits})'}"
        }
    
    def validate(self):
        """
        Run all semantic validations and return comprehensive results.
        
        Returns:
            dict: Complete validation results with overall pass/fail status
        """
        print("\n" + "="*60)
        print("Semantic Validation")
        print("="*60)
        
        # Run all validation checks
        count_validation = self.validate_at_least_n_exhibit_visits()
        category_validation = self.validate_category_coverage()
        specific_validation = self.validate_specific_exhibits()
        
        # Run attribute-based validations if exhibit CSV was loaded
        date_validation = self.validate_date_constraint()
        material_validation = self.validate_material_constraint()
        find_spot_validation = self.validate_find_spot_constraint()
        technique_validation = self.validate_technique_constraint()
        combined_validation = self.validate_combined_constraint()
        
        # Overall pass requires all checks to pass
        overall_passed = (
            count_validation['passed'] and 
            category_validation['passed'] and 
            specific_validation['passed'] and
            date_validation['passed'] and
            material_validation['passed'] and
            find_spot_validation['passed'] and
            technique_validation['passed'] and
            combined_validation['passed']
        )
        
        # Print results
        print(f"\n{count_validation['message']}")
        print(f"{category_validation['message']}")
        
        # Print detailed category results
        if category_validation.get('category_results'):
            for category, result in category_validation['category_results'].items():
                print(f"  {result['message']}")
                if result['passed'] and result.get('visited_from_category'):
                    print(f"    Visited: {result['visited_from_category']}")
        
        print(f"{specific_validation['message']}")
        
        # Print attribute validation results
        if self.exhibits_df is not None:
            print("\n" + "-"*60)
            print("Attribute-Based Validations:")
            print(f"{date_validation['message']}")
            print(f"{material_validation['message']}")
            print(f"{find_spot_validation['message']}")
            print(f"{technique_validation['message']}")
            print(f"{combined_validation['message']}")
        
        print("\n" + "-"*60)
        print(f"Overall Semantic Validation: {'✅ PASSED' if overall_passed else '❌ FAILED'}")
        print("="*60)
        
        # Build exhibit_category_coverage dictionary with per-category results
        exhibit_category_coverage = {}
        if category_validation.get('category_results'):
            for category, result in category_validation['category_results'].items():
                exhibit_category_coverage[category] = {
                    "valid": result['passed'],
                    "percentage": result.get('percentage', 0.0)
                }
        
        return {
            "specific_exhibit_coverage": {
                "valid": specific_validation['passed'],
                "percentage": specific_validation.get('percentage', 0.0)
            },
            "at_least_n_exhibits_coverage": {
                "valid": count_validation['passed'],
                "percentage": count_validation.get('percentage', 0.0)
            },
            "exhibit_category_coverage": exhibit_category_coverage,
            "attribute_validations": {
                "date_constraint": {
                    "valid": date_validation['passed'],
                    "percentage": date_validation.get('percentage', 0.0)
                },
                "material_constraint": {
                    "valid": material_validation['passed'],
                    "percentage": material_validation.get('percentage', 0.0)
                },
                "find_spot_constraint": {
                    "valid": find_spot_validation['passed'],
                    "percentage": find_spot_validation.get('percentage', 0.0)
                },
                "technique_constraint": {
                    "valid": technique_validation['passed'],
                    "percentage": technique_validation.get('percentage', 0.0)
                },
                "combined_constraint": {
                    "valid": combined_validation['passed'],
                    "percentage": combined_validation.get('percentage', 0.0)
                }
            }
        }, {
            'overall_passed': overall_passed,
            'visited_exhibits': self.visited_exhibits,
            'count_validation': count_validation,
            'category_validation': category_validation,
            'specific_validation': specific_validation,
            'date_validation': date_validation,
            'material_validation': material_validation,
            'find_spot_validation': find_spot_validation,
            'technique_validation': technique_validation,
            'combined_validation': combined_validation,
            'summary': {
                'total_exhibits_visited': len(self.visited_exhibits),
                'required_minimum': self.at_least_n_exhibits_to_cover,
                'required_categories': self.exhibit_categories_to_cover,
                'categories_covered': sum(1 for r in category_validation.get('category_results', {}).values() if r['passed']),
                'required_specific_exhibits': len(self.specific_exhibit_to_cover),
                'specific_exhibits_visited': len(specific_validation['visited_exhibits'])
            }
        }
