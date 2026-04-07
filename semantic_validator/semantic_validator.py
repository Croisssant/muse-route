import cv2
import numpy as np
import json
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import math

class SemanticValidator:
    def __init__(self, config_file, visited_exhibits, exhibits_by_section_file=None):
        """
        Initialize the semantic validator.
        
        Args:
            config_file: Path to JSON file with exhibit configurations
            visited_exhibits: List of visited exhibit IDs (can be strings or integers)
            exhibits_by_section_file: Optional path to JSON file mapping categories to exhibit numbers
        """

        with open(config_file, 'r') as f:
            self.config = json.load(f)
        
        # Convert visited exhibits to integers for consistent comparison
        self.visited_exhibits = [int(e) if isinstance(e, str) else e for e in visited_exhibits]
        
        self.exhibit_categories_to_cover = self.config.get("exhibit_categories_to_cover", [])
        self.at_least_n_exhibits_to_cover = self.config.get("at_least_n_exhibits_to_cover", 0)
        self.specific_exhibit_to_cover = self.config.get("specific_exhibit_to_cover", [])
        
        # Load exhibit category mappings if provided
        self.exhibits_by_section = {}
        if exhibits_by_section_file:
            with open(exhibits_by_section_file, 'r') as f:
                self.exhibits_by_section = json.load(f)

    
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
        Validate that at least one exhibit from each required category was visited.
        
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
            passed = len(visited_from_category) > 0
            
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
        
        # Overall pass requires all checks to pass
        overall_passed = (
            count_validation['passed'] and 
            category_validation['passed'] and 
            specific_validation['passed']
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
        }, {
            'overall_passed': overall_passed,
            'visited_exhibits': self.visited_exhibits,
            'count_validation': count_validation,
            'category_validation': category_validation,
            'specific_validation': specific_validation,
            'summary': {
                'total_exhibits_visited': len(self.visited_exhibits),
                'required_minimum': self.at_least_n_exhibits_to_cover,
                'required_categories': self.exhibit_categories_to_cover,
                'categories_covered': sum(1 for r in category_validation.get('category_results', {}).values() if r['passed']),
                'required_specific_exhibits': len(self.specific_exhibit_to_cover),
                'specific_exhibits_visited': len(specific_validation['visited_exhibits'])
            }
        }
