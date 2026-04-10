"""
Test script for attribute-based semantic validation.
Demonstrates how to validate exhibit attributes like production date, materials, etc.

NOTE: This test requires preprocessed CSV file.
Run: python exhibits_construction_helpers/preprocess_exhibits.py first
"""

from semantic_validator import SemanticValidator
import pandas as pd
import json

def main():
    print("="*70)
    print("Testing Attribute-Based Semantic Validation")
    print("="*70)
    
    # Check if preprocessed CSV exists
    import os
    csv_file = './original_floorplans/museum_layout_01/layout_01_exhibits.csv'
    if not os.path.exists(csv_file):
        print(f"\n❌ ERROR: {csv_file} not found!")
        print("Please run preprocessing first:")
        print("  python exhibits_construction_helpers/preprocess_exhibits.py")
        return
    
    # Load the preprocessed exhibits CSV
    df = pd.read_csv(csv_file)
    print(f"\n✓ Loaded {len(df)} exhibits from {csv_file}")
    
    # Show some statistics using preprocessed columns
    print(f"\nExhibit Statistics:")
    print(f"  Sections: {df['Section'].unique().tolist()}")
    print(f"  Total Greek exhibits: {len(df[df['Section'] == 'Greek'])}")
    print(f"  Total pottery exhibits: {len(df[df['Materials'].str.contains('pottery', case=False, na=False)])}")
    print(f"  Exhibits from Egypt (preprocessed): {len(df[df['find_spot_country'] == 'Egypt'])}")
    print(f"  Painted exhibits: {len(df[df['Technique'].str.contains('painted', case=False, na=False)])}")
    
    # Count exhibits before year 600 using preprocessed columns
    before_600_count = len(df[(df['production_year_max'].notna()) & (df['production_year_max'] < 600)])
    print(f"  Exhibits before year 600 AD (preprocessed): {before_600_count}")
    
    # Test Case 1: Visit ALL required exhibits (should PASS)
    print("\n" + "="*70)
    print("TEST CASE 1: Visit ALL required exhibits")
    print("="*70)
    
 
    
   
    visited_exhibits = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    print(f"Simulating visit to {len(visited_exhibits)} Greek exhibits")
    print(f"Visited IDs: {visited_exhibits}... (showing first 10)")
    
    # Create validator with attribute constraints using preprocessed CSV
    validator = SemanticValidator(
        config_file='config_with_attributes.json',
        visited_exhibits=visited_exhibits,
        exhibits_csv_file=csv_file
    )
    
    # Run validation
    results, detailed = validator.validate()
    
   
    print("\n" + "="*70)
    print("Testing Complete!")
    print("="*70)
    print("\nSummary:")
    print(f"Results: \n{json.dumps(results, indent=4)}")
    print(f"Details : \n{json.dumps(detailed, indent=4)}")

if __name__ == '__main__':
    main()