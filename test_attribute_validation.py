"""
Test script for attribute-based semantic validation.
Demonstrates how to validate exhibit attributes like production date, materials, etc.
"""

from semantic_validator import SemanticValidator
import pandas as pd

def main():
    print("="*70)
    print("Testing Attribute-Based Semantic Validation")
    print("="*70)
    
    # Load the exhibits CSV to see what we're working with
    df = pd.read_csv('selected_exhibits.csv')
    print(f"\n✓ Loaded {len(df)} exhibits from selected_exhibits.csv")
    
    # Show some statistics
    print(f"\nExhibit Statistics:")
    print(f"  Sections: {df['Section'].unique().tolist()}")
    print(f"  Total Greek exhibits: {len(df[df['Section'] == 'Greek'])}")
    print(f"  Total pottery exhibits: {len(df[df['Materials'].str.contains('pottery', case=False, na=False)])}")
    print(f"  Exhibits from Naukratis: {len(df[df['Find spot'].str.contains('Naukratis', case=False, na=False)])}")
    print(f"  Painted exhibits: {len(df[df['Technique'].str.contains('painted', case=False, na=False)])}")
    
    # Parse dates to show how many are before year 600
    from semantic_validator.semantic_validator import SemanticValidator as SV
    validator_temp = SV('config.json', [])
    
    before_600_count = 0
    for _, row in df.iterrows():
        start_year, end_year = validator_temp.parse_production_date_to_year(row['Production date'])
        if start_year is not None and end_year is not None and end_year < 600:
            before_600_count += 1
    
    print(f"  Exhibits before year 600 AD: {before_600_count}")
    
    # Test Case 1: Visit ALL required exhibits (should PASS)
    print("\n" + "="*70)
    print("TEST CASE 1: Visit ALL required exhibits")
    print("="*70)
    
    # Get all Greek exhibit IDs
    greek_exhibits = df[df['Section'] == 'Greek']['id'].tolist()
    
    # For this test, let's visit all Greek exhibits
    visited_exhibits = greek_exhibits[:20]  # Visit first 20 Greek exhibits
    
    print(f"Simulating visit to {len(visited_exhibits)} Greek exhibits")
    print(f"Visited IDs: {visited_exhibits[:10]}... (showing first 10)")
    
    # Create validator with attribute constraints
    validator = SemanticValidator(
        config_file='config_with_attributes.json',
        visited_exhibits=visited_exhibits,
        exhibits_csv_file='selected_exhibits.csv'
    )
    
    # Run validation
    results, detailed = validator.validate()
    
    # Test Case 2: Visit only SOME exhibits (should FAIL some constraints)
    print("\n\n" + "="*70)
    print("TEST CASE 2: Visit only 10 exhibits (should fail some checks)")
    print("="*70)
    
    visited_exhibits_partial = greek_exhibits[:10]
    
    print(f"Simulating visit to only {len(visited_exhibits_partial)} exhibits")
    print(f"Visited IDs: {visited_exhibits_partial}")
    
    validator2 = SemanticValidator(
        config_file='config_with_attributes.json',
        visited_exhibits=visited_exhibits_partial,
        exhibits_csv_file='selected_exhibits.csv'
    )
    
    results2, detailed2 = validator2.validate()
    
    # Test Case 3: Custom date constraint
    print("\n\n" + "="*70)
    print("TEST CASE 3: Test with different date constraint (before 400 BC)")
    print("="*70)
    
    # Create a custom config for this test
    import json
    with open('config_with_attributes.json', 'r') as f:
        config = json.load(f)
    
    config['exhibit_attribute_constraints']['date_constraint']['before_year'] = -400  # 400 BC
    
    with open('config_test_bc.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    validator3 = SemanticValidator(
        config_file='config_test_bc.json',
        visited_exhibits=visited_exhibits,
        exhibits_csv_file='selected_exhibits.csv'
    )
    
    results3, detailed3 = validator3.validate()
    
    print("\n" + "="*70)
    print("Testing Complete!")
    print("="*70)
    print("\nSummary:")
    print(f"  Test 1 - Visit all required: {'PASSED' if detailed['overall_passed'] else 'FAILED'}")
    print(f"  Test 2 - Visit partial: {'PASSED' if detailed2['overall_passed'] else 'FAILED'}")
    print(f"  Test 3 - BC date constraint: {'PASSED' if detailed3['overall_passed'] else 'FAILED'}")

if __name__ == '__main__':
    main()
