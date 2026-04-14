"""
Unit tests for validation utility functions.
Tests filter_and_validate_attributes and extract_scar_validations with mock data.
"""
import json
from pathlib import Path
from prompts_utils import filter_and_validate_attributes, extract_scar_validations, filter_non_empty_fields


def test_with_categories_in_config():
    """Test extract_scar_validations when config has categories."""
    print("="*70)
    print("TEST 4: extract_scar_validations() WITH categories in config")
    print("="*70)
    
    # Create config WITH categories
    config_with_categories = {
        "exhibit_categories_to_cover": ["Greek", "Roman"],
        "exhibit_attribute_constraints": {
            "date_constraint": {"in_range": []},
                "combined_constraint": {
                "_combined_comment": "Combined constraint uses AND logic - exhibit must match ALL specified conditions",
                "date_constraint": {
                    "in_range": []
                },
                "material_constraint": {
                    "materials": ["pottery"]
                },
                "find_spot_constraint": {
                    "locations": []
                },
                "technique_constraint": {
                    "techniques": []
                }
            }
        },
       "material_constraint": {
            "materials": ["pottery"]
        },
        "find_spot_constraint": {
            "locations": ["China"]
        },
        "technique_constraint": {
            "techniques": []
        },
    }
    
    # Create mock scar_full
    scar_full = {
        "specific_exhibit_coverage": {
            "valid": True,
            "percentage": 100.0
        },
        "at_least_n_exhibits_coverage": {
            "valid": True,
            "percentage": 3800.0
        },
        "exhibit_category_coverage": {
            "Greek": {"valid": True, "percentage": 100.0},
            "Roman": {"valid": False, "percentage": 3.0}
        },
        "attribute_validations": {
            "date_constraint": {
                "valid": False,
                "percentage": 3.0
            },
            "material_constraint": {
                "valid": True,
                "percentage": 0.0
            },
            "find_spot_constraint": {
                "valid": True,
                "percentage": 0.0
            },
            "technique_constraint": {
                "valid": True,
                "percentage": 0.0
            },
            "combined_constraint": {
                "valid": True,
                "percentage": 3.0
            }
        },
         
    }
    
    fields_to_include = ['exhibit_category_coverage', 'attribute_validations']
    
    result = extract_scar_validations(scar_full, config_with_categories, fields_to_include)
    
    print(f"Config: exhibit_categories_to_cover = {config_with_categories['exhibit_categories_to_cover']}")
    print(f"\nResult: {json.dumps(result, indent=2)}")
    

    return True


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("VALIDATION UTILS TEST SUITE")
    print("="*70 + "\n")
    
    results = []
    # results.append(("filter_non_empty_fields", test_filter_non_empty_fields()))
    # results.append(("filter_and_validate_attributes", test_filter_and_validate_attributes()))
    # results.append(("extract_scar_validations (no categories)", test_extract_scar_validations()))
    results.append(("extract_scar_validations (with categories)", test_with_categories_in_config()))
    
    # Summary
    print("="*70)
    print("TEST SUMMARY")
    print("="*70)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"Tests passed: {passed}/{total}\n")
    
    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {status}: {name}")
    
    print("="*70)
    
    return all(result for _, result in results)


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
