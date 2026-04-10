"""
Exhibit List Generator (CSV Version v4.0)
Generates exhibit list from preprocessed CSV with exhibit number assignments.
Outputs:
  1. CSV file with exhibit_number column + all original columns
  2. JSON file with exhibit_number and description only
"""

import json
from pathlib import Path
from collections import OrderedDict
import pandas as pd


def parse_input_line(line):
    """
    Parse a line like "1-10: Roman" or "8: maya"
    Returns: (exhibit_numbers, section)
    Note: range(start, end + 1) ensures 1-5 includes [1, 2, 3, 4, 5]
    """
    line = line.strip()
    if not line or line.startswith('#'):
        return None, None
    
    # Split by colon
    parts = line.split(':', 1)
    if len(parts) != 2:
        print(f"⚠️  Invalid format: {line}")
        return None, None
    
    number_part = parts[0].strip()
    section = parts[1].strip()
    
    numbers = []
    
    # Check if it's a range (e.g., "1-10")
    if '-' in number_part:
        try:
            start, end = number_part.split('-', 1)
            start = int(start.strip())
            end = int(end.strip())
            numbers = list(range(start, end + 1))  # Includes both start and end
        except ValueError:
            print(f"⚠️  Invalid range: {number_part}")
            return None, None
    else:
        # Single number
        try:
            numbers = [int(number_part)]
        except ValueError:
            print(f"⚠️  Invalid number: {number_part}")
            return None, None
    
    return numbers, section


def load_artifacts_dataframe(csv_file='selected_exhibits_preprocessed.csv'):
    """
    Load artifacts from preprocessed CSV file.
    Returns: pandas DataFrame
    """
    csv_path = Path(csv_file)
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Ensure required columns exist
    required_cols = ['id', 'Section', 'description']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in CSV file: {missing_cols}")
    
    return df


def count_available_artifacts(section, df):
    """
    Count total artifacts available in a section from dataframe.
    Returns: artifact_count
    """
    # Case-insensitive section matching
    section_lower = section.lower()
    matching_rows = df[df['Section'].str.lower() == section_lower]
    return len(matching_rows)


def get_artifacts_from_section(section, df):
    """
    Get ALL artifacts from a section in the dataframe.
    Returns: DataFrame of matching artifacts
    """
    # Case-insensitive section matching
    section_lower = section.lower()
    matching_artifacts = df[df['Section'].str.lower() == section_lower].copy()
    
    # Sort by id for consistency
    matching_artifacts = matching_artifacts.sort_values('id')
    
    return matching_artifacts


def validate_artifact_availability(section_requests, df):
    """
    Validate that each section has enough artifacts.
    Returns: (is_valid, validation_report)
    """
    print("\n" + "=" * 60)
    print("🔍 Validating Artifact Availability...")
    print("=" * 60)
    
    validation_report = []
    is_valid = True
    
    # Get available sections
    available_sections = df['Section'].unique().tolist()
    
    for section, exhibit_numbers in section_requests.items():
        needed = len(exhibit_numbers)
        available = count_available_artifacts(section, df)
        
        if available == 0:
            is_valid = False
            msg = f"❌ {section}: Section not found"
            print(msg)
            validation_report.append(msg)
            print(f"   Available sections: {', '.join(sorted(available_sections))}")
        elif available < needed:
            is_valid = False
            shortage = needed - available
            msg = f"❌ {section}: Needs {needed} artifacts, only {available} available (shortage: {shortage})"
            print(msg)
            validation_report.append(msg)
        else:
            msg = f"✅ {section}: Needs {needed} artifacts, {available} available"
            print(msg)
            validation_report.append(msg)
    
    return is_valid, validation_report


def generate_exhibit_list(input_file=None, csv_file='selected_exhibits_preprocessed.csv'):
    """
    Generate exhibit list from user input or file with duplicate prevention.
    Reads artifact data from preprocessed CSV file.
    Returns: (exhibit_json_list, exhibit_dataframe_with_numbers)
    """
    print("🏛️  Exhibit List Generator (CSV Version v4.0)")
    print("=" * 60)
    
    # Load artifacts dataframe
    print(f"📊 Loading artifacts from: {csv_file}")
    try:
        df = load_artifacts_dataframe(csv_file)
        print(f"✅ Loaded {len(df)} artifacts from CSV file")
        print(f"   Columns: {', '.join(df.columns.tolist())}")
    except Exception as e:
        print(f"❌ Error loading CSV file: {e}")
        return None
    
    if input_file and Path(input_file).exists():
        print(f"📄 Reading input from file: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            input_lines = f.readlines()
    else:
        print("\n📝 Enter exhibit assignments (one per line)")
        print("   Format: 'number-range: Section' or 'number: Section'")
        print("   Examples:")
        print("     1-10: Roman")
        print("     8: maya")
        print("     9-12: Tang Dynasty")
        print("   Enter a blank line when done.\n")
        
        input_lines = []
        while True:
            line = input("➤ ")
            if not line.strip():
                break
            input_lines.append(line)
    
    print("\n" + "=" * 60)
    print("📊 Parsing Input...")
    print("-" * 60)
    
    # Parse all inputs and track assignments
    exhibit_assignments = OrderedDict()
    
    for line in input_lines:
        numbers, section = parse_input_line(line)
        if numbers is None:
            continue
        
        for num in numbers:
            if num in exhibit_assignments:
                print(f"ℹ️  Exhibit {num} already assigned, skipping duplicate")
            else:
                exhibit_assignments[num] = section
    
    if not exhibit_assignments:
        print("❌ No valid exhibit assignments found")
        return None
    
    print(f"\n✅ Parsed {len(exhibit_assignments)} unique exhibit assignments")
    
    # Group by section
    section_requests = {}
    for exhibit_num, section in exhibit_assignments.items():
        if section not in section_requests:
            section_requests[section] = []
        section_requests[section].append(exhibit_num)
    
    print(f"📂 Sections requested: {len(section_requests)}")
    
    # VALIDATION: Check artifact availability
    is_valid, report = validate_artifact_availability(section_requests, df)
    
    if not is_valid:
        print("\n" + "=" * 60)
        print("❌ VALIDATION FAILED")
        print("=" * 60)
        print("Cannot generate exhibit list due to insufficient artifacts.")
        print("Please adjust your input and try again.")
        return None
    
    print("\n" + "=" * 60)
    print("✅ Validation Passed - Generating Exhibit List...")
    print("=" * 60)
    
    # Track used artifact IDs globally to prevent duplicates
    used_artifact_ids = set()
    used_artifact_ids_list = []  # Ordered list for display
    exhibit_json_list = []  # For JSON output (exhibit_number + description only)
    exhibit_rows_list = []  # For CSV output (full rows with exhibit_number)
    
    # Process each section
    for section, exhibit_numbers in section_requests.items():
        print(f"\n📁 Processing section: {section}")
        print(f"   Requested exhibits: {len(exhibit_numbers)}")
        
        # Get ALL artifacts from this section
        section_artifacts = get_artifacts_from_section(section, df)
        
        # Map exhibit numbers to unique artifacts
        artifacts_assigned = 0
        artifact_index = 0
        
        for exhibit_num in sorted(exhibit_numbers):
            # Find next unused artifact
            while artifact_index < len(section_artifacts):
                artifact_row = section_artifacts.iloc[artifact_index]
                artifact_index += 1
                
                artifact_id = str(artifact_row['id'])
                
                # Check if this artifact was already used
                if artifact_id in used_artifact_ids:
                    print(f"   ⊘ Skipping duplicate artifact ID: {artifact_id}")
                    continue
                
                # This artifact is unique, use it
                used_artifact_ids.add(artifact_id)
                used_artifact_ids_list.append(artifact_id)
                
                # Create JSON entry with only exhibit_number and description
                exhibit_json_entry = {
                    'exhibit_number': exhibit_num,
                    'description': artifact_row['description']
                }
                exhibit_json_list.append(exhibit_json_entry)
                
                # Create CSV entry with exhibit_number + all original columns
                row_dict = artifact_row.to_dict()
                row_dict['exhibit_number'] = exhibit_num
                exhibit_rows_list.append(row_dict)
                
                artifacts_assigned += 1
                print(f"   ✓ Exhibit {exhibit_num}: {artifact_id}")
                break
            else:
                # Ran out of unique artifacts
                print(f"   ✗ Exhibit {exhibit_num}: No more unique artifacts available")
        
        print(f"   → Assigned {artifacts_assigned}/{len(exhibit_numbers)} exhibits")
    
    # Create DataFrame for CSV output with exhibit_number as first column
    exhibit_df = pd.DataFrame(exhibit_rows_list)
    
    # Reorder columns to put exhibit_number first
    cols = exhibit_df.columns.tolist()
    cols.remove('exhibit_number')
    exhibit_df = exhibit_df[['exhibit_number'] + cols]
    
    # Sort by exhibit number
    exhibit_df = exhibit_df.sort_values('exhibit_number').reset_index(drop=True)
    exhibit_json_list.sort(key=lambda x: x['exhibit_number'])
    
    return exhibit_json_list, exhibit_df


def save_exhibit_json(exhibit_list, output_file='exhibit_list.json'):
    """
    Save exhibit list to JSON file (exhibit_number + description only).
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(exhibit_list, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Saved JSON to: {output_file}")


def save_exhibit_csv(exhibit_df, output_file='selected_exhibits_final.csv'):
    """
    Save exhibit DataFrame to CSV file (exhibit_number + all columns).
    """
    exhibit_df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"💾 Saved CSV to: {output_file}")


def main():
    """
    Main function with command-line argument parsing.
    """
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Generate exhibit list from preprocessed CSV with exhibit number assignments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python generate_exhibit_list.py -i exhibit_input.txt -c selected_exhibits_preprocessed.csv -o selected_exhibits_final.csv
  
  # Specify custom JSON output directory
  python generate_exhibit_list.py -i exhibit_input.txt -c selected_exhibits_preprocessed.csv -o final.csv -j output_dir/
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help='Input file with exhibit assignments (format: "number: section" or "start-end: section")'
    )
    
    parser.add_argument(
        '-c', '--csv-input',
        type=str,
        default='selected_exhibits_preprocessed.csv',
        help='Input CSV file with preprocessed exhibit data (default: selected_exhibits_preprocessed.csv)'
    )
    
    parser.add_argument(
        '-o', '--csv-output',
        type=str,
        required=True,
        help='Output CSV file with exhibit_number column (required)'
    )
    
    parser.add_argument(
        '-j', '--json-dir',
        type=str,
        default=None,
        help='Output directory for JSON file (default: same as CSV output directory)'
    )
    
    args = parser.parse_args()
    
    # Determine JSON output directory
    if args.json_dir:
        json_dir = Path(args.json_dir)
    else:
        # Default to same directory as CSV output
        json_dir = Path(args.csv_output).parent
    
    # Create output directory if it doesn't exist
    json_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate exhibit list
    result = generate_exhibit_list(args.input, args.csv_input)
    
    if result:
        exhibit_json_list, exhibit_df = result
        
        print("\n" + "=" * 60)
        print("✨ Generation Complete!")
        print("=" * 60)
        print(f"📊 Total exhibits in list: {len(exhibit_json_list)}")
        
        # Show summary by section
        section_counts = exhibit_df['Section'].value_counts().to_dict()
        
        print("\n📈 Exhibits by section:")
        for section, count in sorted(section_counts.items()):
            print(f"   • {section}: {count} exhibit(s)")
        
        # Display artifact IDs used (for cross-checking)
        artifact_ids = exhibit_df['id'].tolist()
        print(f"\n🔢 Artifact IDs Used ({len(artifact_ids)} total):")
        print(f"   {', '.join(map(str, artifact_ids))}")
        
        # Verify uniqueness
        unique_ids = exhibit_df['id'].nunique()
        print(f"\n✅ Artifact Uniqueness: {unique_ids}/{len(artifact_ids)} unique artifacts")
        
        if unique_ids < len(artifact_ids):
            print("⚠️  Warning: Some duplicate artifacts detected!")
        
        # Save CSV file
        save_exhibit_csv(exhibit_df, args.csv_output)
        
        # Save JSON file
        json_file = json_dir / 'exhibit_list.json'
        save_exhibit_json(exhibit_json_list, str(json_file))
        
        print(f"\n✅ Done! Two files generated:")
        print(f"   • {args.csv_output} - Full exhibit data with exhibit_number column")
        print(f"   • {json_file} - Exhibit list with descriptions (JSON)")
    else:
        print("\n❌ Failed to generate exhibit list")


if __name__ == '__main__':
    main()
