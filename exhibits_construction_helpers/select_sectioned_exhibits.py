"""
Select Sectioned Exhibits from Timetravel Parquet
Filters data by required fields and numeric production dates.
Selects top 5 sections with 20 exhibits each and outputs to CSV.
"""

import pandas as pd
import argparse
import re
from pathlib import Path


def has_numeric_date(date_str):
    """
    Check if a production date contains extractable numbers.
    Accepts pure numeric dates and dates with BC/AD, but rejects century formats.
    Examples:
        "253-268" -> True
        "307" -> True
        "41-45" -> True
        "317-318 (about)" -> True
        "500BC-490BC" -> True (has numeric year with BC)
        "200 BC-160 BC (circa)" -> True (has numeric year with BC)
        "1stC(mid)" -> False (century format)
        "9thC-10thC(early)" -> False (century format)
    """
    if pd.isna(date_str):
        return False
    
    date_str = str(date_str).strip()
    
    # Reject if contains century indicators like 1stC, 9thC, etc.
    if re.search(r'\d+[stndrh]{2}C', date_str):
        return False
    
    # Accept if it contains numeric patterns (years)
    # This includes formats like:
    # - Pure numbers: "307", "253-268"
    # - With BC/AD: "500BC-490BC", "200 BC-160 BC"
    # - With descriptors: "317-318 (about)", "200 BC (circa)"
    
    # Check if it starts with digits (with optional BC/AD and descriptive text)
    # Pattern matches: number or range, optionally with BC/AD/spaces, optionally with parenthetical text
    pattern = r'^\d+(\s*(BC|AD|bc|ad))?(-\d+(\s*(BC|AD|bc|ad))?)?(\s*\([^)]+\))?'
    
    return bool(re.search(pattern, date_str))


def load_and_filter_data(parquet_file):
    """
    Load parquet file and apply all filters.
    Returns: filtered DataFrame
    """
    print("=" * 70)
    print("📊 Loading and Filtering Data")
    print("=" * 70)
    
    parquet_path = Path(__file__).parent / parquet_file
    
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
    
    # Load data
    df = pd.read_parquet(parquet_path)
    print(f"✓ Loaded {len(df)} total artifacts from parquet file")
    print(f"  Columns: {', '.join(df.columns.tolist())}")
    
    # Filter 1: Remove rows with null values in required fields
    required_fields = ['Production date', 'Find spot', 'Materials', 'Technique']
    print(f"\n🔍 Filtering for complete records...")
    print(f"  Required fields: {', '.join(required_fields)}")
    
    initial_count = len(df)
    df_filtered = df.dropna(subset=required_fields)
    
    removed = initial_count - len(df_filtered)
    print(f"  ✓ Removed {removed} rows with null values")
    print(f"  ✓ Remaining: {len(df_filtered)} rows")
    
    # Filter 2: Keep only rows with numeric production dates
    print(f"\n🔍 Filtering for numeric production dates...")
    df_filtered = df_filtered[df_filtered['Production date'].apply(has_numeric_date)]
    
    removed_non_numeric = initial_count - removed - len(df_filtered)
    print(f"  ✓ Removed {removed_non_numeric} rows with non-numeric dates")
    print(f"  ✓ Remaining: {len(df_filtered)} rows")
    
    # Show sample of valid dates
    sample_dates = df_filtered['Production date'].head(10).tolist()
    print(f"\n  Sample valid dates: {', '.join([str(d) for d in sample_dates[:5]])}")
    
    return df_filtered


def select_top_sections(df, num_sections=5):
    """
    Select top N sections by count of qualifying exhibits.
    Returns: list of section names
    """
    print("\n" + "=" * 70)
    print("📂 Analyzing Sections")
    print("=" * 70)
    
    section_counts = df['Section'].value_counts()
    print(f"\nSection counts (all {len(section_counts)} sections):")
    for section, count in section_counts.items():
        print(f"  • {section}: {count} qualifying exhibits")
    
    top_sections = section_counts.head(num_sections).index.tolist()
    
    print(f"\n✓ Selected top {num_sections} sections:")
    for i, section in enumerate(top_sections, 1):
        count = section_counts[section]
        print(f"  {i}. {section}: {count} exhibits")
    
    return top_sections


def sample_exhibits_by_section(df, sections, exhibits_per_section=20, seed=None):
    """
    Sample exhibits from each section.
    Returns: DataFrame with sampled exhibits
    """
    print("\n" + "=" * 70)
    print("🎲 Sampling Exhibits")
    print("=" * 70)
    
    if seed is not None:
        print(f"Using random seed: {seed}")
    
    sampled_dfs = []
    
    for section in sections:
        section_df = df[df['Section'] == section]
        
        if len(section_df) < exhibits_per_section:
            print(f"\n⚠️  Warning: {section} has only {len(section_df)} exhibits (need {exhibits_per_section})")
            sampled = section_df.copy()
        else:
            sampled = section_df.sample(n=exhibits_per_section, random_state=seed)
        
        sampled_dfs.append(sampled)
        print(f"  ✓ {section}: Selected {len(sampled)} exhibits")
    
    result_df = pd.concat(sampled_dfs, ignore_index=True)
    
    print(f"\n✓ Total exhibits selected: {len(result_df)}")
    
    return result_df


def save_to_csv(df, output_file):
    """
    Save DataFrame to CSV file.
    """
    print("\n" + "=" * 70)
    print("💾 Saving Results")
    print("=" * 70)
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV (exclude Image column as it contains binary data)
    columns_to_save = [col for col in df.columns if col != 'Image']
    df[columns_to_save].to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"✓ Saved {len(df)} rows to: {output_path}")
    print(f"  Columns: {', '.join(columns_to_save)}")
    print(f"  Note: 'Image' column excluded (binary data)")


def main():
    """
    Main function with command-line argument parsing.
    """
    parser = argparse.ArgumentParser(
        description='Select sectioned exhibits from timetravel.parquet',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with defaults
  python select_sectioned_exhibits.py
  
  # Custom output file
  python select_sectioned_exhibits.py -o my_exhibits.csv
  
  # With random seed for reproducibility
  python select_sectioned_exhibits.py --seed 42
  
  # Custom number of sections and exhibits
  python select_sectioned_exhibits.py --sections 3 --exhibits 25
        """
    )
    
    parser.add_argument(
        '-d', '--data',
        type=str,
        default='data/timetravel.parquet',
        help='Path to parquet data file (default: data/timetravel.parquet)'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='selected_exhibits.csv',
        help='Output CSV file path (default: selected_exhibits.csv)'
    )
    
    parser.add_argument(
        '--sections',
        type=int,
        default=5,
        help='Number of sections to select (default: 5)'
    )
    
    parser.add_argument(
        '--exhibits',
        type=int,
        default=20,
        help='Number of exhibits per section (default: 20)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility (default: None)'
    )
    
    args = parser.parse_args()
    
    print("\n🏛️  SECTIONED EXHIBIT SELECTOR")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  • Data file: {args.data}")
    print(f"  • Sections: {args.sections}")
    print(f"  • Exhibits per section: {args.exhibits}")
    print(f"  • Output: {args.output}")
    if args.seed:
        print(f"  • Random seed: {args.seed}")
    
    try:
        # Step 1: Load and filter data
        df_filtered = load_and_filter_data(args.data)
        
        # Step 2: Select top sections
        top_sections = select_top_sections(df_filtered, args.sections)
        
        # Step 3: Sample exhibits from each section
        sampled_df = sample_exhibits_by_section(
            df_filtered, 
            top_sections, 
            args.exhibits,
            args.seed
        )
        
        # Step 4: Save to CSV
        save_to_csv(sampled_df, args.output)
        
        print("\n" + "=" * 70)
        print("✨ SUCCESS!")
        print("=" * 70)
        print(f"Selected {len(sampled_df)} exhibits from {args.sections} sections")
        print(f"Output saved to: {args.output}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
