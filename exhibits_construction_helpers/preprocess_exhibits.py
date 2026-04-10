"""
Preprocess Exhibits CSV
Adds parsed columns for production dates and find spot locations.
Makes semantic validation faster and more reliable.
"""

import pandas as pd
import re
from pathlib import Path

# City to Country mapping dictionary
CITY_TO_COUNTRY = {
    # Italy (Etruscan/Roman sites)
    "Nola": "Italy",
    "Vulci": "Italy",
    "Etruria": "Italy",
    "Cerveteri": "Italy",
    "Tarquinia": "Italy",
    "Spina": "Italy",
    "Puglia": "Italy",
    
    # Egypt (Greek trading posts & ancient sites)
    "Naukratis": "Egypt",
    "Alexandria": "Egypt",
    "Memphis": "Egypt",
    "Egypt": "Egypt",
    "Tomb of Sety I": "Egypt",
    "Scarab Factory": "Egypt",
    
    # Greece (mainland & islands)
    "Athens": "Greece",
    "Corinth": "Greece",
    "Thebes": "Greece",
    "Olympia": "Greece",
    "Delphi": "Greece",
    "Rhodes": "Greece",
    "Samos": "Greece",
    "Fikellura": "Greece",  # Rhodes area
    "Fikellura grave 179": "Greece",
    "Kamiros": "Greece",  # Rhodes
    "Kalavarda": "Greece",  # Rhodes
    "Crete": "Greece",
    "Sanctuary of Aphrodite": "Cyprus",
    "Sanctuary of Apollo": "Greece",
    "Blacas Tomb": "Italy",
    
    # Cyprus
    "Marion-Arsinoe": "Cyprus",
    "Marion": "Cyprus",
    "Arsinoe": "Cyprus",
    "Salamis": "Cyprus",
    
    # Turkey (ancient Greek sites)
    "Ephesus": "Turkey",
    "Miletus": "Turkey",
    "Halicarnassus": "Turkey",
    
    # Libya (ancient Cyrenaica)
    "Cyrenaica": "Libya",
    "Cyrene": "Libya",
    
    # Iraq (ancient Mesopotamia)
    "Babylon": "Iraq",
    "Ur": "Iraq",
    "Uruk": "Iraq",
    "Nippur": "Iraq",
    "Nineveh": "Iraq",
    "Iraq": "Iraq",
    "Abu Habba": "Iraq",  # Sippar
    "Kouyunjik": "Iraq",  # Nineveh
    "Nimrud": "Iraq",
    "Arpachiyah": "Iraq",
    "North Palace": "Iraq",  # Nineveh
    "South East Palace": "Iraq",  # Nineveh
    "Royal Cemetery": "Iraq",  # Ur
    "Tell Arsh": "Iraq",
    "Warka": "Iraq",  # Uruk
    "Marduk Temple": "Iraq",  # Babylon
    "Temple of Ishtar": "Iraq",
    "Dar al-Khilafa": "Iraq",  # Samarra
    "Samarra": "Iraq",
    
    # Iran (ancient Persia)
    "Persepolis": "Iran",
    "Susa": "Iran",
    "Pasargadae": "Iran",
    
    # Syria
    "Palmyra": "Syria",
    "Damascus": "Syria",
    
    # China
    "Chang'an": "China",
    "Luoyang": "China",
    "Xi'an": "China",
    "China": "China",
    "Changsha": "China",
    "Shanxi": "China",
    
    # Japan
    "Nara": "Japan",
    "Kyoto": "Japan",
    "Kamakura": "Japan",
    "Akasaka-cho": "Japan",
    "Hattorigawa": "Japan",
    "Kamatari Ko": "Japan",
    "Kinshozan": "Japan",
    "Misaoyama Dolmen": "Japan",
    "Nijo-jo": "Japan",  # Kyoto
    "Okamachi": "Japan",
    "Omi-Hachiman": "Japan",
    "Ryoseki": "Japan",
    "Seto-shi": "Japan",
    "Shibayama dolmen": "Japan",
    "Tanba": "Japan",
    "Toyonaka-shi": "Japan",
    "Yamato": "Japan",
    "Domyoji": "Japan",
    "Yasui rock tomb": "Japan",
    
    # Mexico (Maya sites)
    "Tikal": "Guatemala",
    "Palenque": "Mexico",
    "Chichen Itza": "Mexico",
    "Copan": "Honduras",
}


def parse_production_date_to_year(date_str):
    """
    Parse production date string to numeric year(s) for comparison.
    Handles BC/AD, ranges, and circa notations.
    
    Args:
        date_str: Production date string (e.g., "500BC-490BC", "618-906", "307")
    
    Returns:
        tuple: (min_year, max_year, is_range, status)
               - min_year/max_year: integers (BC years are negative) or None
               - is_range: boolean (True if different years)
               - status: "success", "failed", or "partial"
    """
    if pd.isna(date_str):
        return (None, None, False, "failed")
    
    date_str = str(date_str).strip()
    original_str = date_str
    
    # Remove parenthetical notations like (circa), (about)
    date_str = re.sub(r'\s*\([^)]+\)', '', date_str)
    date_str = date_str.strip()
    
    # Check for range: "500BC-490BC" or "618-906"
    range_match = re.match(r'^(\d+)\s*(BC|AD|bc|ad)?\s*-\s*(\d+)\s*(BC|AD|bc|ad)?$', date_str, re.IGNORECASE)
    if range_match:
        start_num = int(range_match.group(1))
        start_bc = range_match.group(2) and range_match.group(2).upper() == 'BC'
        end_num = int(range_match.group(3))
        end_bc = range_match.group(4) and range_match.group(4).upper() == 'BC'
        
        start_year = -start_num if start_bc else start_num
        end_year = -end_num if end_bc else end_num
        
        min_year = min(start_year, end_year)
        max_year = max(start_year, end_year)
        
        return (min_year, max_year, True, "success")
    
    # Check for single year: "307", "500BC", "200 AD"
    single_match = re.match(r'^(\d+)\s*(BC|AD|bc|ad)?$', date_str, re.IGNORECASE)
    if single_match:
        year_num = int(single_match.group(1))
        is_bc = single_match.group(2) and single_match.group(2).upper() == 'BC'
        
        year = -year_num if is_bc else year_num
        return (year, year, False, "success")
    
    # Unable to parse
    return (None, None, False, "failed")


def extract_city_from_findspot(find_spot):
    """
    Extract city name from find spot string.
    
    Args:
        find_spot: Find spot string (e.g., "Excavated/Findspot: Vulci (said to be)")
    
    Returns:
        tuple: (city_name, status)
               - city_name: extracted city or None
               - status: "success" or "failed"
    """
    if pd.isna(find_spot):
        return (None, "failed")
    
    find_spot = str(find_spot).strip()
    
    # Pattern: "Excavated/Findspot: [CITY]" or "Excavated/Findspot: [CITY] (qualifier)"
    # Also handle "Found/Acquired: [CITY]"
    patterns = [
        r'(?:Excavated/Findspot|Found/Acquired|Findspot):\s*([^(,;]+)',
        r':\s*([A-Za-z\s\-]+)',  # Fallback: anything after colon
    ]
    
    for pattern in patterns:
        match = re.search(pattern, find_spot, re.IGNORECASE)
        if match:
            city = match.group(1).strip()
            # Clean up common suffixes
            city = re.sub(r'\s+\(.*\)$', '', city)  # Remove trailing parentheses
            city = re.sub(r'\s+(said to be|possibly|probably|modern|historic).*$', '', city, flags=re.IGNORECASE)
            city = city.strip()
            
            if city:
                return (city, "success")
    
    return (None, "failed")


def map_city_to_country(city):
    """
    Map city name to country using lookup dictionary.
    
    Args:
        city: City name
    
    Returns:
        tuple: (country_name, status)
               - country_name: mapped country or None
               - status: "success" or "unknown"
    """
    if pd.isna(city) or not city:
        return (None, "unknown")
    
    city = str(city).strip()
    
    # Direct lookup
    if city in CITY_TO_COUNTRY:
        return (CITY_TO_COUNTRY[city], "success")
    
    # Try case-insensitive lookup
    city_lower = city.lower()
    for known_city, country in CITY_TO_COUNTRY.items():
        if known_city.lower() == city_lower:
            return (country, "success")
    
    # Try partial match (city name contains known city)
    for known_city, country in CITY_TO_COUNTRY.items():
        if known_city.lower() in city_lower or city_lower in known_city.lower():
            return (country, "success")
    
    # Unknown city
    return (None, "unknown")


def preprocess_exhibits(input_csv, output_csv):
    """
    Preprocess exhibits CSV by adding parsed date and location columns.
    
    Args:
        input_csv: Path to input CSV file
        output_csv: Path to output CSV file
    
    Returns:
        dict: Statistics about preprocessing
    """
    print("="*70)
    print("Exhibit Preprocessing")
    print("="*70)
    
    # Load CSV
    print(f"\n📂 Loading: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"✓ Loaded {len(df)} exhibits")
    
    # Initialize new columns
    df['production_year_min'] = None
    df['production_year_max'] = None
    df['is_date_range'] = False
    df['date_parse_status'] = 'unknown'
    df['find_spot_city'] = None
    df['find_spot_country'] = None
    df['location_parse_status'] = 'unknown'
    
    # Statistics
    stats = {
        'total': len(df),
        'date_success': 0,
        'date_failed': 0,
        'location_success': 0,
        'location_unknown': 0,
        'location_failed': 0,
        'failed_items': [],
        'unknown_cities': set()
    }
    
    # Process each row
    print("\n🔄 Processing exhibits...")
    for idx, row in df.iterrows():
        # Parse production date
        min_year, max_year, is_range, date_status = parse_production_date_to_year(row['Production date'])
        df.at[idx, 'production_year_min'] = min_year
        df.at[idx, 'production_year_max'] = max_year
        df.at[idx, 'is_date_range'] = is_range
        df.at[idx, 'date_parse_status'] = date_status
        
        if date_status == 'success':
            stats['date_success'] += 1
        else:
            stats['date_failed'] += 1
            stats['failed_items'].append({
                'id': row['id'],
                'reason': f"Date parsing failed: {row['Production date']}"
            })
        
        # Extract and map location
        city, city_status = extract_city_from_findspot(row['Find spot'])
        df.at[idx, 'find_spot_city'] = city
        
        if city_status == 'success':
            country, country_status = map_city_to_country(city)
            df.at[idx, 'find_spot_country'] = country
            df.at[idx, 'location_parse_status'] = country_status
            
            if country_status == 'success':
                stats['location_success'] += 1
            else:
                stats['location_unknown'] += 1
                stats['unknown_cities'].add(city)
        else:
            df.at[idx, 'location_parse_status'] = 'failed'
            stats['location_failed'] += 1
            stats['failed_items'].append({
                'id': row['id'],
                'reason': f"Location parsing failed: {row['Find spot']}"
            })
    
    # Save preprocessed CSV
    print(f"\n💾 Saving: {output_csv}")
    df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"✓ Saved {len(df)} exhibits with {7} new columns")
    
    # Print statistics
    print("\n" + "="*70)
    print("Preprocessing Statistics")
    print("="*70)
    print(f"\n📊 Total Exhibits: {stats['total']}")
    
    print(f"\n📅 Production Date Parsing:")
    print(f"  ✓ Success: {stats['date_success']} ({stats['date_success']/stats['total']*100:.1f}%)")
    print(f"  ✗ Failed:  {stats['date_failed']} ({stats['date_failed']/stats['total']*100:.1f}%)")
    
    print(f"\n🌍 Location Mapping:")
    print(f"  ✓ Success: {stats['location_success']} ({stats['location_success']/stats['total']*100:.1f}%)")
    print(f"  ? Unknown: {stats['location_unknown']} ({stats['location_unknown']/stats['total']*100:.1f}%)")
    print(f"  ✗ Failed:  {stats['location_failed']} ({stats['location_failed']/stats['total']*100:.1f}%)")
    
    if stats['unknown_cities']:
        print(f"\n⚠️  Unknown Cities ({len(stats['unknown_cities'])}):")
        for city in sorted(stats['unknown_cities']):
            print(f"     - {city}")
        print("\n  → Add these to CITY_TO_COUNTRY dictionary for better mapping")
    
    # Save report
    report_path = Path(output_csv).parent / 'preprocessing_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("Exhibit Preprocessing Report\n")
        f.write("="*70 + "\n\n")
        f.write(f"Total Exhibits: {stats['total']}\n\n")
        f.write(f"Date Parsing:\n")
        f.write(f"  Success: {stats['date_success']} ({stats['date_success']/stats['total']*100:.1f}%)\n")
        f.write(f"  Failed:  {stats['date_failed']}\n\n")
        f.write(f"Location Mapping:\n")
        f.write(f"  Success: {stats['location_success']} ({stats['location_success']/stats['total']*100:.1f}%)\n")
        f.write(f"  Unknown: {stats['location_unknown']}\n")
        f.write(f"  Failed:  {stats['location_failed']}\n\n")
        
        if stats['unknown_cities']:
            f.write(f"Unknown Cities:\n")
            for city in sorted(stats['unknown_cities']):
                f.write(f"  - {city}\n")
        
        if stats['failed_items']:
            f.write(f"\nItems Needing Manual Review:\n")
            for item in stats['failed_items']:
                f.write(f"  ID {item['id']}: {item['reason']}\n")
    
    print(f"\n📄 Report saved to: {report_path}")
    
    # Save manual review CSV if needed
    if stats['failed_items'] or stats['unknown_cities']:
        failed_ids = [item['id'] for item in stats['failed_items']]
        unknown_city_rows = df[df['location_parse_status'] == 'unknown']
        review_df = df[(df['id'].isin(failed_ids)) | (df['id'].isin(unknown_city_rows['id']))]
        
        review_path = Path(output_csv).parent / 'manual_review_needed.csv'
        review_df.to_csv(review_path, index=False, encoding='utf-8')
        print(f"📄 Manual review CSV: {review_path} ({len(review_df)} items)")
    
    print("\n" + "="*70)
    print("✨ Preprocessing Complete!")
    print("="*70)
    
    return stats


def main():
    """
    Main function with command-line argument parsing.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Preprocess exhibits CSV by adding parsed date and location columns',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python preprocess_exhibits.py -i ../selected_exhibits.csv -o ../selected_exhibits_preprocessed.csv
  
  # From exhibits_construction_helpers directory
  python preprocess_exhibits.py
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        type=str,
        default='../selected_exhibits.csv',
        help='Input CSV file (default: ../selected_exhibits.csv)'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='../selected_exhibits_preprocessed.csv',
        help='Output CSV file (default: ../selected_exhibits_preprocessed.csv)'
    )
    
    args = parser.parse_args()
    
    # Run preprocessing
    stats = preprocess_exhibits(args.input, args.output)
    
    return 0 if stats else 1


if __name__ == '__main__':
    exit(main())
