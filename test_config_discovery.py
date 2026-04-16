"""Test script to verify config file discovery works correctly."""

from pathlib import Path
from prompts_utils import discover_config_files

# Test layout
layout_folder = Path("floorplans/complex/layout_03")

print("="*70)
print("Testing Config Discovery")
print("="*70)

# Test 1: Discover medium_*.json files
print("\n1. Testing medium_*.json pattern:")
medium_configs = discover_config_files(layout_folder, "medium_*.json")
print(f"   Found {len(medium_configs)} config(s):")
for variant, path in medium_configs:
    print(f"   - {variant}: {path}")

# Test 2: Discover hard_*.json files
print("\n2. Testing hard_*.json pattern:")
hard_configs = discover_config_files(layout_folder, "hard_*.json")
print(f"   Found {len(hard_configs)} config(s):")
for variant, path in hard_configs:
    print(f"   - {variant}: {path}")

# Test 3: Single file (backward compatibility)
print("\n3. Testing medium.json (single file):")
single_config = discover_config_files(layout_folder, "medium.json")
print(f"   Found {len(single_config)} config(s):")
for variant, path in single_config:
    print(f"   - {variant}: {path}")

print("\n" + "="*70)
print("Test complete!")
print("="*70)
