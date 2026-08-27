"""
Complete Layout Setup Pipeline
Master pipeline that combines annotation and exhibit setup workflows.

This script runs both pipelines sequentially:
1. Annotation Pipeline (2 steps) - Convert VIA and generate annotated image
2. Exhibit Pipeline (3 steps) - Generate exhibits, visualize, create configs

Usage:
    python setup_complete_pipeline.py -c simple -l layout_02
    python setup_complete_pipeline.py --complexity complex --layout layout_03
"""

import argparse
import subprocess
import sys
from pathlib import Path
from glob import glob


class CompletePipelineRunner:
    """Manages the complete layout setup pipeline execution."""
    
    def __init__(self, complexity, layout):
        self.complexity = complexity
        self.layout = layout
        self.layout_dir = Path(f"floorplans/{complexity}/{layout}")
        self.errors = []
    
    def validate_all_prerequisites(self):
        """Validate all required files for both pipelines."""
        print("=" * 70)
        print("🔍 Validating All Prerequisites...")
        print("=" * 70)
        
        all_valid = True
        
        # Check for VIA project JSON (exactly one)
        via_json_pattern = str(self.layout_dir / 'via_project_*.json')
        via_json_files = glob(via_json_pattern)
        
        if len(via_json_files) == 0:
            print(f"❌ Missing: VIA project JSON file")
            print(f"   Expected pattern: {via_json_pattern}")
            all_valid = False
        elif len(via_json_files) > 1:
            print(f"❌ Error: Multiple VIA project JSON files found ({len(via_json_files)})")
            for f in via_json_files:
                print(f"   • {Path(f).name}")
            all_valid = False
        else:
            print(f"✅ Found: VIA project JSON ({Path(via_json_files[0]).name})")
        
        # Check for layout image
        layout_image = self._find_layout_image()
        if layout_image:
            print(f"✅ Found: layout image ({layout_image.name})")
        else:
            print(f"❌ Missing: layout image ({self.layout}.jpg or .png)")
            all_valid = False
        
        # Check for exhibit_input.txt
        exhibit_input = self.layout_dir / 'exhibit_input.txt'
        if exhibit_input.exists():
            print(f"✅ Found: exhibit_input.txt")
        else:
            print(f"❌ Missing: exhibit_input.txt")
            print(f"   This file defines exhibit number assignments to sections")
            print(f"   Format: 'number-range: Section' (e.g., '1-10: Greek')")
            all_valid = False
        
        # Check for global database
        db_path = Path('exhibits_construction_helpers/data/selected_exhibits_preprocessed.csv')
        if db_path.exists():
            print(f"✅ Found: selected_exhibits_preprocessed.csv")
        else:
            print(f"❌ Missing: selected_exhibits_preprocessed.csv")
            all_valid = False
        
        print()
        
        if not all_valid:
            print("❌ Prerequisite validation failed!")
            print("   Please ensure all required files exist before running.")
        else:
            print("✅ All prerequisites validated!")
        
        print()
        return all_valid
    
    def _find_layout_image(self):
        """Auto-detect layout image in .jpg or .png format."""
        jpg_path = self.layout_dir / f"{self.layout}.jpg"
        png_path = self.layout_dir / f"{self.layout}.png"
        
        if jpg_path.exists():
            return jpg_path
        elif png_path.exists():
            return png_path
        else:
            return None
    
    def run_pipeline(self, pipeline_name, command):
        """Execute a sub-pipeline with error handling."""
        print("=" * 70)
        print(f"🚀 Running {pipeline_name}")
        print("=" * 70)
        print(f"Command: {' '.join(command)}")
        print()
        
        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=False,
                text=True
            )
            print()
            print(f"✅ {pipeline_name} completed successfully!")
            print()
            return True
        
        except subprocess.CalledProcessError as e:
            print()
            print(f"❌ {pipeline_name} failed with exit code {e.returncode}")
            self.errors.append(f"{pipeline_name} failed")
            return False
        
        except Exception as e:
            print()
            print(f"❌ {pipeline_name} failed with error: {e}")
            self.errors.append(f"{pipeline_name} error: {e}")
            return False
    
    def run_annotation_pipeline(self):
        """Run the annotation setup pipeline."""
        command = [
            'python',
            'setup_annotation_pipeline.py',
            '-c', self.complexity,
            '-l', self.layout
        ]
        
        return self.run_pipeline("Annotation Pipeline", command)
    
    def run_exhibit_pipeline(self):
        """Run the exhibit setup pipeline."""
        command = [
            'python',
            'setup_layout_pipeline.py',
            '-c', self.complexity,
            '-l', self.layout
        ]
        
        return self.run_pipeline("Exhibit Pipeline", command)
    
    def print_final_summary(self):
        """Print final summary of complete pipeline."""
        print("=" * 70)
        print("📊 COMPLETE PIPELINE SUMMARY")
        print("=" * 70)
        
        if not self.errors:
            print("✨ Complete layout setup finished successfully!")
            print()
            print(f"📁 Layout Directory: {self.layout_dir}")
            print()
            print("📄 All Generated Files (29 total):")
            print()
            print("   Annotation Files:")
            
            annotation_files = [
                'layout_annotations.json',
                'annotated_layout.png'
            ]
            for file in annotation_files:
                filepath = self.layout_dir / file
                status = "✅" if filepath.exists() else "⚠️"
                print(f"   {status} {file}")
            
            print()
            print("   Exhibit Files:")
            
            exhibit_files = [
                'exhibits.csv',
                'exhibit_list.json',
                'exhibit_visualization.png'
            ]
            for file in exhibit_files:
                filepath = self.layout_dir / file
                status = "✅" if filepath.exists() else "⚠️"
                print(f"   {status} {file}")
            
            print()
            print("   Config Files:")
            
            # Check static easy config files
            easy_configs = ['easy_spatial.json', 'easy_semantic.json']
            for file in easy_configs:
                filepath = self.layout_dir / file
                status = "✅" if filepath.exists() else "⚠️"
                print(f"   {status} {file}")
            
            # Dynamically discover numbered medium and hard config files
            medium_configs = sorted(glob(str(self.layout_dir / 'medium_*.json')))
            hard_configs = sorted(glob(str(self.layout_dir / 'hard_*.json')))
            
            # Display discovered medium configs
            for config_path in medium_configs:
                config_file = Path(config_path)
                status = "✅" if config_file.exists() else "⚠️"
                print(f"   {status} {config_file.name}")
            
            # Display discovered hard configs
            for config_path in hard_configs:
                config_file = Path(config_path)
                status = "✅" if config_file.exists() else "⚠️"
                print(f"   {status} {config_file.name}")
            
            print()
            print("🎉 Layout is now ready for route validation!")
            print()
            print("Next steps:")
            print("  1. Review exhibit_visualization.png for exhibit distribution")
            print("  2. Review annotated_layout.png for annotation accuracy")
            print("  3. Use config files with validation_pipeline.py")
        else:
            print("❌ Complete pipeline failed with errors:")
            for error in self.errors:
                print(f"   • {error}")
            print()
            print("Please fix the errors and run the complete pipeline again.")
        
        print("=" * 70)
    
    def run(self):
        """Execute the complete pipeline."""
        print("\n" + "=" * 70)
        print("🏛️  COMPLETE LAYOUT SETUP PIPELINE")
        print("=" * 70)
        print(f"Complexity: {self.complexity}")
        print(f"Layout: {self.layout}")
        print(f"Directory: {self.layout_dir}")
        print()
        print("This will run:")
        print("  • Annotation Pipeline (2 steps)")
        print("  • Exhibit Pipeline (3 steps)")
        print("  • Total: 5 automated steps")
        print()
        
        # Validate all prerequisites upfront
        if not self.validate_all_prerequisites():
            self.print_final_summary()
            return False
        
        # Run annotation pipeline
        if not self.run_annotation_pipeline():
            print("\n⚠️  Annotation pipeline failed. Stopping complete pipeline.")
            self.print_final_summary()
            return False
        
        # Run exhibit pipeline
        if not self.run_exhibit_pipeline():
            print("\n⚠️  Exhibit pipeline failed. Stopping complete pipeline.")
            self.print_final_summary()
            return False
        
        # Print final summary
        self.print_final_summary()
        return True


def main():
    parser = argparse.ArgumentParser(
        description='Complete layout setup: annotations + exhibits + configs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python setup_complete_pipeline.py -c simple -l layout_01
  python setup_complete_pipeline.py -c simple -l layout_02
  python setup_complete_pipeline.py -c complex -l layout_03
  
Complete Pipeline Steps:
  Annotation Pipeline:
    1. Convert VIA project JSON to museum layout format
    2. Generate annotated layout image (conditional forbidden_area)
  
  Exhibit Pipeline:
    3. Generate exhibit list from exhibit_input.txt
    4. Visualize exhibits by section
    5. Generate difficulty-specific config files

Required Input Files:
  • via_project_*.json (exactly 1)
  • {layout}.jpg or .png
  • exhibit_input.txt
  • selected_exhibits_preprocessed.csv (global)

Generated Files (29 total):
  Annotations:
    • layout_annotations.json
    • annotated_layout.png

  Exhibits:
    • exhibits.csv
    • exhibit_list.json
    • exhibit_visualization.png

  Configs:
    • easy_spatial.json
    • easy_semantic.json
    • medium_01.json ... medium_11.json
    • hard_01.json ... hard_11.json
        """
    )
    
    parser.add_argument(
        '-c', '--complexity',
        type=str,
        required=True,
        help='Complexity level (e.g., "simple", "complex")'
    )
    
    parser.add_argument(
        '-l', '--layout',
        type=str,
        required=True,
        help='Layout name (e.g., "layout_01", "layout_02")'
    )
    
    args = parser.parse_args()
    
    # Run complete pipeline
    runner = CompletePipelineRunner(args.complexity, args.layout)
    success = runner.run()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
