"""
Annotation Setup Pipeline
Automates the VIA annotation conversion and annotated image generation workflow.

This script chains together two steps:
1. Convert VIA project JSON to museum layout format
2. Generate annotated layout image (with conditional forbidden_area)

Usage:
    python setup_annotation_pipeline.py -c simple -l layout_01
    python setup_annotation_pipeline.py --complexity complex --layout layout_03
    python setup_annotation_pipeline.py -c simple -l layout_01 -t entrance exit wall
"""

import argparse
import subprocess
import sys
import json
from pathlib import Path
from glob import glob


class AnnotationPipelineRunner:
    """Manages the annotation pipeline execution."""
    
    def __init__(self, complexity, layout, custom_types=None):
        self.complexity = complexity
        self.layout = layout
        self.layout_dir = Path(f"floorplans/{complexity}/{layout}")
        self.custom_types = custom_types
        self.errors = []
        self.generated_files = []
        self.via_json_path = None
        self.layout_image = None
        self.annotations_json = None
        self.forbidden_areas_count = 0
    
    def validate_prerequisites(self):
        """Validate all required input files exist."""
        print("=" * 70)
        print("🔍 Validating Prerequisites...")
        print("=" * 70)
        
        all_valid = True
        
        # Find VIA project JSON file
        via_json_pattern = str(self.layout_dir / 'via_project_*.json')
        via_json_files = glob(via_json_pattern)
        
        if len(via_json_files) == 0:
            print(f"❌ Missing: VIA project JSON file")
            print(f"   Expected pattern: {via_json_pattern}")
            self.errors.append(f"No VIA project JSON found in {self.layout_dir}")
            all_valid = False
        elif len(via_json_files) > 1:
            print(f"❌ Error: Multiple VIA project JSON files found")
            print(f"   Found {len(via_json_files)} files:")
            for f in via_json_files:
                print(f"   • {Path(f).name}")
            print(f"   Expected: Exactly 1 VIA project JSON file")
            self.errors.append(f"Multiple VIA JSON files found in {self.layout_dir}")
            all_valid = False
        else:
            self.via_json_path = Path(via_json_files[0])
            print(f"✅ Found: VIA project JSON ({self.via_json_path.name})")
        
        # Auto-detect layout image
        layout_image = self._find_layout_image()
        if layout_image:
            print(f"✅ Found: layout image ({layout_image.name})")
            self.layout_image = layout_image
        else:
            print(f"❌ Missing: layout image (.jpg or .png)")
            print(f"   Expected at: {self.layout_dir}/{self.layout}.jpg or .png")
            self.errors.append(f"No layout image found in {self.layout_dir}")
            all_valid = False
        
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
    
    def run_step(self, step_number, step_name, command):
        """Execute a pipeline step with error handling."""
        print("=" * 70)
        print(f"📋 Step {step_number}: {step_name}")
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
            print(f"✅ Step {step_number} completed successfully!")
            return True
        
        except subprocess.CalledProcessError as e:
            print()
            print(f"❌ Step {step_number} failed with exit code {e.returncode}")
            self.errors.append(f"Step {step_number} ({step_name}) failed")
            return False
        
        except Exception as e:
            print()
            print(f"❌ Step {step_number} failed with error: {e}")
            self.errors.append(f"Step {step_number} ({step_name}) error: {e}")
            return False
    
    def step1_convert_via_to_museum_layout(self):
        """Step 1: Convert VIA project JSON to museum layout format."""
        self.annotations_json = self.layout_dir / 'layout_annotations.json'
        
        command = [
            'python',
            'via_musuem_helpers/convert_via_to_museum_layout.py',
            str(self.via_json_path),
            str(self.annotations_json)
        ]
        
        success = self.run_step(1, "Convert VIA to Museum Layout", command)
        
        if success:
            self.generated_files.append(self.annotations_json)
            # Detect forbidden areas
            self._detect_forbidden_areas()
        
        return success
    
    def _detect_forbidden_areas(self):
        """Read layout_annotations.json and detect forbidden areas."""
        try:
            with open(self.annotations_json, 'r') as f:
                data = json.load(f)
            
            counts = data.get('metadata', {}).get('counts', {})
            self.forbidden_areas_count = counts.get('forbidden_areas', 0)
            
            print()
            print("=" * 70)
            print("🔍 Forbidden Area Detection")
            print("=" * 70)
            if self.forbidden_areas_count > 0:
                print(f"✅ Detected {self.forbidden_areas_count} forbidden area(s)")
                print("   → Will include 'forbidden_area' in annotated image")
            else:
                print("ℹ️  No forbidden areas detected")
                print("   → Will exclude 'forbidden_area' from annotated image")
            print()
        
        except Exception as e:
            print(f"⚠️  Warning: Could not read forbidden areas from {self.annotations_json}: {e}")
            self.forbidden_areas_count = 0
    
    def step2_generate_annotated_image(self):
        """Step 2: Generate annotated layout image."""
        # Determine annotation types
        if self.custom_types:
            # User provided custom types
            types = self.custom_types
            print(f"Using custom annotation types: {', '.join(types)}")
        else:
            # Auto-determine based on forbidden areas
            base_types = ['entrance', 'exit', 'wall', 'gallery', 'exhibit']
            if self.forbidden_areas_count > 0:
                types = base_types + ['forbidden_area']
            else:
                types = base_types
        
        # Build command
        command = [
            'python',
            'via_musuem_helpers/generate_custom_annotations.py',
            str(self.layout_image),
            str(self.annotations_json),
            '-t'
        ] + types + [
            '--labels', '1',
            '-o', str(self.layout_dir / 'annotated_layout.png')
        ]
        
        success = self.run_step(2, "Generate Annotated Image", command)
        
        if success:
            self.generated_files.append(self.layout_dir / 'annotated_layout.png')
        
        return success
    
    def print_summary(self):
        """Print pipeline execution summary."""
        print("\n" + "=" * 70)
        print("📊 PIPELINE SUMMARY")
        print("=" * 70)
        
        if not self.errors:
            print("✨ All steps completed successfully!")
            print()
            print(f"📁 Layout Directory: {self.layout_dir}")
            print()
            print(f"📄 Generated Files ({len(self.generated_files)}):")
            for file in self.generated_files:
                if file.exists():
                    print(f"   ✅ {file.name}")
                else:
                    print(f"   ⚠️  {file.name} (expected but not found)")
            
            print()
            print(f"📊 Forbidden Areas: {self.forbidden_areas_count}")
            if self.forbidden_areas_count > 0:
                print("   → Annotated image includes forbidden_area annotations")
            else:
                print("   → Annotated image excludes forbidden_area annotations")
        else:
            print("❌ Pipeline completed with errors:")
            for error in self.errors:
                print(f"   • {error}")
            print()
            print("Please fix the errors and run the pipeline again.")
        
        print("=" * 70)
    
    def run(self):
        """Execute the complete pipeline."""
        print("\n" + "=" * 70)
        print("🏛️  ANNOTATION SETUP PIPELINE")
        print("=" * 70)
        print(f"Complexity: {self.complexity}")
        print(f"Layout: {self.layout}")
        print(f"Directory: {self.layout_dir}")
        print()
        
        # Step 0: Validate prerequisites
        if not self.validate_prerequisites():
            self.print_summary()
            return False
        
        # Step 1: Convert VIA to museum layout
        if not self.step1_convert_via_to_museum_layout():
            self.print_summary()
            return False
        
        # Step 2: Generate annotated image
        if not self.step2_generate_annotated_image():
            self.print_summary()
            return False
        
        # Print summary
        self.print_summary()
        return True


def main():
    parser = argparse.ArgumentParser(
        description='Automate the VIA annotation conversion and image generation workflow',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python setup_annotation_pipeline.py -c simple -l layout_01
  python setup_annotation_pipeline.py -c simple -l layout_02
  python setup_annotation_pipeline.py -c complex -l layout_03
  
  # Override annotation types
  python setup_annotation_pipeline.py -c simple -l layout_01 \\
    -t entrance exit wall gallery exhibit
  
Pipeline Steps:
  1. Convert VIA project JSON to museum layout format
  2. Generate annotated layout image (conditional forbidden_area)
  
Generated Files:
  • layout_annotations.json
  • annotated_layout.png
  
Forbidden Area Detection:
  - Automatically detects forbidden areas in layout_annotations.json
  - If forbidden_areas > 0: Includes 'forbidden_area' in annotated image
  - If forbidden_areas = 0: Excludes 'forbidden_area' from annotated image
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
    
    parser.add_argument(
        '-t', '--types',
        type=str,
        nargs='+',
        help='Override annotation types (default: auto-detect based on forbidden_areas)'
    )
    
    args = parser.parse_args()
    
    # Run pipeline
    runner = AnnotationPipelineRunner(args.complexity, args.layout, args.types)
    success = runner.run()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
