"""
Layout Setup Pipeline
Automates the complete exhibit setup workflow for a museum layout.

This script chains together three steps:
1. Generate exhibit list from input file
2. Visualize exhibits by section
3. Generate difficulty-specific config files

Usage:
    python setup_layout_pipeline.py -c simple -l layout_01
    python setup_layout_pipeline.py --complexity complex --layout layout_03
"""

import argparse
import subprocess
import sys
from pathlib import Path


class PipelineRunner:
    """Manages the exhibit setup pipeline execution."""
    
    def __init__(self, complexity, layout):
        self.complexity = complexity
        self.layout = layout
        self.layout_dir = Path(f"floorplans/{complexity}/{layout}")
        self.errors = []
        self.generated_files = []
    
    def validate_prerequisites(self):
        """Validate all required input files exist."""
        print("=" * 70)
        print("🔍 Validating Prerequisites...")
        print("=" * 70)
        
        required_files = {
            'exhibit_input.txt': self.layout_dir / 'exhibit_input.txt',
            'layout_annotations.json': self.layout_dir / 'layout_annotations.json',
            'selected_exhibits_preprocessed.csv': Path('exhibits_construction_helpers/data/selected_exhibits_preprocessed.csv')
        }
        
        all_valid = True
        
        # Check required files
        for name, path in required_files.items():
            if path.exists():
                print(f"✅ Found: {name}")
            else:
                print(f"❌ Missing: {name}")
                print(f"   Expected at: {path}")
                self.errors.append(f"Missing required file: {path}")
                all_valid = False
        
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
                capture_output=False,  # Show output in real-time
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
    
    def step1_generate_exhibit_list(self):
        """Step 1: Generate exhibit list from input file."""
        command = [
            'python',
            'exhibits_construction_helpers/generate_exhibit_list.py',
            '-i', str(self.layout_dir / 'exhibit_input.txt'),
            '-c', 'exhibits_construction_helpers/data/selected_exhibits_preprocessed.csv',
            '-o', str(self.layout_dir / 'exhibits.csv'),
            '-j', str(self.layout_dir)
        ]
        
        success = self.run_step(1, "Generate Exhibit List", command)
        
        if success:
            self.generated_files.extend([
                self.layout_dir / 'exhibits.csv',
                self.layout_dir / 'exhibit_list.json'
            ])
        
        return success
    
    def step2_visualize_exhibits(self):
        """Step 2: Visualize exhibits by section."""
        command = [
            'python',
            'exhibits_construction_helpers/visualize_exhibits_by_section.py',
            '-f', str(self.layout_image),
            '-a', str(self.layout_dir / 'layout_annotations.json'),
            '-c', str(self.layout_dir / 'exhibits.csv'),
            '-o', str(self.layout_dir / 'exhibit_visualization.png')
        ]
        
        success = self.run_step(2, "Visualize Exhibits by Section", command)
        
        if success:
            self.generated_files.append(self.layout_dir / 'exhibit_visualization.png')
        
        return success
    
    def step3_generate_config_files(self):
        """Step 3: Generate difficulty-specific config files."""
        command = [
            'python',
            'exhibits_construction_helpers/generate_config_files.py',
            '-l', str(self.layout_dir),
            '-c', 'exhibits.csv'
        ]
        
        success = self.run_step(3, "Generate Config Files", command)
        
        if success:
            self.generated_files.extend([
                self.layout_dir / 'easy_spatial.json',
                self.layout_dir / 'easy_semantic.json',
                self.layout_dir / 'medium.json',
                self.layout_dir / 'hard.json'
            ])
        
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
        print("🏛️  LAYOUT SETUP PIPELINE")
        print("=" * 70)
        print(f"Complexity: {self.complexity}")
        print(f"Layout: {self.layout}")
        print(f"Directory: {self.layout_dir}")
        print()
        
        # Step 0: Validate prerequisites
        if not self.validate_prerequisites():
            self.print_summary()
            return False
        
        # Step 1: Generate exhibit list
        if not self.step1_generate_exhibit_list():
            self.print_summary()
            return False
        
        # Step 2: Visualize exhibits
        if not self.step2_visualize_exhibits():
            self.print_summary()
            return False
        
        # Step 3: Generate config files
        if not self.step3_generate_config_files():
            self.print_summary()
            return False
        
        # Print summary
        self.print_summary()
        return True


def main():
    parser = argparse.ArgumentParser(
        description='Automate the complete exhibit setup workflow for a museum layout',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python setup_layout_pipeline.py -c simple -l layout_01
  python setup_layout_pipeline.py -c simple -l layout_02
  python setup_layout_pipeline.py -c complex -l layout_03
  
Pipeline Steps:
  1. Generate exhibit list from exhibit_input.txt
  2. Visualize exhibits by section on layout image
  3. Generate difficulty-specific config files
  
Generated Files:
  • exhibits.csv
  • exhibit_list.json
  • exhibit_visualization.png
  • easy_spatial.json
  • easy_semantic.json
  • medium.json
  • hard.json
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
    
    # Run pipeline
    runner = PipelineRunner(args.complexity, args.layout)
    success = runner.run()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
