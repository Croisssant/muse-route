
import asyncio
import argparse
import logging
import sys
import io
from pathlib import Path

# -------- Force UTF-8 encoding for stdout on Windows --------
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


def parse_arguments():
    """Parse command-line arguments for the master script."""
    parser = argparse.ArgumentParser(
        description='Master Script - Run all prompt scripts (Easy Semantic, Easy Spatial, Medium, Hard) concurrently',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with model and reasoning
  python run_all_prompts.py --model gpt-4 --reasoning extended

  # Run without reasoning
  python run_all_prompts.py --model gpt-5.4

  # Process specific complexities
  python run_all_prompts.py --model gpt-4o --complexity-mode list --complexity simple complex

  # Process specific layouts
  python run_all_prompts.py --model gpt-4 --layout-mode list --layouts layout_01 layout_02
        """
    )
    
    # Model configuration
    parser.add_argument('--model', type=str, required=True,
                       help='Model name to use (e.g., gpt-4, gpt-5.4, gpt-4o)')
    
    parser.add_argument('--reasoning', type=str, default=None,
                       help='Reasoning mode (e.g., extended, standard, or None/omit for no reasoning)')
    
    # Complexity selection
    parser.add_argument('--complexity-mode', type=str, 
                       choices=['all', 'single', 'list'],
                       default='all',
                       help='Complexity selection mode (default: all)')
    
    parser.add_argument('--complexity', nargs='+', type=str,
                       default=['simple'],
                       help='Specific complexity level(s) when not using "all" mode (default: simple)')
    
    # Layout selection
    parser.add_argument('--layout-mode', type=str,
                       choices=['all', 'single', 'list'],
                       default='all',
                       help='Layout selection mode (default: all)')
    
    parser.add_argument('--layouts', nargs='+', type=str,
                       default=None,
                       help='Specific layout(s) when using single or list mode')
    
    # Directory paths
    parser.add_argument('--floorplan-dir', type=str, default='floorplans',
                       help='Base directory for floorplans (default: floorplans)')
    
    parser.add_argument('--results-dir', type=str, default='results',
                       help='Base directory for results (default: results)')
    
    return parser.parse_args()


def build_output_folder_name(model, reasoning):
    """Build output folder name based on model and reasoning."""
    if reasoning:
        return f"{model}({reasoning})"
    return model


def setup_master_logger(log_file_path):
    """Setup the master execution logger (file only, no console to avoid tqdm interference)."""
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger('master_script')
    logger.setLevel(logging.INFO)
    logger.handlers = []
    
    # File handler only - console output handled by tqdm
    file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                                 datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    
    return logger


async def run_script(script_name, script_label, model, reasoning, args, logger, position=0):
    """Run a single prompt script asynchronously, streaming output to console.
    
    Args:
        script_name: Name of the Python script to run
        script_label: Label for display (e.g., "Semantic", "Spatial")
        model: Model name
        reasoning: Reasoning mode
        args: Parsed arguments from main script
        logger: Master logger instance
        position: Progress bar position for tqdm (default: 0)
    
    Returns:
        tuple: (script_name, success, error_message)
    """
    logger.info(f"[{script_label}] Starting {script_name}")
    
    # Build command
    cmd = [
        sys.executable,  # Use same Python interpreter
        '-u',  # Unbuffered output for real-time progress updates
        script_name,
        '--model', model,
        '--complexity-mode', args.complexity_mode,
        '--layout-mode', args.layout_mode,
        '--floorplan-dir', args.floorplan_dir,
        '--results-dir', args.results_dir,
    ]
    
    # Add reasoning if provided
    if reasoning:
        cmd.extend(['--reasoning', reasoning])
    
    # Add complexity arguments
    if args.complexity_mode != 'all':
        cmd.append('--complexity')
        cmd.extend(args.complexity)
    
    # Add layout arguments
    if args.layouts:
        cmd.append('--layouts')
        cmd.extend(args.layouts)
    
    # Add progress position for concurrent tqdm display
    cmd.extend(['--progress-position', str(position)])
    
    logger.info(f"[{script_label}] Command: {' '.join(cmd)}")
    
    try:
        # Create subprocess - tqdm writes to stderr, so we don't capture it
        # stdout is for regular print statements, stderr for progress bars
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,  # Capture stdout for status messages
            stderr=None  # Don't capture stderr - let tqdm write directly
        )
        
        # Read stdout while process runs to avoid blocking
        stdout_lines = []
        while True:
            line = await process.stdout.readline()
            if not line:
                break
            line_text = line.decode('utf-8', errors='replace').strip()
            if line_text:
                stdout_lines.append(line_text)
        
        # Wait for process to finish
        await process.wait()
        
        if process.returncode == 0:
            logger.info(f"[{script_label}] ✓ Completed successfully")
            return (script_name, True, None)
        else:
            error_msg = f"Process exited with code {process.returncode}"
            logger.error(f"[{script_label}] ✗ Failed: {error_msg}")
            if stdout_lines:
                logger.error(f"[{script_label}] Output:\n" + "\n".join(stdout_lines))
            return (script_name, False, error_msg)
            
    except Exception as e:
        error_msg = str(e)
        logger.error(f"[{script_label}] ✗ Exception: {error_msg}")
        import traceback
        logger.error(f"[{script_label}] Traceback:\n{traceback.format_exc()}")
        return (script_name, False, error_msg)


async def run_all_scripts(args, logger):
    """Run all prompt scripts concurrently, streaming their native progress bars."""
    
    # Define scripts to run
    scripts = [
        ('prompts_easy_semantic.py', 'Semantic'),
        ('prompts_easy_spatial.py', 'Spatial'),
        ('prompts_medium.py', 'Medium'),
        ('prompts_hard.py', 'Hard')
    ]
    
    logger.info("="*70)
    logger.info("Starting concurrent execution of all prompt scripts")
    logger.info("="*70)
    
    # Launch all scripts concurrently with positioned progress bars
    tasks = [
        run_script(script_name, label, args.model, args.reasoning, args, logger, position=idx)
        for idx, (script_name, label) in enumerate(scripts)
    ]
    
    # Wait for all to complete (even if some fail)
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    return results


def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Build output folder name
    output_folder = build_output_folder_name(args.model, args.reasoning)
    
    # Setup master log
    base_results = Path(args.results_dir)
    master_log_path = base_results / output_folder / "master_execution.log"
    logger = setup_master_logger(master_log_path)
    
    # Display configuration
    print("="*70)
    print("MASTER SCRIPT - CONCURRENT PROMPT EXECUTION")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Model: {args.model}")
    print(f"  Reasoning: {args.reasoning if args.reasoning else 'None'}")
    print(f"  Output Folder: {output_folder}")
    print(f"  Complexity Mode: {args.complexity_mode}")
    if args.complexity_mode != 'all':
        print(f"    Specific: {args.complexity}")
    print(f"  Layout Mode: {args.layout_mode}")
    if args.layouts:
        print(f"    Specific: {args.layouts}")
    print(f"  Floorplan Dir: {args.floorplan_dir}")
    print(f"  Results Dir: {args.results_dir}")
    print(f"  Master Log: {master_log_path}")
    print("="*70)
    print()
    
    logger.info("Master script started")
    logger.info(f"Model: {args.model}, Reasoning: {args.reasoning}, Output folder: {output_folder}")
    
    # Run scripts concurrently
    print("Running both scripts concurrently...\n")
    results = asyncio.run(run_all_scripts(args, logger))
    
    # Process results
    print("\n" + "="*70)
    print("MASTER EXECUTION SUMMARY")
    print("="*70)
    
    successful = 0
    failed = 0
    
    for result in results:
        if isinstance(result, tuple):
            script_name, success, error_msg = result
            if success:
                print(f"✓ {script_name}: Success")
                logger.info(f"✓ {script_name}: Success")
                successful += 1
            else:
                print(f"✗ {script_name}: Failed - {error_msg}")
                logger.error(f"✗ {script_name}: Failed - {error_msg}")
                failed += 1
        else:
            # Exception was raised
            print(f"✗ Unexpected error: {result}")
            logger.error(f"✗ Unexpected error: {result}")
            failed += 1
    
    print(f"\nTotal Scripts: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print("="*70)
    print(f"\nMaster log saved to: {master_log_path}")
    print("\nExecution complete!")
    
    logger.info("="*70)
    logger.info(f"Master execution complete - Success: {successful}, Failed: {failed}")
    logger.info("="*70)
    
    # Exit with appropriate code
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
