#!/usr/bin/env python3
"""
Calculate average inference runtime per task for each evaluated model.
Groups easy_semantic and easy_spatial under "Easy" difficulty.
"""

import os
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import re


def parse_timestamp(log_line):
    """Extract timestamp from log line format: '2026-04-29 11:06:49 - INFO - ...'"""
    match = re.match(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', log_line)
    if match:
        timestamp_str = match.group(1)
        return datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
    return None


def calculate_task_runtime(log_file_path):
    """Calculate runtime in seconds from first to last timestamp in processing.log"""
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if len(lines) < 2:
            return None
        
        # Get first and last timestamp
        first_timestamp = None
        last_timestamp = None
        
        for line in lines:
            timestamp = parse_timestamp(line)
            if timestamp:
                if first_timestamp is None:
                    first_timestamp = timestamp
                last_timestamp = timestamp
        
        if first_timestamp and last_timestamp:
            runtime = (last_timestamp - first_timestamp).total_seconds()
            return runtime
        
        return None
    except Exception as e:
        print(f"Error reading {log_file_path}: {e}")
        return None


def remove_outliers_iqr(data):
    """Remove outliers using the IQR method (Q1 - 1.5*IQR, Q3 + 1.5*IQR)"""
    if len(data) < 4:  # Need at least 4 points for meaningful IQR
        return data, []
    
    sorted_data = sorted(data)
    n = len(sorted_data)
    
    # Calculate Q1 and Q3
    q1_idx = n // 4
    q3_idx = 3 * n // 4
    q1 = sorted_data[q1_idx]
    q3 = sorted_data[q3_idx]
    
    # Calculate IQR
    iqr = q3 - q1
    
    # Define outlier bounds
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    # Separate clean data from outliers
    clean_data = [x for x in data if lower_bound <= x <= upper_bound]
    outliers = [x for x in data if x < lower_bound or x > upper_bound]
    
    return clean_data, outliers


def scan_model_results(results_dir):
    """Scan all model folders and calculate average runtime per difficulty"""
    results_dir = Path(results_dir)
    model_stats = {}
    
    # Get all model directories
    model_dirs = [d for d in results_dir.iterdir() if d.is_dir()]
    
    for model_dir in sorted(model_dirs):
        model_name = model_dir.name
        difficulty_runtimes = defaultdict(list)
        
        # Scan all difficulty folders
        for difficulty_dir in model_dir.iterdir():
            if not difficulty_dir.is_dir():
                continue
            
            difficulty_name = difficulty_dir.name
            
            # Map difficulty to grouped categories
            if difficulty_name in ['easy_semantic', 'easy_spatial']:
                grouped_difficulty = 'Easy'
            elif difficulty_name == 'medium':
                grouped_difficulty = 'Medium'
            elif difficulty_name == 'hard':
                grouped_difficulty = 'Hard'
            else:
                continue  # Skip unknown difficulty levels
            
            # Scan simple and complex subdirectories
            for complexity_dir in difficulty_dir.iterdir():
                if not complexity_dir.is_dir():
                    continue
                
                # Scan all task folders
                for task_dir in complexity_dir.iterdir():
                    if not task_dir.is_dir():
                        continue
                    
                    log_file = task_dir / 'processing.log'
                    if log_file.exists():
                        runtime = calculate_task_runtime(log_file)
                        if runtime is not None:
                            difficulty_runtimes[grouped_difficulty].append(runtime)
        
        # Calculate averages for this model (with outlier removal)
        model_stats[model_name] = {}
        for difficulty, runtimes in difficulty_runtimes.items():
            if runtimes:
                # Remove outliers using IQR method
                clean_runtimes, outliers = remove_outliers_iqr(runtimes)
                
                if clean_runtimes:
                    avg_runtime = sum(clean_runtimes) / len(clean_runtimes)
                    model_stats[model_name][difficulty] = {
                        'average': avg_runtime,
                        'count': len(clean_runtimes),
                        'total_tasks': len(runtimes),
                        'outliers_removed': len(outliers),
                        'min': min(clean_runtimes),
                        'max': max(clean_runtimes),
                        'outlier_values': sorted(outliers) if outliers else []
                    }
                else:
                    # All values were outliers (unlikely but handle it)
                    model_stats[model_name][difficulty] = {
                        'average': sum(runtimes) / len(runtimes),
                        'count': 0,
                        'total_tasks': len(runtimes),
                        'outliers_removed': len(runtimes),
                        'min': min(runtimes),
                        'max': max(runtimes),
                        'outlier_values': sorted(runtimes)
                    }
    
    return model_stats


def print_results(model_stats):
    """Print results in a clear format"""
    print("\n" + "="*80)
    print("AVERAGE INFERENCE RUNTIME PER TASK BY MODEL AND DIFFICULTY")
    print("(Outliers removed using IQR method: Q1 - 1.5×IQR, Q3 + 1.5×IQR)")
    print("="*80 + "\n")
    
    for model_name in sorted(model_stats.keys()):
        print(f"Model: {model_name}")
        print("-" * 80)
        
        stats = model_stats[model_name]
        
        # Print in order: Easy, Medium, Hard
        for difficulty in ['Easy', 'Medium', 'Hard']:
            if difficulty in stats:
                avg = stats[difficulty]['average']
                count = stats[difficulty]['count']
                total = stats[difficulty]['total_tasks']
                outliers_removed = stats[difficulty]['outliers_removed']
                min_time = stats[difficulty]['min']
                max_time = stats[difficulty]['max']
                
                # Build the main line
                main_line = f"  {difficulty:8} avg={avg:6.2f}s  (n={count:3}/{total:3} tasks)"
                main_line += f"  [min={min_time:6.2f}s, max={max_time:6.2f}s]"
                print(main_line)
                
                # If outliers were removed, show details
                if outliers_removed > 0:
                    outlier_values = stats[difficulty]['outlier_values']
                    outlier_str = ', '.join([f"{v:.1f}s" for v in outlier_values[:5]])
                    if len(outlier_values) > 5:
                        outlier_str += f", ... ({len(outlier_values)} total)"
                    print(f"           ⚠ Removed {outliers_removed} outlier(s): {outlier_str}")
            else:
                print(f"  {difficulty:8} No data available")
        
        print()
    
    print("="*80)


def main():
    results_dir = Path(__file__).parent / 'results'
    
    if not results_dir.exists():
        print(f"Error: Results directory not found at {results_dir}")
        return
    
    print("Scanning results folder and calculating runtimes...")
    model_stats = scan_model_results(results_dir)
    
    if not model_stats:
        print("No results found!")
        return
    
    print_results(model_stats)


if __name__ == '__main__':
    main()
