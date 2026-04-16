import json
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import argparse

# Configure matplotlib for table rendering
matplotlib.use('Agg')


def calculate_svr_score(svr_data):
    """
    Calculate SVR (Spatial Validation Rules) score.
    Checks: connectivity (True=pass), wall_crossings (False=pass), 
            exhibit_collision (False=pass), out_of_area_violations (False=pass)
    """
    checks = {
        'connectivity': svr_data.get('connectivity', False) == True,
        'wall_crossings': svr_data.get('wall_crossings', True) == False,
        'exhibit_collision': svr_data.get('exhibit_collision', True) == False,
        'out_of_area_violations': svr_data.get('out_of_area_violations', True) == False
    }
    passed = sum(checks.values())
    total = len(checks)
    return passed, total


def calculate_scsr_score(scsr_data):
    """
    Calculate SCSR (Spatial Constraint Satisfaction Rules) score.
    Checks: start_end_location (True=pass), must_pass_regions (True=pass),
            restricted_area_violations (False=pass), distance_budget (True=pass)
    """
    checks = {}
    
    if 'start_end_location' in scsr_data:
        checks['start_end_location'] = scsr_data['start_end_location'] == True
    if 'must_pass_regions' in scsr_data:
        checks['must_pass_regions'] = scsr_data['must_pass_regions'] == True
    if 'restricted_area_violations' in scsr_data:
        checks['restricted_area_violations'] = scsr_data['restricted_area_violations'] == False
    if 'distance_budget' in scsr_data:
        checks['distance_budget'] = scsr_data['distance_budget'] == True
    
    passed = sum(checks.values())
    total = len(checks)
    return passed, total


def calculate_scar_score(scar_data):
    """
    Calculate SCAR (Semantic Coverage and Requirements) score.
    Checks: specific_exhibit_coverage, at_least_n_exhibits_coverage,
            exhibit_category_coverage, attribute_validations
    """
    checks = {}
    
    if 'specific_exhibit_coverage' in scar_data:
        val = scar_data['specific_exhibit_coverage']
        checks['specific_exhibit_coverage'] = val if isinstance(val, bool) else val.get('valid', False)
    
    if 'at_least_n_exhibits_coverage' in scar_data:
        val = scar_data['at_least_n_exhibits_coverage']
        checks['at_least_n_exhibits_coverage'] = val if isinstance(val, bool) else val.get('valid', False)
    
    if 'exhibit_category_coverage' in scar_data:
        val = scar_data['exhibit_category_coverage']
        if isinstance(val, bool):
            checks['exhibit_category_coverage'] = val
        elif isinstance(val, dict):
            # All categories must be valid
            checks['exhibit_category_coverage'] = all(
                cat_data.get('valid', False) if isinstance(cat_data, dict) else cat_data
                for cat_data in val.values()
            )
    
    if 'attribute_validations' in scar_data:
        val = scar_data['attribute_validations']
        checks['attribute_validations'] = val if isinstance(val, bool) else val.get('valid', False)
    
    passed = sum(checks.values())
    total = len(checks)
    return passed, total


def collect_all_results(results_dir='results'):
    """
    Traverse the results directory and collect all final_results.json files.
    Returns a list of dictionaries with metadata and scores.
    """
    results = []
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"Warning: Results directory '{results_dir}' not found.")
        return results
    
    # Find all final_results.json files
    for json_file in results_path.rglob('final_results.json'):
        # Parse the directory structure
        # Expected: results/{model}/{difficulty}/{complexity}/{layout}/final_results.json
        parts = json_file.parts
        
        try:
            # Get indices relative to results directory
            results_idx = parts.index('results')
            
            if len(parts) >= results_idx + 5:
                model = parts[results_idx + 1]
                difficulty = parts[results_idx + 2]
                complexity = parts[results_idx + 3]
                layout = parts[results_idx + 4]
                
                # Load the JSON data
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Calculate scores
                svr_pass, svr_total = calculate_svr_score(data.get('svr', {}))
                scsr_pass, scsr_total = calculate_scsr_score(data.get('scsr', {}))
                scar_pass, scar_total = calculate_scar_score(data.get('scar', {}))
                
                results.append({
                    'model': model,
                    'difficulty': difficulty,
                    'complexity': complexity,
                    'layout': layout,
                    'svr': f"{svr_pass}/{svr_total}",
                    'scsr': f"{scsr_pass}/{scsr_total}",
                    'scar': f"{scar_pass}/{scar_total}"
                })
        except (ValueError, IndexError) as e:
            print(f"Warning: Could not parse path structure for {json_file}: {e}")
            continue
    
    return results


def build_summary_dataframe(results):
    """
    Build a hierarchical DataFrame from the collected results.
    Multi-index rows: [Complexity, Difficulty, Layout]
    Multi-level columns: [Model, Metric]
    """
    if not results:
        print("No results found to process.")
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Sort for better organization
    # Define custom sort orders
    complexity_order = {'simple': 0, 'complex': 1}
    difficulty_order = {
        'easy_semantic': 0,
        'easy_spatial': 1,
        'medium': 2,
        'hard': 3
    }
    
    df['complexity_sort'] = df['complexity'].map(complexity_order)
    df['difficulty_sort'] = df['difficulty'].map(difficulty_order)
    
    # Sort by complexity, difficulty, then layout
    df = df.sort_values(['complexity_sort', 'difficulty_sort', 'layout'])
    df = df.drop(['complexity_sort', 'difficulty_sort'], axis=1)
    
    # Format names for display
    df['complexity_display'] = df['complexity'].str.capitalize()
    df['difficulty_display'] = df['difficulty'].str.replace('_', ' ').str.title()
    
    # Create pivot table with multi-index
    pivot_df = df.pivot_table(
        index=['complexity_display', 'difficulty_display', 'layout'],
        columns='model',
        values=['svr', 'scsr', 'scar'],
        aggfunc='first'
    )
    
    # Reorder columns: for each model, show SVR, SCSR, SCAR
    if len(pivot_df.columns.levels[1]) > 0:  # Check if we have models
        pivot_df = pivot_df.swaplevel(0, 1, axis=1)
        pivot_df = pivot_df.sort_index(axis=1, level=0)
    
    # Rename index labels
    pivot_df.index.names = ['Layout Complexity', 'Difficulty', 'Layout']
    
    return pivot_df


def export_to_latex(df, output_file='metrics_summary.tex'):
    """
    Export DataFrame to LaTeX table format suitable for research papers.
    """
    if df is None or df.empty:
        print("No data to export to LaTeX.")
        return
    
    latex_lines = []
    
    # Begin table
    latex_lines.append("\\begin{table}[htbp]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{Validation Metrics Summary}")
    latex_lines.append("\\label{tab:metrics_summary}")
    
    # Get models from columns
    models = df.columns.levels[0] if hasattr(df.columns, 'levels') else df.columns.get_level_values(0).unique()
    num_models = len(models)
    
    # Define column alignment
    col_align = "l l l " + " ".join(["c c c"] * num_models)
    latex_lines.append(f"\\begin{{tabular}}{{{col_align}}}")
    latex_lines.append("\\hline")
    
    # Create header rows
    # First row: Model names spanning 3 columns each
    header1 = "\\textbf{Complexity} & \\textbf{Difficulty} & \\textbf{Layout}"
    for model in models:
        header1 += f" & \\multicolumn{{3}}{{c}}{{\\textbf{{{model}}}}}"
    header1 += " \\\\"
    latex_lines.append(header1)
    
    # Second row: Metric names (SVR, SCSR, SCAR) for each model
    header2 = " & & "
    for _ in models:
        header2 += " & \\textbf{SVR} & \\textbf{SCSR} & \\textbf{SCAR}"
    header2 += " \\\\"
    latex_lines.append(header2)
    latex_lines.append("\\hline")
    
    # Add data rows
    prev_complexity = None
    prev_difficulty = None
    
    for idx, row in df.iterrows():
        complexity, difficulty, layout = idx
        
        # Add separator between complexity groups
        if prev_complexity is not None and prev_complexity != complexity:
            latex_lines.append("\\hline")
        
        # Format row
        row_str = ""
        
        # Complexity column (only show if changed)
        if complexity != prev_complexity:
            row_str += f"{complexity}"
        else:
            row_str += ""
        row_str += " & "
        
        # Difficulty column (only show if changed)
        if difficulty != prev_difficulty:
            row_str += f"{difficulty}"
        else:
            row_str += ""
        row_str += " & "
        
        # Layout column
        row_str += f"{layout}"
        
        # Add metrics for each model
        for model in models:
            try:
                svr = row[(model, 'svr')] if (model, 'svr') in row.index else '-'
                scsr = row[(model, 'scsr')] if (model, 'scsr') in row.index else '-'
                scar = row[(model, 'scar')] if (model, 'scar') in row.index else '-'
                row_str += f" & {svr} & {scsr} & {scar}"
            except:
                row_str += " & - & - & -"
        
        row_str += " \\\\"
        latex_lines.append(row_str)
        
        prev_complexity = complexity
        prev_difficulty = difficulty
    
    # End table
    latex_lines.append("\\hline")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")
    
    # Write to file
    latex_content = "\n".join(latex_lines)
    with open(output_file, 'w') as f:
        f.write(latex_content)
    
    print(f"LaTeX table exported to: {output_file}")
    return latex_content


def export_to_image(df, output_file='metrics_summary.png', dpi=300):
    """
    Export DataFrame to image for visualization with merged cells for repeated values.
    """
    if df is None or df.empty:
        print("No data to export to image.")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis('tight')
    ax.axis('off')
    
    # Convert multi-index DataFrame to displayable format
    display_df = df.copy()
    
    # Reset index to make it part of the data
    display_df = display_df.reset_index()
    
    # Create a version with merged cells (empty strings for repeated values)
    display_data = []
    prev_complexity = None
    prev_difficulty = None
    
    for idx, row in display_df.iterrows():
        row_data = list(row)
        
        # Replace repeated complexity values with empty string
        if row_data[0] == prev_complexity:
            row_data[0] = ''
        else:
            prev_complexity = row_data[0]
        
        # Replace repeated difficulty values with empty string
        if row_data[1] == prev_difficulty:
            row_data[1] = ''
        else:
            prev_difficulty = row_data[1]
        
        display_data.append(row_data)
    
    # Create column labels
    col_labels = list(display_df.columns)
    
    # Format column labels for multi-level headers
    formatted_cols = []
    for col in col_labels:
        if isinstance(col, tuple):
            formatted_cols.append(f"{col[0]}\n{col[1]}")
        else:
            formatted_cols.append(col)
    
    # Create the table
    table = ax.table(
        cellText=display_data,
        colLabels=formatted_cols,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 2)
    
    # Style header cells
    for i in range(len(formatted_cols)):
        cell = table[(0, i)]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(weight='bold', color='white')
    
    # Style index columns (first 3 columns) and add visual separation
    for i in range(1, len(display_data) + 1):
        for j in range(3):
            cell = table[(i, j)]
            cell.set_facecolor('#E7E6E6')
            
            # Remove borders from merged cells (empty cells in complexity and difficulty columns)
            if j == 0 and display_data[i-1][0] == '':
                # Empty complexity cell - remove borders
                cell.set_edgecolor('#E7E6E6')  # Match background color
                cell.set_linewidth(0.5)
            elif j == 1 and display_data[i-1][1] == '':
                # Empty difficulty cell - remove borders
                cell.set_edgecolor('#E7E6E6')  # Match background color
                cell.set_linewidth(0.5)
            else:
                # Cells with content - keep normal borders
                cell.set_edgecolor('black')
                cell.set_linewidth(1)
    
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    print(f"Table image exported to: {output_file}")


def main():
    """
    Main function to orchestrate metrics collection and export.
    """
    parser = argparse.ArgumentParser(description='Generate metrics summary from validation results')
    parser.add_argument('--output', choices=['latex', 'image', 'all'], default='all',
                        help='Output format(s) to generate')
    parser.add_argument('--results-dir', default='results',
                        help='Directory containing results (default: results)')
    args = parser.parse_args()
    
    print("=" * 70)
    print("METRICS SUMMARY GENERATOR")
    print("=" * 70)
    print()
    
    # Collect all results
    print(f"Scanning results directory: {args.results_dir}")
    results = collect_all_results(args.results_dir)
    print(f"Found {len(results)} result files")
    print()
    
    if not results:
        print("No results found. Exiting.")
        return
    
    # Build summary DataFrame
    print("Building summary DataFrame...")
    summary_df = build_summary_dataframe(results)
    
    if summary_df is not None:
        print()
        print("=" * 70)
        print("SUMMARY TABLE")
        print("=" * 70)
        print()
        print(summary_df.to_string())
        print()
        
        # Export based on user choice
        if args.output in ['latex', 'all']:
            print("Generating LaTeX output...")
            export_to_latex(summary_df)
            print()
        
        if args.output in ['image', 'all']:
            print("Generating image output...")
            export_to_image(summary_df)
            print()
        
        print("=" * 70)
        print("PROCESSING COMPLETE")
        print("=" * 70)


if __name__ == "__main__":
    main()
