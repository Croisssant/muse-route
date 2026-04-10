# Quick metrics extraction
import json
with open('validation_results.json') as f:
    data = json.load(f)
    summary = data['validation_summary']
    
    # Calculate SVR pass rate (good = true for connectivity, false for violations)
    svr = summary['svr']
    svr_pass = sum([
        svr['connectivity'] == True,
        svr['wall_crossings'] == False,
        svr['exhibit_collision'] == False,
        svr['out_of_area_violations'] == False
    ])
    
    # Calculate SCSR pass rate
    scsr = summary['scsr']
    scsr_pass = sum([
        scsr['start_end_location'] == True,
        scsr['must_pass_regions'] == True,
        scsr['restricted_area_violations'] == False,
        scsr['distance_budget'] == True
    ])
    
    # Calculate SCAR pass rate (Semantic Coverage and Requirements)
    scar = summary.get('scar', {})
    scar_checks = []
    
    # Check specific exhibit coverage
    if 'specific_exhibit_coverage' in scar:
        scar_checks.append(scar['specific_exhibit_coverage'].get('valid', False))
    
    # Check minimum exhibits coverage
    if 'at_least_n_exhibits_coverage' in scar:
        scar_checks.append(scar['at_least_n_exhibits_coverage'].get('valid', False))
    
    # Check all category coverage (each category must be valid)
    if 'exhibit_category_coverage' in scar:
        category_coverage = scar['exhibit_category_coverage']
        if category_coverage:
            # All categories must be valid
            all_categories_valid = all(
                cat_data.get('valid', False) 
                for cat_data in category_coverage.values()
            )
            scar_checks.append(all_categories_valid)
    
    scar_pass = sum(scar_checks)
    scar_total = len(scar_checks)
    
    print(f"is_valid: {summary['is_valid']}")
    print(f"total_violations: {summary['total_violations']}")
    print(f"svr_pass: {svr_pass}/4")
    print(f"scsr_pass: {scsr_pass}/4")
    print(f"scar_pass: {scar_pass}/{scar_total}")
