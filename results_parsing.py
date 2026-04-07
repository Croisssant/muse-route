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
    
    print(f"is_valid: {summary['is_valid']}")
    print(f"total_violations: {summary['total_violations']}")
    print(f"svr_pass: {svr_pass}/4")
    print(f"scsr_pass: {scsr_pass}/4")
