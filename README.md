Metrics:

SVR

- connectivity
- wall_crossings
- exhibit_collision
- out_of_area_violations

SCSR

- start_end_location
- must_pass_regions
- restricted_area_violations
- distance_budget

SCAR

- specific_exhibit_coverage
- at_least_n_exhibits_coverage
- exhibit_category_coverage (visit all exhibits under this category to pass)
- attribute_validations
  - or (visit all exhibits that matches any of these attributes to pass):
    - date
    - material
    - find_spot
    - technique
  - and (visit all exhibits that matches all of the attributes to pass):
    - date
    - material
    - find_spot
    - technique

Validation Metrics By Difficulty:
Easy-spatial:

1. Full SVR
2. SCSR
   - start_end_location
   - must_pass_regions
   - restricted_area_violations
3. SCAR
   - at_least_n_exhibits_coverage (At least 1)

Easy-semantic:

1. Full SVR
2. SCSR
   - start_end_location
3. SCAR
   - exhibit_category_coverage / attribute_validations (or / and)

Medium Task:

1. Full SVR
2. SCSR
   - start_end_location
   - must_pass_regions
   - restricted_area_violations
3. SCAR
   - at_least_n_exhibits_coverage (Higher number around 10)
   - exhibit_category_coverage / attribute_validations (or / and)

Hard Task:

1. Full SVR
2. Full SCSR
3. SCAR
   - specific_exhibit_coverage
   - at_least_n_exhibits_coverage (Higher number around 25)
   - exhibit_category_coverage
   - attribute_validations (or / and)
