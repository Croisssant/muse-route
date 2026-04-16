import json

def build_required_categories(required_categories):

    required_categories_json = json.dumps(required_categories)

    required_category_requirement = (
        f"    - The final route must include all exhibits from each required category in {required_categories_json}.\n"
        if required_categories
        else ""
    )
    required_category_checklist = (
        f"    - At least one exhibit from each required category in {required_categories_json} is present.\n"
        if required_categories
        else ""
    )
    required_category_prompt_line = (
        f"Ensure the selection covers each required category in {required_categories_json}.\n"
        if required_categories
        else ""
    )

    return required_category_requirement, required_category_checklist, required_category_prompt_line


def build_required_OR_attributes(required_exhibit_attributes):
    required_OR_exhibit_attributes_json = None

    if required_exhibit_attributes:

        required_OR_exhibit_attributes = {
            k: cleaned
            for k, v in required_exhibit_attributes.items()
            if k not in ["_comment", "combined_constraint"] and (cleaned := {
                inner_key: inner_value
                for inner_key, inner_value in v.items()
                if not (isinstance(inner_value, list) and len(inner_value) == 0)
            })
        }

        required_OR_exhibit_attributes_json = json.dumps(required_OR_exhibit_attributes)

        required_OR_attribute_requirement = (
            f"""    - The final route must include all exhibits that has attributes stated in {required_OR_exhibit_attributes_json}.
                    Each attribute fields are self explanatory from its corresponding key in JSON.
                    Individual attributes are independent (OR logic between them).\n
            """
            if required_OR_exhibit_attributes
            else ""
        )
        required_OR_attribute_checklist = (
            f"    - All exhibits with the stated attributes in {required_OR_exhibit_attributes_json} is present.\n"
            if required_OR_exhibit_attributes
            else ""
        )
        required_OR_attribute_prompt_line = (
            f"Ensure the selection covers all exhibits with the stated attributes in {required_OR_exhibit_attributes_json}.\n"
            if required_OR_exhibit_attributes
            else ""
        )

    return required_OR_attribute_requirement, required_OR_attribute_checklist, required_OR_attribute_prompt_line


def build_required_AND_attributes(required_exhibit_attributes):

    required_AND_exhibit_attributes_json = None

    if required_exhibit_attributes:

        combined_constraint = required_exhibit_attributes["combined_constraint"]

        filtered_contraints = {
            k: cleaned
            for k, v in combined_constraint.items()
            if k != "_combined_comment" and (cleaned := {
                inner_key: inner_value
                for inner_key, inner_value in v.items()
                if not (isinstance(inner_value, list) and len(inner_value) == 0)
            })
        }

        required_AND_exhibit_attributes_json = json.dumps(filtered_contraints)

        required_AND_attribute_requirement = (
            f"""    - The final route must include all exhibits with matching attributes that satisfies all the attributes 
                      stated in {required_AND_exhibit_attributes_json}.
                      Each attribute fields are self explanatory from its corresponding key in JSON.
                      Attributes are AND logic - exhibit must match ALL specified conditions\n
            """
            if combined_constraint
            else ""
        )
        required_AND_attribute_checklist = (
            f"    - All exhibits with the matching attributes that satisfies all the attributes stated in {required_AND_exhibit_attributes_json} is present.\n"
            if combined_constraint
            else ""
        )
        required_AND_attribute_prompt_line = (
            f"Ensure the selection covers all exhibits with the matching attributes that satisfies all the attributes stated in {required_AND_exhibit_attributes_json}.\n"
            if combined_constraint
            else ""
        )

    return required_AND_attribute_requirement, required_AND_attribute_checklist, required_AND_attribute_prompt_line


def build_visit_distance(visit_distance_mm, distance_entries):
  
    mm_per_px = distance_entries[0].get("mm_per_px") if distance_entries else None
    visit_distance_px = int(round(visit_distance_mm / mm_per_px)) if visit_distance_mm and mm_per_px else None
    visit_distance_text = (
        f"{visit_distance_px} pixels"
        if visit_distance_px is not None
        else f"the configured visit radius derived from {visit_distance_mm} mm"
    )

    return visit_distance_text


def build_specific_exhibit_ids(required_exhibit_ids):

    required_exhibit_count = len(required_exhibit_ids)

    if required_exhibit_count > 0:
        required_exhibit_ids_json = json.dumps(required_exhibit_ids)

        specific_id_requirement = f"- The final route must include exhibit numbers {required_exhibit_ids_json}.\n"
        specific_id_selection_process = f"- Start by locking in every hard-required exhibit {required_exhibit_ids_json}. These required exhibits are mandatory and cannot be removed.\n"
        specific_id_checklist = f"- {required_exhibit_ids_json} are all present.\n"
        specific_id_prompt_line = f"Do not submit any answer that omits one of {required_exhibit_ids_json}.\n"

        return specific_id_requirement, specific_id_selection_process, specific_id_checklist, specific_id_prompt_line, required_exhibit_count
    
    else:
        return "", "", "", "", 0
    


def build_min_num_exhibits_to_cover(at_least_n_exhibits_to_cover, required_exhibit_count):

    selection_target_count = max(at_least_n_exhibits_to_cover, required_exhibit_count)


    if selection_target_count > 0:

        min_exhibit_requirement = f" - The final route must cover at least {selection_target_count} exhibits total."
        min_exhibit_selection_process = f"- Add additional exhibits until there are at least {selection_target_count} unique exhibit numbers."
        min_exhibit_checklist = f"- The list contains at least {selection_target_count} unique integers."
        min_exhibit_prompt_line = f"Return at least {selection_target_count} exhibit numbers that satisfy the benchmark requirements above."
    
        return min_exhibit_requirement, min_exhibit_selection_process, min_exhibit_checklist, min_exhibit_prompt_line


    return "", "", "", ""


def build_travel_distance_prompt(distance_budget_in_mm):
    if distance_budget_in_mm > 0:
        return f"- The total distance of the path must within {distance_budget_in_mm} in mm."
    
    return ""