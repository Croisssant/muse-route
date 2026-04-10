"""
Prompt building utilities for museum route planning at different difficulty levels.
"""

def build_selection_prompt(difficulty, exhibits_json, configs, user_preference):
    """
    Build exhibit selection prompt based on difficulty level.
    
    Args:
        difficulty: 'easy', 'medium', or 'hard'
        exhibits_json: List of all exhibits
        configs: Configuration dictionary
        user_preference: User's exhibit preference string
    
    Returns:
        tuple: (system_prompt, user_prompt) or (None, None) for easy
    """
    if difficulty == 'easy':
        # No exhibit selection for easy level
        return None, None
    
    # Extract config values
    required_exhibit_ids = configs.get("specific_exhibit_to_cover", [])
    required_categories = configs.get("exhibit_categories_to_cover", [])
    min_exhibits_to_cover = configs.get("at_least_n_exhibits_to_cover", len(required_exhibit_ids))
    selection_target_count = max(min_exhibits_to_cover, len(required_exhibit_ids))
    
    # Build category requirement text
    required_category_requirement = ""
    required_category_checklist = ""
    required_category_prompt_line = ""
    
    if required_categories:
        required_category_requirement = f"    3. The final route must include at least one exhibit from each required category in {required_categories}.\n"
        required_category_checklist = f"    2. At least one exhibit from each required category in {required_categories} is present.\n"
        required_category_prompt_line = f"Ensure the selection covers each required category in {required_categories}.\n"
    else:
        required_category_requirement = "    3. There are no required exhibit categories configured for this run.\n"
        required_category_checklist = "    2. There are no required exhibit-category checks for this run.\n"
    
    system_prompt = f"""
    You are a museum assistant AI selecting exhibits for a route-planning benchmark.

    Here is the full exhibit list in JSON:
    {exhibits_json}

    Hard benchmark requirements that override user preference when there is any conflict:
    1. The final route must include exhibit numbers {required_exhibit_ids}.
    2. The final route must cover at least {min_exhibits_to_cover} exhibits total.
    {required_category_requirement}

    Selection process:
    1. Start by locking in every hard-required exhibit {required_exhibit_ids}. These required exhibits are mandatory and cannot be removed.
    2. Add additional exhibits until there are exactly {selection_target_count} unique exhibit numbers.
    3. Strongly prefer exhibits that match the user preference.
    4. When several exhibits satisfy the preference equally well, prefer the subset that forms a compact visit plan with less backtracking and fewer long jumps.
    5. Avoid redundant choices that spread the route across distant regions when a more compact preference-matching option exists.
    6. If there is uncertainty, prefer a conservative set that is easier to route legally and compactly rather than a sprawling set.
    7. If the user preference conflicts with the hard benchmark requirements, satisfy the hard benchmark requirements first and then maximize preference match.
    8. Order the final exhibit numbers in a sensible visiting sequence for a compact walk from entrance to exit.
    9. The order should move smoothly through nearby regions instead of jumping back and forth between distant parts of the museum.
    10. Prefer an order that reduces backtracking and reduces the need to cross the same corridor multiple times.
    11. Prefer optional exhibits that can be covered by one or two compact clusters rather than optional exhibits scattered across many distant regions.

    Validation checklist before responding:
    1. {required_exhibit_ids} are all present.
    {required_category_checklist}    
    3. The list contains exactly {selection_target_count} unique integers.
    4. The order should represent a plausible visit order, not a random order.
    5. If any required exhibit is missing, replace optional exhibits until all required exhibits are present before responding.

    Output rules:
    1. Output ONLY a JSON array of exhibit numbers.
    2. Do not output markdown, labels, commentary, or any other text.
    3. The response must be valid JSON, for example: [1, 5, 9]
    """
    
    user_prompt = f"""
    User preference: {user_preference}

    Return exactly {selection_target_count} exhibit numbers that satisfy the benchmark requirements above.
    Follow the validation checklist before you answer.
    The JSON array order should be the recommended visiting order.
    Avoid orders that bounce between distant exhibit groups.
    Do not submit any answer that omits one of {required_exhibit_ids}.
    Prefer optional exhibits that let the route stay compact instead of visiting many separate clusters.
    {required_category_prompt_line}The JSON array should remain valid even if the config values change in a future run.
    """
    
    return system_prompt, user_prompt


def build_route_planning_prompt(difficulty, img_width, img_height, visit_distance_text, filtered_exhibits=None):
    """
    Build route planning prompt based on difficulty level.
    
    Args:
        difficulty: 'easy', 'medium', or 'hard'
        img_width: Image width in pixels
        img_height: Image height in pixels
        visit_distance_text: Text describing visit distance (e.g., "25 pixels")
        filtered_exhibits: List of selected exhibits (for medium/hard) or None (for easy)
    
    Returns:
        tuple: (system_prompt, user_prompt)
    """
    if difficulty == 'easy':
        system_prompt = f"""
        You are a museum path planning assistant generating a single drawable polyline on top of the museum image.

        Your ONLY goal is to create a simple, valid path from entrance to exit.

        ### Visual legend
        - BLUE outlines = walls and structural barriers. Never cross them.
        - ORANGE boxes = restricted areas. Never enter them.
        - PURPLE boxes = gallery areas. You can choose to visit or ignore it.
        - GREEN box = entrance. The route must start inside it.
        - YELLOW box = exit. The route must end inside it.
        - Numbered circles = exhibits (you can ignore these).

        ### Non-negotiable constraints
        - Start inside the entrance box (GREEN), with the first point clearly inside rather than on the border.
        - End inside the exit box (YELLOW), with the last point clearly inside rather than on the border.
        - Stay inside the valid museum floor area at all times.
        - Never cross walls (BLUE outlines).
        - Never draw through exhibit markers or obstacle geometry.
        - Never enter restricted areas (ORANGE boxes).
        - The path must be continuous and physically plausible.

        ### Planning strategy
        - First, silently identify all visible no-go areas: walls, restricted areas, exhibit markers, and dead-end risky spaces.
        - Second, build a safe corridor skeleton from entrance to exit that stays legal from start to finish.
        - Third, convert the final walk into a single continuous polyline.
        - If uncertain, choose the safer route instead of the shorter shortcut.
        - After drafting the route, trim any detour that does not help connect the legal start-to-exit walk.
        - Focus solely on creating a valid, direct path from entrance to exit.
        - The route should be simple and efficient without unnecessary detours.
        - Keep it short and straightforward.

        ### Path construction rules
        - The path must be one continuous, physically plausible walking route.
        - Use a multi-point polyline with enough waypoints to show the path clearly.
        - Return between 15 and 25 coordinate pairs (fewer is better if path is direct).
        - Consecutive points should trace a sensible walking path through open floor space.
        - Every straight segment between consecutive points must stay in legal open floor space.
        - If a straight segment would clip a wall, restricted area, or exhibit marker, add waypoints to go around.
        - Prefer orthogonal walking segments (horizontal and vertical) where practical.
        - Favor open corridors and wider spaces over risky shortcuts near hazards.
        - Maintain visible clearance from restricted-area borders and walls.

        ### Coordinate constraints
        - Image width: {img_width} pixels
        - Image height: {img_height} pixels
        - Every coordinate must be an integer pair [x, y].
        - Every coordinate must satisfy 0 <= x < {img_width} and 0 <= y < {img_height}.
        - Do not output floats.
        - Do not output tuples, objects, strings, or nested wrappers.

        ### Output format
        - Output ONLY a JSON array of coordinate pairs.
        - The first item must be the start point (inside GREEN entrance box).
        - The last item must be the exit point (inside YELLOW exit box).
        - Valid example: [[120, 410], [145, 410], [170, 405]]
        - Invalid examples: [120, 410], {{"route": [[120, 410]]}}, [[120.5, 410.2]], [[120, 410]]
        - No commentary, no markdown fences, no explanation.
    """
        
        user_prompt = f"""
        Create a simple, direct route from entrance (GREEN box) to exit (YELLOW box).

        Remember:
        - Visit any exhibits as you see fit
        - NO mandatory gallery visits required - do as you see fit
        - the route must be a continuous drawable JSON polyline,
        - the first point must be comfortably inside the entrance (GREEN),
        - the last point must be comfortably inside the exit (YELLOW),
        - avoid walls (BLUE outlines),
        - avoid restricted areas (ORANGE boxes),
        - avoid colliding with exhibit markers,
        - stay within the museum floor area,
        - keep the path simple, direct, and efficient.

        Before answering, verify:
        1. every segment avoids walls, restricted areas, and exhibits,
        2. the first point is inside the entrance,
        3. the last point is inside the exit,
        4. the path is simple and direct without unnecessary wandering.
        """
    
    else:  # medium or hard
        system_prompt = f"""
        You are a museum path planning assistant generating a single drawable polyline on top of the museum image.

        Produce a route that satisfies the benchmark exactly. Treat the following as HARD requirements.

        ### Visual legend
        - BLUE outlines = walls and structural barriers. Never cross them.
        - ORANGE boxes = restricted areas. Never enter them.
        - PURPLE boxes = gallery areas. Some may be must-see or restricted.
        - GREEN box = entrance. The route must start inside it.
        - YELLOW box = exit. The route must end inside it.
        - Numbered circles = exhibits.

        ### Non-negotiable constraints
        - Start inside the entrance box, with the first point clearly inside rather than on the border.
        - End inside the exit box, with the last point clearly inside rather than on the border.
        - Enter every must-see gallery or region.
        - Never enter any restricted gallery or restricted region.
        - Stay inside the valid museum floor area.
        - Never cross walls.
        - Never draw through exhibit markers or obstacle geometry.
        - Visit all selected exhibits and ignore unselected exhibits.
        - Rule priority is: legality first, then correct entrance and exit placement, then required gallery coverage, then selected exhibit coverage, then compactness.
        - Never violate a higher-priority rule to satisfy a lower-priority one.

        ### Visit definition
        - A selected exhibit counts as visited when the path comes within {visit_distance_text} of that exhibit's numbered location.
        - Missing even one selected exhibit is a failure.
        - Missing a required gallery or region is a failure.
        - If a selected exhibit is near a restricted area or obstacle cluster, satisfy the visit from the nearest legal open-floor location instead of entering the risky area.
        - If a selected exhibit would require entering restricted space or crossing a barrier, approach only as closely as the nearest legal open-floor position allows.
        - A required gallery visit only needs legal entry into that gallery. Once the route has legally entered the required gallery, leave it again by the nearest legal continuation instead of wandering through adjacent interiors.

        ### Planning strategy
        - First, silently identify all visible no-go areas: walls, restricted areas, restricted galleries, exhibit markers, and dead-end risky spaces.
        - Second, build a safe corridor skeleton from entrance to exit that stays legal from start to finish.
        - Third, use the selected exhibit list as the preferred visit order, but reorder when needed to preserve legality and reduce backtracking.
        - Fourth, attach short, legal detours from that skeleton to cover the selected exhibits.
        - Fifth, convert the final walk into a single continuous polyline.
        - If uncertain, choose the safer route instead of the shorter shortcut.
        - After drafting the route, trim any detour that does not help cover a selected exhibit, reach a required gallery, or connect the legal start-to-exit walk.

        ### Path construction rules
        - The path must be one continuous, physically plausible walking route.
        - Use a multi-point polyline with many waypoints, not a single point and not just 2 points.
        - Consecutive points should trace a sensible walking path through open floor space.
        - Every straight segment between consecutive points must be directly drawable through legal open floor. If a straight segment would clip a wall, restricted area, restricted gallery, or exhibit marker, add another waypoint instead of cutting through.
        - Keep the route compact: no loops, no retracing, no sightseeing detours, and no long perimeter sweeps.
        - Prefer orthogonal walking segments where practical, using diagonals only for short local adjustments in open floor space.
        - Favor open corridors and wider spaces over risky shortcuts near hazards.
        - Maintain visible clearance from restricted-region borders and exhibit markers rather than skimming right along them.
        - When satisfying a must-see gallery requirement, make the visit as shallow as possible: enter legally, cover the requirement, and exit without crossing into neighboring risky interiors.
        - After the route reaches the exit, stop immediately. Do not overshoot the exit or hook around it.
        - Avoid accidentally passing near large numbers of unselected exhibits. If many unselected exhibits would also be covered, the route is probably too broad and should be tightened.

        ### Coordinate constraints
        - Image width: {img_width} pixels
        - Image height: {img_height} pixels
        - Every coordinate must be an integer pair [x, y].
        - Every coordinate must satisfy 0 <= x < {img_width} and 0 <= y < {img_height}.
        - Do not output floats.
        - Do not output tuples, objects, strings, or nested wrappers.

        ### Output format
        - Output ONLY a JSON array of coordinate pairs.
        - The first item must be the start point and the last item must be the exit point.
        - Valid example: [[120, 410], [145, 410], [170, 405]]
        - Invalid examples: [120, 410], {{"route": [[120, 410]]}}, [[120.5, 410.2]], [[120, 410]]
        - No commentary, no markdown fences, no explanation.
    """
        
        user_prompt = f"""
        Plan a valid route for this exact selected exhibit set:
        {filtered_exhibits}

        Remember:
        - the route must be a continuous drawable JSON polyline,
        - the first point must be inside the entrance,
        - the last point must be inside the exit,
        - keep the first and last points comfortably inside those boxes, not on their borders,
        - the path must include enough waypoints to show the full walk.
        - treat the selected exhibit list as the preferred visiting order, but reorder when needed to stay legal,
        - Keep the route short and deliberate.
        - Avoid sweeping through large parts of the museum just to pass near extra exhibits.
        - Favor a corridor-like Manhattan path made of horizontal and vertical steps.
        - make the first and last coordinates visibly centered inside the green and yellow boxes rather than merely barely inside,
        - if a must-see gallery is close to restricted space, touch the legal portion you need and then leave immediately rather than traversing deeply through nearby gallery interiors,
        - trim any waypoint that does not help legality, selected-exhibit coverage, must-see gallery coverage, or direct progress from entrance to exit,
        - Before answering, silently verify that:
        1. every segment is legal and does not cut through a wall, restricted area, restricted gallery, or exhibit marker,
        2. all selected exhibits are covered from legal open floor,
        3. every must-see gallery is entered,
        4. the first point is inside the entrance,
        5. the last point is inside the exit,
        6. the route is not unnecessarily passing near many unselected exhibits.
        - if an exhibit is near a restricted area, cover it from the nearest legal open-floor position.
        """
    
    return system_prompt, user_prompt
