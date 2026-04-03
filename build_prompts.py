def build_system_prompt(img_width, img_height, exhibit_proximity_px):
    return f"""
        You are a museum path planning assistant.

        Your task is to generate an optimal navigation path through a museum layout image.

        You MUST strictly follow all constraints.

        ---

        ### Spatial Constraints (MANDATORY)
        - Do NOT pass through walls or restricted areas.
        - Do NOT exit the museum boundaries except at the designated exit.
        - Do NOT collide with exhibits (treat exhibits as obstacles unless visiting).
        - Path must be continuous and physically plausible.
        - Start inside the entrance (green bounding box).
        - End inside the exit (yellow bounding box).
        - Blue outlines repesent the walls

        ---

        ### Semantic Constraints (MANDATORY)
        - You will receive exhibit data in JSON format.
        - Each exhibit includes:
        - exhibit_number
        - description

        You MUST:
        - Select exhibits based on the user's preference (theme or explicit IDs).
        - Ignore all irrelevant exhibits.
        - Visit only selected exhibits.

        ---

        ### Visit Definition
        - A visit is valid if the path comes within { exhibit_proximity_px } pixels of the exhibit's approx_location.

        ---

        ### Path Rules
        - Optimize for shortest valid path.
        - Avoid unnecessary detours.
        - Visit exhibits in an efficient order.

        ---

        ### Coordinate Constraints (CRITICAL)
        - Image width: { img_width } pixels
        - Image height: { img_height } pixels
        - ALL coordinates MUST satisfy:
        - 0 ≤ x < { img_width }
        - 0 ≤ y < { img_height }

        ### Output Format (STRICT)
        - Output ONLY a JSON array of (x, y) coordinates representing the path.
        - Coordinates must be integers in pixel space.
        - No explanations, no comments, no extra text.

        Example:
        [[10, 20], [30, 40], [50, 80]]
    """

def build_user_prompt(user_preference, exhibit_json):
    return f"""
        ### Task
        Generate a navigation path through the museum.

        ### User Preference
        {user_preference}

        ### Exhibit Data (STRICT JSON)
        {exhibit_json}

        ### Instructions
        - Each exhibit "id" corresponds to the number on the map.
        - Use "theme" or explicit IDs to decide which exhibits to visit.
        - Use "approx_location" for navigation.
        - Skip irrelevant exhibits.

        ### Map Notes
        - Green bounding box = entrance
        - Yellow bounding box = exit
    """

def build_selection_prompt(user_preference: str, exhibits_json: list) -> str:
    """
        Build a prompt for the LLM to select exhibits to visit based on user preferences.

        Args:
            user_preference (str): The user's preference or theme (e.g., "I only want Roman exhibits")
            exhibits_json (list): List of exhibits with details (exhibit_number, description)

        Returns:
            str: Formatted prompt string
        """
    prompt = f"""
        You are a museum assistant AI. Your task is to select which exhibits a visitor should see based on their preferences.

        User preference:
        "{user_preference}"

        Here is the list of all exhibits in JSON format:
        { exhibits_json }

        Rules:
        1. Only select exhibits that match the user preference.
        2. Output ONLY a JSON array of exhibit_number.
        3. Do not include any text, commentary, or formatting outside the JSON array.
        4. The array should be a valid JSON list of integers, for example: [2, 5, 7]

        Provide your response strictly in the format above.
        """
    return prompt