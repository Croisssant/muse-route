In the attached floorplan, each number corresponds to an exhibit number.
Each exhibit has an associated artifact with its details provided in the JSON list of exhibits below.

INSTRUCTIONS:

1. Analyze the floorplan and identify the walls and spatial constraints that humans cannot pass through.
2. Plan a route that starts from the entrance and exits via the specified exit.
3. Route planned MUST STRICTLY FOLLOW all spatial and semantic constraints.
4. Route planned MUST NOT include paths that are not in the semantic constraints

SPATIAL CONSTRAINTS:
The route should connect the entrance to the exit, visiting all exhibits fullfiling the semantic constraints.
Do not allow the route to cross walls or extend outside the floorplan.
The route line should be continuous, smooth, and clearly visible on the image.
START from the GREEN BOUNDING BOX.
END at the YELLOW BOUNDING BOX.

SEMANTIC CONSTRAINTS:
Only include exhibits where the description clearly indicates Roman origin. Exclude all non-Roman artifacts (e.g., Mauryan coins).

OUTPUT:
Generate and return an image showing the original floorplan with the overlaid route plan.
Output image should have the SAME DIMENSION and RESOLUTION from the original image.
Strictly follow all spatial and semantic constraints.

LIST OF EXHIBITS:
[]
