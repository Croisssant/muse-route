# Museum Route Planning Benchmark

This benchmark evaluates Vision-Language Models (VLMs) on their ability to generate valid museum routes with increasing task complexity.

## Difficulty Levels

### Easy: Basic Spatial Navigation

**Goal**: Navigate from entrance to exit while respecting basic spatial constraints only.

**Constraints**:

- **Basic SVR (Spatial Validity Rate)**
  - Maintain route connectivity (single connected path)
  - Never cross walls (BLUE outlines)
  - Never collide with exhibits (numbered circles)
  - Stay within valid floor area (no out-of-bounds)

- **Basic SCSR (Spatial Constraint Satisfaction Rate)**:
  - Start at entrance (GREEN box)
  - End at exit (YELLOW box)
  - Never enter restricted areas (ORANGE boxes)

- **Basic SCAR (Semantic Constraint Alignment Rate)**:
  - Visit at least ONE exhibit

**Success Criteria**: Generate a connected route from entrance to exit that stays within the floor plan boundaries.

---

### Medium: Partial Spatial and Semantic Constraints

**Goal**: Navigate from entrance to exit while respecting additional spatial and semantics constraints.

**Constraints**:

- **All Easy constraints**, plus:

- **Additional SCSR (Spatial Constraint Satisfaction Rate)**:
  - Must pass through required galleries (PURPLE boxes marked "must_see")

- **Partial SCAR (Semantic Constraint Alignment Rate)**:
  - Visit a larger minimum number of exhibits (e.g., at least 15 exhibits)
  - Cover required exhibit categories (e.g., at least one "Roman" exhibit)

**Success Criteria**: Generate a physically valid route that avoids all obstacles, respects restricted zones, respects user's visit requirements and passes through mandatory gallery regions.

---

### Hard: Full Spatial and Semantic Constraints

**Goal**: Navigate with all spatial constraints while satisfying semantic exhibit requirements.

**Constraints**:

- **All Easy and Medium constraints**, plus:
- **SCAR (Semantic Constraint Alignment Rate)**:
  - Visit specific required exhibits (e.g., exhibits [1, 96, 97, 98, 99, 100])

- **Full SCSR**:
  - Respect distance budget constraints (if applicable)
  - Must avoid restricted galleries (PURPLE boxes marked "restricted") + Must pass through required galleries (PURPLE boxes marked "must_see")

**Success Criteria**: Generate a route that satisfies all spatial validity checks AND all semantic coverage requirements while staying within the distance budget.

---

**Additional Complexity for ALL levels**:

- Exhibits must be visited within detection range (~183 pixels)
- Route must be optimized to visit all required exhibits efficiently
- Semantic requirements may conflict with spatial optimization (e.g., required exhibit near restricted area)
