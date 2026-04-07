# Museum Route Prompt Optimization

This is an automated prompt refinement system to improve LLM-generated museum routes.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr6`). The branch `prompt-opt/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b prompt-opt/<tag>` from current master.
3. **Read the in-scope files**: The repo structure is focused. Read these files for full context:
   - `README.md` — repository context (if exists).
   - `main.py` — validation pipeline (DO NOT MODIFY). This runs route extraction and validation.
   - `config.json` — fixed constants (DO NOT MODIFY). Contains distance budgets, gallery configs, and proximity thresholds.
   - `gpt_2_steps.py` — the file you modify. Contains prompts for exhibit selection and route planning.
   - `validation_results.json` — output metrics after each run.
4. **Verify setup**: Ensure `.env` file exists with valid `OPENAI_API_KEY`, and all required images/annotations are in place.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Key Context

### Project Overview

This system validates LLM-generated museum routes against spatial and semantic constraints:

**SVR (Spatial Validation Rules)**:

- `connectivity`: Route is a single connected path
- `wall_crossings`: Route does NOT pass through walls
- `exhibit_collision`: Route does NOT collide with exhibits
- `out_of_area_violations`: Route stays within valid floor area

**SCSR (Semantic Constraint Satisfaction Rules)**:

- `start_end_location`: Route starts at entrance (GREEN box) and ends at exit (YELLOW box)
- `must_pass_regions`: Route visits all required galleries (see `config.json`)
- `restricted_area_violations`: Route does NOT enter restricted areas (ORANGE boxes)
- `distance_budget`: Total route length is within budget (see `config.json`)

### Visual Annotations

- **BLUE outlines**: Walls (cannot pass through)
- **ORANGE bounding box**: Restricted area (cannot enter)
- **PURPLE bounding box**: Gallery area (may be required or restricted per config)
- **GREEN bounding box**: Entrance (start point)
- **YELLOW bounding box**: Exit (end point)
- **Numbered circles**: Exhibits with associated artifacts

### Important Constants

- `exhibit_see_distance_in_mm`: 1500 (approximately 183 pixels)
- `pixels_per_mm`: ~0.122 (calculated from image metadata)
- Image dimensions: 3195 x 1984 pixels (H x W)
- Coordinate system: (x, y) where x is horizontal, y is vertical
- Valid coordinate range: 0 ≤ x < 3195, 0 ≤ y < 1984

### Gallery Configuration (from config.json)

- `gallery_room_1`: must_see — Route MUST visit this gallery
- `gallery_open_space_1`: restricted — Route CANNOT enter this area
- Distance budget: effectively unlimited (1,000,000,000 mm)

## Experimentation

**What you CAN do:**

- Modify `gpt_2_steps.py` — specifically the prompt strings:
  - `system_prompt_selection` (lines ~40-50)
  - `system_prompt_route` (lines ~80-137)
  - `user_prompt_selection` (lines ~51-53)
  - `user_prompt_route` (lines ~138-140)
- Experiment with prompt wording, structure, emphasis, examples, etc.
- Add or remove constraints, clarifications, or instructions

**What you CANNOT do:**

- Modify `main.py`, `config.json`, validation logic, or annotations
- Change the LLM model or API parameters
- Modify the route extraction or validation algorithms
- Install new packages or dependencies

**The goal is simple: Achieve perfect validation scores.** Specifically:

1. **Primary Goal**: All SVR checks pass (true for good constraints, false for violations)
2. **Secondary Goal**: All SCSR checks pass
3. **Tertiary Goal**: Minimize `total_violations` count

Note that in the validation output:

- SVR `connectivity: true` means connected (GOOD)
- SVR `wall_crossings: true` means there ARE wall crossings (BAD)
- SVR `exhibit_collision: true` means there ARE collisions (BAD)
- SVR `out_of_area_violations: true` means there ARE violations (BAD)

So the ideal SVR is: `{connectivity: true, wall_crossings: false, exhibit_collision: false, out_of_area_violations: false}`

**The first run**: Your very first run should always be to establish the baseline, so you will run the script as is.

## Output Format

The script runs through three steps:

1. Exhibit selection based on user preference
2. Route planning using GPT with vision
3. Validation using `main.py`

After validation completes, check `validation_results.json`:

```json
{
  "validation_summary": {
    "is_valid": false,
    "total_violations": 3257,
    "svr": {
      "connectivity": true,
      "wall_crossings": true,
      "exhibit_collision": true,
      "out_of_area_violations": true
    },
    "scsr": {
      "start_end_location": true,
      "must_pass_regions": false,
      "restricted_area_violations": false,
      "distance_budget": true
    }
  }
}
```

Extract key metrics:

```bash
grep '"is_valid":\|"total_violations":\|"svr":\|"scsr":' validation_results.json
```

## Logging Results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated).

The TSV has a header row and 7 columns:

```
commit	is_valid	total_violations	svr_pass	scsr_pass	status	description
```

1. git commit hash (short, 7 chars)
2. is_valid (true/false)
3. total_violations count
4. svr_pass: "4/4" if all pass, "2/4" if 2 out of 4 pass, etc.
5. scsr_pass: "4/4" if all pass, "3/4" if 3 out of 4 pass, etc.
6. status: `keep`, `discard`, or `crash`
7. short text description of what this experiment tried

Example:

```
commit	is_valid	total_violations	svr_pass	scsr_pass	status	description
a1b2c3d	false	3257	1/4	3/4	keep	baseline - original prompts
b2c3d4e	false	1842	2/4	3/4	keep	emphasize "do not cross walls" 3x in prompt
c3d4e5f	false	4103	1/4	2/4	discard	removed coordinate constraints - made worse
d4e5f6g	false	0	0/0	0/0	crash	invalid JSON response from GPT
e5f6g7h	true	0	4/4	4/4	keep	added step-by-step reasoning + visual examples
```

To calculate pass rates:

- SVR: count how many of {connectivity: true, wall_crossings: false, exhibit_collision: false, out_of_area_violations: false} are satisfied
- SCSR: count how many of {start_end_location: true, must_pass_regions: true, restricted_area_violations: false, distance_budget: true} are satisfied

## The Experiment Loop

The experiment runs on a dedicated branch (e.g. `prompt-opt/apr6`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune the prompts in `gpt_2_steps.py` with an experimental idea
3. git commit with a descriptive message
4. Run the experiment: `python gpt_2_steps.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read the validation results: `cat validation_results.json` or parse specific fields
6. If the script crashes, run `tail -n 50 run.log` to read the error and attempt a fix. If you can't fix after a few attempts, log as crash and move on
7. Record the results in `results.tsv` (NOTE: do not commit results.tsv, leave it untracked by git)
8. If validation improved (higher is_valid, lower total_violations, better pass rates), you "advance" the branch, keeping the git commit
9. If validation is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. You're advancing the branch so you can iterate. If you feel stuck, you can rewind but do this very sparingly.

**Timeout**: Each experiment should take ~30-60 seconds (GPT API call time + validation). If a run exceeds 5 minutes, kill it and treat it as a failure.

**Crashes**: If a run crashes (API error, JSON parse error, etc.), use judgment: If it's easy to fix (typo, formatting), fix and re-run. If the idea is fundamentally broken, log "crash" and move on.

**NEVER STOP**: Once the experiment loop has begun (after initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" The human might be asleep, and expects you to continue working indefinitely until manually stopped. You are autonomous. If you run out of ideas, think harder — try different prompt structures, add examples, emphasize critical constraints, try chain-of-thought, combine previous near-misses, etc. The loop runs until the human interrupts you, period.

## Prompt Engineering Strategies

Some ideas to try:

1. **Emphasis & Repetition**: Repeat critical constraints multiple times
2. **Structure**: Break complex instructions into numbered lists or sections
3. **Examples**: Provide concrete examples of valid/invalid paths
4. **Reasoning**: Ask the LLM to reason step-by-step before generating coordinates
5. **Visual cues**: Reference the color-coding explicitly and repeatedly
6. **Negative examples**: Explicitly state what NOT to do
7. **Constraints as requirements**: Frame constraints as success criteria
8. **Precision**: Be extremely specific about coordinate systems, boundaries, etc.
9. **Context**: Provide more context about why constraints matter
10. **Simplification**: Sometimes removing clutter makes instructions clearer

Remember: The only way to know if a prompt change works is to test it. Theory doesn't matter — only measured results.
