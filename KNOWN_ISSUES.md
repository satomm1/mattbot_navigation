# Known issues and limitations: observation re-checking (importance, detours, policy)

This file collects the limitations found while building obstacle importance, detour cost, the detour re-check policy, and its execution. It started on 2026-10-07.

Each item says what the issue is, why it matters, and a possible fix. Severity:
- **High**: wrong behaviour likely in normal use.
- **Med**: wrong or suboptimal in some situations.
- **Low**: an approximation, or cleanup.

Code references are relative to `mattbot_navigation/` unless another package is named.

---

## 1. Not yet verified on the robot

| # | Sev | Issue | Next step |
|---|---|---|---|
| 1.1 | High | Navigator detour execution has only been tested in a kinematic simulator: perfect TF and odometry, fake amcl, no camera, evaluator or people. Covered: full detour, abandonment by timeout, new goal during a detour. | Run the on-robot test in `README.md`. Also check: `/stop` during a detour, a viewpoint blocked by another ledger object, a person stop during OBSERVE, and lost localization during a detour. |
| 1.2 | Med | The navigator (`scripts/localize_and_navigate.py`) and mapper (`scripts/occupancy_grid_mapper.py`) have no unit tests. Detour state handling (`_active_goal`, `_start_detour`, `_abandon_detour`, `_cancel_detour`) is checked only by simulation. | Turn the scratch simulator (unicycle TF/odometry plus a fake `amcl` dynamic_reconfigure server) into a committed rostest. |
| 1.3 | Low | Reported timings were each measured once on the Jetson: roadmap build ~1 s, importance ~70 ms median per obstacle (up to ~190 ms), detour query ~15 ms, select service ~35 ms. | Measure properly before quoting in a paper. |

## 2. Opportunistic vs. detour interaction

| # | Sev | Issue | Possible fix |
|---|---|---|---|
| 2.1 | **High** | **Some objects are never checked.** An object counts as "checkable from the path" if any path point sees it, including the skip zones near the start (`skip_start_m`) and goal (`skip_goal_m`). Such objects are excluded from detours, but `select_stops` never places a stop in those zones. An object visible only near the start or goal is therefore never checked, by either mechanism. | Define "checkable" as visible from a *usable* path point. That means the same skip zones and cost limit (`max_cost_s`) that `select_stops` applies. |
| 2.2 | Med | The same gap occurs when an opportunistic stop exists geometrically but is dropped because `max_cost_s` is exceeded: the object is excluded from detours anyway. | Same fix as 2.1. |
| 2.3 | Med | The two mechanisms use different decision rules. Opportunistic stops still use a fixed belief threshold (`check_below_belief`, 0.5) and the value I_o·(1−b). Detours use the value-of-information rule. A visible object with b > 0.5 waits; a hidden one may be detoured to at any belief where V > C. | Gate opportunistic stops with the same V − C rule. Their C is just v·(turn + dwell). |
| 2.4 | Low | `ObservationStop.values` means different things by kind: importance·(1−b) for OPPORTUNISTIC, V in metres for DETOUR. This is documented in `mattbot_dds/msg/ObservationStop.msg`, but easy to misuse. | Add separate msg fields when the msg is next changed. |

## 3. Detour decision policy (`src/navigation_utils/observation_policy.py`)

| # | Sev | Issue | Possible fix |
|---|---|---|---|
| 3.1 | Med | N, the number of benefiting trips, is one constant (default 5). It doesn't scale with fleet size, time of day, or how soon someone would see the object anyway. | Use N = trip rate × time until the next expected observation. Reduce it when a peer's planned path (`/path_from_agent`) passes a viewpoint of the object. |
| 3.2 | Med | q, the probability a check is conclusive, is fixed at 1.0. INCONCLUSIVE outcomes are ignored. | Estimate it from `/observation/results`, per class and distance. |
| 3.3 | Med | p_gone = (1−b)/2 makes "unknown" mean 50/50. Some objects (furniture) rarely move, others (carts, people-moved items) often do. | Learn a per-class prior p₀ from PRESENT/ABSENT outcomes and use (1−b)(1−p₀). |
| 3.4 | Med | The model assumes the fleet routes around every ledger object until it is removed. That is only true with `ledger_blockout:=true`, which defaults to `observe_detour`. With blockout off, V describes a cost that isn't actually paid. | Warn, or refuse detours, when detours are on and ledger blockout is off. |
| 3.5 | Med | **No multi-robot coordination.** Two robots can detour for the same object at once, and the ledger has no claims. | Broadcast a short-lived claim on the object through the ledger or DDS before leaving the path. |
| 3.6 | Low | At most one detour per path (`max_detours_per_path`). Bundling only gathers candidates around *each object's own best* viewpoint. That is not a set cover, so a viewpoint seeing several objects but best for none can be missed. | Search over viewpoints, not objects, for the best combined value. Allow several detours under a time budget, ranked by V/C. |
| 3.7 | Low | D* (search limit) uses one object's value and zero turn. A bundle can justify a longer detour than any of its members' D*, so those detours are never found. | Raise the limit using nearby candidates' combined value, or run a second pass. |
| 3.8 | Low | Cost is distance only. It ignores mission urgency, battery, the robot's own deadlines, and passengers or deliveries. `margin_m` is the only knob. | Add a per-mission budget or urgency factor. |
| 3.9 | Low | Only the error "the fleet avoids an object that's gone" is modelled. Believing something absent that's actually present is out of scope for re-checking, by design. I_o_disc is computed but unused. | Fine for now. Revisit if objects are ever routed through at low belief. |
| 3.10 | Med | **No detour branches off at the start or goal.** A detour whose route leaves the path within `detour_skip_start_m` / `detour_skip_goal_m` (1 m) of the start or goal is never taken (`ThresholdPolicy.detour_blocked`, using `DetourResult.branch_arc_m`). An object whose only detours branch off at a patrol endpoint is therefore never checked on that patrol. In the sim, `detour_present` had to move its west waypoint past the passage. | Accept it for patrols. Otherwise let a robot that is idle at a goal check such objects as a separate short task. |

## 4. Detour distance (`src/navigation_utils/detour.py`)

| # | Sev | Issue | Possible fix |
|---|---|---|---|
| 4.1 | Med | The detour grid uses the static C-space only. Other ledger objects, people and peers are ignored, so a chosen viewpoint may be unreachable. The navigator then abandons the detour, after paying part of it. | Subtract current `/object_map` blockouts from the detour grid for each query. |
| 4.2 | Med | The round-robot C-space (radius `robot_clearance`, 0.4 m) differs from the navigator's square footprint (`robot_d` = 0.8 m, corners at 0.57 m). A viewpoint can be free for the planner but not for the navigator. Currently mitigated by snapping the goal to the nearest free cell within 0.4 m (`replan()`), which can cost line of sight. | Use the navigator's footprint, or a larger radius, when building C-space for detours. Re-check line of sight after snapping. |
| 4.3 | Low | A coarse 0.2 m cell is a viewpoint if *any* fine viewshed cell lies in it. That is optimistic by up to ~0.14 m, so the viewpoint can be slightly beyond `r_max`. | Require the cell centre to be in the viewshed, or sample the fine cell nearest the centre. |
| 4.4 | Low | 8-connected grid distances overestimate true path length by up to ~8%. The navigator's A* plus spline differs again. | Use any-angle distances (e.g. a fast-marching method) if accuracy matters. |
| 4.5 | Low | Path points off the C-space are snapped within 0.5 m or dropped. If the goal isn't in C-space, no detours are computed for that path at all. | Snap the goal with a larger radius. |
| 4.6 | Low | Leave-point recovery is a tie-breaking heuristic: the last route cell with F ≥ arc − 0.3 m. Odd path shapes could pick a slightly early or late leave point. | Fine for now. |
| 4.7 | Low | Viewsheds use 2D line of sight on the static map plus known objects as squares. They ignore people, glass, object height versus camera height, and the camera's field of view. The evaluator's `max_range_m` (4.5 m) is larger than `r_max` (3.5 m), which is consistent. | Add the camera FOV and height model if checks often turn out INCONCLUSIVE. |

## 5. Obstacle importance (`roadmap.py`, `edge_blocking.py`, `obstacle_importance.py`)

| # | Sev | Issue | Possible fix |
|---|---|---|---|
| 5.1 | Med | **`/map_mod` is ignored.** The roadmap, viewsheds and detour grid are built from `/map` only, while navigation also uses `/map_mod` (manual edits). Passages closed in `map_mod` still count as open. | Build from max(`/map`, `/map_mod`), and include `map_mod` in the fingerprint. |
| 5.2 | Med | Each obstacle is evaluated alone against the obstacle-free map, which is the user's chosen semantics. Two obstacles that only cut off an area together each get a low I_o. | Optionally compute a "joint" importance for clusters of nearby obstacles. |
| 5.3 | Med | The trip set defaults to 50 random roadmap nodes (fixed seed). `config/landmarks.example.yaml` holds placeholder points, and weighted trips from fleet history are only a TODO (`obstacle_importance.py`, `mattbot_database` goals table). Two landmarks can snap to the same node. | Create real landmark files for each robot and map. Implement the history-based weights. |
| 5.4 | Med | Footprints are axis-aligned squares of the *largest* width ever seen, centred on the *mean* sighting. Orientation and shape are ignored, and the width only grows. | Store an oriented box or polygon in the ledger, since `footprint_cells(polygon=...)` already exists. |
| 5.5 | Low | D_max = 50 m (cost of a trip cut off entirely) is arbitrary, and dead-end obstacles are dominated by it. | Calibrate it, e.g. as the cost of a failed mission. |
| 5.6 | Low | I_o_disc assumes the robot discovers the obstacle only at the vertex before the blocked edge. Real robots see it earlier, about 4.5 m away, so this is pessimistic. | Model discovery at the first path point within sensing range of the obstacle. |
| 5.7 | Low | Only the largest connected roadmap component is kept. Smaller free regions are dropped with a warning: on the current map, a 12-node component near (3, 2.5) and a few tiny ones. Objects there get I_o from snapping alone. | Inspect the dropped regions on each map. Keep components above a size threshold. |
| 5.8 | Low | Coarse 0.2 m grid: clearance is sampled at the cell centre, which is accurate to about ±0.1 m. Doorways near 2 × `robot_clearance` may be opened or closed wrongly. The split-region warning only fires for pieces of at least 1 m². | Use `~roadmap_robot_radius` per map, or 0.1 m resolution on maps with tight doors. |
| 5.9 | Low | The edge-blocking cluster/arm heuristic is approximate. Diagonal squeezes between an obstacle corner and a wall corner can count as passable at coarse resolution. A 0.4 m cart at a 1.6 m T counts as passable, which matches the geometry but is marginal for a real robot. | Add a small safety margin to the inflation used for blocking. |
| 5.10 | Low | Each robot computes importance on its own map and landmarks. Values differ between robots, and the work is duplicated. | Compute on the reference map and share importance through the ledger. |

## 6. Navigator execution (`scripts/localize_and_navigate.py`)

| # | Sev | Issue | Possible fix |
|---|---|---|---|
| 6.1 | Med | A new goal resets the per-goal bookkeeping (`_observed_this_goal`, `_detour_skipped`). A detour cancelled by a new goal can be offered again right away; this was seen in simulation. Cooldown only starts after a completed check. | Keep an "offered or abandoned" cooldown across goals in the planner. |
| 6.2 | Med | Arrival at the viewpoint uses `near_thresh` (0.35 m). That's coarse, and with the 0.4 m snapping (4.2) the robot may look from up to ~0.75 m away from the planned viewpoint. | Check line of sight from the actual pose before OBSERVE_TURN, or use a tighter arrival threshold for detours. |
| 6.3 | Low | The abandonment deadline (2 × estimated time + 20 s) is a heuristic. The estimate (`cost_s`) assumes cruising speed without alignment pauses. | Calibrate it from on-robot runs. |
| 6.4 | Low | `path_still_valid` now ignores path points within `robot_d` of the robot, so a ledger object appearing right next to it doesn't cause stop/replan loops. A real new blockout within 0.8 m ahead is also ignored by this check, though other safety layers such as person stops still apply. | Ignore only blockouts that already overlapped the robot when they appeared. |
| 6.5 | Low | Detours (and all observation) are disabled during multi-agent missions (`localize_and_navigate_multi_agent.py`, `_observation_allowed`). | Integrate with the multi-agent timing first, if wanted. |

**Pre-existing navigator issues noticed along the way.** None of these were changed.
- `replan()` failure loop: outside IDLE and STOPPED_FOR_AGENT, a failed `solve()` loops again, and its `else` branch reads `problem.path` as if planning had succeeded.
- `cmd_nav_callback` cannot interrupt TRACK, because `replan()` returns early in TRACK.
- The goal check in `external_goal_callback` and the post-recovery start check use the static `occupancy` only, so a goal inside an object blockout is accepted.
- `map_callback` builds `occupancy` only once; later `/navigation_map` updates are ignored (marked FIXME in the code).
- `aligned_goal` is defined twice.
- The "Path too short to track" branch parks at a stale `park_x/y` (the refresh line is commented out).
- The second planning failure clears `x_g/y_g/theta_g`. Detours are guarded against this, but other callers aren't.

## 7. Mapper and ledger blockout (`scripts/occupancy_grid_mapper.py`)

| # | Sev | Issue | Possible fix |
|---|---|---|---|
| 7.1 | Med | The map origin is still not subtracted in `map_update_callback`, `person_callback`, or the moving-person window. Only `_detected_object_blockout_indices` was fixed. This is harmless while maps have origin (0, 0); `y2e2_5_5_25.yaml` has (−99.6, −62.8). | Apply the same origin fix in those places. |
| 7.2 | Med | Ledger blockout keeps every object with b ≥ 0 blocked forever, until someone sees it gone. Objects that are never re-checked, e.g. ones visible only near path ends (2.1), stay blocked indefinitely. | Fix 2.1. Optionally expire very old unknown objects. |
| 7.3 | Low | Ledger footprints use the object's full width with no cap, while confirmed-object blockouts are capped at `max_object_blockout_width` (0.5 m). The robot's own detections appear in both layers (max-merged, harmless). | Decide on one width policy. |
| 7.4 | Low | `/navigation_map` is only republished at start-up, on `/localized`, and from depth callbacks. `/map_update` edits and object changes never reach it. The navigator ignores updates anyway (see map_callback above). | Republish on change, together with the navigator FIXME. |
| 7.5 | Low | TTL expiry of confirmed-object blockouts only runs while the robot is IDLE. This is pre-existing behaviour. | Fine for now. |
| 7.6 | Low | A robot standing next to a newly blocked ledger object relies on start snapping within `nearest_free_search_radius` (1 m). Otherwise it goes IDLE. | Ignore blockout cells under the robot's own footprint. |

## 8. Visualization, tools and housekeeping

| # | Sev | Issue | Possible fix |
|---|---|---|---|
| 8.1 | Low | Detour and importance PNGs are drawn on the planner's worker thread. Python's global interpreter lock once slowed a `/observation/select` call to ~170 ms while a PNG was being drawn. | Draw in a subprocess, or reduce the rate (`~*_png_min_interval_s`). |
| 8.2 | Low | `plot_observation_detours.py` omits objects with D* ≤ 0 from the PNG (they appear only in the table). | Draw them as "never worth a detour". |
| 8.3 | Low | Example figures in the workspace root are not in any repo and can be deleted: `importance_example.png`, `detour_example.png`, `detour_decision_example.png`. | Delete when no longer needed. |
| 8.4 | Low | The `TripSet.from_pairs` hook ("explicit" trips) is unused, because live fleet trips were rejected by design. | Keep or remove. |
| 8.5 | Low | `startup_script.py` has a local fix (written into robot_env.sh) that is overwritten on the next software update unless it is also made in dds_robot_platform. | Make the same fix in dds_robot_platform. |
| 8.6 | Low | Stale tests in `path_planning`: `tests/test_a_star.py` imports the old `path_planning.*` name, and `tests/test_subsample_path_stride.py` imports the missing `social_path_planning.path_subsample`. | Fix or delete them. |

## 9. Caveats for the paper write-up

- **The example numbers come from simulation.** They are from one offline run and one simulated run with an idealized robot (north corridor: predicted 13.8 m extra, driven 13 m). Replace them with robot results or label them as simulation.
- **State the assumptions.** The discovery-cost assumption (5.6), N as a tuned constant (3.1), fixed q (3.2), the lack of coordination (3.5) and the one-detour limit (3.6) should be listed as limitations.
