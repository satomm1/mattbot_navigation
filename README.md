# Navigation

Implements navigation for the mobile robot. Uses A* to plan paths, smooths them with splines, and tracks them with differential-flatness control. Parts of these scripts were developed as part of the CS 237A course at Stanford University.

### Scripts

- **localize_and_navigate.py**: Primary navigator (localize, ALIGN/TRACK/PARK, replanning, social A* optional).
- **localize_and_navigate_multi_agent.py**: Extends the primary navigator with coordinated multi-robot timing and DDS planned-path handoff. Optional distributed collision-pair detection (`~multi_agent_distributed_constraints`).
- **localize_and_map.py**: Navigator variant with mapping/localization workflow.
- **occupancy_grid_mapper.py**: Builds `/navigation_map` from SLAM + dynamic layers; optional depth occupancy grid via `enable_depth_occupancy_grid` ROS param (default false; set from bringup launch files).
- **patrol.py**: Patrol behavior (optional in combined-map launches).
- **observation_planner.py**: Picks stops along the planned path to re-check ledger objects (`observe:=true`). With `observe_importance:=true` it values each object by its obstacle importance I_o (below).
- **plot_obstacle_importance.py**: Offline I_o for a map and a CSV of objects; prints a table and writes a PNG.
- **plot_observation_detours.py**: Offline detour decisions (detour, V, C, GO/skip) for a path (or start/goal) and a CSV of objects; prints a table and writes a PNG.

### Obstacle importance I_o

`navigation_utils/roadmap.py`, `edge_blocking.py`, `obstacle_importance.py`, `importance_viz.py` (pure Python, no ROS):

- **Roadmap**: sparse topological graph of the free space of the static `/map` (0.2 m grid, C-space inflated by `robot_clearance`, skeleton junctions/endpoints/corridor nodes, lattice nodes in open areas). Built once per map and cached in `~/.ros/mattbot_roadmap/` keyed by a hash of the map and parameters.
- **Edge blocking**: an obstacle blocks a roadmap edge only if the robot cannot get past it inside the edge's corridor (a cart against one wall of a wide hallway blocks nothing).
- **I_o** (m per trip) = expected extra travel distance over a trip set if the obstacle is there: all landmark pairs (`config/landmarks.example.yaml`, or 50 random roadmap nodes) or pairs weighted by a CSV of counts. Also reported: `I_o_disc` (cost when the obstacle is only discovered on arrival), the fraction of trips affected, and the mean detour of those.
- **Computed once**: each obstacle is evaluated alone against the obstacle-free roadmap; other obstacles and belief never change it. It is recomputed only if its own footprint changes the set of blocked edges.
- **Outputs** (`observe_importance:=true`): `/observation/roadmap` and `/observation/importance` (RViz MarkerArrays), `~/.ros/mattbot_roadmap/importance_latest.png` and `.json`.

### Endpoint checks

On-path stops are never placed within `skip_start_m` (0.5 m) of the start or `skip_goal_m` (0.8 m) of the goal. An eligible object that gets no on-path stop is checked while the robot is stopped anyway:
- **AT_START**: from the start pose before ALIGN. Only on the first plan for a goal (`SelectObservations.allow_start_check`), never on a mid-trip replan.
- **AT_GOAL**: from the park pose, after PARK_POSE and before the final heading turn (which is also the turn back). `patrol.py` counts a waypoint as reached on the IDLE after parking, so its wait starts after these checks.

The cost is only the turn and dwell beyond the turn the robot makes anyway. A bundle over `max_cost_s` drops its least valuable targets. Both kinds are `ObservationStop` kinds (`mattbot_dds`), so the planner and navigator must be deployed together.

### Observation detours

`navigation_utils/detour.py`: extra driving needed to see an object that **no opportunistic or endpoint stop serves**: it got no stop on this path, or (belief still above `check_below_belief`) no usable path point, start pose or park pose sees it. Each object's viewshed (the same ray casting as opportunistic stops) is mapped to reachable C-space cells; the robot leaves the path anywhere, drives to a viewpoint, looks, and replans to the goal. detour(v) = F(v) + G(v) - L, with F one Dijkstra from the whole path (offset by distance driven along it) and G one from the goal, so a path query costs two grid Dijkstras for all objects (~15 ms on the y2e2 map).

**Decision** (`observation_policy.py`, value of information, all in metres): for a ledger object with belief b in [0, 1],
`p_gone = (1 - b)/2`, `V = q * N * p_gone * I_o` (driving the fleet saves over the next N trips if the check finds it gone), `C = detour + v_cruise * (turn + dwell)`. A detour is taken if `V - C > margin`, at most one per path (objects seen from the same viewpoint are bundled; their V adds). `D* = V - v_cruise * dwell` bounds each object's search; objects with D* <= 0 (just seen, or blocking nothing) are never searched.

**Execution** (`observe_detour:=true`): the observation planner returns the chosen DETOUR stop (`path_index` = where to leave the path, `x, y` = viewpoint); the navigator drives to the viewpoint as a sub-goal (the real goal is never overwritten), observes, then replans to the goal. Arriving counts within `near_thresh` (0.35 m), but a viewpoint is only guaranteed to see the object within 0.1 m of it (`viewpoints.LOS_POSE_TOLERANCE_M`; beside a wall corner the view is lost a few centimetres further), so a robot further than `~observe_viewpoint_tol_m` (0.1 m) first parks on the viewpoint (`PARK_POSE`, at most 15 s) and then looks. A detour is abandoned (and not offered again for this goal) if the viewpoint cannot be planned to or it takes longer than `~observe_detour_timeout_factor` x the planned drive to the viewpoint (plus the align pause and the looks; never less than the factor x its estimated extra time) + 20 s; a new goal or `/stop` cancels it. Detour (and scout) distances are computed on the driving grid: max(`/map`, `/map_mod`), with the cells the navigator currently avoids (`/object_map`, inflated by the robot radius) left out, so a route through a passage another object closes, or a viewpoint inside a blockout, is never chosen (`detour.BlockedPlanners`). `observe_detour` also turns on importance and, by default, `ledger_blockout`: the occupancy grid mapper adds every ledger object to `/object_map` until it is removed, which is what the value model assumes. Tune with `observe_detour_n_trips` (N, default 5), `observe_detour_margin_m` and `observe_max_detour_m` (hard cap, 15 m). Outputs: `/observation/detours` (V, C, GO/skip per candidate) and `~/.ros/mattbot_roadmap/detour_latest.png` / `.json`.

```bash
rosrun mattbot_navigation plot_observation_detours.py --map $(rospack find mattbot_mcl)/map_json/current_map.json \
    --objects objects.csv --start 10,18.9 --goal 80,18.9 --belief 0.0 --n-trips 5 --out detours.png
rosrun mattbot_navigation plot_obstacle_importance.py --map $(rospack find mattbot_mcl)/map_json/current_map.json \
    --objects objects.csv --out importance.png          # objects.csv rows: x,y,width[,id] (local map frame)
python3 -m pytest mattbot_navigation/test               # from src/
```

### Scout re-observation

`navigation_utils/scout.py`: a robot with **no goal of its own** drives out only to check an object. It shares the detour C-space grid; one Dijkstra from the robot gives the distance to every object's nearest viewpoint (`to_view_m`; `return_m` back to the start is the same on the symmetric grid). The value is the detour V with its own N and q (`~scout_n_trips`, `~scout_conclusive_prob`). The cost C comes from a pluggable `ScoutCostModel` (`observation_policy.py`), selected by name with `~scout_cost_model` and built with the kwargs in `~scout_cost_params`. The default model `travel_time` adds the opportunity cost of being away to the physical cost, all in metres (times converted at `v_cruise`):

```
travel = to_view (+ return if round_trip)          T_away = travel / v + turn / turn_rate + dwell * n   (s)
C      = travel + v * (turn / turn_rate + dwell * n) + v * fixed_s + v * per_s_away * T_away
```

`fixed_s` is the cost of committing to a scout at all and `per_s_away` the cost per second the robot is unavailable (both 0 until calibrated; launch args `observe_scout_fixed_s`, `observe_scout_per_s_away`, and `observe_scout_round_trip` in the navigation launch files). Objects are searched only up to the distance at which even the cheapest scout (one target, no turn) costs V. A scout is worth sending if `V - C > ~scout_margin_m`, at most `~max_scouts` (default 1) per evaluation. Each `ScoutCandidate` reports `opportunity_m` and `t_away_s`.

With `observe_scout:=true` (needs `ledger:=true observe:=true`; turns on importance and, by default, `ledger_blockout`), the observation planner evaluates scouts every `~scout_period_s` (5 s) once `/robot_mode` has been IDLE for `~scout_min_idle_s` (10 s), from the TF pose `map -> base_footprint`. Outputs: `/observation/scout` (`mattbot_dds/ScoutDecision`, latched: every candidate with V, C, GO/skip, and `go`) and `/observation/scout_markers`. Evaluate from any pose with:

```bash
rosservice call /observation/scout_query "{start: {x: 2.0, y: 1.8, theta: 0.0}, require_idle: false}"
```

**Which robots scout:** `scout_robot_heights` (default `short`) lists the `ROBOT_HEIGHT` values allowed to scout, e.g. `scout_robot_heights:=short,tall`; `any` (or empty) allows every robot. On any other robot the planner logs `scouts off on this robot (...)` and evaluates and publishes no scouts. A robot whose `ROBOT_HEIGHT` is unset never matches a restricted list.

**Execution** (`observe_scout_execute:=true`, default off): the navigator drives a GO decision when it is IDLE, has no goal, and has had none received or active for `observe_scout_min_idle_s` (also the planner's idle time; raise it above patrol pauses). A decision is used once, only while fresh (`~scout_decision_max_age_s`, 12 s) and only if it was evaluated within 0.5 m of the robot. The scout is a goal of its own:
- It drives to the viewpoint (`"Scout to (x, y) to check X (est. N s)"`), decides what is still due, parks on the viewpoint (`~observe_viewpoint_tol_m`, 0.1 m, at most 15 s) and looks (`SCOUT` stops on `/observation/events`), as at a detour viewpoint (`"Reached scout viewpoint"`, `"Scout done"`).
- Then it drives back to where it started and turns to its old heading (`"Back from scout"`; `observe_scout_round_trip`, default true). The no-goal time starts again there.
- Nothing is due on arrival (e.g. the camera re-sighted the object on the way): `"Scout done (targets no longer due)"`, and back.
- No on-path stops on either leg; every object may be checked once on the scout, and the next real goal starts fresh.
- Times come from the decision's metres at `/cruising_velocity`. Deadline of the way out, as for a detour: `observe_detour_timeout_factor` x (the drive + align pause + dwell per target) + 20 s, never less than the estimate.
- **Cancelled** (`"Scout cancelled (new goal)"` / `(stop)`): any goal (`/external_goal`, `/voice_goal`, `/cmd_nav`, RViz), including the one the robot had before, or `/stop`; the robot takes the new goal or stays.
- **Abandoned on the way out** (`"Scout abandoned (reason); returning"`): no path or no free start cell, viewpoint not free (snapped up to 0.4 m), deadline passed, or stuck in IDLE for 3 s (TF loss, failed recovery). It drives back.
- **Return failed** (`"Scout return failed (reason); staying here"`): no path back; the robot stays.
- While on the way out it announces its check (`SCOUT`, committed) and gives way en route as below (`"Scout to check X yielded to robot N"`, then back).

Known limitations and open issues of importance, detours, scouts and the re-check policy are tracked in [KNOWN_ISSUES.md](KNOWN_ISSUES.md).

### Check timing and revalidation

The rules are evaluated for the time a check would happen, not for when the path is requested. The navigator sends the trajectory time of every request pose (`SelectObservations.times`). The planner predicts each object's belief with the belief model of `/object_beliefs` (hold `hold_s` after `last_stamp`, then linear decay over `decay_s`):
- An object gets an on-path stop only where it will be due when the robot gets there (`ThresholdPolicy.t_due`: belief ≤ `check_below_belief` and cooldown over). An object that decays while the robot drives is therefore checked on this leg, not the next one.
- AT_GOAL takes objects due on arrival. Detour and scout values V use the belief at the moment the robot would look.
- `check_etas` gives each stop's check time: trajectory time, plus the cost of the checks before it, plus leave point → viewpoint for a detour.

When the robot gets to a stop, it asks `/observation/revalidate` (`mattbot_dds/RevalidateStop`) whether the stop still passes the rules with the current ledger, evaluated at max(now, planned time). An object that was re-sighted (own camera or a peer), removed, or checked in the meantime is skipped (`"Skipping check of X (reason)"`). A detour is also dropped at its leave point if no target is still worth it, and its viewpoint check is skipped if nothing is left. If the service cannot be called, the robot does the check.

### Fleet coordination

Robots announce their planned checks so that two robots do not check the same object at once. The earliest check wins.
- **Announcing:** the navigator publishes its complete current set (`/observation/intents_local`, `mattbot_dds/ObservationIntentArray`: object, kind, ETA, committed) on every change and at 1 Hz. The ledger node sends it to the other robots on DDS `IntentTopic{id}` every 2 s and on change. ETAs travel as seconds from the send time, so robots need no synced clocks. It also publishes every live intent of the fleet on `/observation/intents`. A robot whose set has not been refreshed for 10 s is forgotten.
- **Who gives way** (`observation_policy.yields_to`):
  - **Opportunistic and endpoint checks** never give way. They are skipped only by revalidation.
  - **A detour** gives way to another robot's check of any kind that is earlier by more than `observe_intent_margin_s` (10 s). Within the margin, a committed check (en route or observing) beats an uncommitted one, then the lower robot id wins. This is tested at plan time (`skip (robot N checks first)`), every cycle before the leave point, and while driving to the viewpoint (`"Detour to check X yielded to robot N"`, then the robot replans to its goal). It no longer gives way within `observe_intent_commit_s` (15 s) of the viewpoint.
  - **A scout**, the most expensive check, gives way to any check that is not later by more than the margin, unless that check is also a scout and its robot has a higher id. This blocks scout decisions (`robot N checks first`) and, with `observe_scout_execute`, a scout on its way out (it drives back). Two idle robots that decide in the same second may both set off; the en-route test sends one back once their announcements arrive (1–2 s).
- A yielded object is not offered again while the winner still plans its check. If the winner withdraws, a pending detour comes back at once.
- Launch args: `observe_coordinate` (default true; false = no announcing or giving way), `observe_intent_margin_s`, `observe_intent_commit_s`.
- All robots must run this version. A robot with the old stack announces nothing, so newer robots never give way to it.

### Launch

- **localize_and_navigate_combined_map_short.launch** / **localize_and_navigate_combined_map_tall.launch**: `occupancy_grid_mapper.py` + `localize_and_navigate.py` (+ optional `patrol.py`).
- **localize_and_navigate_multi_agent_short.launch**: Same stack with `localize_and_navigate_multi_agent.py`.
- **localize_and_occupancy_mapper_short.launch** / **localize_and_occupancy_mapper_tall.launch**: Mapper + `localize_and_map.py`.

**Author**: Matthew Sato, Engineering Informatics Lab, Stanford University

**License**: This package is released under the [MIT license](LICENSE).
