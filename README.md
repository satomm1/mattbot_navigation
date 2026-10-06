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

### Obstacle importance I_o

`navigation_utils/roadmap.py`, `edge_blocking.py`, `obstacle_importance.py`, `importance_viz.py` (pure Python, no ROS):

- **Roadmap**: sparse topological graph of the free space of the static `/map` (0.2 m grid, C-space inflated by `robot_clearance`, skeleton junctions/endpoints/corridor nodes, lattice nodes in open areas). Built once per map and cached in `~/.ros/mattbot_roadmap/` keyed by a hash of the map and parameters.
- **Edge blocking**: an obstacle blocks a roadmap edge only if the robot cannot get past it inside the edge's corridor (a cart against one wall of a wide hallway blocks nothing).
- **I_o** (m per trip) = expected extra travel distance over a trip set if the obstacle is there: all landmark pairs (`config/landmarks.example.yaml`, or 50 random roadmap nodes) or pairs weighted by a CSV of counts. Also reported: `I_o_disc` (cost when the obstacle is only discovered on arrival), the fraction of trips affected, and the mean detour of those.
- **Computed once**: each obstacle is evaluated alone against the obstacle-free roadmap; other obstacles and belief never change it. It is recomputed only if its own footprint changes the set of blocked edges.
- **Outputs** (`observe_importance:=true`): `/observation/roadmap` and `/observation/importance` (RViz MarkerArrays), `~/.ros/mattbot_roadmap/importance_latest.png` and `.json`.

```bash
rosrun mattbot_navigation plot_obstacle_importance.py --map $(rospack find mattbot_mcl)/map_json/current_map.json \
    --objects objects.csv --out importance.png          # objects.csv rows: x,y,width[,id] (local map frame)
python3 -m pytest mattbot_navigation/test               # from src/
```

### Launch

- **localize_and_navigate_combined_map_short.launch** / **localize_and_navigate_combined_map_tall.launch**: `occupancy_grid_mapper.py` + `localize_and_navigate.py` (+ optional `patrol.py`).
- **localize_and_navigate_multi_agent_short.launch**: Same stack with `localize_and_navigate_multi_agent.py`.
- **localize_and_occupancy_mapper_short.launch** / **localize_and_occupancy_mapper_tall.launch**: Mapper + `localize_and_map.py`.

**Author**: Matthew Sato, Engineering Informatics Lab, Stanford University

**License**: This package is released under the [MIT license](LICENSE).
