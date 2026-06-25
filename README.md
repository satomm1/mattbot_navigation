# Navigation

Implements navigation for the mobile robot. Uses A* to plan paths, smooths them with splines, and tracks them with differential-flatness control. Parts of these scripts were developed as part of the CS 237A course at Stanford University.

### Scripts

- **localize_and_navigate.py**: Primary navigator (localize, ALIGN/TRACK/PARK, replanning, social A* optional).
- **localize_and_navigate_multi_agent.py**: Extends the primary navigator with coordinated multi-robot timing and DDS planned-path handoff. Optional distributed collision-pair detection (`~multi_agent_distributed_constraints`).
- **localize_and_map.py**: Navigator variant with mapping/localization workflow.
- **occupancy_grid_mapper.py**: Builds `/navigation_map` from SLAM + dynamic layers; optional depth occupancy grid via `enable_depth_occupancy_grid` ROS param (default false; set from bringup launch files).
- **patrol.py**: Patrol behavior (optional in combined-map launches).

### Launch

- **localize_and_navigate_combined_map_short.launch** / **localize_and_navigate_combined_map_tall.launch**: `occupancy_grid_mapper.py` + `localize_and_navigate.py` (+ optional `patrol.py`).
- **localize_and_navigate_multi_agent_short.launch**: Same stack with `localize_and_navigate_multi_agent.py`.
- **localize_and_occupancy_mapper_short.launch** / **localize_and_occupancy_mapper_tall.launch**: Mapper + `localize_and_map.py`.

**Author**: Matthew Sato, Engineering Informatics Lab, Stanford University

**License**: This package is released under the [MIT license](LICENSE).
