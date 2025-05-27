# Navigation

Implements navigation for the mobile robot. Uses the A* algorithm to find an optimal, smoothes the path using splines, and provides the control for reaching the goal. Parts of these scripts were developed as part of the CS 237A course at Stanford University.

### Scripts
- **navigator.py**: Implements the navigator.
- **localize_and_navigate.py**: First rotates the robot until localized, then proceeds with the navigator.
- **occupancy_grid_mapper.py**: Implements an occupancy grid mapper based on the Astra RGB-D Camera

### Launch
- **navigator.launch**: Launches the `navigator.py` file.
- **localize_and_navigate.launch**: Launches the `localize_and_navigate.py` file.
- **localize_and_navigate_combined_map.launch**: Launches `localize_and_navigate.py` and `occupancy_grid_mapper.py` files.

**Author**: Matthew Sato, Engineering Informatics Lab, Stanford University

**License**: This package is released under the [MIT license](LICENSE).