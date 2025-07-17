import numpy as np
from shapely.geometry import Point, Polygon

def load_zones_from_file(filename="/workspace/catkin_ws/src/mattbot_navigation/scripts/zones.txt"):
    """
    Load zones from a text file.
    Each line in the file should contain a zone name followed by its polygon vertices.
    Example: "zone1 0 0 1 0 1 1 0 1"
                    x y x y x y x y
    """
    zones = {}
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            parts = line.split()
            zone_name = parts[0]
            # Extract coordinate pairs
            coords = [(float(parts[i]), float(parts[i+1])) 
                      for i in range(1, len(parts), 2)]
            
            zones[zone_name] = Polygon(coords)
    return zones

def get_zone_for_point(point, zones):
    """
    Get the zone name for a given point.
    
    :param point: A tuple (x, y) representing the point coordinates.
    :param zones: A dictionary of zones where keys are zone names and values are shapely Polygon objects.
    :return: The name of the zone that contains the point, or None if the point is not in any zone.
    """
    point = Point(point)
    for zone_name, polygon in zones.items():
        if polygon.contains(point):
            return zone_name
    return None