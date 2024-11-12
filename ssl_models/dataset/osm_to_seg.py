"""OSM to segmentation module."""

import json
from pathlib import Path

import cv2
import numpy as np


def map_category_to_meta(category: str) -> int:
    """Encode categories."""
    mapping = {
        # Buildings and Structures
        "building": 0,
        "flagpole": 0,
        "lighthouse": 0,
        "obelisk": 0,
        "observatory": 0,
        # Transportation Infrastructure
        "aerialway_pylon": 1,
        "airport": 1,
        "gas_station": 1,
        "helipad": 1,
        "parking": 1,
        "road": 1,
        "runway": 1,
        "taxiway": 1,
        # Industrial and Energy Infrastructure (chimneys go here)
        "chimney": 2,
        "petroleum_well": 2,
        "power_plant": 2,
        "power_substation": 2,
        "power_tower": 2,
        "satellite_dish": 2,
        "silo": 2,
        "storage_tank": 2,
        "wind_turbine": 2,
        "works": 2,
        # Water Features
        "river": 3,
        "fountain": 3,
        # Other
        "leisure": 4,
    }
    return mapping.get(category, 5)


def process_geojson_file(path: str | Path) -> np.ndarray:  # noqa: C901
    """Create raster from geojson file."""
    image = np.zeros((512, 512), dtype=np.uint8)

    with Path(path).open(encoding="utf-8") as f:
        data = json.load(f)
        if not data:
            return image
        for feature in data.get("features", []):
            properties = feature.get("properties", {})
            category = properties.get("category")

            # Map category to meta category value
            meta_value = map_category_to_meta(category)

            geometry = feature.get("geometry", {})
            geom_type = geometry.get("type")
            coordinates = geometry.get("coordinates", [])

            match geom_type:
                case "LineString":
                    # Draw lines on the image
                    points = np.array(coordinates, dtype=np.int32)
                    # Ensure points are within image bounds
                    points = np.clip(points, 0, 511)
                    # Draw the line with thickness 1
                    cv2.polylines(
                        image,
                        [points],
                        isClosed=False,
                        color=meta_value,
                        thickness=2,
                    )

                case "MultiLineString":
                    # Draw multiple lines
                    for line in coordinates:
                        points = np.array(line, dtype=np.int32)
                        points = np.clip(points, 0, 511)
                        cv2.polylines(
                            image,
                            [points],
                            isClosed=False,
                            color=meta_value,
                            thickness=2,
                        )

                case "Polygon":
                    # Draw filled polygon on the image
                    # Polygons may have multiple rings ; \
                    # coordinates[0] is the exterior ring
                    for polygon in coordinates:
                        points = np.array(polygon, dtype=np.int32)
                        # Ensure points are within image bounds
                        points = np.clip(points, 0, 511)
                        if np.size(points) != 0:
                            cv2.fillPoly(image, [points], color=meta_value)

                case "MultiPolygon":
                    # Draw multiple polygons
                    for multipolygon in coordinates:
                        for polygon in multipolygon:
                            points = np.array(polygon, dtype=np.int32)
                            points = np.clip(points, 0, 511)
                            if np.size(points) != 0:
                                cv2.fillPoly(image, [points], color=meta_value)

                case "Point":
                    # Draw a point on the image
                    point = np.array(coordinates, dtype=np.int32)
                    point = np.clip(point, 0, 511)
                    # Draw a small circle to represent the point
                    cv2.circle(
                        image,
                        tuple(point),
                        radius=1,
                        color=meta_value,
                        thickness=-1,
                    )

                case "MultiPoint":
                    # Draw multiple points
                    for coord in coordinates:
                        point = np.array(coord, dtype=np.int32)
                        point = np.clip(point, 0, 511)
                        cv2.circle(
                            image,
                            tuple(point),
                            radius=1,
                            color=meta_value,
                            thickness=-1,
                        )

                case _:
                    msg = f"Unsupported geometry type '{geom_type}' in file {path}"
                    raise ValueError(msg)

    return image
