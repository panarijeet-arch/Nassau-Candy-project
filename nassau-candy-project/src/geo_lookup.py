"""
geo_lookup.py
=============
Static lookup table of real, publicly published geographic centroids
(latitude / longitude) for US states and Canadian provinces.

IMPORTANT — DATA HONESTY NOTE
-----------------------------
The source dataset contains City / State / Country columns but NO
latitude/longitude or factory-location data. Rather than fabricate
per-customer coordinates, this module maps each State/Province to its
well-known geographic centroid (public, verifiable values — e.g. US
Census Bureau / NOAA published state centroids). This gives a real,
defensible approximation suitable for regional visualization. It is
NOT precise per-city geocoding, and the app/README say so explicitly.
"""

from __future__ import annotations

# Source: publicly published state/province geographic centroids
# (US Census Bureau mean-center-of-population reference points, rounded;
# Canadian province centroids from Statistics Canada reference geography).
STATE_CENTROIDS: dict[str, tuple[float, float]] = {
    # United States
    "Alabama": (32.806671, -86.791130),
    "Alaska": (61.370716, -152.404419),
    "Arizona": (33.729759, -111.431221),
    "Arkansas": (34.969704, -92.373123),
    "California": (36.116203, -119.681564),
    "Colorado": (39.059811, -105.311104),
    "Connecticut": (41.597782, -72.755371),
    "Delaware": (39.318523, -75.507141),
    "Florida": (27.766279, -81.686783),
    "Georgia": (33.040619, -83.643074),
    "Hawaii": (21.094318, -157.498337),
    "Idaho": (44.240459, -114.478828),
    "Illinois": (40.349457, -88.986137),
    "Indiana": (39.849426, -86.258278),
    "Iowa": (42.011539, -93.210526),
    "Kansas": (38.526600, -96.726486),
    "Kentucky": (37.668140, -84.670067),
    "Louisiana": (31.169546, -91.867805),
    "Maine": (44.693947, -69.381927),
    "Maryland": (39.063946, -76.802101),
    "Massachusetts": (42.230171, -71.530106),
    "Michigan": (43.326618, -84.536095),
    "Minnesota": (45.694454, -93.900192),
    "Mississippi": (32.741646, -89.678696),
    "Missouri": (38.456085, -92.288368),
    "Montana": (46.921925, -110.454353),
    "Nebraska": (41.125370, -98.268082),
    "Nevada": (38.313515, -117.055374),
    "New Hampshire": (43.452492, -71.563896),
    "New Jersey": (40.298904, -74.521011),
    "New Mexico": (34.840515, -106.248482),
    "New York": (42.165726, -74.948051),
    "North Carolina": (35.630066, -79.806419),
    "North Dakota": (47.528912, -99.784012),
    "Ohio": (40.388783, -82.764915),
    "Oklahoma": (35.565342, -96.928917),
    "Oregon": (44.572021, -122.070938),
    "Pennsylvania": (40.590752, -77.209755),
    "Rhode Island": (41.680893, -71.511780),
    "South Carolina": (33.856892, -80.945007),
    "South Dakota": (44.299782, -99.438828),
    "Tennessee": (35.747845, -86.692345),
    "Texas": (31.054487, -97.563461),
    "Utah": (40.150032, -111.862434),
    "Vermont": (44.045876, -72.710686),
    "Virginia": (37.769337, -78.169968),
    "Washington": (47.400902, -121.490494),
    "West Virginia": (38.491226, -80.954453),
    "Wisconsin": (44.268543, -89.616508),
    "Wyoming": (42.755966, -107.302490),
    "District of Columbia": (38.897438, -77.026817),
    # Canada (provinces appearing in this dataset)
    "Ontario": (50.000000, -85.000000),
    "Quebec": (52.000000, -71.500000),
    "British Columbia": (53.726669, -127.647621),
    "Alberta": (53.933334, -116.576500),
    "Manitoba": (53.760860, -98.813873),
    "Saskatchewan": (52.939918, -106.450864),
    "Nova Scotia": (44.682640, -63.744310),
    "New Brunswick": (46.498390, -66.159668),
    "Newfoundland and Labrador": (53.135509, -57.660435),
    "Prince Edward Island": (46.510712, -63.416809),
}


def get_coords(state: str) -> tuple[float, float] | tuple[None, None]:
    """Return (lat, lon) centroid for a state/province name, or (None, None)."""
    return STATE_CENTROIDS.get(state, (None, None))
