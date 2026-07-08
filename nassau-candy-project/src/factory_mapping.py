"""
factory_mapping.py
===================
Defines the factory network used throughout this project.

** ASSUMPTION DISCLOSURE — READ THIS **
----------------------------------------
The source dataset (Nassau_Candy_Distributor.csv) does NOT contain any
factory identifiers, factory names, or factory coordinates. No factory
table was supplied with this project.

To make factory-reallocation optimization possible at all, this project
derives a plausible 3-factory production network from the one real
production-grouping signal in the data: the `Division` column
(Chocolate / Sugar / Other). This mirrors how confectionery distributors
commonly organize plants — by product family — and lets every downstream
feature (lead time, cost, distance) be computed from a real, consistent
rule rather than randomly assigned.

These factory names and coordinates are ILLUSTRATIVE, not Nassau Candy's
real facilities. They are placed at real US logistics hub coordinates
(Chicago, Atlanta, Allentown) purely so that distance-based features are
geographically meaningful. If you have Nassau Candy's actual factory
list, replace FACTORY_NETWORK below and every downstream module will
pick up the change automatically.
"""

from __future__ import annotations

# Factory_ID -> (display name, lat, lon, primary division, city hub)
FACTORY_NETWORK: dict[str, dict] = {
    "FAC-CHI-01": {
        "name": "Chicago Chocolate Works",
        "division": "Chocolate",
        "lat": 41.8781,
        "lon": -87.6298,
        "city": "Chicago, IL",
        "base_capacity_units_per_day": 4000,
    },
    "FAC-ATL-02": {
        "name": "Atlanta Sugar & Candy Plant",
        "division": "Sugar",
        "lat": 33.7490,
        "lon": -84.3880,
        "city": "Atlanta, GA",
        "base_capacity_units_per_day": 2500,
    },
    "FAC-ALN-03": {
        "name": "Allentown Specialty Confections",
        "division": "Other",
        "lat": 40.6084,
        "lon": -75.4902,
        "city": "Allentown, PA",
        "base_capacity_units_per_day": 1800,
    },
}

DIVISION_TO_FACTORY = {info["division"]: fid for fid, info in FACTORY_NETWORK.items()}

ALL_FACTORY_IDS = list(FACTORY_NETWORK.keys())


def default_factory_for_division(division: str) -> str:
    """Return the assumed 'current' factory assignment for a division."""
    return DIVISION_TO_FACTORY.get(division, "FAC-CHI-01")


def factory_coords(factory_id: str) -> tuple[float, float]:
    info = FACTORY_NETWORK[factory_id]
    return info["lat"], info["lon"]
