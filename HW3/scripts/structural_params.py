"""Frozen mechanical assumptions for the HW3 rigid-diaphragm calculations."""

from __future__ import annotations


E_KN_M2 = 30_000_000.0
POISSON = 0.2
G_KN_M2 = E_KN_M2 / (2.0 * (1.0 + POISSON))
STOREY_HEIGHT_M = 3.0
SHEAR_KAPPA = 5.0 / 6.0
GRAVITY_M_S2 = 9.81
FLOOR_LOAD_KN_M2 = 8.75
STOREYS = 5
DESIGN_FORCE_KN = 1000.0
DESIGN_DIRECTION = "+x at CM"

# Signed, non-overlapping rectangles: (name, sign, x0, x1, y0, y1).
# The two openings are negative regions.  This is the accepted HW1 geometry.
MASS_REGIONS = (
    ("main floor", 1.0, 0.0, 10.9, 1.0, 16.9),
    ("lower balcony", 1.0, 1.0, 9.9, 0.0, 1.0),
    ("upper balcony", 1.0, 1.0, 9.9, 16.9, 17.7),
    ("opening west", -1.0, 3.47, 4.57, 9.11, 12.15),
    ("opening east", -1.0, 4.57, 7.32, 9.11, 11.24),
)
