import csv
import math
from pathlib import Path

FOLDER = Path(__file__).resolve().parent

GROUND_TRUTH = {
    "tag_0": (1.49, 2.10),
    "tag_1": (0.82, 2.10),
    "tag_3": (-0.40, 1.37),
    "tag_4": (-0.40, 0.24),
    "tag_5": (0.82, -0.42),
    "tag_6": (1.54, -0.42),
    "tag_7": (2.03, 0.20),
    "tag_8": (2.03, 0.92),
}


def load_map(filename):
    with (FOLDER / filename).open(newline="") as f:
        rows = list(csv.DictReader(f))

    positions = {
        row["tag"]: (float(row["x"]), float(row["y"]))
        for row in rows
    }

    if len(rows) != len(positions):
        raise ValueError(f"{filename}: duplicate tag rows")
    if set(positions) != set(GROUND_TRUTH):
        raise ValueError(f"{filename}: expected the same eight ground-truth tags")

    return positions


def mean_distance(map_a, map_b):
    """Mean Euclidean distance between corresponding tags, in meters."""
    return sum(
        math.dist(map_a[tag], map_b[tag])
        for tag in GROUND_TRUTH
    ) / len(GROUND_TRUTH)


for filename_shape, label in [
    ("square", "Square"),
    ("octogon", "Octagon"),  # Matches your CSV filename spelling.
]:
    one_lap = load_map(f"tag_positions_{filename_shape}_1.csv")
    two_laps = load_map(f"tag_positions_{filename_shape}_2.csv")

    error_1 = 100 * mean_distance(one_lap, GROUND_TRUTH)
    error_2 = 100 * mean_distance(two_laps, GROUND_TRUTH)
    estimate_change = 100 * mean_distance(one_lap, two_laps)

    print(f"\n{label}")
    print(f"  Average error after 1 lap:  {error_1:.2f} cm")
    print(f"  Average error after 2 laps: {error_2:.2f} cm")
    print(f"  Error reduction:           {error_1 - error_2:+.2f} cm")
    print(f"  Average estimate shift:    {estimate_change:.2f} cm")

print("\nPositive error reduction means improvement; negative means worse.")