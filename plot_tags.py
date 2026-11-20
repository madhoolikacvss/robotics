import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt


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

CSV_PATH = Path(__file__).with_name("tag_positions.csv")


def load_tag_positions(path):
    with path.open(newline="") as f:
        rows = csv.DictReader(f)
        positions = {}
        for row in rows:
            tag = row["tag"]
            if tag in positions:
                raise ValueError(f"{path}: duplicate row for {tag}")
            positions[tag] = (float(row["x"]), float(row["y"]))

    missing = set(GROUND_TRUTH) - set(positions)
    unexpected = set(positions) - set(GROUND_TRUTH)
    if missing or unexpected:
        raise ValueError(
            f"{path}: tag mismatch; missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    return positions


def main():
    estimated = load_tag_positions(CSV_PATH)
    errors = {}

    print(f"Comparing {CSV_PATH.name} with reference tag positions:")
    print("tag    estimated (x, y)       reference (x, y)        dx       dy    distance")
    for tag in GROUND_TRUTH:
        x, y = estimated[tag]
        gx, gy = GROUND_TRUTH[tag]
        dx, dy = x - gx, y - gy
        distance = math.hypot(dx, dy)
        errors[tag] = distance
        print(
            f"{tag:<6} ({x:7.3f}, {y:7.3f})      "
            f"({gx:7.3f}, {gy:7.3f})      "
            f"{dx:+7.3f}  {dy:+7.3f}  {distance:7.3f} m"
        )

    mean_error = sum(errors.values()) / len(errors)
    print(f"\nMean distance error: {mean_error:.3f} m ({mean_error * 100:.1f} cm)")

    fig, ax = plt.subplots(figsize=(7, 7))
    for tag, (gx, gy) in GROUND_TRUTH.items():
        x, y = estimated[tag]
        ax.scatter(x, y, color="tab:blue", marker="x",
                   label="Estimated" if tag == next(iter(GROUND_TRUTH)) else None)
        ax.scatter(gx, gy, color="tab:green", marker="o",
                   label="Reference" if tag == next(iter(GROUND_TRUTH)) else None)
        ax.plot([gx, x], [gy, y], "k--", alpha=0.4)
        ax.annotate(tag, (gx, gy), xytext=(5, 5),
                    textcoords="offset points")

    ax.scatter(0, 0, color="tab:red", marker="*", s=150, label="Robot start")
    ax.set(
        xlabel="X (m)",
        ylabel="Y (m)",
        title=f"AprilTag Map Comparison (mean error: {mean_error:.3f} m)",
    )
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
