import csv
from pathlib import Path
import matplotlib.pyplot as plt

ground_truth = {
    "tag_0": (1.49, 2.10),
    "tag_1": (0.82, 2.10),
    "tag_3": (-0.40, 1.37),
    "tag_4": (-0.40, 0.24),
    "tag_5": (0.82, -0.42),
    "tag_6": (1.54, -0.42),
    "tag_7": (2.03, 0.20),
    "tag_8": (2.03, 0.92),
}

csv_path = Path(__file__).with_name("tag_positions_square_2.csv")
with csv_path.open(newline="") as f:
    rows = list(csv.DictReader(f))

fig, ax = plt.subplots(figsize=(7, 7))

for i, row in enumerate(rows):
    tag = row["tag"]
    x, y = float(row["x"]), float(row["y"])
    gx, gy = ground_truth[tag]

    ax.scatter(x, y, color="blue", marker="x",
               label="Estimated" if i == 0 else None)
    ax.scatter(gx, gy, color="green", marker="o",
               label="Ground truth" if i == 0 else None)
    ax.plot([gx, x], [gy, y], "k--", alpha=0.3)
    ax.annotate(tag, (gx, gy), xytext=(5, 5),
                textcoords="offset points")

ax.scatter(0, 0, color="red", marker="*", s=150, label="Robot start")
ax.set(xlabel="X (m)", ylabel="Y (m)", title="AprilTag Map Comparison")
ax.set_aspect("equal", adjustable="box")
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.show()