# calibrate_camera.py

from pathlib import Path
import cv2
import numpy as np

# ----------------------------
# Checkerboard parameters
# ----------------------------

CHECKERBOARD = (10,7)       # inner corners
SQUARE_SIZE = 0.020         # meters (20 mm)

IMAGE_DIR = Path("/Users/jonah/Downloads/calibration-img")

# termination criteria

criteria = (
    cv2.TERM_CRITERIA_EPS +
    cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001,
)

# 3D object points

objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)

objp[:, :2] = np.mgrid[
    0:CHECKERBOARD[0],
    0:CHECKERBOARD[1]
].T.reshape(-1, 2)

objp *= SQUARE_SIZE

objpoints = []
imgpoints = []

images = sorted(IMAGE_DIR.glob("*.png"))
print(f"Found {len(images)} images")

for image_path in images:

    img = cv2.imread(str(image_path))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    found, corners = cv2.findChessboardCorners(
        gray,
        CHECKERBOARD,
        None,
    )
    print(image_path.name, found)

    if found:

        corners2 = cv2.cornerSubPix(
            gray,
            corners,
            (11,11),
            (-1,-1),
            criteria,
        )

        objpoints.append(objp)
        imgpoints.append(corners2)

        cv2.drawChessboardCorners(
            img,
            CHECKERBOARD,
            corners2,
            found,
        )

        cv2.imshow("Corners", img)
        cv2.waitKey(100)

cv2.destroyAllWindows()

ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,
    imgpoints,
    gray.shape[::-1],
    None,
    None,
)

print()

print("RMS reprojection error:")
print(ret)

print()

print("Camera Matrix")

print(K)

print()

print("Distortion")

print(dist)

fs = cv2.FileStorage(
    "camera_intrinsics.yaml",
    cv2.FILE_STORAGE_WRITE,
)

fs.write("camera_matrix", K)
fs.write("distortion_coefficients", dist)

fs.release()

print()
print("Saved camera_intrinsics.yaml")