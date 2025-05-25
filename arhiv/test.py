
import cv2
import numpy as np
# Define the source points (original triangle)
src_pts = np.float32([[50, 50],
                      [200, 50],
                      [50, 200]])

# Define the destination points (how the triangle should be transformed)
dst_pts = np.float32([[10, 100],
                      [200, 50],
                      [100, 250]])

# Calculate the Affine Transformation matrix using cv2.getAffineTransform
M = cv2.getAffineTransform(src_pts, dst_pts).round(2)
print('Affine Transformation Matrix (M) =', M)
