
import cv2
import numpy as np
import os

# Define base paths
image_dir = 'data/val_selection_cropped/image/'
depth_dir = 'data/val_selection_cropped/groundtruth_depth/'

# --- Example for specific drive & frame ---
date = '2011_09_26'
drive = '0002'
frame = '0000000005'

# Construct the filenames
image_filename = f"{date}_drive_{drive}_sync_image_{frame}_image_02.png"
depth_filename = f"{date}_drive_{drive}_sync_groundtruth_depth_{frame}_image_02.png"

image_path = os.path.join(image_dir, image_filename)
depth_path = os.path.join(depth_dir, depth_filename)

#Load the image & depth map
image = cv2.imread(image_path, cv2.IMREAD_COLOR)                # cv2.IMREAD_COLOUR loads the image in BGR format
depth_map_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)    # cv2.IMREAD_UNCHANGED loads the image as is (16-bit PNG for KITTI depth maps)
depth_meters = np.zeros_like(depth_map_raw, dtype=np.float32)
prior_locations = depth_map_raw > 0
depth_meters[prior_locations] = depth_map_raw[prior_locations] / 256.0 

print(image.shape)
print(depth_meters.shape)