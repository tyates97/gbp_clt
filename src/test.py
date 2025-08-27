# External libraries
import cv2
import numpy as np
import os
from scipy.stats import t
import matplotlib.pyplot as plt
import collections

# Internal modules
import config as cfg
import factor_graph as fg
import distribution_management as dm
import belief_propagation as bp
import optimisation as opt

print("Loading image...")
# Define base paths
image_dir = 'data/val_selection_cropped/image/'
depth_dir = 'data/val_selection_cropped/groundtruth_depth/'

# --- Example for specific drive & frame ---
date = '2011_09_26'
drive = '0002'
frame = '0000000068'

# Construct the filenames
image_filename = f"{date}_drive_{drive}_sync_image_{frame}_image_03.png"
depth_filename = f"{date}_drive_{drive}_sync_groundtruth_depth_{frame}_image_03.png"

image_path = os.path.join(image_dir, image_filename)
depth_path = os.path.join(depth_dir, depth_filename)

# Load the image & depth map
cfg.image = cv2.imread(image_path, cv2.IMREAD_COLOR)                # cv2.IMREAD_COLOUR loads the image in BGR format
cfg.depth_map_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)    # cv2.IMREAD_UNCHANGED loads the image as is (16-bit PNG for KITTI depth maps)

# Crop the image
cropped_y_width = 50
cropped_x_width = 50
img_y_mean = cfg.image.shape[0] // 2
img_x_mean = cfg.image.shape[1] // 2
cfg.image = cfg.image[img_y_mean-(cropped_y_width//2):img_y_mean+(cropped_y_width//2), img_x_mean-(cropped_x_width//2):img_x_mean+(cropped_x_width//2)]
cfg.depth_map_raw = cfg.depth_map_raw[img_y_mean-(cropped_y_width//2):img_y_mean+(cropped_y_width//2), img_x_mean-(cropped_x_width//2):img_x_mean+(cropped_x_width//2)]

# #DEBUGGING
# cv2.imshow("Image", cfg.image)
# cv2.imshow("Depth Map (raw)", cfg.depth_map_raw)
# cv2.waitKey(0)

cfg.depth_map_meters = np.zeros_like(cfg.depth_map_raw, dtype=np.float32)
cfg.prior_location = cfg.depth_map_raw > 0
cfg.depth_map_meters[cfg.prior_location] = cfg.depth_map_raw[cfg.prior_location] / 256.0

# Set some variables
cfg.belief_discretisation = 256
cfg.prior_width = int(cfg.belief_discretisation/8)
cfg.smoothing_width = int(cfg.belief_discretisation/8)

# Define variables from the data
cfg.num_variables = cfg.depth_map_meters.shape[0]*cfg.depth_map_meters.shape[1] 
cfg.num_priors = np.sum(cfg.prior_location, axis=(0,1))
cfg.graph_type = 'Grid'
cfg.min_measurement = np.min(cfg.depth_map_meters)     # approximately 0.5m
cfg.max_measurement = np.max(cfg.depth_map_meters)     # approximately 80m
cfg.measurement_range = np.linspace(cfg.min_measurement, cfg.max_measurement, cfg.belief_discretisation)

# Build the factor graph
print("image processed. Building factor graph...")
graph = fg.build_factor_graph(cfg.num_variables, cfg.num_priors, cfg.num_loops, cfg.graph_type, cfg.measurement_range, cfg.prior_location)

print("Factor graph built. Running belief propagation...")
# Run belief propagation
graph = bp.run_belief_propagation(graph, cfg.num_iterations, cfg.bp_pass_direction)

print("Belief propagation complete. Calculating length to nearest prior for each variable...")
### Plotting MSE vs distance to prior
length_to_priors = list(opt.find_all_nearest_priors(graph).values())
# get all MSEs to best-fit gaussian
gauss_mse = []

print("Shortest path calculations complete. Calculating MSE from best-fit Gaussian for each variable...")
for variable in graph.variables:
    min_mse,_,_ = opt.optimise_gaussian(variable.belief, cfg.measurement_range)
    gauss_mse.append(min_mse)

print("MSE calculations complete. Averaging MSEs by distance to prior...")
# Group MSEs by distance
mse_by_distance = collections.defaultdict(list)
for dist, mse in zip(length_to_priors, gauss_mse):
    mse_by_distance[dist].append(mse)

# Compute mean MSE for each distance
mean_mse = {dist: np.mean(mse_list) for dist, mse_list in mse_by_distance.items()}

# Sort by distance for plotting
sorted_distances = sorted(mean_mse.keys())
sorted_mse = [mean_mse[dist] for dist in sorted_distances]

print("Average MSE per distance to prior calculated. Plotting MSE vs distance to prior...")
# Create the plot
fig2, ax2 = plt.subplots()
ax2.plot(sorted_distances, sorted_mse, marker='o')
ax2.set_xlabel("Distance to nearest prior")
ax2.set_ylabel("MSE to best-fit Gaussian")
ax2.set_title("MSE vs. Distance to Prior")

plt.show()