# External libraries
import cv2
import numpy as np
from scipy.stats import t
import matplotlib.pyplot as plt
import copy

# Internal modules
import config as cfg
import factor_graph as fg
import distribution_management as dm
import belief_propagation as bp
import optimisation as opt
import image_processing as ip
import graphics as gx


''' 1. Load images '''

np.random.seed(cfg.random_seed)
print("Loading images...")
# Define base paths
image_dir = 'data/stereo/teddy/'
left_image_filename = "im2.png"
right_image_filename = "im6.png"
left_ground_truth_filename = "disp2.png"

# Load the image & depth map
left_image = cv2.imread(image_dir+left_image_filename, cv2.IMREAD_GRAYSCALE)                
right_image = cv2.imread(image_dir+right_image_filename, cv2.IMREAD_GRAYSCALE)
ground_truth = cv2.imread(image_dir+left_ground_truth_filename, cv2.IMREAD_GRAYSCALE)
print("Images loaded")

# #DEBUGGING
# cv2.imshow("Left Image", left_image)
# cv2.imshow("Right Image", right_image)
# cv2.waitKey(0)

# ''' 2. Calculate Costs '''

# # Calculate the costs
cfg.max_measurement = 64 # int(np.ceil(max(ground_truth.flatten()/4)))
cfg.belief_discretisation = cfg.max_measurement
patch_size = 5*5
cost_volume = ip.get_cost_volume(left_image, right_image, patch_size, cfg.belief_discretisation)

# Plot the cost functions for various pixels
gx.interactive_pixel_inspector(left_image, cost_volume, cfg.max_measurement)
plt.show()
# print(f"Plotting cost variance heatmap...")
# gx.plot_variance_heatmap(cost_volume)


''' 3. Convert Costs to PDFs '''
# Convert the costs to a probability distribution
pdf_volume = ip.get_pdfs_from_costs(cost_volume)


# Plot the pdfs for various pixels
gx.interactive_pixel_inspector(left_image, pdf_volume, cfg.max_measurement)
plt.show()
print(f"Plotting prior variance heatmap...")
gx.plot_variance_heatmap(pdf_volume)


''' 4. Build Factor Graph '''

# Get disparity histogram
ground_truth_signed = ground_truth.astype(np.int16)
all_diffs = dm.get_histogram_from_truth(ground_truth_signed)
hist, bin_edges = np.histogram(all_diffs, bins=2*cfg.belief_discretisation-1)
smoothing_kernel = dm.normalise(hist)

# Plot disparity pdf
gx.plot_disparity_histogram(smoothing_kernel, bin_edges)
plt.show()

# # DEBUGGING: visualise smoothing factors
# dm.create_smoothing_factor_distribution(cfg.belief_discretisation, hist=smoothing_kernel)

# Build factor graph from image
cfg.num_variables = pdf_volume.shape[0]*pdf_volume.shape[1]
cfg.min_measurement = 0
cfg.measurement_range = np.linspace(cfg.min_measurement, cfg.max_measurement-1, cfg.belief_discretisation)


graph = fg.get_graph_from_pdf_hist(pdf_volume, smoothing_kernel)        # with kernel defined
# graph = fg.get_graph_from_pdf_hist(pdf_volume)                          # with triangular/default kernel


''' 4.5 Save Pre-BP State '''

# HACKY WORKAROUND - initialise every variable belief with its prior
for variable in graph.variables:
    for factor in variable.neighbors:
        if factor.factor_type == "prior":
            variable.belief = factor.function

initial_beliefs = {var: var.belief.copy() for var in graph.variables}


# graph_before_bp = copy.deepcopy(graph)
disparity_vol_pre_bp = ip.get_disparity_from_graph(graph)


''' 5. Run Belief Propagation '''

# Run Belief Propagation on graph
graph = bp.run_belief_propagation(graph, cfg.num_iterations)
graph_after_bp = graph


''' 6. Show Results '''

# # Plot some of the beliefs that you've calculated
# num_curves = 100
# gx.plot_graph_beliefs(graph, num_curves, cfg.max_measurement)

# Show results - depth map results
disparity_vol_post_bp = ip.get_disparity_from_graph(graph)

#TODO: plot all the below together, where you can click on a pixel to show its belief before and after
gx.plot_depth_estimate(disparity_vol_post_bp, "Disparity Post-BP")
gx.plot_depth_estimate(disparity_vol_pre_bp, "Disparity Pre-BP")

# gx.interactive_pixel_inspector(disparity_vol_post_bp, cost_volume, cfg.max_measurement)
# gx.interactive_pixel_inspector(disparity_vol_pre_bp, cost_volume, cfg.max_measurement)

# Show results - gaussian heatmaps
gx.plot_gaussian_heatmap(graph_after_bp, "Heatmap 2: Post-BP")
# --- Temporarily restore initial beliefs for "before" plots ---
for var, initial_belief in initial_beliefs.items():
    var.belief = initial_belief
gx.plot_gaussian_heatmap(graph, "Heatmap 1: Pre-BP")


cv2.imshow("Left Image", left_image)
plt.show()