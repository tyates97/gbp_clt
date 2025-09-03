# External libraries
import cv2
import numpy as np
from scipy.stats import t
import matplotlib.pyplot as plt

# Internal modules
import config as cfg
import factor_graph as fg
import distribution_management as dm
import belief_propagation as bp
import optimisation as opt
import image_processing as ip
import graphics as gx


np.random.seed(cfg.random_seed)
print("Loading images...")
# Define base paths
image_dir = 'data/stereo/teddy/'
left_image_filename = "im2.png"
right_image_filename = "im6.png" 

# Load the image & depth map
left_image = cv2.imread(image_dir+left_image_filename, cv2.IMREAD_GRAYSCALE)                
right_image = cv2.imread(image_dir+right_image_filename, cv2.IMREAD_GRAYSCALE)
print("Images loaded")

# #DEBUGGING
# cv2.imshow("Left Image", left_image)
# cv2.imshow("Right Image", right_image)
# cv2.waitKey(0)

# Calculate the costs
max_disparity = 64
patch_size = 7*7
cost_volume = ip.get_cost_volume(left_image, right_image, patch_size, cfg.belief_discretisation)

# # DEBUGGING: Plot some of the costs that you've calculated
# num_curves = 100
# gx.plot_volume_curves(cost_volume, num_curves)
print(f"Plotting cost variance heatmap...")
gx.plot_variance_heatmap(cost_volume)

# Convert the costs to a probability distributionz
pdf_volume = ip.get_pdfs_from_costs(cost_volume)

# #DEBUGGING: Plot some of the pdfs that you've calculated
# num_curves = 100
# gx.plot_volume_curves(pdf_volume, num_curves)
print(f"Plotting cost variance heatmap...")
gx.plot_variance_heatmap(pdf_volume)

# Build factor graph from image
cfg.num_variables = pdf_volume.shape[0]*pdf_volume.shape[1]
cfg.min_measurement = 0
cfg.max_measurement = max_disparity
cfg.belief_discretisation = max_disparity
cfg.measurement_range = np.linspace(cfg.min_measurement, cfg.max_measurement-1, cfg.belief_discretisation)

# ## DEBUGGING: visualise smoothing factors
# # dm.create_smoothing_factor_distribution(cfg.belief_discretisation)

graph = fg.get_graph_from_pdf(pdf_volume)


# HACKY WORKAROUND - make a 2nd graph, and put all beliefs as the variable's prior to begin with. Then use that to find mse for best Gaussian.
graph_2 = graph
for variable in graph_2.variables:
    for factor in variable.neighbors:
        if factor.factor_type == "prior":
            variable.belief = factor.function
mse_volume_1 = opt.get_mse_from_graph(graph_2)
# gx.plot_graph_beliefs(graph_2, 100, max_disparity)


# Run Belief Propagation on graph
graph = bp.run_belief_propagation(graph, cfg.num_iterations)

# # # Plot some of the beliefs that you've calculated
# # num_curves = 100
# # gx.plot_graph_beliefs(graph, num_curves, max_disparity)

### Plotting MSE for each variable
mse_volume_2 = opt.get_mse_from_graph(graph)

# Show results
cv2.imshow("Left Image", left_image)

plt.figure("Heatmap 1: Pre-BP")
im_1 = plt.imshow(mse_volume_1, cmap='RdYlGn_r')
plt.colorbar(im_1, label='MSE (lower = more Gaussian, pre_BP)')
plt.axis('off')
plt.tight_layout()


plt.figure("Heatmap 2: Post-BP")
im_2 = plt.imshow(mse_volume_2, cmap='RdYlGn_r')
plt.colorbar(im_2, label='MSE (lower = more Gaussian, post_BP)')
plt.axis('off')
plt.tight_layout()

plt.show()