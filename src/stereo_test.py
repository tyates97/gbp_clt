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
patch_size = 10
cost_volume = ip.get_cost_volume(left_image, right_image, patch_size, cfg.belief_discretisation)

# Convert the costs to a probability distribution
# TODO: make sure you're happy with the softmax function used here, and the lambda value
# TODO: speed this up with numba
pdf_volume = ip.get_pdfs_from_costs(cost_volume)

# #DEBUGGING: Plot some of the pdfs that you've calculated
# num_curves = 100
# gx.plot_pdf_volume(pdf_volume, num_curves)

# Build factor graph from image
cfg.num_variables = pdf_volume.shape[0]*pdf_volume.shape[1]
cfg.min_measurement = 0
cfg.max_measurement = max_disparity
cfg.belief_discretisation = max_disparity
cfg.measurement_range = np.linspace(cfg.min_measurement, cfg.max_measurement-1, cfg.belief_discretisation)

## DEBUGGING: visualise smoothing factors
# dm.create_smoothing_factor_distribution(cfg.belief_discretisation)

graph = fg.get_graph_from_pdf(pdf_volume)

# Run Belief Propagation on graph
graph = bp.run_belief_propagation(graph, cfg.num_iterations)

# Plot some of the beliefs that you've calculated
num_curves = 100
gx.plot_graph_beliefs(graph, num_curves, max_disparity)

### Plotting MSE for each variable
mse_volume = opt.get_mse_from_graph(graph)

# Show results
cv2.imshow("Right Image", right_image)
im = plt.imshow(mse_volume, cmap='RdYlGn_r') # Red (high MSE) to Green (low MSE)
plt.colorbar(im, label='MSE (lower = more Gaussian)')
plt.axis('off')
plt.tight_layout()
plt.show()