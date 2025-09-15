''' config file for global parameters '''

import numpy as np


# Graph Shape
num_variables = 15
num_priors = 1
graph_type = 'Loopy'
prior_location = 'random'

# Loopy Graphs
num_loops = 3

# Tree Graphs
bp_pass_direction = 'Both'
branching_probability = 1.0
branching_factor = 2

## Grid Graphs
# num_cols = int(np.ceil(np.sqrt(num_variables)))

# UI
show_comparison = True
show_heatmap = False
max_subplots = 12

# Loopy BP
num_iterations = 10
belief_discretisation = 64

# Distributions
prior_width = int(belief_discretisation/2)
smoothing_width = int(belief_discretisation)//2
min_measurement = -5
max_measurement = 5
measurement_range = np.linspace(min_measurement, max_measurement, belief_discretisation)
random_seed = 42
rng = np.random.default_rng(seed=42)

# Real Data
left_image = None
right_image = None
depth_map_raw = None
depth_map_meters = None

# Stereo Paramters
cost_function = 'SSD'               # options: 'NCC', 'SAD', 'SSD'
lambda_param = 0.000001                  # Cost to pdf
smoothing_function = 'histogram'    # options: 'histogram', 'triangular' #TODO: implement option to change