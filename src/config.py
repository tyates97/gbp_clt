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

# Distributions
# min_measurement = -5
# max_measurement = 5
min_measurement = 0
max_measurement = 52
measurement_range = np.arange(min_measurement, max_measurement, 0.25)
belief_discretisation = len(measurement_range)

random_seed = 42
rng = np.random.default_rng(seed=42)
prior_width = int(max_measurement/2)
smoothing_width = int(max_measurement)//2

# Real Data
left_image = None
right_image = None
depth_map_raw = None
depth_map_meters = None

# Stereo Paramters
cost_function = 'SSD'               # options: 'NCC', 'SAD', 'SSD'
lambda_param = 0.0020                  # Cost to pdf
smoothing_function = 'histogram'    # options: 'histogram', 'triangular' #TODO: implement option to change