''' config file for global parameters '''

import numpy as np

num_variables = 15
graph_type = 'Loopy'
show_comparison = True
prior_location = 'random'
num_priors = 1
num_loops = 3
bp_pass_direction = 'Both'
branching_probability = 1.0
branching_factor = 2
show_heatmap = False
bp_pass_direction = 'Both'
branching_probability = 1.0
branching_factor = 2
show_heatmap = False
num_iterations = 50
belief_discretisation = 52
random_seed = 42
min_measurement = -5
max_measurement = 5
max_subplots = 12
measurement_range = np.linspace(min_measurement, max_measurement, belief_discretisation)
rng = np.random.default_rng(seed=42)
