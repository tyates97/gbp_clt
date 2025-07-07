
# External libraries
import numpy as np
import matplotlib.pyplot as plt

# Local modules
import distribution_management as dm
from factor_graph import build_factor_graph
from belief_propagation import run_belief_propagation
from graphics import plot_results
from optimisation import optimise_q_gaussian, optimise_gaussian


### Variables ###
# Define factor graph topology
num_variables = 12
num_loops = 3
num_priors = 1
is_tree = False
prior_distribution_type = 'random'       # can be 'random', 'random symmetric' 'gaussian', 'top hat', 'horns' or 'skew'
comparison_distribution = 'gaussian'     # can be 'gaussian' or 'q gaussian'
num_iterations = 50
identical_smoothing_functions = False  # If True, all smoothing functions will be identical, otherwise they will be different

# Plot range variables
min_measurement = -5
max_measurement = 5
belief_discretisation = 52             # number of buckets for the beliefs: must be divisible by 4
max_subplots = 12

# Calculated variables
measurement_range = np.linspace(min_measurement, max_measurement, belief_discretisation)
measurement_mean = (min_measurement + max_measurement) / 2


### Running the script ###
print('Building factor graph...')
graph = build_factor_graph(
    num_variables,
    num_priors,
    num_loops,
    is_tree,
    identical_smoothing_functions,
    measurement_range,
    prior_distribution_type,
    gauss_sigma=2.3, # temporary, in case needed for the definition of a gaussian prior
    
)

print('Factor graph built. Running belief propagation...')
graph = run_belief_propagation(graph, num_iterations)

# optimise q-gaussian distribution for variables
print('Belief propagation finished. Calculating optimal q-gaussian distribution...')
# TODO: use more clever logic to find the furthest variable from the prior
if num_loops == 0:
    variable_to_optimise_for = graph.variables[-1]  
else:
    variable_to_optimise_for = graph.variables[int(len(graph.variables)//2)]
target_belief = variable_to_optimise_for.belief
print(f'Optimising for variable {variable_to_optimise_for.name}...')

# min_mse, optimal_q, optimal_sigma = optimise_q_gaussian(variable_to_optimise_for.belief, measurement_range)
min_mse, optimal_sigma, optimal_mu = optimise_gaussian(variable_to_optimise_for.belief, measurement_range)
print("Optimisation Complete.")
if optimal_sigma is not None:
#     print(f"\nOptimal q: {optimal_q:.4f}")
    print(f"Optimal sigma: {optimal_sigma:.4f}")
    print(f"Minimum MSE achieved: {min_mse:.6f}")
else:
    print("Optimization did not find a valid solution. Check search ranges or target belief.")

print('\nPlotting results...')
plot_results(
    graph
    , max_subplots
    , measurement_range
    , comparison_distribution
    #, optimal_sigma
    #, optimal_q
)