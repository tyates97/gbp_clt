
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
num_variables = 10
num_loops = 0
num_priors = 1
prior_distribution_type = 'random'       # can be 'random', 'random symmetric' 'gaussian', 'top hat', 'horns' or 'skew'
comparison_distribution = 'gaussian'     # can be 'gaussian' or 'q gaussian'
num_iterations = 50

# Plot range variables
min_measurement = -5
max_measurement = 5
belief_discretisation = 100             # number of buckets for the beliefs
max_subplots = 12

# Calculated variables
measurement_range = np.linspace(min_measurement, max_measurement, belief_discretisation)
measurement_mean = (min_measurement + max_measurement) / 2


''' Functions '''
def sweep_optimal_q(param_list, sweep_type, mean_sample_size, fixed_num_variables, fixed_num_loops):
    """Sweep over parameter values and return a list of the mean optimal q for each value."""
    results = []
    for param in param_list:
        q_values = []
        for sample in range(mean_sample_size):
            if sweep_type == "variables":
                num_variables = param
                num_loops = fixed_num_loops
            elif sweep_type == "loops":
                num_variables = fixed_num_variables
                num_loops = param
            else:
                raise ValueError("sweep_type must be 'variables' or 'loops'")

            graph = build_factor_graph(
                num_variables,
                num_priors,
                num_loops,
                measurement_range,
                prior_distribution_type,
                gauss_sigma=2.3
            )
            graph = run_belief_propagation(graph, num_iterations)
            variable_to_optimise_for = graph.variables[len(graph.variables)//2]
            target_belief = variable_to_optimise_for.belief
            sample, optimal_q, _ = optimise_q_gaussian(target_belief, measurement_range)
            q_values.append(optimal_q)
        results.append(np.mean(q_values))
    return results



### Running the script ###
print('Building factor graph...')
graph = build_factor_graph(
    num_variables,
    num_priors,
    num_loops,
    measurement_range,
    prior_distribution_type,
    gauss_sigma=2.3 # temporary, in case needed for the definition of a gaussian prior
)

print('Factor graph built. Running belief propagation...')
graph = run_belief_propagation(graph, num_iterations)

# optimise q-gaussian distribution for variables
print('Belief propagation finished. Calculating optimal q-gaussian distribution...')
# TODO: use more clever logic to find the furthest variable from the prior
variable_to_optimise_for = graph.variables[int(len(graph.variables)//2)]
target_belief = variable_to_optimise_for.belief
print(f'Optimising for variable {variable_to_optimise_for.name}...')

# min_mse, optimal_q, optimal_sigma = optimise_q_gaussian(variable_to_optimise_for.belief, measurement_range)
min_mse, optimal_sigma = optimise_gaussian(variable_to_optimise_for.belief, measurement_range)
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



# '''
# vary the number of loops/variables, and measure q. Plot how gaussian is the belief?
# '''
# num_variables_list = [2,4,6,8,10,12,14,16,18,20]
# num_loops_list = [0,1,2,3,4,5]
# average_samples = 5

# loop_optimised_q = sweep_optimal_q(num_loops_list, "loops", average_samples, num_variables, 0)
# variable_optimised_q = sweep_optimal_q(num_variables_list, "variables", average_samples, 0, num_loops)

# make_plot(num_loops_list, loop_optimised_q, "number of loops", "optimal q value")
# make_plot(num_variables_list, variable_optimised_q, "number of variables", "optimal q value")