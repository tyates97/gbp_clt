
# External libraries
import numpy as np
import networkx as nx
import numba
import cv2
# from sklearn.metrics import mean_squared_error

# Local modules
import distribution_management as dm
import config as cfg


# normalises a set of distribution values so their sum adds to 1
@numba.jit(nopython=True)
def normalise(distribution_values):
    sum_value = np.sum(distribution_values)
    if sum_value > 0:
        normalised_values = distribution_values / sum_value
        return normalised_values

# creates a gaussian distribution
@numba.jit(nopython=True)
def create_gaussian_distribution(x, sigma, mu=0):
    mean = mu
    coef = 1 / (np.sqrt(2 * np.pi) * sigma)
    exponent = -((x - mean) ** 2) / (2 * sigma ** 2)
    return normalise(coef * np.exp(exponent))


@numba.jit(nopython=True)
def mean_squared_error_numba(y_true, y_pred):
    """A Numba-compatible implementation of Mean Squared Error."""
    mse = 0.0
    for i in range(y_true.shape[0]):
        mse += (y_true[i] - y_pred[i])**2
    return mse / y_true.shape[0]


def optimise_q_gaussian(target_belief, measurement_range):
    num_q_steps = 50
    q_search_values = np.linspace(0, 3, num_q_steps)                   # q must be 0 < q < 3

    num_sigma_steps = cfg.belief_discretisation
    sigma_search_values = np.linspace(0, cfg.belief_discretisation-1, num_sigma_steps)

    min_mse = float('inf')
    optimal_q = None
    optimal_sigma = None

    for q_candidate in q_search_values:
        # print('here1')
        for sigma_candidate in sigma_search_values:
            # print('here2')
            if sigma_candidate <= 0:
                continue

            y_q_gauss = dm.create_q_gaussian_distribution(measurement_range, q_candidate, sigma_candidate)
            current_mse = mean_squared_error_numba(target_belief, y_q_gauss)

            if np.isnan(current_mse):
                continue

            if current_mse < min_mse:
                min_mse = current_mse
                optimal_q = q_candidate
                optimal_sigma = sigma_candidate

    return min_mse, optimal_q, optimal_sigma

@numba.jit(nopython=True)
def optimise_gaussian(target_belief, measurement_range):
    # num_sigma_steps = 50
    # sigma_min = 0.1
    # sigma_max = 5.0
    
    sigma_min = 0.1 #1e6 to avoid dividing by zero
    sigma_max = (measurement_range[-1] - measurement_range[0]) / 4.0
    num_sigma_steps = 100 # cfg.belief_discretisation
    
    optimal_mean = np.sum(measurement_range * target_belief)

    sigma_search_values = np.linspace(sigma_min, sigma_max, num_sigma_steps)
    min_mse = float('inf')
    optimal_sigma = None
        
    for sigma_candidate in sigma_search_values:
        y_gauss = create_gaussian_distribution(measurement_range, sigma_candidate, mu=optimal_mean)
        current_mse = mean_squared_error_numba(target_belief, y_gauss)

        if np.isnan(current_mse):
            continue

        if current_mse < min_mse:
            min_mse = current_mse
            optimal_sigma = sigma_candidate

    return min_mse, optimal_sigma, optimal_mean


# @numba.jit(nopython=True, parallel=True)
# def _calculate_all_mses(all_beliefs, measurement_range):
#     """
#     Calculates the best-fit Gaussian MSE for all beliefs in parallel.
#     Operates only on NumPy arrays.
#     """
#     num_variables = all_beliefs.shape[0]
#     mse_values_flat = np.zeros(num_variables)

#     # Using prange for parallel execution
#     for i in numba.prange(num_variables):
#         min_mse, _, _ = optimise_gaussian(all_beliefs[i, :], measurement_range)
#         mse_values_flat[i] = min_mse
        
#     return mse_values_flat


# def get_mse_from_graph(graph):
#     # num_variables = len(graph.variables)
#     # width = graph.grid_cols
#     # height = int(np.ceil(num_variables/width))
    
#     # mse_values = np.zeros((height, width))

#     # for i, variable in enumerate(graph.variables):
#     #     min_mse,_,_ = optimise_gaussian(variable.belief, cfg.measurement_range)
#     #     row = i // graph.grid_cols
#     #     col = i % graph.grid_cols
#     #     mse_values[row][col] = min_mse
    
#     # return mse_values
#     """
#     Orchestrator function to calculate the MSE for every variable's belief.
#     It extracts data into NumPy arrays and calls the fast Numba helper.
#     """
#     print("Calculating best-fit Gaussian for each variable...")
#     num_variables = len(graph.variables)
#     width = graph.grid_cols
#     height = int(np.ceil(num_variables / width))
    
#     # 1. Extract all beliefs into a single NumPy array
#     discretisation = len(graph.variables[0].belief)
#     all_beliefs = np.empty((num_variables, discretisation), dtype=np.float64)
#     for i, variable in enumerate(graph.variables):
#         all_beliefs[i, :] = variable.belief

#     # 2. Call the fast, parallel Numba function
#     mse_values_flat = _calculate_all_mses(all_beliefs, cfg.measurement_range)
    
#     # 3. Reshape the flat results back into a 2D grid
#     mse_values = mse_values_flat.reshape((height, width))
    
#     return mse_values



def get_mse_from_graph(graph):
    """
    Calculate MSE for every variable's belief without Numba parallel processing.
    """
    print("Calculating best-fit Gaussian for each variable...")
    num_variables = len(graph.variables)
    width = graph.grid_cols
    height = int(np.ceil(num_variables / width))
    
    mse_values = np.zeros((height, width))

    for i, variable in enumerate(graph.variables):
        # print(f"variable belief type: {type(variable.belief)}, shape: {variable.belief.shape}")
        # print(f"measurement range: {type(cfg.measurement_range)}, shape: {cfg.measurement_range.shape}")
        min_mse, _, _ = optimise_gaussian(variable.belief, cfg.measurement_range)
        row = i // graph.grid_cols
        col = i % graph.grid_cols
        mse_values[row][col] = min_mse
    
    return mse_values



''' function to find the nearest prior to a variable in a factor graph '''
# def find_nearest_prior(variable, graph):
    
def find_all_nearest_priors(graph):
    """
    Returns a dict mapping variable names to their shortest inter-variable distance to any variable with a prior factor.
    """
    # Find variable nodes with a prior factor neighbor
    prior_variables = set()
    for factor in graph.factors:
        if factor.factor_type == 'prior' and factor.neighbors:
            prior_variables.add(factor.neighbors[0].name)

    # Compute shortest path lengths from all variables to all others
    # Only count steps between variable nodes (skip factor nodes)
    variable_names = [v.name for v in graph.variables]
    G = graph.graph

    # Precompute all shortest paths between variables
    all_shortest_paths = dict(nx.all_pairs_shortest_path(G))

    # For each variable, find the minimum number of variable-to-variable steps to any prior variable
    distances = {}
    for var in variable_names:
        min_steps = float('inf')
        for prior_var in prior_variables:
            if prior_var == var:
                min_steps = 0
                break
            # Path alternates variable-factor-variable... so steps = (len(path)-1)//2
            try:
                path = all_shortest_paths[var][prior_var]
                var_steps = (len(path)-1)//2
                if var_steps < min_steps:
                    min_steps = var_steps
            except KeyError:
                continue
        distances[var] = min_steps if min_steps != float('inf') else None
    return distances


def find_nearest_prior(variable, graph):
    return find_all_nearest_priors(graph).get(variable.name)



@numba.jit(nopython=True)
def kl_divergence_numba(p, q):
    """KL divergence between belief p and Gaussian q"""
    epsilon = 1e-10
    kl = 0.0
    for i in range(p.shape[0]):
        p_i = max(p[i], epsilon)
        q_i = max(q[i], epsilon)
        kl += p_i * np.log(p_i / q_i)
    return kl

@numba.jit(nopython=True)
def optimise_gaussian_kl(target_belief, measurement_range):
    """Find Gaussian that minimizes KL divergence from target_belief"""
    sigma_min = 0.25
    sigma_max = (measurement_range[-1] - measurement_range[0]) / 4.0
    num_sigma_steps = 100
    
    optimal_mean = np.sum(measurement_range * target_belief)
    sigma_search_values = np.linspace(sigma_min, sigma_max, num_sigma_steps)
    min_kl = float('inf')
    optimal_sigma = None
        
    for sigma_candidate in sigma_search_values:
        y_gauss = create_gaussian_distribution(measurement_range, sigma_candidate, mu=optimal_mean)
        current_kl = kl_divergence_numba(target_belief, y_gauss)

        if np.isnan(current_kl):
            continue

        if current_kl < min_kl:
            min_kl = current_kl
            optimal_sigma = sigma_candidate

    return min_kl, optimal_sigma, optimal_mean

def get_kl_from_graph(graph):
    """
    Calculate KL divergence for every variable's belief.
    """
    print("Calculating KL divergence for each variable...")
    num_variables = len(graph.variables)
    width = graph.grid_cols
    height = int(np.ceil(num_variables / width))
    
    kl_values = np.zeros((height, width))

    for i, variable in enumerate(graph.variables):
        min_kl, _, _ = optimise_gaussian_kl(variable.belief, cfg.measurement_range)
        row = i // graph.grid_cols
        col = i % graph.grid_cols
        kl_values[row][col] = min_kl
    
    return kl_values