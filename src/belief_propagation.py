# External libraries
import numpy as np
import numba

# Local modules
import distribution_management as dm
import optimisation as opt
import config as cfg

@numba.jit(nopython=True)
def _run_bp_numba(num_iterations, discretisation,
                  factor_to_var_msgs, var_to_factor_msgs, beliefs,
                  factor_connections, var_neighbors, factor_functions, priors, prior_indices,
                  var_neighbor_to_factor_neighbor_idx, factor_to_var_neighbor_idx):
    """
    A Numba-jitted function to run the core belief propagation loop.
    This function uses sparse message arrays (indexed by neighbor, not by node ID)
    to handle large graphs efficiently.
    """
    num_variables = var_neighbors.shape[0]
    num_factors = factor_connections.shape[0]

    # Handle prior factors (which are unary)
    for i in range(prior_indices.shape[0]):
        prior_factor_idx = prior_indices[i]
        # The message from a prior is just the prior itself. It goes to neighbor 0.
        factor_to_var_msgs[prior_factor_idx, 0, :] = priors[i]

    for iteration in range(num_iterations):
        ### STEP 1: Update messages from variables to factors (direct computation)
        for i in range(num_variables):
            for j in range(var_neighbors.shape[1]):
                factor_idx = var_neighbors[i, j]
                if factor_idx == -1: break
                
                # Compute message by multiplying all OTHER incoming messages
                outgoing_msg = np.ones(discretisation)
                for k in range(var_neighbors.shape[1]):
                    other_factor_idx = var_neighbors[i, k]
                    if other_factor_idx == -1: break
                    if other_factor_idx != factor_idx:  # Skip the target factor
                        other_factor_n_idx = var_neighbor_to_factor_neighbor_idx[i, k]
                        outgoing_msg *= factor_to_var_msgs[other_factor_idx, other_factor_n_idx, :]
                
                # Normalize
                s = np.sum(outgoing_msg)
                if s > 0:
                    var_to_factor_msgs[i, j, :] = outgoing_msg / s

        ### STEP 2: Update messages from factors to variables
        for i in range(num_factors):
            # Handle pairwise smoothing factors
            if factor_connections[i, 1] != -1:
                var1_idx = factor_connections[i, 0]
                var2_idx = factor_connections[i, 1]
                
                # Find the incoming messages from each variable
                var1_n_idx = factor_to_var_neighbor_idx[i, 0]
                var2_n_idx = factor_to_var_neighbor_idx[i, 1]
                
                incoming_from_var1 = var_to_factor_msgs[var1_idx, var1_n_idx, :]
                incoming_from_var2 = var_to_factor_msgs[var2_idx, var2_n_idx, :]
                
                # Message to var1 (neighbor 0 of the factor)
                msg_to_var1 = np.dot(factor_functions[i], incoming_from_var2)
                s1 = np.sum(msg_to_var1)
                if s1 > 0:
                    factor_to_var_msgs[i, 0, :] = msg_to_var1 / s1
                
                # Message to var2 (neighbor 1 of the factor)
                msg_to_var2 = np.dot(factor_functions[i].T, incoming_from_var1)
                s2 = np.sum(msg_to_var2)
                if s2 > 0:
                    factor_to_var_msgs[i, 1, :] = msg_to_var2 / s2

    ### STEP 3: Update final beliefs
    for i in range(num_variables):
        belief = np.ones(discretisation)
        for j in range(var_neighbors.shape[1]):
            factor_idx = var_neighbors[i, j]
            if factor_idx == -1: break
            
            factor_n_idx = var_neighbor_to_factor_neighbor_idx[i, j]
            belief *= factor_to_var_msgs[factor_idx, factor_n_idx, :]
        
        s = np.sum(belief)
        if s > 0:
            beliefs[i, :] = belief / s
            
    return beliefs


def run_belief_propagation(graph, num_iterations):
    """
    Wrapper function to run belief propagation.
    It converts the object-oriented graph into NumPy arrays, calls the fast
    Numba-jitted core function, and then updates the graph objects with the results.
    """
    print("BP Stage 1: Converting graph to numerical representation...")

    num_variables = len(graph.variables)
    num_factors = len(graph.factors)
    discretisation = len(graph.variables[0].belief)

    var_map = {var: i for i, var in enumerate(graph.variables)}
    factor_map = {factor: i for i, factor in enumerate(graph.factors)}

    factor_connections = np.zeros((num_factors, 2), dtype=np.int32) - 1
    factor_functions = np.zeros((num_factors, discretisation, discretisation))
    priors, prior_indices = [], []

    max_neighbors = 0
    for var in graph.variables:
        if len(var.neighbors) > max_neighbors:
            max_neighbors = len(var.neighbors)
    var_neighbors = np.zeros((num_variables, max_neighbors), dtype=np.int32) - 1

    for factor, i in factor_map.items():
        if factor.factor_type == 'prior':
            factor_connections[i, 0] = var_map[factor.neighbors[0]]
            priors.append(factor.function)
            prior_indices.append(i)
        else:
            factor_connections[i, 0] = var_map[factor.neighbors[0]]
            factor_connections[i, 1] = var_map[factor.neighbors[1]]
            factor_functions[i, :, :] = factor.function

    var_to_neighbor_map = [{} for _ in range(num_variables)]
    for var, i in var_map.items():
        for j, neighbor_factor in enumerate(var.neighbors):
            f_idx = factor_map[neighbor_factor]
            var_neighbors[i, j] = f_idx
            var_to_neighbor_map[i][f_idx] = j

    priors = np.array(priors)
    prior_indices = np.array(prior_indices, dtype=np.int32)

    # Create lookup tables for sparse indexing
    var_neighbor_to_factor_neighbor_idx = np.zeros_like(var_neighbors)
    for v_idx in range(num_variables):
        for n_idx in range(max_neighbors):
            f_idx = var_neighbors[v_idx, n_idx]
            if f_idx == -1: break
            if factor_connections[f_idx, 0] == v_idx:
                var_neighbor_to_factor_neighbor_idx[v_idx, n_idx] = 0
            elif factor_connections[f_idx, 1] == v_idx:
                var_neighbor_to_factor_neighbor_idx[v_idx, n_idx] = 1

    factor_to_var_neighbor_idx = np.zeros_like(factor_connections)
    for f_idx in range(num_factors):
        v1_idx = factor_connections[f_idx, 0]
        if v1_idx != -1:
            factor_to_var_neighbor_idx[f_idx, 0] = var_to_neighbor_map[v1_idx][f_idx]
        v2_idx = factor_connections[f_idx, 1]
        if v2_idx != -1:
            factor_to_var_neighbor_idx[f_idx, 1] = var_to_neighbor_map[v2_idx][f_idx]

    # **FIX**: Initialize message arrays with sparse shapes to avoid memory error
    factor_to_var_msgs = np.ones((num_factors, 2, discretisation)) / discretisation
    var_to_factor_msgs = np.ones((num_variables, max_neighbors, discretisation)) / discretisation
    beliefs = np.ones((num_variables, discretisation)) / discretisation

    print("BP Stage 2: Running BP iterations...")
    
    final_beliefs = _run_bp_numba(num_iterations, discretisation,
                                  factor_to_var_msgs, var_to_factor_msgs, beliefs,
                                  factor_connections, var_neighbors, factor_functions,
                                  priors, prior_indices,
                                  var_neighbor_to_factor_neighbor_idx, factor_to_var_neighbor_idx)

    print("BP Stage 3: Updating graph objects with final beliefs...")
    for i, variable in enumerate(graph.variables):
        variable.belief = final_beliefs[i, :]

    return graph

def convert_graph_to_gaussian(graph):
    """
    Convert all variable beliefs and factor functions to Gaussian approximations
    """
    print("Converting factor graph to Gaussian approximations...")
    
    # Convert variable beliefs to Gaussian
    for i, variable in enumerate(graph.variables):
        if i % 10000 == 0:  # Progress indicator
            print(f"Converting variable beliefs: {i}/{len(graph.variables)}")
            
        # Find best Gaussian fit for this variable's belief
        _, optimal_sigma, optimal_mean = opt.optimise_gaussian_kl(
            variable.belief, cfg.measurement_range
        )
        
        # Replace belief with Gaussian approximation
        variable.belief = dm.create_gaussian_distribution(
            cfg.measurement_range, optimal_sigma, mu=optimal_mean
        )
        
        # Store original for comparison if needed
        if not hasattr(variable, 'original_belief'):
            variable.original_belief = variable.belief.copy()
    
    # Convert factor functions to Gaussian
    print("Converting factor functions...")
    for i, factor in enumerate(graph.factors):
        if factor.factor_type == "prior":
            _, opt_sig, opt_mean = opt.optimise_gaussian_kl(factor.function, cfg.measurement_range)
            
            factor.function = dm.create_gaussian_distribution(cfg.measurement_range, opt_sig, mu=opt_mean)
            
            
        elif factor.factor_type == "smoothing":
            # Convert pairwise smoothing factors
            # For 2D factor functions, we need a different approach
            factor.function = convert_pairwise_factor_to_gaussian(factor.function)
    
    print("Gaussian conversion complete!")
    return graph

def convert_pairwise_factor_to_gaussian(factor_matrix):
    """
    Convert a 2D pairwise factor matrix to a Gaussian form
    """
    height, width = factor_matrix.shape
    original_kernel = factor_matrix[height//2,:]
    
    kernel_range = np.linspace(cfg.min_measurement, cfg.max_measurement, width, dtype=np.float64)
    
    _, opt_sig, opt_mean = opt.optimise_gaussian_kl(original_kernel, kernel_range)
    
    gaussian_kernel = dm.create_gaussian_distribution(kernel_range, opt_sig, mu=opt_mean)
    gaussian_smoothing_function = dm.create_smoothing_factor_distribution(len(kernel_range), kernel=gaussian_kernel)
    
    return dm.normalise_rows(gaussian_smoothing_function)


def run_gaussian_belief_propagation(graph, num_iterations):
    """
    Main function to run Gaussian BP
    """
    gaussian_graph = convert_graph_to_gaussian(graph)
    result_graph = run_belief_propagation(gaussian_graph, num_iterations)
    return result_graph

def compare_gaussian_vs_original(graph):
    """
    Compare Gaussian approximations with original distributions
    """
    kl_divergences = []
    
    for variable in graph.variables:
        if hasattr(variable, 'original_belief'):
            kl_div = opt.kl_divergence_numba(variable.original_belief, variable.belief)
            kl_divergences.append(kl_div)
    
    print(f"Average KL divergence from original: {np.mean(kl_divergences):.4f}")
    print(f"Max KL divergence: {np.max(kl_divergences):.4f}")
    
    return kl_divergences

# # Optional: More sophisticated Gaussian factor approximation
# def fit_2d_gaussian_to_factor(factor_matrix):
#     """
#     Fit a 2D Gaussian to a pairwise factor matrix
#     """
#     rows, cols = factor_matrix.shape
    
#     # Create coordinate grids
#     x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    
#     # Flatten for fitting
#     coords = np.column_stack([x.ravel(), y.ravel()])
#     values = factor_matrix.ravel()
    
#     # Calculate weighted mean and covariance
#     total_weight = np.sum(values)
#     if total_weight == 0:
#         return factor_matrix
    
#     mean_x = np.sum(coords[:, 0] * values) / total_weight
#     mean_y = np.sum(coords[:, 1] * values) / total_weight
    
#     # Calculate covariance matrix
#     diff_x = coords[:, 0] - mean_x
#     diff_y = coords[:, 1] - mean_y
    
#     var_xx = np.sum(diff_x * diff_x * values) / total_weight
#     var_yy = np.sum(diff_y * diff_y * values) / total_weight
#     var_xy = np.sum(diff_x * diff_y * values) / total_weight
    
#     # Create 2D Gaussian
#     gaussian_2d = np.zeros_like(factor_matrix)
    
#     det_cov = var_xx * var_yy - var_xy * var_xy
#     if det_cov <= 0:
#         return factor_matrix  # Fallback to original
    
#     inv_det = 1.0 / det_cov
    
#     for i in range(rows):
#         for j in range(cols):
#             dx = j - mean_x
#             dy = i - mean_y
            
#             exponent = -0.5 * inv_det * (var_yy * dx * dx - 2 * var_xy * dx * dy + var_xx * dy * dy)
#             gaussian_2d[i, j] = np.exp(exponent)
    
    # # Normalize
    # return dm.normalise(gaussian_2d.flatten()).reshape(gaussian_2d.shape)