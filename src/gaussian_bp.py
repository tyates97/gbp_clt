# Add this to a new file: gaussian_bp.py
import numpy as np
import optimisation as opt
import distribution_management as dm
import config as cfg

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
    
    return dm.normalise(gaussian_smoothing_function)


def run_gaussian_belief_propagation(graph, num_iterations):
    """
    Main function to run Gaussian BP
    """
    gaussian_graph = convert_graph_to_gaussian(gaussian_graph)

    # Run standard belief propagation on Gaussian approximations
    # Import your existing BP module
    import belief_propagation as bp
    result_graph = bp.run_belief_propagation(gaussian_graph, num_iterations)
    
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

# Optional: More sophisticated Gaussian factor approximation
def fit_2d_gaussian_to_factor(factor_matrix):
    """
    Fit a 2D Gaussian to a pairwise factor matrix
    """
    rows, cols = factor_matrix.shape
    
    # Create coordinate grids
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    
    # Flatten for fitting
    coords = np.column_stack([x.ravel(), y.ravel()])
    values = factor_matrix.ravel()
    
    # Calculate weighted mean and covariance
    total_weight = np.sum(values)
    if total_weight == 0:
        return factor_matrix
    
    mean_x = np.sum(coords[:, 0] * values) / total_weight
    mean_y = np.sum(coords[:, 1] * values) / total_weight
    
    # Calculate covariance matrix
    diff_x = coords[:, 0] - mean_x
    diff_y = coords[:, 1] - mean_y
    
    var_xx = np.sum(diff_x * diff_x * values) / total_weight
    var_yy = np.sum(diff_y * diff_y * values) / total_weight
    var_xy = np.sum(diff_x * diff_y * values) / total_weight
    
    # Create 2D Gaussian
    gaussian_2d = np.zeros_like(factor_matrix)
    
    det_cov = var_xx * var_yy - var_xy * var_xy
    if det_cov <= 0:
        return factor_matrix  # Fallback to original
    
    inv_det = 1.0 / det_cov
    
    for i in range(rows):
        for j in range(cols):
            dx = j - mean_x
            dy = i - mean_y
            
            exponent = -0.5 * inv_det * (var_yy * dx * dx - 2 * var_xy * dx * dy + var_xx * dy * dy)
            gaussian_2d[i, j] = np.exp(exponent)
    
    # Normalize
    return dm.normalise(gaussian_2d.flatten()).reshape(gaussian_2d.shape)
