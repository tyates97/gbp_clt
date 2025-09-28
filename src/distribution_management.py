
# External Libraries
import numpy as np
import numba
import matplotlib.pyplot as plt

 # Internal modules
import config as cfg
import optimisation as opt

# normalises a set of distribution values so their sum adds to 1
@numba.jit(nopython=True)
def normalise(distribution_values):
    sum_value = np.sum(distribution_values)
    if sum_value > 0:
        normalised_values = distribution_values / sum_value
        return normalised_values

    else:
        # Return uniform distribution instead
        return np.ones_like(distribution_values) / len(distribution_values)
    
# normalises a 2d distribution so its rows sum to 1. (commented below is a non-numba, simplified version)
@numba.jit(nopython=True)
def normalise_rows(distribution_values):
    height, width = distribution_values.shape
    result = np.empty_like(distribution_values)
    for row in range(height):
        row_sum = 0.0
        for col in range(width):
            row_sum += distribution_values[row, col]
        if row_sum == 0.0:
            scaling_factor = 1.0 / width
            for col in range(width):
                result[row, col] = scaling_factor
        else:
            scaling_factor = 1.0 / row_sum
            for col in range(width):
                result[row, col] = distribution_values[row, col] * scaling_factor
    return result

def calculate_distributional_variance(pdf_volume):
    """Calculates the true variance of the disparity distributions."""
    height, width, _ = pdf_volume.shape
    variance_vol = np.zeros((height, width))

    # The x-values of our distribution (i.e., the disparity values)
    disparity_values = cfg.measurement_range

    for y in range(height):
        for x in range(width):
            pdf = pdf_volume[y, x, :]

            # E[X] = sum(x * p(x))
            mean = np.sum(disparity_values * pdf)

            # E[X^2] = sum(x^2 * p(x))
            mean_sq = np.sum((disparity_values**2) * pdf)

            # Var(X) = E[X^2] - (E[X])^2
            variance = mean_sq - (mean**2)
            variance_vol[y, x] = variance

    return variance_vol


# creates a gaussian distribution
@numba.jit(nopython=True)
def create_gaussian_distribution(x, sigma, mu=0):
    mean = mu
    coef = 1 / (np.sqrt(2 * np.pi) * sigma)
    exponent = -((x - mean) ** 2) / (2 * sigma ** 2)
    return normalise(coef * np.exp(exponent))



# @numba.jit(nopython=True)
# creates a random discrete distribution for the variable priors
def create_random_prior_distribution(x_range, mean=None, prior_width=None):
    if prior_width is None:
        prior_width = cfg.prior_width
    x = np.asarray(x_range)
    discretisation = x.size
    if mean is None:
        centre_idx = discretisation // 2
    elif isinstance(mean, (int, np.integer)):
        centre_idx = int(mean)
    else:
        # map measurement value to nearest discretisation index
        centre_idx = int(np.argmin(np.abs(x - float(mean))))
    
    centre_idx = max(0, min(discretisation-1, centre_idx)) # clamp to valid range
    half_width = prior_width // 2
    start = max(0, centre_idx-half_width)
    end = min(discretisation, centre_idx+half_width)
    
    
    unnormalised_prior = np.zeros(discretisation)
    unnormalised_prior[start:end] = cfg.rng.random(end-start)    
    return normalise(unnormalised_prior)


@numba.jit(nopython=True)
def get_histogram_from_truth(ground_truth):
    """
    Calculates disparity differences with Numba-accelerated for loops.
    Ignores occluded pixels (value 0).
    """
    height, width = ground_truth.shape
    # Pre-allocate a list with an estimated size for performance
    disp_diff_list = []

    for row in range(height):
        for col in range(width):
            pixel_disparity = ground_truth[row, col]
            
            # Ignore occluded/invalid pixels - very important
            if pixel_disparity == 0:
                continue

            # Check right neighbor
            if col + 1 < width:
                neighbour_disparity = ground_truth[row, col + 1]
                if neighbour_disparity > 0:
                    disp_diff_list.append(pixel_disparity - neighbour_disparity)
            
            # Check neighbor below
            if row + 1 < height:
                neighbour_disparity = ground_truth[row + 1, col]
                if neighbour_disparity > 0:
                    disp_diff_list.append(pixel_disparity - neighbour_disparity)
    
    # We want this to be centred on zero, so will count all differences both ways (e.g. from pixel 1 to 2 and from 2 to 1)
    diffs_array = np.array(disp_diff_list)
    all_diffs_symmetric = np.concatenate((diffs_array, -diffs_array))

    return all_diffs_symmetric






''' TEST '''
@numba.jit(nopython=True)
def _make_default_triangular_kernel(width):
    # width >= 1; triangle with peak 1 at center, linearly decaying to 0 at edges
    # Works for odd/even widths.
    kernel = np.zeros(width, dtype=np.float64)
    centre_idx = width // 2
    denominator = max(1.0, centre_idx)  # avoid div by zero when width=1 or 2
    for i in range(width):
        # symmetric triangular shape
        kernel[i] = max(0.0, 1.0 - abs(i - centre_idx) / denominator)
    
    # normalise to sum 1
    _normalise_vector_inplace(kernel)

    return kernel


@numba.jit(nopython=True)
def _normalise_vector_inplace(vector):

    sum = np.sum(vector)
    if sum > 0.0:
        for i in range(vector.shape[0]):
            vector[i] /= sum
    else:
        uniform_val = 1.0 / vector.shape[0]
        for i in range(vector.shape[0]):
            vector[i] = uniform_val

@numba.njit
def _reflect_index(j, N):
    # mirror padding: 0 1 2 ... N-2 N-1 | N-2 ... 2 1 | 0 1 ...
    if N <= 1:
        return 0
    while j < 0 or j >= N:
        if j < 0:
            j = -j - 1
        else:
            j = 2 * N - 1 - j
    return j

@numba.njit
def _normalise_row_inplace(v):
    s = 0.0
    for i in range(v.size):
        s += v[i]
    if s > 0.0:
        inv = 1.0 / s
        for i in range(v.size):
            v[i] *= inv



@numba.jit(nopython=True)
def create_smoothing_factor_distribution(discretisation, kernel=np.ones(1, dtype=np.float64), mrange=0, hist=None, smoothing_function='triangular', triangular_width=26):
    """
    Diagonal pairwise factor:
      f(x1, x2) = k((x2 - x1) mod N)
    where k is a 1-D kernel:
      - if `hist` is given, use it (1-D array) as the kernel support,
      - elif `kernel` is given, use it (1-D array),
      - else build a triangular kernel of width cfg.smoothing_width.
    The returned matrix has identical row sums (each row is a rotation of k),
    so uniform messages remain uniform; repeated BP corresponds to repeated
    *circular* convolutions, giving the Gaussianising effect cleanly.
    """
    N = discretisation

    # --- choose base 1-D kernel source (hist > kernel > default triangle)
    # if hist is not None:
    if smoothing_function == 'histogram' and hist is not None:
        base = hist.astype(np.float64)
    elif smoothing_function == "triangular":
        
        width = max(1,min(triangular_width, N))
        base = _make_default_triangular_kernel(width)
    else:
        # fallback to uniform if nothing else works
        base = kernel

    # --- embed the base kernel on a circle of length N
    # We center `base` and wrap its mass onto a length-N circular vector k.
    # kernel = np.zeros(N, dtype=np.float64)
    Length = base.shape[0]
    centre = Length // 2
    
    # --- build reflected Toeplitz-like matrix
    mat = np.zeros((N, N), dtype=np.float64)

    for x1 in range(N):
        # place the kernel centered at x1 with reflection at boundaries
        for i in range(Length):
            j = x1 + (i - centre)
            if 0 <= j < N:
                mat[x1, j] += base[i]
            # o = i - centre         # signed offset
            # j = x1 + o
            # j_ref = _reflect_index(j, N)
            # mat[x1, j_ref] += base[i]

        # ensure each row sums to 1 (important near boundaries where folding occurs)
        _normalise_row_inplace(mat[x1])

    return mat

def convert_graph_to_gaussian(graph):
    """
    Convert all variable beliefs and factor functions to Gaussian approximations
    """
    print("Converting factor graph to Gaussian approximations...")
    
    # Convert variable beliefs to Gaussian
    for i, variable in enumerate(graph.variables):
        if i % 10000 == 0:  # Progress indicator
            print(f"Converting variable beliefs: {i}/{len(graph.variables)}")

        original = variable.belief.copy()

        # Find best Gaussian fit for this variable's belief
        _, optimal_sigma, optimal_mean = opt.optimise_gaussian_kl(
            variable.belief, cfg.measurement_range
        )
        
        # Replace belief with Gaussian approximation
        variable.belief = create_gaussian_distribution(
            cfg.measurement_range, optimal_sigma, mu=optimal_mean
        )
        
        # Store original for comparison if needed
        if not hasattr(variable, 'original_belief'):
            variable.original_belief = original
    
    # Convert factor functions to Gaussian
    print("Converting factor functions...")
    for i, factor in enumerate(graph.factors):
        if factor.factor_type == "prior":
            _, opt_sig, opt_mean = opt.optimise_gaussian_kl(factor.function, cfg.measurement_range)
            
            factor.function = create_gaussian_distribution(cfg.measurement_range, opt_sig, mu=opt_mean)
            
            
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
    
    gaussian_kernel = create_gaussian_distribution(kernel_range, opt_sig, mu=opt_mean)
    gaussian_smoothing_function = create_smoothing_factor_distribution(len(kernel_range), kernel=gaussian_kernel)
    
    return normalise_rows(gaussian_smoothing_function)
