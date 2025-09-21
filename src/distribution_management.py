
# External Libraries
import numpy as np
import numba
import matplotlib.pyplot as plt

 # Internal modules
import config as cfg

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
    n, m = distribution_values.shape
    out = np.empty_like(distribution_values)
    for i in range(n):
        s = 0.0
        for j in range(m):
            s += distribution_values[i, j]
        if s == 0.0:
            inv = 1.0 / m
            for j in range(m):
                out[i, j] = inv
        else:
            inv = 1.0 / s
            for j in range(m):
                out[i, j] = distribution_values[i, j] * inv
    return out


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

# @numba.jit(nopython=True)
# def create_smoothing_factor_distribution(discretisation, kernel=None, mrange=0, hist=None, smoothing_function='histogram', triangular_width=26):
#     """
#     Reflective (mirrored) pairwise factor:
#       f(x1, x2) = k(reflect_diff(x2 - x1, N))
#     where k is a 1-D kernel and reflect_diff handles boundary reflection
#     instead of wrapping.
#     """
#     N = discretisation

#     # --- choose base 1-D kernel source (hist > kernel > default triangle)
#     if smoothing_function == 'histogram' and hist is not None:
#         base = hist.astype(np.float64)
#     elif smoothing_function == "triangular":
#         width = max(1, min(triangular_width, N))
#         base = _make_default_triangular_kernel(width)
#     else:
#         # fallback to uniform if nothing else works
#         base = np.ones(1, dtype=np.float64)

#     # --- embed the base kernel on a reflected range
#     # We need to handle both positive and negative differences with reflection
#     max_diff = N - 1  # maximum possible difference
#     kernel_size = 2 * max_diff + 1  # size to accommodate -max_diff to +max_diff
#     kernel = np.zeros(kernel_size, dtype=np.float64)
    
#     Length = base.shape[0]
#     centre = Length // 2
#     kernel_centre = max_diff  # center of the new kernel array
    
#     for i in range(Length):
#         diff = i - centre  # signed offset from base kernel center
#         kernel_idx = kernel_centre + diff
#         if 0 <= kernel_idx < kernel_size:
#             kernel[kernel_idx] += base[i]

#     # normalise k to sum 1 (robust even if base was all zeros)
#     _normalise_vector_inplace(kernel)

#     # --- build the matrix with reflection: each row uses reflected differences
#     mat = np.empty((N, N), dtype=np.float64)
#     for x1 in range(N):
#         for x2 in range(N):
#             # Calculate raw difference
#             diff = x2 - x1
            
#             # # Apply reflection instead of wrapping
#             # # Reflect differences that go beyond boundaries
#             # if raw_diff > max_diff:
#             #     reflected_diff = 2 * max_diff - raw_diff
#             # elif raw_diff < -max_diff:
#             #     reflected_diff = -2 * max_diff - raw_diff
#             # else:
#             #     reflected_diff = raw_diff
            
#             # Map to kernel index
#             kernel_idx = kernel_centre + diff
#             if 0 <= kernel_idx < kernel_size:
#                 mat[x1, x2] = kernel[kernel_idx]
#             else:
#                 mat[x1, x2] = 0.0

#     return mat


###### Testing a reflected version instead of wraparound ######
@numba.jit(nopython=True)
def create_smoothing_factor_distribution_reflected(discretisation, kernel=None, mrange=0, hist=None, smoothing_function='histogram', triangular_width=26):
    """
    Non-circular pairwise factor with reflection at boundaries:
      f(x1, x2) = k(x2 - x1) where k is reflected at domain boundaries
    
    This ensures that disparities at the edges (0 and discretisation-1) don't 
    wrap around to each other, which makes more sense for disparity estimation.
    """
    N = discretisation

    # --- choose base 1-D kernel source (same as before)
    if smoothing_function == 'histogram' and hist is not None:
        base = hist.astype(np.float64)
    elif smoothing_function == "triangular":
        width = max(1, min(triangular_width, N))
        base = _make_default_triangular_kernel(width)
    else:
        base = np.ones(1, dtype=np.float64)

    # --- build the matrix with reflection at boundaries
    mat = np.empty((N, N), dtype=np.float64)
    base_length = base.shape[0]
    base_center = base_length // 2
    
    for x1 in range(N):
        # Initialize row
        row = np.zeros(N, dtype=np.float64)
        
        # Apply kernel centered at x1, with reflection at boundaries
        for i in range(base_length):
            offset = i - base_center  # signed offset from center
            target_idx = x1 + offset
            
            # Handle boundaries with reflection
            if target_idx < 0:
                # Reflect: if we'd go to -1, go to 1 instead
                target_idx = -target_idx
            elif target_idx >= N:
                # Reflect: if we'd go to N, go to N-2 instead
                target_idx = 2 * N - 2 - target_idx
            
            # Clamp to valid range (in case of multiple reflections)
            target_idx = max(0, min(N - 1, target_idx))
            
            row[target_idx] += base[i]
        
        # Normalize the row to sum to 1
        _normalise_vector_inplace(row)
        mat[x1, :] = row

    return mat
