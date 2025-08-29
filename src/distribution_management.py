
# External Libraries
import numpy as np
import numba
import matplotlib.pyplot as plt

 # Internal modules
import config as cfg

cfg.rng = np.random.default_rng(seed=42)

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

# creates a gaussian distribution
@numba.jit(nopython=True)
def create_gaussian_distribution(x, sigma, mu=0):
    mean = mu
    coef = 1 / (np.sqrt(2 * np.pi) * sigma)
    exponent = -((x - mean) ** 2) / (2 * sigma ** 2)
    return normalise(coef * np.exp(exponent))

def downsample_variance(distribution, target_width):
    max_width = len(distribution)
    current_width = np.count_nonzero(distribution)
    adjusted_distribution = distribution
    if target_width > max_width:
        raise ValueError("target distribution width is greater than max distribution width")
    if target_width < current_width:
        #downsampling code
        adjusted_distribution[0: int((len(distribution)/2)-target_width//2)] = np.zeros(int((len(distribution)/2)-target_width//2))
        adjusted_distribution[int((len(distribution)/2)+target_width//2+target_width%2):] = np.zeros(int((len(distribution)/2)-target_width//2-target_width%2))
    elif target_width >= current_width:
        # TODO: write below
        # upsampling code
        pass
    return normalise(adjusted_distribution)


@numba.jit(nopython=True)
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
    unnormalised_prior[start:end] = np.random.rand(end-start)    
    return normalise(unnormalised_prior)


@numba.jit(nopython=True)
# creates a smoothing factor that encourages neighbouring variables to have the same factor
def create_smoothing_factor_distribution(discretisation, kernel=None, mrange=cfg.measurement_range):
    # if kernel is None:
    #     if mrange is None:
    #         raise ValueError("measurement_range required (pass it or set config.measurement_range).")
    #     kernel = create_random_prior_distribution(mrange, mean=None, prior_width=cfg.smoothing_width)
    
    # Create triangular kernel favoring small disparity differences
    x = np.linspace(-cfg.smoothing_width/2, cfg.smoothing_width/2, cfg.smoothing_width)
    kernel = np.maximum(0, 1 - np.abs(x)/(cfg.smoothing_width/4))

    kernel = np.asarray(kernel)
    extended_len = 2*discretisation-1
    extended_kernel = np.zeros(extended_len)

    # Place original kernel in the center of the extended kernel
    start_idx = (extended_len-len(kernel)) // 2
    extended_kernel[start_idx:start_idx+len(kernel)] = kernel

    # ## Show extended_kernel
    # plt.plot(range(2*cfg.belief_discretisation-1), extended_kernel)
    # plt.show()

    # Create pairwise factor matrix
    unnormalised_factor_values = np.zeros((discretisation, discretisation))

    center = extended_len//2
    # Fill in the factor matrix constraining variables to be similar
    for x1_row in range(discretisation):
        for x2_col in range(discretisation):
            diff = x2_col-x1_row
            idx = diff + center
            # if 0 <= idx < extended_len:
            # unnormalised_factor_values[x2_col, x1_row] = extended_kernel[idx]
            unnormalised_factor_values[x1_row, x2_col] = extended_kernel[idx]
    
    ## DEBUGGING: Show 2d array
    # plt.figure()
    # plt.imshow(unnormalised_factor_values)
    # plt.figure()
    # plt.plot(range(2*cfg.belief_discretisation-1), extended_kernel)
    # plt.show()

    return normalise(unnormalised_factor_values)


