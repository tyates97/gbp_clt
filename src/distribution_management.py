
# External Libraries
import numpy as np

 # Internal modules
import config as cfg

cfg.rng = np.random.default_rng(seed=42)

# normalises a set of distribution values so their sum adds to 1
def normalise(distribution_values):
    sum_value = np.sum(distribution_values)
    if sum_value > 0:
        normalised_values = distribution_values / sum_value
        return np.clip(normalised_values, 1e-10, 1e10)  # Clamp values to avoid extreme ranges

    else:
        # Return uniform distribution instead
        return np.ones_like(distribution_values) / len(distribution_values)

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

# creates a gaussian distribution
def create_gaussian_distribution(x, sigma, mu=0):
    mean = mu
    coef = 1 / (np.sqrt(2 * np.pi) * sigma)
    exponent = -((x - mean) ** 2) / (2 * sigma ** 2)
    return normalise(coef * np.exp(exponent))


# creates a random discrete distribution for the variable priors
def create_random_prior_distribution(x_range, prior_width=None):
    if prior_width is None:
        prior_width = cfg.prior_width

    discretisation = len(x_range)
    unnormalised_prior = np.zeros(discretisation)
    # unnormalised_prior[discretisation//4:3*discretisation//4] = cfg.rng.random(discretisation//2)
    start = max(0, discretisation//2-prior_width//2)
    end = min(discretisation, discretisation//2+prior_width//2)
    unnormalised_prior[start:end] = np.ones(prior_width)
    # unnormalised_prior[start:end] = cfg.rng.random(prior_width)
    return normalise(unnormalised_prior)


# creates a smoothing factor that encourages neighbouring variables to have the same factor
def create_smoothing_factor_distribution(discretisation, kernel=None, mrange=cfg.measurement_range):
    if kernel is None:
        if mrange is None:
            raise ValueError("measurement_range required (pass it or set config.measurement_range).")
        kernel = create_random_prior_distribution(mrange, cfg.smoothing_width)
    
    # kernel = downsample_variance(kernel, target_width=cfg.smoothing_width)  # Adjust kernel width to be 3/4 of the belief range

    # Now build the pairwise factor matrix: center index corresponds to diff == 0
    center = (len(kernel)-1) // 2
    unnormalised_factor_values = np.zeros((discretisation, discretisation))
    
    # Fill in the factor matrix constraining variables to be similar
    for x1_rows in range(discretisation):
        for x2_cols in range(discretisation):
            diff = x2_cols-x1_rows
            idx = diff + center
            if 0 <= idx < discretisation:
                unnormalised_factor_values[x1_rows, x2_cols] = kernel[-idx]
    return normalise(unnormalised_factor_values)


