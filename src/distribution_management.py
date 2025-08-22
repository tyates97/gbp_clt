
# External Libraries
import numpy as np

# creates a discrete distribution for the variable priors
def create_prior_distribution(
        # distribution_type,
         measurement_range
        #, gauss_sigma=2.3
    ):

    discretisation = len(measurement_range)
    unnormalised_prior = np.zeros(discretisation)
    measurement_mean = np.mean(measurement_range)

    # if distribution_type == 'random':
    unnormalised_prior[discretisation//4:3*discretisation//4] = np.random.rand(discretisation//2)

    # if distribution_type == 'random symmetric':
    #     unnormalised_prior[discretisation//4:discretisation//2] = np.random.rand(discretisation//4)
    #     unnormalised_prior[discretisation//2:3*discretisation//4] = unnormalised_prior[discretisation//4:discretisation//2][::-1]

    # if distribution_type == 'horns':
    #     width = (max(measurement_range)-measurement_mean)/5         # If horns are too far apart, convolution won't work
    #     unnormalised_prior = np.where(np.abs(measurement_range - measurement_mean) < width, 0.0, 1.0)

    # if distribution_type == 'skew':
    #     unnormalised_prior = np.where(measurement_range < measurement_mean, 1.0, 0.0)

    # if distribution_type == 'gaussian':
    #     unnormalised_prior = create_gaussian_distribution(measurement_range, gauss_sigma)

    # if distribution_type == 'top hat':
    #     top_hat_width = (max(measurement_range) - measurement_mean)/2
    #     unnormalised_prior = np.where(np.abs(measurement_range - measurement_mean) < top_hat_width, 1.0, 0.0)
    return normalise(unnormalised_prior)


# creates a smoothing factor that encourages neighbouring variables to have the same factor
def create_smoothing_factor_distribution(
        discretisation,
        prior=None
    ):
    unnormalised_factor_values = np.zeros((discretisation, discretisation))
    # Fill in the factor matrix
    for x1_rows in range(discretisation):
        for x2_cols in range(discretisation):
            diff = x2_cols-x1_rows
            idx = diff + discretisation//2
            if 0 <= idx < discretisation:
                unnormalised_factor_values[x1_rows, x2_cols] = prior[-idx]
    return normalise(unnormalised_factor_values)


# normalises a set of distribution values so their sum adds to 1
def normalise(distribution_values):
    sum_value = np.sum(distribution_values)
    if sum_value > 0:
        normalised_values = distribution_values / sum_value
        return np.clip(normalised_values, 1e-10, 1e10)  # Clamp values to avoid extreme ranges

    else:
        # Return uniform distribution instead
        return np.ones_like(distribution_values) / len(distribution_values)

# creates a gaussian distribution
def create_gaussian_distribution(x, sigma, mu=0):
    mean = mu
    coef = 1 / (np.sqrt(2 * np.pi) * sigma)
    exponent = -((x - mean) ** 2) / (2 * sigma ** 2)
    return normalise(coef * np.exp(exponent))

# creates a q-gaussian distribution for a given q, beta
def create_q_gaussian_distribution(x, q, sigma):
    beta = 1/(2*sigma**2)
    if q == 1:
        return normalise(np.exp(-beta * (x ** 2)))
    else:
        factor = 1 - (1-q)*beta * x**2
        safe_factor = np.maximum(factor, 0)
        exponent = 1/(1-q)
        unnormalised_distribution = safe_factor ** exponent
        return normalise(unnormalised_distribution)