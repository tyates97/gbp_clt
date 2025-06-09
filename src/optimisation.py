# TODO: how does changing the number of loops/variables/priors affect the optimal q?
# does it tend towards a gaussian (q=1) with more loops? More variables?
# External libraries
import numpy as np
from sklearn.metrics import mean_squared_error

# Local modules
import distribution_management as dm

''' function to optimise the q and sigma parameters of a q-gaussian to fit a target belief '''
def optimise_q_gaussian(target_belief, measurement_range):
    num_q_steps = 50
    q_search_values = np.linspace(0, 3, num_q_steps)                   # q must be 0 < q < 3

    num_sigma_steps = 50
    sigma_search_values = np.linspace(0.1, 5, num_sigma_steps)

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
            current_mse = mean_squared_error(target_belief, y_q_gauss)

            if np.isnan(current_mse):
                continue

            if current_mse < min_mse:
                min_mse = current_mse
                optimal_q = q_candidate
                optimal_sigma = sigma_candidate

    return min_mse, optimal_q, optimal_sigma