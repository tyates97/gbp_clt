import numpy as np
import numba
import cv2

import distribution_management as dm
import config as cfg

# Crop the image
def crop_image(image, cropped_dimensions, centre=None):
    if centre is None:                      # if centre of crop isn't provided, use middle of image
        img_y_mean = image.shape[0] // 2
        img_x_mean = image.shape[1] // 2
    else:                                   # otherwise use tuple provided
        img_y_mean = centre[0]
        img_x_mean = centre[1]
    
    cropped_y_width = cropped_dimensions[0]
    cropped_x_width = cropped_dimensions[1]

    return image[img_y_mean-(cropped_y_width//2):img_y_mean+(cropped_y_width//2), img_x_mean-(cropped_x_width//2):img_x_mean+(cropped_x_width//2)]

# Get the depth cost volume from two images
@numba.jit(nopython=True)
def get_cost_volume(left_img, right_img, patch_size, max_disparity, cost_function_str):
    left_image = left_img.astype(np.int32)
    right_image = right_img.astype(np.int32)
    # print("Calculating cost volume...")
    
    # Crop the image
    patch_width = int(patch_size//2)
    height, width = left_image.shape
    cost_volume = np.zeros((height, width, max_disparity+1))
    epsilon = 1e-6  # NCC

    # Calculate the cost
    for y in range(patch_width, height-patch_width):
        for x in range(patch_width, width-patch_width):
            left_patch = left_image[y - patch_width : y + patch_width + 1,
                                    x - patch_width : x + patch_width + 1]
            
            # NCC
            left_mu = np.mean(left_patch)
            left_sigma = np.std(left_patch)

            for d in range(max_disparity+1):
                if x-d >= patch_width:
                    right_patch = right_image[y - patch_width     : y + patch_width + 1, 
                                              x - d - patch_width :  x - d + patch_width + 1]
                    
                    # Normalised Cross Correlation (NCC)
                    if cost_function_str == "NCC":
                        right_mu = np.mean(right_patch)
                        right_sigma = np.std(right_patch)
                        numerator = np.mean((left_patch-left_mu)*(right_patch-right_mu))
                        denominator = left_sigma*right_sigma + epsilon
                        ncc_score = numerator/denominator
                        cost = max(1-ncc_score, 0)
                    
                    # Sum of Absolute Differences (SAD)
                    elif cost_function_str == "SAD":
                        cost = np.sum(np.abs(left_patch-right_patch))
                    
                    # Sum of Squared Differences (SSD)
                    elif cost_function_str == "SSD":
                        cost = np.sum(np.square(left_patch-right_patch))

                    cost_volume[y,x,d] = cost
    
    return cost_volume


def get_pdfs_from_costs(cost_volume):
    print("Converting cost volume to pdf volume...")
    height,width,max_disp_int = cost_volume.shape
    lambda_param = cfg.lambda_param
    
    # create pdf with 0.25 discretisation
    pdf_volume = np.zeros((height, width, cfg.belief_discretisation))

    for y in range(height):
        for x in range(width):
            cost = cost_volume[y,x,:]

            integer_disparities = np.arange(max_disp_int)
            interpolated_costs = np.interp(cfg.measurement_range, integer_disparities, cost)
            
            pdf = np.exp(-lambda_param*interpolated_costs)
            pdf = dm.normalise(pdf) 
            pdf_volume[y,x,:] = pdf
    return pdf_volume


def get_disparity_from_graph(graph):
    num_variables = len(graph.variables)
    num_cols = graph.grid_cols
    num_rows = int(np.ceil(num_variables/num_cols))
    
    disparity_volume = np.zeros((num_rows, num_cols))
    for i, variable in enumerate(graph.variables):
        row = i // num_cols
        col = i % num_cols

        MAP_index = np.argmax(variable.belief)
        MAP_disparity = cfg.measurement_range[MAP_index]
        disparity_volume[row][col] = MAP_disparity
    
    return disparity_volume
