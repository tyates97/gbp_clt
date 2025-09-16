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
                        # # NCC, OpenCV implementation
                        # ncc_score = cv2.matchTemplate(left_patch, right_patch, cv2.TM_CCOEFF_NORMED)[0,0]
                        # cost = max(1-ncc_score, 0)
                    
                    # Sum of Absolute Differences (SAD)
                    elif cost_function_str == "SAD":
                        cost = np.sum(np.abs(left_patch-right_patch))
                    
                    # Sum of Squared Differences (SSD)
                    elif cost_function_str == "SSD":
                        # if (y == 193) and (x == 225) and (d == 45):
                        #     print(f'patch_size: {patch_size}')
                        #     print(f"left_patch dtype: {left_patch.dtype}")
                        #     print(f"right_patch dtype: {right_patch.dtype}")
                        #     print(f"difference: {left_patch - right_patch}")
                        #     print(f"squared difference: {(left_patch - right_patch)**2}")
                        #     print(f"sum check: {np.sum((left_patch - right_patch)**2)}")
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






# # --- CENSUS TRANSFORM IMPLEMENTATION ---

# @numba.jit(nopython=True)
# def _census_transform_numba(image, window_size):
#     """
#     Computes the Census transform of an image.
#     For each pixel, it creates a bitstring descriptor based on the intensity
#     comparison of the center pixel with its neighbors.
#     """
#     height, width = image.shape
#     border = window_size // 2
#     # The descriptor will be a 64-bit integer
#     transformed_image = np.zeros((height, width), dtype=np.uint64)

#     for y in range(border, height - border):
#         for x in range(border, width - border):
#             center_pixel_val = image[y, x]
#             descriptor = np.uint64(0)
#             bit_pos = 0

#             # Iterate through the window around the center pixel
#             for v in range(-border, border + 1):
#                 for u in range(-border, border + 1):
#                     if v == 0 and u == 0:
#                         continue # Skip the center pixel

#                     # If neighbor is brighter, set the bit to 1
#                     if image[y + v, x + u] > center_pixel_val:
#                         descriptor |= (np.uint64(1) << bit_pos)
                    
#                     bit_pos += 1
            
#             transformed_image[y, x] = descriptor
            
#     return transformed_image

# @numba.jit(nopython=True)
# def _hamming_distance_numba(val1, val2):
#     """
#     Calculates the Hamming distance between two integer descriptors
#     by XORing them and counting the number of set bits.
#     """
#     xor_val = np.uint64(val1) ^ np.uint64(val2)
#     distance = 0
#     while xor_val > np.uint64(0):
#         # This idiom counts the number of set bits (popcount)
#         xor_val = xor_val & (xor_val - np.uint64(1))
#         distance += 1
#     return distance


# def get_cost_volume(left_image, right_image, patch_size, max_disparity):
#     """
#     Calculates the disparity cost volume using the Census transform
#     and Hamming distance.
#     """
#     print("Calculating cost volume using Census transform...")
#     height, width = left_image.shape
    
#     # A 64-bit descriptor can't support a window larger than 7x7
#     if patch_size > 7:
#         raise ValueError("patch_size cannot be greater than 7 for a 64-bit descriptor.")

#     # Ensure patch_size is odd to have a definite center
#     if patch_size % 2 == 0:
#         patch_size += 1
    
#     border = patch_size // 2

#     # 1. Pre-compute the Census transform for both images
#     print("Pre-computing Census transforms...")
#     left_census = _census_transform_numba(left_image, patch_size)
#     right_census = _census_transform_numba(right_image, patch_size)

#     # 2. Compute cost volume using Hamming distance
#     print("Calculating Hamming distances for cost volume...")
#     cost_volume = np.zeros((height, width, max_disparity), dtype=np.float32)

#     for y in range(border, height - border):
#         for x in range(border, width - border):
#             left_descriptor = left_census[y, x]
#             for d in range(max_disparity):
#                 if x - d >= border:
#                     right_descriptor = right_census[y, x - d]
                    
#                     # The cost is the Hamming distance
#                     cost = _hamming_distance_numba(left_descriptor, right_descriptor)
#                     cost_volume[y, x, d] = cost
#                 else:
#                     # Set a high cost if out of bounds
#                     cost_volume[y, x, d] = float(patch_size * patch_size -1)

#     return cost_volume

# # Crop the image
# def crop_image(image, cropped_dimensions, centre=None):
#     if centre is None:                      # if centre of crop isn't provided, use middle of image
#         img_y_mean = image.shape[0] // 2
#         img_x_mean = image.shape[1] // 2
#     else:                                   # otherwise use tuple provided
#         img_y_mean = centre[0]
#         img_x_mean = centre[1]
    
#     cropped_y_width = cropped_dimensions[0]
#     cropped_x_width = cropped_dimensions[1]

#     return image[img_y_mean-(cropped_y_width//2):img_y_mean+(cropped_y_width//2), img_x_mean-(cropped_x_width//2):img_x_mean+(cropped_x_width//2)]


# def get_pdfs_from_costs(cost_volume):
#     print("Converting cost volume to pdf volume...")
#     height, width, _ = cost_volume.shape
#     lambda_param = 0.01             # You will need to re-tune this (5.0 for NCC)
#     pdf_volume = np.zeros(cost_volume.shape)

#     for y in range(height):
#         for x in range(width):
#             cost = cost_volume[y,x,:]
            
#             # --- This is the essential step for controllable priors ---
#             stable_cost = cost - np.min(cost)
#             pdf = np.exp(-lambda_param * stable_cost)
#             # ---------------------------------------------------------

#             pdf = dm.normalise(pdf) 
#             pdf_volume[y, x, :] = pdf
            
#     return pdf_volume
