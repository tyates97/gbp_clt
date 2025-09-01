import numpy as np
import numba

import distribution_management as dm

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
def get_cost_volume(left_image, right_image, patch_size, max_disparity):
    print("Calculating cost volume...")
    # Crop the image
    patch_width = patch_size//2
    height, width = left_image.shape
    cost_volume = np.zeros((height, width, max_disparity))

    # Calculate the disparity
    for y in range(patch_width, height-patch_width):
        for x in range(patch_width, width-patch_width):
            left_patch = left_image[y-patch_width:y+patch_width+1, x-patch_width:x+patch_width+1]
            for d in range(max_disparity):
                if x-d >= patch_width:
                    right_patch = right_image[y-patch_width:y+patch_width+1, x-d-patch_width:x-d+patch_width+1]
                    cost = np.sum(np.abs(left_patch-right_patch))
                    cost_volume[y,x,d] = cost
    
    return cost_volume


def get_pdfs_from_costs(cost_volume):
    print("Converting cost volume to pdf volume...")
    height,width,_ = cost_volume.shape
    # lambda_param = 0.1            # TESTING
    lambda_param = 0.2              # TESTING
    # lambda_param = 1.0            # TESTING
    pdf_volume = np.zeros(cost_volume.shape)

    for y in range(height):
        for x in range(width):
            cost = cost_volume[y,x,:]
            pdf = np.exp(-lambda_param*cost)
            pdf = dm.normalise(pdf) 
            pdf_volume[y,x,:] = pdf
    return pdf_volume