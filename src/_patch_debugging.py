import streamlit as st
import cv2
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Import your modules (assuming they're in the same directory)
import config as cfg
import image_processing as ip

# Configure Streamlit page
st.set_page_config(
    page_title="Stereo Vision Belief Propagation",
    page_icon="🔸",
    layout="wide"
)

def calculate_distributional_variance(pdf_volume):
    """Calculates the true variance of the disparity distributions."""
    height, width, max_disparity = pdf_volume.shape
    variance_vol = np.zeros((height, width))

    # The x-values of our distribution (i.e., the disparity values)
    # disparity_values = np.arange(max_disparity)
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

            # if y == 187 and x == 260:
                # print(f"mean: {mean}")
                # print(f"mean of squares: {mean_sq}")
                # print(f"variance: {variance}")

    # print(f"variance volume shape: {variance_vol.shape}")
    # print(f"max variance: {np.max(variance_vol)}")
    # print(f"max variance: {np.max(variance_vol)}")

    return variance_vol

def extract_patch(image, x, y, patch_size):
    """Extract a patch from the image centered at (x, y)"""
    half_patch = patch_size // 2
    height, width = image.shape
    
    # Calculate patch boundaries with bounds checking
    y_start = max(0, y - half_patch)
    y_end = min(height, y + half_patch + 1)
    x_start = max(0, x - half_patch)
    x_end = min(width, x + half_patch + 1)
    
    patch = image[y_start:y_end, x_start:x_end]
    
    # Pad patch if near boundaries to maintain consistent size
    if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
        padded_patch = np.zeros((patch_size, patch_size), dtype=patch.dtype)
        pad_y_start = (patch_size - patch.shape[0]) // 2
        pad_x_start = (patch_size - patch.shape[1]) // 2
        padded_patch[pad_y_start:pad_y_start + patch.shape[0], 
                    pad_x_start:pad_x_start + patch.shape[1]] = patch
        return padded_patch
    
    return patch

def create_image_patch_selector(image_data, title, key_prefix, patch_size, disparity=0, cmap="gray"):
    """Create an image with patch selector and display the magnified patch below"""
    st.write(f"**{title}**")
    
    # Create sliders for coordinate selection
    col1, col2 = st.columns(2)
    
    with col1:
        x_coord = st.slider(
            "X coordinate", 
            min_value=0, 
            max_value=image_data.shape[1]-1, 
            value=image_data.shape[1]//2,
            key=f"{key_prefix}_x"
        )
    
    with col2:
        y_coord = st.slider(
            "Y coordinate", 
            min_value=0, 
            max_value=image_data.shape[0]-1, 
            value=image_data.shape[0]//2,
            key=f"{key_prefix}_y"
        )
    
    # Create the main image plot
    fig = px.imshow(image_data, color_continuous_scale=cmap)
    
    # Calculate patch boundaries
    half_patch = patch_size // 2
    height, width = image_data.shape
    
    # For right image, adjust x coordinate by disparity
    if key_prefix == "right":
        display_x = x_coord - disparity
    else:
        display_x = x_coord
    
    # Calculate actual patch boundaries (with bounds checking)
    y_start = max(0, y_coord - half_patch)
    y_end = min(height, y_coord + half_patch + 1)
    x_start = max(0, display_x - half_patch)
    x_end = min(width, display_x + half_patch + 1)
    
    # Add red square around the patch area
    fig.add_shape(
        type="rect",
        x0=x_start - 0.5,
        y0=y_start - 0.5,
        x1=x_end - 0.5,
        y1=y_end - 0.5,
        line=dict(color="red", width=3),
        fillcolor="rgba(255,0,0,0.1)"  # Semi-transparent red fill
    )
    
    # Add center point for precise coordinate reference
    fig.add_trace(go.Scatter(
        x=[display_x],
        y=[y_coord],
        mode='markers',
        marker=dict(
            size=8,
            color='red',
            symbol='cross',
            line=dict(color='white', width=1)
        ),
        name=f'Center ({display_x}, {y_coord})',
        showlegend=False
    ))
    
    if key_prefix == "right":
        fig.update_layout(
            title=f"Selected: ({display_x}, {y_coord}) | Patch: {patch_size}×{patch_size} | Disparity: {disparity}",
            xaxis_title="X",
            yaxis_title="Y",
            height=400,
            width=500,
            showlegend=False
        )
    else:
        fig.update_layout(
            title=f"Selected: ({display_x}, {y_coord}) | Patch: {patch_size}×{patch_size}",
            xaxis_title="X",
            yaxis_title="Y",
            height=400,
            width=500,
            showlegend=False
        )
    
    # Remove the color scale bar
    fig.update_coloraxes(showscale=False)
    
    # Display the image
    st.plotly_chart(fig, use_container_width=True, key=f"image_selector_{key_prefix}")
    
    # Extract and display the patch below the image
    patch = extract_patch(image_data, display_x, y_coord, patch_size)
    
    # Create patch visualization with absolute color scaling
    fig_patch = px.imshow(
        patch, 
        color_continuous_scale='gray',
        zmin=0,  # Set minimum value to 0
        zmax=64,  # Set maximum value to 255 (assuming 8-bit images)
        title=f"Patch {patch_size}×{patch_size}"
    )
    
    fig_patch.update_layout(
        height=300,
        width=300,
        showlegend=False
    )
    fig_patch.update_coloraxes(showscale=False)
    
    # Add grid lines to show patch structure
    for i in range(patch_size + 1):
        fig_patch.add_hline(y=i-0.5, line_color="red", line_width=0.5, opacity=0.3)
        fig_patch.add_vline(x=i-0.5, line_color="red", line_width=0.5, opacity=0.3)
    
    st.plotly_chart(fig_patch, use_container_width=True, key=f"patch_{key_prefix}")
    
    # Show patch statistics
    st.write(f"**Patch Statistics:**")
    st.write(f"Mean: {np.mean(patch):.2f}")
    st.write(f"Std: {np.std(patch):.2f}")
    
    # Calculate similarity metrics if both patches are available
    if key_prefix == "right" and hasattr(st.session_state, 'left_patch'):
        left_patch = st.session_state.left_patch
        
        if cfg.cost_function == 'NCC':
            # Normalized Cross Correlation
            left_norm = left_patch - np.mean(left_patch)
            right_norm = patch - np.mean(patch)
            if np.std(left_norm) > 0 and np.std(right_norm) > 0:
                ncc = np.sum(left_norm * right_norm) / (np.sqrt(np.sum(left_norm**2)) * np.sqrt(np.sum(right_norm**2)))
                st.write(f"**NCC: {ncc:.4f}**")
            else:
                st.write("**NCC: undefined (zero variance)**")
        elif cfg.cost_function == 'SAD':
            sad = np.sum(np.abs(left_patch - patch))
            st.write(f"**SAD: {sad:.2f}**")
        elif cfg.cost_function == 'SSD':
            ssd = np.sum((left_patch.astype(np.int32) - patch.astype(np.int32))**2)
            # print(f"left_patch dtype: {left_patch.dtype}")
            # print(f"right_patch dtype: {patch.dtype}")
            # print(f"difference: {left_patch - patch}")
            # print(f"squared difference: {(left_patch - patch)**2}")
            # print(f"sum check: {np.sum((left_patch - patch)**2)}")
            st.write(f"**SSD: {ssd}**")
    
    # Store patch in session state for comparison
    if key_prefix == "left":
        st.session_state.left_patch = patch
    
    return x_coord, y_coord, patch

@st.cache_data
def load_images():
    """Load and cache the stereo images and ground truth"""
    try:
        # Define base paths
        image_dir = 'data/stereo/teddy/'
        left_image_filename = "im2.png"
        right_image_filename = "im6.png"
        left_ground_truth_filename = "disp2.png"

        # Load the images
        left_image = cv2.imread(image_dir + left_image_filename, cv2.IMREAD_GRAYSCALE)/4
        right_image = cv2.imread(image_dir + right_image_filename, cv2.IMREAD_GRAYSCALE)/4
        ground_truth = cv2.imread(image_dir + left_ground_truth_filename, cv2.IMREAD_GRAYSCALE)/4
        
        # resizing for faster processing
        left_image = ip.crop_image(left_image, (150, 200))
        right_image = ip.crop_image(right_image, (150, 200))
        ground_truth = ip.crop_image(ground_truth, (150, 200))


        cfg.max_measurement = int(np.ceil(np.max(ground_truth)))
        # print("test")
        # print(f"max disparity in ground truth: {np.max(ground_truth)/4}")

        if left_image is None or right_image is None or ground_truth is None:
            st.error("Could not load images. Please check the file paths.")
            return None, None, None
            
        return left_image, right_image, ground_truth
    except Exception as e:
        st.error(f"Error loading images: {e}")
        return None, None, None

@st.cache_data
def compute_cost_volume(left_image, right_image, patch_size, max_disparity, cost_function):
    return ip.get_cost_volume(left_image, right_image, patch_size, max_disparity, cost_function)

@st.cache_data
def compute_pdf_volume(cost_volume, lambda_param):
    return ip.get_pdfs_from_costs(cost_volume)

def plot_pixel_data(data_volume, x, y, title, x_label="Index", y_label="Value", highlight_disparity=None):
    """Plot data for a specific pixel with optional disparity highlighting"""
    if x < data_volume.shape[1] and y < data_volume.shape[0]:
        pixel_data = data_volume[y, x, :]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=np.linspace(cfg.min_measurement, cfg.max_measurement, data_volume.shape[2]),
            y=pixel_data,
            mode='lines+markers',
            name='Data',
            line=dict(color='blue'),
            marker=dict(color='blue', size=6)
        ))
        
        # Highlight the selected disparity if provided
        if highlight_disparity is not None and highlight_disparity < len(pixel_data):
            # Find the closest index in measurement_range to highlight_disparity
            disparity_idx = np.argmin(np.abs(cfg.measurement_range - highlight_disparity))
            
            if disparity_idx < len(pixel_data):
                fig.add_trace(go.Scatter(
                    x = [cfg.measurement_range[disparity_idx]],
                    y = [pixel_data[disparity_idx]],
                    
                    # x=[highlight_disparity],
                    # y=[pixel_data[highlight_disparity]],
                    mode='markers',
                    name=f'Current Disparity ({highlight_disparity})',
                    marker=dict(color='red', size=12, symbol='circle')
                ))
        
        fig.update_layout(
            title=f"{title} at Pixel (x={x}, y={y})",
            xaxis_title=x_label,
            yaxis_title=y_label,
            yaxis=dict(range=[0, None]),  # Forces y-axis to start at 0
            height=500,
            width=500
        )
        
        return fig
    return None


def main():
    st.title("🔍 Stereo Vision Belief Propagation Interactive App")
    st.sidebar.title("Controls")
    # print(f"here0: {cfg.max_measurement}")

    # Load images
    left_image, right_image, ground_truth = load_images()
    # print(f"here1: {cfg.max_measurement}")
    
    if left_image is None:
        st.stop()
    
    # Configuration parameters
    st.sidebar.header("Parameters")
    patch_area = st.sidebar.selectbox("Patch Size", [9, 25, 49], index=1)
    patch_size = int(np.sqrt(patch_area))  # Convert area to actual patch dimension
    num_iterations = st.sidebar.slider("BP Iterations", 1, 20, 10)

    # Display the actual patch dimensions for clarity
    st.sidebar.write(f"Actual patch: {patch_size}×{patch_size}")

    # Store selected parameters (don't apply them yet)
    selected_cost_function = st.sidebar.selectbox("Select Cost Function", 
                                        ['NCC', 'SAD', 'SSD'], 
                                        index=['NCC', 'SAD', 'SSD'].index(getattr(cfg, 'cost_function', 'SSD'))
                                        )
    selected_lambda_param = st.sidebar.number_input("Lambda (λ)", 
                                        min_value=0.00000001, 
                                        max_value=10.00000000, 
                                        value=getattr(cfg, 'lambda_param', 0.00200000), 
                                        step=0.00000001,
                                        format="%.8f"
                                        )

    # Check if parameters have changed
    current_cost_function = getattr(cfg, 'cost_function', 'NCC')
    current_lambda_param = getattr(cfg, 'lambda_param', 5.0)

    params_changed = (selected_cost_function != current_cost_function or 
                    selected_lambda_param != current_lambda_param)

    # Show current vs selected parameters
    if params_changed:
        st.sidebar.warning("⚠️ Parameters changed - click Update to apply")
        st.sidebar.write(f"Current: {current_cost_function}, λ={current_lambda_param}")
        st.sidebar.write(f"Selected: {selected_cost_function}, λ={selected_lambda_param}")

    # Add refresh button
    update_clicked = st.sidebar.button("🔄 Update Parameters", 
                                    type="primary" if params_changed else "secondary",
                                    disabled=not params_changed)

    if update_clicked:
        # Apply the new parameters
        cfg.cost_function = selected_cost_function
        cfg.lambda_param = selected_lambda_param
        
        # Clear relevant caches to force recomputation
        compute_cost_volume.clear()
        compute_pdf_volume.clear()
        
        # Show success message
        st.sidebar.success("✅ Parameters updated!")
        
        # Rerun the app to apply changes
        st.rerun()

    # If parameters haven't been set yet, initialize them
    if not hasattr(cfg, 'cost_function'):
        cfg.cost_function = 'NCC'
    if not hasattr(cfg, 'lambda_param'):
        cfg.lambda_param = 1.0
    
    # Set up configuration
    cfg.min_measurement = 0
    cfg.max_measurement = int(np.ceil(np.max(ground_truth)))
    cfg.measurement_range = np.arange(cfg.min_measurement, cfg.max_measurement+0.25, 0.25)
    cfg.belief_discretisation = len(cfg.measurement_range)
    
    # Compute cost and PDF volumes
    with st.spinner("Computing cost volume..."):
        cost_volume = compute_cost_volume(left_image, right_image, patch_size, cfg.max_measurement, cfg.cost_function)
    # print(cfg.max_measurement)
    # print(cost_volume.shape)
    
    with st.spinner("Converting costs to PDFs..."):
        pdf_volume = compute_pdf_volume(cost_volume, cfg.lambda_param)

    # Interactive cost function inspector
    st.header("💰 Cost Function Inspector")
    
    # Add disparity slider at the top
    # print(f"here3: {cfg.max_measurement}")
    disparity = st.slider(
        "Disparity for Right Image Patch", 
        min_value=0, 
        max_value=cfg.max_measurement, 
        value=0,
        key="main_disparity",
        help="Shift the right patch horizontally to see different disparity comparisons"
    )
    
    # Create two columns for left and right images with patches
    col1, col2 = st.columns(2)
    
    with col1:
        x_left, y_left, left_patch = create_image_patch_selector(
            left_image, 
            "Left Image - Select coordinates for cost inspection", 
            "left", 
            patch_size, 
            cmap="gray"
        )
    
    with col2:
        x_right, y_right, right_patch = create_image_patch_selector(
            right_image, 
            "Right Image - Shows patch shifted by disparity", 
            "right", 
            patch_size,
            disparity,
            cmap="gray"
        )

    # Cost function equations and plots
    st.header("📊 Cost and Probability Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        if cfg.cost_function == 'NCC':
            st.write("Cost function: Normalised Cross Correlation")
            st.latex(r'''C(x_i) = \frac{\sum_{patch} [I_L-\mathbb{E}(I_L)][I_R-\mathbb{E}(I_R)]}
                        {\sqrt{\sum_{patch} (I_l-\mathbb{E}(I_L))^2}\sqrt{\sum_{patch} (I_R-\mathbb{E}(I_R))^2}}
                ''')
        elif cfg.cost_function == 'SAD':
            st.write("Cost function: Sum of Absolute Differences")
            st.latex(r'''C(x_i) = \sum_{patch} |I_L - I_R|''')
        elif cfg.cost_function == 'SSD':
            st.write("Cost function: Sum of Squared Differences")
            st.latex(r'''C(x_i) = \sum_{patch} (I_L - I_R)^2''')
        
        # Generate and display the cost plot with disparity highlighting
        cost_plot = plot_pixel_data(
            cost_volume, 
            x_left, 
            y_left, 
            "Cost Function", 
            "Disparity", 
            "Cost",
            highlight_disparity=disparity
        )
        if cost_plot:
            st.plotly_chart(cost_plot, use_container_width=True)
    
    with col2: 
        st.write("Probability Conversion: Softmax")
        st.latex(r''' p(x_i) = \frac{e^{-\lambda C(x_i)}}{\sum_{j} e^{-\lambda C(x_j)} }
                ''')
        
        # Generate and display the PDF plot with disparity highlighting
        pdf_plot = plot_pixel_data(
            pdf_volume, 
            x_left, 
            y_left, 
            "Probability Distribution", 
            "Disparity", 
            "Probability",
            highlight_disparity=disparity
        )
        if pdf_plot:
            st.plotly_chart(pdf_plot, use_container_width=True)

    # Calculate and display variance heatmap
    st.header("📈 Disparity Variance Analysis")
    variance_volume = calculate_distributional_variance(pdf_volume)
    # print(f"variance volume shape: {variance_volume.shape}")
    # print(f"max variance: {np.max(variance_volume)}")
    
    # Create variance heatmap plot
    variance_fig = px.imshow(
        variance_volume,
        color_continuous_scale='RdYlGn',  # Red=Low Var, Green=High Var
        title="Disparity Variance Per Pixel",
        labels=dict(x="X", y="Y", color="Variance"),
    )
    variance_fig.update_layout(
        height=500,
        width=500
    )
    st.plotly_chart(variance_fig, use_container_width=True)

if __name__ == "__main__":
    # Initialize session state
    if 'bp_completed' not in st.session_state:
        st.session_state['bp_completed'] = False
    
    main()