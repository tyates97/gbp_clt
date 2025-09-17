import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import plotly.graph_objects as go
import plotly.express as px
import threading

# Import your modules (assuming they're in the same directory)
import config as cfg
import factor_graph as fg
import distribution_management as dm
import belief_propagation as bp
import optimisation as opt
import image_processing as ip

# defining a custom colour scale
excellent_threshold = 0.001      # Very Gaussian (green)
good_threshold = 0.01           # Reasonably Gaussian
poor_threshold = 0.05           # Somewhat non-Gaussian (yellow)
bad_threshold = 0.2             # Non-Gaussian (orange)
custom_colorscale = [
    [0.0, "green"],                                    # 0.0 = perfect (green)
    [excellent_threshold/1.0, "green"],               # 0.001 = still green  
    [good_threshold/1.0, "lightgreen"],               # 0.01 = light green
    [poor_threshold/1.0, "yellow"],                   # 0.05 = yellow
    [bad_threshold/1.0, "orange"],                    # 0.5 = orange
    [1.0, "red"]                                      # 1.0 = red
]

# Configure Streamlit page
st.set_page_config(
    page_title="Stereo Vision Belief Propagation",
    page_icon="🔸",
    layout="wide"
)

class RealTimeConsoleCapture:
    """Capture console output in real-time for display in Streamlit"""
    def __init__(self, placeholder):
        self.contents = []
        self.placeholder = placeholder
        self.lock = threading.Lock()
    
    def write(self, data):
        with self.lock:
            self.contents.append(data)
            # Update the display in real-time
            self.placeholder.code(''.join(self.contents))
    
    def flush(self):
        pass
    
    def get_output(self):
        with self.lock:
            return ''.join(self.contents)

def run_belief_propagation_with_progress(graph, num_iterations, console_placeholder):
    """Run belief propagation with real-time console updates"""
    # Create a real-time console capture
    console_capture = RealTimeConsoleCapture(console_placeholder)
    
    # Redirect stdout to our custom capture
    original_stdout = sys.stdout
    try:
        sys.stdout = console_capture
        result_graph = bp.run_belief_propagation(graph, num_iterations)
        return result_graph
    finally:
        sys.stdout = original_stdout

def calculate_distributional_variance(pdf_volume):
    """Calculates the true variance of the disparity distributions."""
    height, width, max_disparity = pdf_volume.shape
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

def create_coordinate_selector(image_data, title, key_prefix, cmap="gray", global_min=None, global_max=None):
    """Create a coordinate selector with image display and slider controls, with overlay point"""
    scale_label = "KL Divergence" if cmap == custom_colorscale else "Color"

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
    
    # Create the heatmap with fixed scale
    fig = px.imshow(
        image_data,
        color_continuous_scale=cmap,  # Red=high KL, Green=low KL
        zmin=0,
        title=title,
        labels=dict(x="X", y="Y", color=scale_label)
    )
    
    # Add a red point at the selected coordinates
    fig.add_trace(go.Scatter(
        x=[x_coord],
        y=[y_coord],
        mode='markers',
        marker=dict(
            size=15,
            color='red',
            symbol='circle',
            line=dict(color='white', width=2)
        ),
        name=f'Selected Point ({x_coord}, {y_coord})',
        showlegend=False
    ))
    
    fig.update_layout(
        title=f"Selected: ({x_coord}, {y_coord})",
        xaxis_title="X",
        yaxis_title="Y",
        height=500,
        width=500,
        showlegend=False
    )

    # Display the interactive plot
    st.plotly_chart(fig, use_container_width=True, key=f"coord_selector_{key_prefix}")
    
    return x_coord, y_coord

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
        
        # reszing for faster processing
        left_image = ip.crop_image(left_image, (150, 200))
        right_image = ip.crop_image(right_image, (150, 200))
        ground_truth = ip.crop_image(ground_truth, (150, 200))

        cfg.max_measurement = int(np.ceil(np.max(ground_truth)))

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

@st.cache_resource
def build_factor_graph_cached(pdf_volume, smoothing_kernel):
    """Build and cache the factor graph"""
    return fg.get_graph_from_pdf_hist(pdf_volume, smoothing_kernel)

def plot_pixel_data(data_volume, x, y, title, x_label="Index", y_label="Value"):
    """Plot data for a specific pixel"""
    if x < data_volume.shape[1] and y < data_volume.shape[0]:
        pixel_data = data_volume[y, x, :]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=np.linspace(cfg.min_measurement, cfg.max_measurement, data_volume.shape[2]),
            y=pixel_data,
            mode='lines',
            name='Data'
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

def create_heatmap_plot(image_data, title, colorscale='gray'):
    """Create a heatmap for MSE visualization"""
    fig = px.imshow(
        image_data,
        color_continuous_scale=colorscale,
        title=title,
        labels=dict(x="X", y="Y", color="Value")
    )
    
    fig.update_layout(height=400)
    return fig

# Update the belief plotting function to use KL:
def plot_pixel_belief_with_gaussian(graph, x, y, measurement_range, metric="KL"):
    """Plot pixel belief with optimal Gaussian overlay using KL divergence"""

    if not hasattr(graph, 'grid_cols'):
        return None
        
    pixel_idx = y * graph.grid_cols + x
    if pixel_idx >= len(graph.variables):
        return None
    
    variable = graph.variables[pixel_idx]
    belief = variable.belief
    
    # Calculate optimal Gaussian using KL divergence
    if metric == "KL":
        min, optimal_sigma, optimal_mean = opt.optimise_gaussian_kl(belief, measurement_range)
    else:
        min, optimal_sigma, optimal_mean = opt.optimise_gaussian(belief, measurement_range)
    gaussian_fit = dm.create_gaussian_distribution(measurement_range, optimal_sigma, mu=optimal_mean)
    
    fig = go.Figure()
    
    # Plot belief
    fig.add_trace(go.Scatter(
        x=measurement_range,
        y=belief,
        mode='lines+markers',
        name='Belief',
        line=dict(color='blue')
    ))
    
    # Plot Gaussian fit
    fig.add_trace(go.Scatter(
        x=measurement_range,
        y=gaussian_fit,
        mode='lines',
        name=f'Gaussian Fit (σ={optimal_sigma:.2f}, {metric}={min:.2e})',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title=f"Belief vs Gaussian Fit at Pixel (x={x}, y={y})",
        xaxis_title="Disparity",
        yaxis_title="Probability",
        height=400
    )
    
    return fig

def main():
    # Initialize session state variables
    if 'bp_method' not in st.session_state:
        st.session_state['bp_method'] = None
    
    st.title("🔍 Stereo Vision Belief Propagation Interactive App")
    st.sidebar.title("Controls")
    
    # Load images
    left_image, right_image, ground_truth = load_images()
    
    if left_image is None:
        st.stop()
    
    # Display basic images
    st.header("📷 Input Images")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Left Image")
        st.image(left_image/np.max(left_image), use_container_width=True, clamp=True)
    
    with col2:
        st.subheader("Right Image")
        st.image(right_image/np.max(right_image), use_container_width=True, clamp=True)
     
    # Configuration parameters
    st.sidebar.header("Parameters")
    patch_size = int(np.sqrt(st.sidebar.selectbox("Patch Size", [9, 25, 49], index=1)))
    use_gaussian_bp = st.sidebar.checkbox("Use Gaussian Belief Propagation", value=False)
    cfg.num_iterations = st.sidebar.slider("BP Iterations", 1, 20, 10)

    # Store selected parameters (don't apply them yet)
    selected_cost_function = st.sidebar.selectbox("Select Cost Function", 
                                        ['NCC', 'SAD', 'SSD'], 
                                        index=['NCC', 'SAD', 'SSD'].index(getattr(cfg, 'cost_function', 'SSD'))
                                        )
    selected_lambda_param = st.sidebar.number_input("Lambda (λ)", 
                                        min_value=0.0001, 
                                        max_value=10.0000, 
                                        value=getattr(cfg, 'lambda_param', 0.0020), 
                                        step=0.0001,
                                        format="%.4f"
                                        )
    selected_smoothing_kernel = st.sidebar.selectbox("Smoothing Kernel",
                                        ['histogram', 'triangular'], 
                                        index=['histogram', 'triangular'].index(getattr(cfg, 'smoothing_function', 'histogram'))
                                        )
    if selected_smoothing_kernel != 'histogram':
        selected_smoothing_width = st.sidebar.slider("Smoothing Width", 
                                           1, 
                                           cfg.max_measurement, 
                                           value=getattr(cfg, 'smoothing_width'),
                                           step=1
                                           )

    # Check if parameters have changed
    current_params = {
            'cost_function': getattr(cfg, 'cost_function', 'NCC'),
    'lambda_param': getattr(cfg, 'lambda_param', 5.0),
    'smoothing_function': getattr(cfg, 'smoothing_function', 'histogram'),
    'smoothing_width': getattr(cfg, 'smoothing_width', None)
    }

    selected_params = {
    'cost_function': selected_cost_function,
    'lambda_param': selected_lambda_param,
    'smoothing_function': selected_smoothing_kernel,
    'smoothing_width': selected_smoothing_width if selected_smoothing_kernel != 'histogram' else None
    }

    params_changed = current_params != selected_params

    if params_changed:
        st.sidebar.warning("⚠️ Parameters changed - click Update to apply")
        st.sidebar.write(f"Current: {current_params}")
        st.sidebar.write(f"Selected: {selected_params}")

    # Add refresh button
    update_clicked = st.sidebar.button("🔄 Update Parameters", 
                                    type="primary" if params_changed else "secondary",
                                    disabled=not params_changed)

    if update_clicked:
        # Apply the new parameters
        cfg.cost_function = selected_cost_function
        cfg.lambda_param = selected_lambda_param
        cfg.smoothing_function = selected_smoothing_kernel
        cfg.smoothing_width = selected_smoothing_width if selected_smoothing_kernel != 'histogram' else None
        
        # Clear relevant caches to force recomputation
        compute_cost_volume.clear()
        compute_pdf_volume.clear()
        build_factor_graph_cached.clear()
        
        # Show success message
        st.sidebar.success("✅ Parameters updated!")
        
        # Rerun the app to apply changes
        st.rerun()

    # If parameters haven't been set yet, initialize them
    if not hasattr(cfg, 'cost_function'):
        cfg.cost_function = 'NCC'
    if not hasattr(cfg, 'lambda_param'):
        cfg.lambda_param = 0.002
    
    # Set up configuration
    cfg.min_measurement = 0
    cfg.max_measurement = int(np.ceil(np.max(ground_truth)))
    cfg.measurement_range = np.arange(cfg.min_measurement, cfg.max_measurement+0.25, 0.25)
    cfg.belief_discretisation = len(cfg.measurement_range)
    
    # Compute cost and PDF volumes
    with st.spinner("Computing cost volume..."):
        cost_volume = compute_cost_volume(left_image, right_image, patch_size, cfg.max_measurement, cfg.cost_function)
    
    with st.spinner("Converting costs to PDFs..."):
        pdf_volume = compute_pdf_volume(cost_volume, cfg.lambda_param)
    
    # Interactive cost function inspector
    st.header("💰 Cost Function Inspector")
    
    # Create two columns: left for coordinate selector, right for variance heatmap
    col1, col2 = st.columns(2)
    
    with col1:
        # Use the coordinate selector with overlay
        x_coord, y_coord = create_coordinate_selector(left_image, "Left Image - Use sliders to select coordinates for cost inspection", "cost", cmap="gray")
    
    with col2:
        # Calculate and display variance heatmap
        st.write("**Prior Variance Heatmap**")
        # Add empty space to align with the image below
        st.write("")  # Add some vertical spacing
        st.write("")  # Add more spacing if needed
        st.write("")  # Add more spacing if needed
        st.write("")  # Add more spacing if needed
        variance_volume = calculate_distributional_variance(pdf_volume)
        
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
        # Generate and display the cost plot (full width)
        cost_plot = plot_pixel_data(cost_volume, x_coord, y_coord, "Cost Function", "Disparity", "Cost")
        if cost_plot:
            st.plotly_chart(cost_plot, use_container_width=True)
    with col2: 
        st.write("Probability Conversion: Softmax")
        st.latex(r''' p(x_i) = \frac{e^{-\lambda C(x_i)}}{\sum_{j} e^{-\lambda C(x_j)} }
                ''')
        # Generate and display the PDF plot
        pdf_plot = plot_pixel_data(pdf_volume, x_coord, y_coord, "Probability Distribution", "Disparity", "Probability")
        if pdf_plot:
            st.plotly_chart(pdf_plot, use_container_width=True)
    
    # Smoothing kernel visualization
    st.header("🔧 Smoothing Kernel")

    # Generate kernel data based on smoothing function selection
    if cfg.smoothing_function == 'histogram':

        # Get disparity histogram for smoothing kernel
        ground_truth_signed = ground_truth.astype(np.int16)
        all_diffs = dm.get_histogram_from_truth(ground_truth_signed)
        
        # Create integer bins centered on integer values
        max_diff = int(np.max(np.abs(all_diffs)))
        bins = np.arange(-max_diff - 0.5, max_diff + 1.5, 1)  # Creates bins centered on integers
        hist, bin_edges = np.histogram(all_diffs, bins=bins)
        
        # Use bin centers for plotting (integers)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        histogram_kernel = dm.normalise(hist)

        # Prepare data for plotting
        kernel_data = histogram_kernel
        kernel_x_values = bin_centers

    else:  # triangular
        # Create triangular kernel for display
        width = cfg.smoothing_width
        triangular_kernel = dm._make_default_triangular_kernel(width)

        # Create x-values that span from -63 to +63 (full disparity range)
        max_disparity = cfg.max_measurement - 1  # 63 for belief_discretisation = 64
        kernel_x_values = np.arange(-max_disparity, max_disparity + 1)  # -63 to +63
    
        # Create zero-padded kernel data to match the full range
        kernel_data = np.zeros(len(kernel_x_values))
        
        # Place the triangular kernel in the center
        center_idx = len(kernel_x_values) // 2
        half_width = width // 2
        start_idx = max(0, center_idx - half_width)
        end_idx = min(len(kernel_x_values), center_idx + len(triangular_kernel) - half_width)
        
        # Insert the triangular kernel values
        tri_start = max(0, half_width - center_idx)
        tri_end = tri_start + (end_idx - start_idx)
        kernel_data[start_idx:end_idx] = triangular_kernel[tri_start:tri_end]

    col1, col2 = st.columns(2)

    with col1:
        # Plot smoothing kernel
        kernel_fig = go.Figure()
        if cfg.smoothing_function == 'histogram':
            kernel_fig.add_trace(go.Scatter(
                x=kernel_x_values,
                y=kernel_data,
                name='Smoothing Kernel'
            ))
        else:
        # Use line plot for triangular
            kernel_fig.add_trace(go.Scatter(
                x=kernel_x_values,
                y=kernel_data,
                mode='lines+markers',
                name='Smoothing Kernel',
                line=dict(shape='linear')
            ))

        if use_gaussian_bp:
            _, optimal_sigma, optimal_mean = opt.optimise_gaussian_kl(kernel_data, kernel_x_values)
            gaussian_fit = dm.create_gaussian_distribution(kernel_x_values, optimal_sigma, mu=optimal_mean)
            
            kernel_fig.add_trace(go.Scatter(
                x=kernel_x_values,
                y=gaussian_fit,
                mode='lines',
                name='Gaussian Approximation',
                line=dict(dash='dash')
            ))

        kernel_fig.update_layout(
            title="Smoothing Kernel (Disparity Difference Histogram)",
            xaxis_title="Disparity Difference",
            yaxis_title="Probability",
            height=500,
            width=500,
            showlegend=True
        )
        st.plotly_chart(kernel_fig, use_container_width=True)
    
    with col2:
        # Generate the 2D pairwise factor matrix based on selected smoothing function
        if cfg.smoothing_function == 'histogram':
            pairwise_factor_matrix = dm.create_smoothing_factor_distribution(
                cfg.max_measurement, 
                kernel=None, 
                hist=histogram_kernel,
                smoothing_function="histogram"
            )
        else:  # triangular
            pairwise_factor_matrix = dm.create_smoothing_factor_distribution(
                cfg.max_measurement, 
                kernel=None, 
                hist=None,  # This will trigger the triangular kernel creation
                smoothing_function="triangular"
            )

        if use_gaussian_bp:
            gaussian_matrix = bp.convert_pairwise_factor_to_gaussian(pairwise_factor_matrix)
                
            # Create 2D heatmap
            factor_2d_fig = px.imshow(
                gaussian_matrix,
                color_continuous_scale='Viridis',
                title="f(x1, x2) - Pairwise Factor Function",
                labels=dict(x="x2 (Disparity)", y="x1 (Disparity)", color="Factor Value"),
                aspect='equal',
                origin='lower'
            )
        else:
            # Create 2D heatmap
            factor_2d_fig = px.imshow(
                pairwise_factor_matrix,
                color_continuous_scale='Viridis',
                title="f(x1, x2) - Pairwise Factor Function",
                labels=dict(x="x2 (Disparity)", y="x1 (Disparity)", color="Factor Value"),
                aspect='equal',
                origin='lower'
            )
        factor_2d_fig.update_layout(
            height=500,
            width=500,
            xaxis_title="x2 (Disparity Level)",
            yaxis_title="x1 (Disparity Level)"
        )
        st.plotly_chart(factor_2d_fig, use_container_width=True)
    
    # Optional: Add explanation text below the plots
    st.write("""
    **Interpretation:**
    - **Left plot**: Shows the 1D kernel derived from ground truth disparity differences
    - **Right plot**: Shows how this kernel creates pairwise relationships between disparity levels
    - Bright diagonal indicates strong preference for similar disparities between neighboring pixels
    - Off-diagonal elements show how much smoothing occurs between different disparity levels
    """)

    
    # Build factor graph
    cfg.num_variables = pdf_volume.shape[0] * pdf_volume.shape[1]
    
    with st.spinner("Building factor graph..."):
        if cfg.smoothing_function == 'histogram':
            graph = build_factor_graph_cached(pdf_volume, histogram_kernel)
        elif cfg.smoothing_function == 'triangular':
            graph = build_factor_graph_cached(pdf_volume, triangular_kernel)
    
    # Initialize beliefs
    for variable in graph.variables:
        for factor in variable.neighbors:
            if factor.factor_type == "prior":
                variable.belief = factor.function
    
    initial_beliefs = {i: var.belief.copy() for i, var in enumerate(graph.variables)}
    st.session_state['graph_before_bp'] = graph
    disparity_vol_pre_bp = ip.get_disparity_from_graph(graph)


    # Belief Propagation Section
    st.header("🧠 Belief Propagation")
    
    # Console Output section (always visible)
    st.subheader("Console Output")
    console_placeholder = st.empty()
    
    # Initialize console display
    if 'console_output' not in st.session_state:
        st.session_state['console_output'] = "Factor Graph built. Ready to run belief propagation..."
    
    console_placeholder.code(st.session_state['console_output'])
    
    if st.button("Run Belief Propagation", type="primary"):
        # Clear previous results when switching methods
        current_method = 'gbp' if use_gaussian_bp else 'regular'
        # previous_method = 'regular' if use_gaussian_bp else 'gbp'
        
        # clear cached results
        if 'disparity_vol_post_bp' in st.session_state:
            del st.session_state['disparity_vol_post_bp']
        if 'graph_after_bp' in st.session_state:
            del st.session_state['graph_after_bp']
        
        st.session_state['bp_method'] = current_method

        # Reset to initial beliefs 
        for i, variable in enumerate(graph.variables):
            variable.belief = initial_beliefs[i].copy()

        st.session_state['console_output'] = "Starting belief propagation...\n"
        console_placeholder.code(st.session_state['console_output'])

        with st.spinner("Running Belief Propagation..."):
            if use_gaussian_bp:
                result_graph = bp.run_gaussian_belief_propagation(graph, cfg.num_iterations)
            else:
                result_graph = run_belief_propagation_with_progress(graph, cfg.num_iterations, console_placeholder)
        
        console_placeholder.code("Belief propagation completed successfully!")
        
        # Store results in session state
        st.session_state['graph_after_bp'] = result_graph
        st.session_state['disparity_vol_post_bp'] = ip.get_disparity_from_graph(result_graph)
        st.session_state['bp_completed'] = True

        # Final update
        console_placeholder.code("Belief propagation completed successfully!")
    
    # Results section
    if st.session_state.get('bp_completed', False):
        st.header("📈 Results")
        
        # Disparity comparison
        st.subheader("Disparity Maps")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("Ground Truth")
            st.image(ground_truth/np.max(ground_truth), use_container_width=True, clamp=True)
        
        with col2:
            st.write("Before BP")
            fig, ax = plt.subplots()
            ax.imshow(disparity_vol_pre_bp, cmap='gray', vmin=0, vmax=cfg.max_measurement)
            ax.set_axis_off()
            st.pyplot(fig)
        
        with col3:
            st.write("After BP")
                        
            fig, ax = plt.subplots()
            ax.imshow(
                st.session_state['disparity_vol_post_bp'],
                cmap='gray', 
                vmin=0, 
                vmax=cfg.max_measurement
            )
            ax.set_axis_off()
            st.pyplot(fig)
        
        # Interactive Gaussian heatmaps
        st.subheader("🎯 Gaussian Fit Analysis")
        
        graph_pre_bp = st.session_state.get('graph_before_bp')
        graph_post_bp = st.session_state.get('graph_after_bp')

        if graph_pre_bp is not None:
            kl_pre_bp = opt.get_kl_from_graph(graph_pre_bp).copy()
        else:
            kl_pre_bp = np.zeros_like(disparity_vol_pre_bp)

        if graph_post_bp is not None:
            kl_post_bp = opt.get_kl_from_graph(graph_post_bp).copy()
        else:
            kl_post_bp = np.zeros_like(kl_pre_bp)
            st.info("Run Belief Propagation to generate the post-BP heatmap.")

        
        # Calculate global min/max for consistent scaling
        global_min = min(np.min(kl_pre_bp), np.min(kl_post_bp))
        global_max = max(np.max(kl_pre_bp), np.max(kl_post_bp))
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Pre-BP Gaussian KL Divergence**")
            
            # Coordinate selector for pre-BP with overlay
            pre_x, pre_y = create_coordinate_selector(kl_pre_bp, "Select coordinates for Pre-BP belief analysis", "pre_bp", cmap=custom_colorscale, global_min=global_min, global_max=global_max)
            
            belief_fig_pre = None
            if graph_pre_bp is not None:
                belief_fig_pre = plot_pixel_belief_with_gaussian(graph_pre_bp, pre_x, pre_y, cfg.measurement_range, metric="KL")
            if belief_fig_pre:
                st.plotly_chart(belief_fig_pre, use_container_width=True)
        
        with col2:
            st.write("**Post-BP Gaussian KL Divergence**")
            # Coordinate selector for post-BP with overlay
            post_x, post_y = create_coordinate_selector(kl_post_bp, "Select coordinates for Post-BP belief analysis", "post_bp", cmap=custom_colorscale, global_min=global_min, global_max=global_max)
            
            belief_fig_post = None
            if graph_post_bp is not None:
                belief_fig_post = plot_pixel_belief_with_gaussian(graph_post_bp, post_x, post_y, cfg.measurement_range, metric="KL")
            if belief_fig_post:
                st.plotly_chart(belief_fig_post, use_container_width=True, key ="post_bp_mse")

if __name__ == "__main__":
    # Initialize session state
    if 'bp_completed' not in st.session_state:
        st.session_state['bp_completed'] = False
    
    main()