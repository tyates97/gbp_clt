# Quick test in Python console
import numpy as np
import config as cfg
import distribution_management as dm
import matplotlib.pyplot as plt

# Setup test
cfg.belief_discretisation = 64
cfg.smoothing_width = 32
cfg.measurement_range = np.linspace(0, 10, cfg.belief_discretisation)

# Create and inspect smoothing factor
factor = dm.create_smoothing_factor_distribution(cfg.belief_discretisation)

print(f"Factor shape: {factor.shape}")
print(f"Sum: {factor.sum():.6f}")
print(f"Max value location: {np.unravel_index(factor.argmax(), factor.shape)}")

# Create figure and plot heatmap
plt.figure(figsize=(10, 8))
plt.imshow(factor, cmap='viridis', origin='lower')
plt.colorbar(label='Factor Value')
plt.title('Smoothing Factor Matrix')
plt.xlabel('x2 (cols)')
plt.ylabel('x1 (rows)')

# Show the plot
plt.show()