import matplotlib.pyplot as plt
import numpy as np

# Sample data for demonstration
plane1_theta_h = np.linspace(-10, 10, 400)
plane2_theta_h = np.linspace(-10, 10, 400)
plane1_theta_h, plane2_theta_h = np.meshgrid(plane1_theta_h, plane2_theta_h)

# Sample function for demonstration: pred_fixed_h = sin(x) + cos(y)
pred_fixed_h = np.sin(plane1_theta_h) * np.cos(plane2_theta_h)
# pred_fixed_h = pred_fixed_h.ravel()
# pred_fixed_h = pred_fixed_h.reshape(400, 400)

# Plotting the 2D function
plt.figure(figsize=(8, 6))
plt.contourf(plane1_theta_h, plane2_theta_h, pred_fixed_h, cmap='viridis')
plt.colorbar(label='pred_fixed_h value')
plt.xlabel('plane1_theta_h (x)')
plt.ylabel('plane2_theta_h (y)')
plt.title('2D Function Visualization: pred_fixed_h = sin(x) + cos(y)')
plt.show()
