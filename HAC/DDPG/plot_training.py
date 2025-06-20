import matplotlib.pyplot as plt
import numpy as np

# First run data
run1_episodes = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200])
run1_rewards = np.array([-944.23, -1623.29, -1639.65, -1348.88, -1267.54, -1180.44, -934.83, -797.43, -958.64, -511.92, 
                        -632.22, -129.32, -125.80, -126.34, -260.50, -119.54, -125.38, -240.32, -124.15, -117.30, -122.66])

# Second run data
run2_episodes = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230])
run2_rewards = np.array([-1552.80, -1617.67, -1869.69, -1264.24, -1114.12, -910.05, -887.82, -633.08, -914.55, -630.02,
                        -506.18, -498.53, -0.17, -125.99, -0.48, -248.01, -504.78, -126.50, -124.95, -704.45, -618.54,
                        -369.72, -123.03, -121.65])

# Create the plot
plt.figure(figsize=(12, 6))
plt.plot(run1_episodes, run1_rewards, 'b-', label='Run 1', alpha=0.7)
plt.plot(run2_episodes, run2_rewards, 'r-', label='Run 2', alpha=0.7)

# Add labels and title
plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.title('DDPG Training Progress Comparison')
plt.grid(True, alpha=0.3)
plt.legend()

# Add a horizontal line at y=0 for reference
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)

# Show the plot
plt.tight_layout()
plt.show() 