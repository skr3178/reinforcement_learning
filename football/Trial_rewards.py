# Parameters (as in the wrapper)
num_checkpoints = 5
checkpoint_reward = 0.1
collected_checkpoints = 0  # Start with zero collected
reward = 0.0

# Simulate ball positions (x, y) as it moves toward the goal (x=1, y=0)
ball_positions = [
    (0.0, 0.0),   # far from goal
    (0.5, 0.0),   # halfway
    (0.7, 0.0),   # closer
    (0.85, 0.0),  # very close
    (0.95, 0.0),  # almost at goal
    (1.0, 0.0),   # at goal
]

for pos in ball_positions:
    x, y = pos
    d = ((x - 1) ** 2 + y ** 2) ** 0.5
    print(f"\nBall at ({x:.2f}, {y:.2f}), distance to goal: {d:.3f}")
    # Checkpoints logic
    while collected_checkpoints < num_checkpoints:
        if num_checkpoints == 1:
            threshold = 0.99 - 0.8
        else:
            threshold = (0.99 - 0.8 / (num_checkpoints - 1) * collected_checkpoints)
        print(f"  Checkpoint {collected_checkpoints}: threshold {threshold:.3f}")
        if d > threshold:
            print("   Not close enough for next checkpoint.")
            break
        reward += checkpoint_reward
        collected_checkpoints += 1
        print(f"   Collected checkpoint {collected_checkpoints}, reward now {reward:.2f}")

print(f"\nTotal reward collected: {reward:.2f}")