import numpy as np
import random
import matplotlib.pyplot as plt

# Environment parameters
A = 100    # baseline demand
B = 2      # price sensitivity
NOISE_STD = 5
PRICE_OPTIONS = [10, 12, 15, 18, 20]

class PricingEnv:
    def __init__(self, a, b, noise_std, prices):
        self.a = a
        self.b = b
        self.noise_std = noise_std
        self.prices = prices

    def reset(self):
        # Initialize with random previous price and corresponding demand
        price = random.choice(self.prices)
        demand = max(0, self.a - self.b * price + np.random.randn() * self.noise_std)
        return (price, int(demand))

    def step(self, action):
        price = self.prices[action]
        demand = max(0, self.a - self.b * price + np.random.randn() * self.noise_std)
        revenue = price * demand
        next_state = (price, int(demand))
        return next_state, revenue, False, {}

# Convert continuous state to discrete indices
def state_to_index(state):
    price, demand = state
    p_idx = PRICE_OPTIONS.index(price)
    d_bucket = min(demand // 10, 9)
    return p_idx, d_bucket

# Main training loop
def main():
    epsilon = 0.1    # exploration rate
    alpha = 0.1      # learning rate
    gamma = 0.99     # discount factor
    num_episodes = 5000

    env = PricingEnv(A, B, NOISE_STD, PRICE_OPTIONS)
    Q = np.zeros((len(PRICE_OPTIONS), 10, len(PRICE_OPTIONS)))
    rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        for _ in range(100):
            p_idx, d_idx = state_to_index(state)
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = random.randrange(len(PRICE_OPTIONS))
            else:
                action = np.argmax(Q[p_idx, d_idx])

            next_state, reward, done, _ = env.step(action)
            p2, d2 = state_to_index(next_state)

            # Q-learning update
            best_next = np.max(Q[p2, d2])
            Q[p_idx, d_idx, action] += alpha * (reward + gamma * best_next - Q[p_idx, d_idx, action])

            state = next_state
            total_reward += reward
            if done:
                break

        rewards.append(total_reward)

    # Plot cumulative revenue per episode
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Revenue')
    plt.title('Training Progress')
    plt.show()

if __name__ == "__main__":
    main()
