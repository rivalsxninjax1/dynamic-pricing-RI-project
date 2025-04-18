
# Dynamic Pricing with Reinforcement Learning

This project demonstrates a real-life reinforcement learning problem: optimizing dynamic pricing for an e-commerce store.

## Problem Statement

An online retailer wants to maximize revenue by adjusting product prices dynamically based on simulated customer demand. The goal is to learn a pricing policy that selects the best price from a set of discrete price points to maximize cumulative revenue over time.

## Environment & Data

- We simulate customer demand as a function of price: higher prices yield lower demand and vice versa.
- Demand model: demand = max(0, a - b * price + noise)
    - `a`, `b` are parameters controlling baseline demand and price sensitivity.
    - `noise` is Gaussian to reflect randomness.
- Price options: [10, 12, 15, 18, 20]

## RL Setup

- **State**: previous price and observed demand (discretized).
- **Actions**: choose one of the price options.
- **Reward**: revenue = price × demand.
- **Algorithm**: Q-learning with epsilon-greedy exploration.

## Files

- `dynamic_pricing_rl.py`: main script defining the environment, Q-learning agent, and training loop.

## How to Run

```bash
python dynamic_pricing_rl.py
```# dynamic-pricing-RI-project
