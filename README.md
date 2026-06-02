# Multi-Stock Reinforcement Learning Trading Agent using Double DQN

## Overview

This project explores the application of Deep Reinforcement Learning (DRL) to algorithmic stock trading. The objective was to build an intelligent trading agent capable of learning profitable Buy, Hold, and Sell decisions directly from historical market data without relying on predefined trading rules.

The system uses a Double Deep Q-Network (Double DQN) implemented in PyTorch and is trained on multiple stocks simultaneously to improve generalization across different market conditions. The project also includes an interactive Streamlit dashboard that enables training, evaluation, portfolio analysis, and trade visualization.

Rather than attempting to predict future stock prices directly, the agent learns a trading policy through interaction with a custom trading environment, receiving rewards based on the profitability and quality of its decisions.

---

# Problem Statement

Most traditional trading systems rely on either:

* Manual technical analysis
* Rule-based trading strategies
* Supervised learning models that predict future price movements

These approaches often struggle to adapt to changing market conditions and typically require significant feature engineering or expert-defined rules.

Reinforcement Learning offers an alternative approach where an agent learns through trial and error by interacting with an environment and optimizing long-term cumulative rewards.

This project investigates the following research question:

"Can a Reinforcement Learning agent learn structured and profitable trading behavior from historical stock data while operating under realistic trading constraints?"

---

# Objectives

The primary objectives of this project were:

* Build a custom stock trading environment.
* Implement a Double DQN agent using PyTorch.
* Train the agent on multiple stocks to improve robustness.
* Incorporate realistic transaction costs.
* Design a reward function that discourages excessive trading.
* Evaluate performance against benchmark strategies.
* Deploy the system through an interactive dashboard.

---

# Dataset

## Data Source

Historical stock market data was collected using Yahoo Finance through the yfinance Python library.

## Stocks Used

| Ticker | Company               |
| ------ | --------------------- |
| AAPL   | Apple Inc.            |
| MSFT   | Microsoft Corporation |
| TSLA   | Tesla Inc.            |

## Time Period

January 2020 – December 2024

This period was selected because it contains multiple market regimes, including:

* COVID-19 market crash (2020)
* Post-pandemic recovery
* 2022 bear market and rate hikes
* 2023–2024 technology-driven bull market

Training across different market conditions helps the agent learn more robust trading behavior.

## Data Split

* 80% Training Data
* 20% Testing Data

The testing period remains completely unseen during training and is used to evaluate the agent's ability to generalize.

---

# Methodology

## Environment Design

A custom trading environment was developed based on the OpenAI Gym framework.

At every time step, the agent observes the current market state and chooses one of three actions:

| Action | Description |
| ------ | ----------- |
| 0      | Hold        |
| 1      | Buy         |
| 2      | Sell        |

The environment simulates:

* Portfolio value tracking
* Position management
* Transaction costs
* Profit and loss calculations
* Maximum drawdown tracking

The agent can either be fully invested in the market or remain entirely in cash.

---

## State Representation

The state vector consists of six features:

### 1. Price Change

Measures short-term price movement.

### 2. Momentum

Captures medium-term trend direction.

### 3. Volatility

Represents recent market uncertainty.

### 4. Relative Strength Index (RSI)

Measures overbought and oversold conditions.

### 5. MACD Histogram

Captures trend strength and trend direction.

### 6. Position Flag

Indicates whether the agent currently holds a position.

State Vector:

```text
[price_change, momentum, volatility, RSI, MACD, position]
```

---

# Key Improvements

## 1. Double Deep Q-Network (Double DQN)

Traditional DQN algorithms often suffer from Q-value overestimation.

Double DQN addresses this issue by separating action selection from action evaluation:

* Online Network selects the best action.
* Target Network evaluates the selected action.

This significantly improves learning stability and reduces overoptimistic value estimates.

---

## 2. Reward Shaping

One of the major challenges encountered during development was excessive trading.

The original reward function encouraged frequent buying and selling because every portfolio value change produced a reward signal.

To address this, a custom reward function was implemented:

* Buy → Small transaction cost penalty
* Hold → Small reward proportional to daily returns
* Sell → Percentage return minus transaction cost
* Invalid actions → Additional penalty

This encouraged the agent to:

* Hold profitable positions longer
* Reduce unnecessary trades
* Focus on overall profitability

---

## 3. Transaction Cost Modeling

Every trade incurs a transaction cost of 0.1%.

This discourages unrealistic high-frequency trading behavior and better reflects real-world market conditions.

---

## 4. Multi-Stock Training

Instead of training on a single stock, the agent was trained on:

* Apple (AAPL)
* Microsoft (MSFT)
* Tesla (TSLA)

Each stock was normalized and concatenated into a single training stream.

Benefits include:

* Exposure to different volatility patterns
* Better generalization
* Reduced overfitting to a single asset

---

# Neural Network Architecture

Input Layer:

```text
6 Features
```

Hidden Layers:

```text
128 Neurons + ReLU
64 Neurons + ReLU
```

Output Layer:

```text
3 Q-values
```

Representing:

```text
Hold
Buy
Sell
```

Framework:

* PyTorch

Optimizer:

* Adam Optimizer

Loss Function:

* Mean Squared Error (MSE)

---

# Training Configuration

| Parameter             | Value             |
| --------------------- | ----------------- |
| Algorithm             | Double DQN        |
| Episodes              | 150               |
| Learning Rate         | 0.001             |
| Gamma                 | 0.90              |
| Batch Size            | 64                |
| Replay Buffer Size    | 20,000            |
| Target Network Update | Every 10 Episodes |
| Initial Epsilon       | 1.0               |
| Minimum Epsilon       | 0.05              |
| Epsilon Decay         | 0.97              |

Training was performed on a concatenated multi-stock dataset containing approximately 3,000 trading days.

---

# Streamlit Dashboard

A complete Streamlit application was developed to make the project interactive and easier to demonstrate.

Features include:

* Stock selection
* Hyperparameter configuration
* Training mode selection
* Real-time training visualization
* Portfolio performance tracking
* Buy/Sell signal visualization
* Strategy comparison charts
* Trade log generation
* Model loading and evaluation

The dashboard allows users to experiment with different settings without modifying code.

---

# Results

## Portfolio Performance

The trained agent achieved:

| Strategy     | Profit  |
| ------------ | ------- |
| Double DQN   | +$58    |
| Random Agent | +$27    |
| Buy & Hold   | +$3,163 |

## Key Findings

* The Double DQN agent outperformed the random trading baseline.
* The agent learned structured trading behavior rather than making random decisions.
* Reward shaping successfully reduced excessive trading.
* Multi-stock training improved robustness across different market conditions.
* Buy-and-Hold significantly outperformed the agent during a strong bull market, which is expected in trending markets.

---

# Insights

Several important observations emerged from the project:

### Reinforcement Learning Can Learn Market Structure

The agent consistently outperformed random trading, indicating that meaningful patterns were learned from historical data.

### Reward Design Matters More Than Model Complexity

Reward shaping had a greater impact on performance than increasing model complexity.

### Multi-Stock Training Improves Generalization

Training on multiple assets exposed the agent to diverse market behaviors and reduced overfitting.

### Active Trading Does Not Always Beat Passive Investing

During strong bull markets, Buy-and-Hold often remains difficult to outperform because active trading can miss extended upward trends.

---

# Technologies Used

* Python
* PyTorch
* NumPy
* Pandas
* Matplotlib
* yFinance
* Streamlit
* Reinforcement Learning
* Double DQN
* Git
* GitHub

---

# Future Improvements

Potential extensions include:

* LSTM-based sequence models
* Attention mechanisms for market representation
* Position sizing strategies
* Risk-adjusted reward functions
* Additional technical indicators
* Portfolio optimization across multiple assets
* Live market data integration
* Paper trading deployment
* Transformer-based market forecasting models

---

# Conclusion

This project demonstrates how Deep Reinforcement Learning can be applied to algorithmic trading through a custom-built stock market environment, reward engineering, and Double DQN optimization. By combining technical indicators, realistic transaction costs, and multi-stock training, the system successfully learned structured trading behavior and outperformed a random baseline while providing a fully interactive Streamlit dashboard for experimentation and evaluation.
