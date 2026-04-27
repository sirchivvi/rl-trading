"""
train.py - Training & Evaluation Script
==========================================
Trains a DQN agent on historical stock data, evaluates it, and compares
performance against a Buy-and-Hold baseline.

Usage:
    python train.py

The script will:
  1. Download AAPL data via yfinance (free, no API key needed)
  2. Train the DQN agent
  3. Evaluate on a held-out test period
  4. Plot results (saves to results/ folder)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (works on Colab/Kaggle)
import matplotlib.pyplot as plt

# ── Try yfinance; fall back to synthetic data if unavailable ──────────
try:
    import yfinance as yf
    USE_YFINANCE = True
except ImportError:
    USE_YFINANCE = False

# Local modules
from env   import TradingEnv
from agent import DQNAgent


# ═══════════════════════════════════════════════════════════════════════
# Configuration  (tweak these if you want to experiment)
# ═══════════════════════════════════════════════════════════════════════
CONFIG = {
    # Data
    "ticker"      : "AAPL",
    "start_date"  : "2018-01-01",
    "end_date"    : "2023-12-31",
    "train_ratio" : 0.80,          # 80% train, 20% test

    # Environment
    "window"           : 10,
    "transaction_cost" : 0.001,    # 0.1% per trade (our novelty)

    # Agent
    "episodes"      : 100,         # keep low for CPU
    "target_update" : 10,          # sync target net every N episodes
    "lr"            : 1e-3,
    "gamma"         : 0.9,
    "epsilon"       : 1.0,
    "epsilon_min"   : 0.05,
    "epsilon_decay" : 0.97,        # faster decay for 100 episodes
    "batch_size"    : 64,
}

os.makedirs("results", exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════
# 1. Data Loading
# ═══════════════════════════════════════════════════════════════════════
def load_prices(ticker: str, start: str, end: str) -> np.ndarray:
    """
    Download closing prices from Yahoo Finance.
    Falls back to synthetic GBM data if yfinance is not available.
    """
    if USE_YFINANCE:
        print(f"[Data] Downloading {ticker} from {start} to {end} …")
        df = yf.download(ticker, start=start, end=end, progress=False)
        prices = df["Close"].dropna().values.flatten().astype(np.float32)
        print(f"[Data] {len(prices)} trading days loaded.")
    else:
        print("[Data] yfinance not found – using synthetic GBM prices.")
        np.random.seed(42)
        n      = 1500
        dt     = 1 / 252
        mu     = 0.10
        sigma  = 0.20
        returns = np.random.normal((mu - 0.5 * sigma**2) * dt,
                                   sigma * np.sqrt(dt), n)
        prices = 150.0 * np.exp(np.cumsum(returns))
        prices = prices.astype(np.float32)
        print(f"[Data] {len(prices)} synthetic price points generated.")

    return prices


# ═══════════════════════════════════════════════════════════════════════
# 2. Baselines
# ═══════════════════════════════════════════════════════════════════════
def buy_and_hold(prices: np.ndarray, initial_cash: float = 10_000.0) -> float:
    """Buy on day 0, sell on last day. Returns total PnL."""
    shares = initial_cash / prices[0]
    return shares * prices[-1] - initial_cash


def random_agent_pnl(env: TradingEnv, n_runs: int = 5) -> float:
    """Average PnL of a uniformly random agent over multiple runs."""
    pnls = []
    for _ in range(n_runs):
        state = env.reset()
        done  = False
        while not done:
            action = np.random.randint(env.action_size)
            state, _, done, info = env.step(action)
        pnls.append(info["portfolio_value"] - env.initial_cash)
    return float(np.mean(pnls))


# ═══════════════════════════════════════════════════════════════════════
# 3. Evaluation Helper
# ═══════════════════════════════════════════════════════════════════════
def evaluate_agent(agent: DQNAgent, env: TradingEnv):
    """
    Run the agent greedily (ε=0) on the environment.
    Returns: (total_pnl, win_rate, portfolio_values, actions_taken)
    """
    old_eps     = agent.epsilon
    agent.epsilon = 0.0           # pure exploitation

    state     = env.reset()
    done      = False
    portfolio_values = [env.initial_cash]
    actions_taken    = []
    trades           = []         # list of (buy_price, sell_price)
    current_buy      = None

    while not done:
        action = agent.select_action(state)
        state, _, done, info = env.step(action)
        portfolio_values.append(info["portfolio_value"])
        actions_taken.append(action)

        if action == TradingEnv.BUY and current_buy is None:
            current_buy = info["price"]
        elif action == TradingEnv.SELL and current_buy is not None:
            trades.append((current_buy, info["price"]))
            current_buy = None

    agent.epsilon = old_eps

    total_pnl = portfolio_values[-1] - env.initial_cash
    win_rate  = (sum(1 for b, s in trades if s > b) / len(trades)) if trades else 0.0

    return total_pnl, win_rate, portfolio_values, actions_taken


# ═══════════════════════════════════════════════════════════════════════
# 4. Training Loop
# ═══════════════════════════════════════════════════════════════════════
def train(agent: DQNAgent, env: TradingEnv, episodes: int,
          target_update: int) -> dict:
    """
    Main training loop.
    Returns a dict of training metrics for plotting.
    """
    history = {
        "episode_rewards": [],
        "episode_pnl"    : [],
        "epsilons"       : [],
        "losses"         : [],
    }

    print("\n[Train] Starting training …")
    print(f"        Episodes : {episodes}")
    print(f"        Buffer warms up after {agent.batch_size} steps\n")

    for ep in range(1, episodes + 1):
        state      = env.reset()
        done       = False
        ep_reward  = 0.0
        ep_losses  = []

        while not done:
            action                      = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            # Store transition
            agent.memory.push(state, action, reward, next_state, float(done))

            # Learn
            loss = agent.learn()
            if loss > 0:
                ep_losses.append(loss)

            state      = next_state
            ep_reward += reward

        # Sync target network periodically
        if ep % target_update == 0:
            agent._sync_target()

        # Decay epsilon
        agent.decay_epsilon()

        # Compute episode PnL (fast eval, no plot)
        pnl, _, _, _ = evaluate_agent(agent, env)

        history["episode_rewards"].append(ep_reward)
        history["episode_pnl"].append(pnl)
        history["epsilons"].append(agent.epsilon)
        history["losses"].append(np.mean(ep_losses) if ep_losses else 0.0)

        if ep % 10 == 0 or ep == 1:
            print(f"  Ep {ep:>3}/{episodes} | "
                  f"Reward: {ep_reward:>8.2f} | "
                  f"PnL: ${pnl:>7.2f} | "
                  f"ε: {agent.epsilon:.3f} | "
                  f"Loss: {history['losses'][-1]:.4f}")

    print("\n[Train] Done.\n")
    return history


# ═══════════════════════════════════════════════════════════════════════
# 5. Plotting
# ═══════════════════════════════════════════════════════════════════════
def plot_training(history: dict, save_path: str = "results/training.png"):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("DQN Training Progress", fontsize=14, fontweight="bold")

    episodes = range(1, len(history["episode_rewards"]) + 1)

    axes[0, 0].plot(episodes, history["episode_rewards"], color="steelblue")
    axes[0, 0].set_title("Episode Reward")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Total Reward")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(episodes, history["episode_pnl"], color="green")
    axes[0, 1].axhline(0, color="red", linestyle="--", linewidth=0.8)
    axes[0, 1].set_title("Episode PnL ($)")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("PnL ($)")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(episodes, history["epsilons"], color="orange")
    axes[1, 0].set_title("Exploration Rate (ε)")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("ε")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(episodes, history["losses"], color="red")
    axes[1, 1].set_title("Average Loss per Episode")
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].set_ylabel("MSE Loss")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Plot] Training curves saved → {save_path}")


def plot_evaluation(portfolio_values: list, prices: np.ndarray,
                    actions: list, bnh_pnl: float,
                    save_path: str = "results/evaluation.png"):
    """Plot portfolio value over time with buy/sell markers."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle("DQN Agent – Test Period Evaluation", fontsize=14,
                 fontweight="bold")

    # ── Portfolio value ──────────────────────────────────────────────
    ax = axes[0]
    ax.plot(portfolio_values, label="DQN Agent", color="steelblue", linewidth=1.5)

    # Buy-and-hold line
    bnh_curve = np.linspace(portfolio_values[0],
                            portfolio_values[0] + bnh_pnl,
                            len(portfolio_values))
    ax.plot(bnh_curve, label="Buy & Hold", color="orange",
            linestyle="--", linewidth=1.5)

    ax.set_title("Portfolio Value ($)")
    ax.set_ylabel("Value ($)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Price + action markers ───────────────────────────────────────
    ax2 = axes[1]
    n   = min(len(prices), len(actions))
    ax2.plot(prices[:n], color="gray", linewidth=1, label="Price")

    buy_idx  = [i for i, a in enumerate(actions[:n]) if a == TradingEnv.BUY]
    sell_idx = [i for i, a in enumerate(actions[:n]) if a == TradingEnv.SELL]

    ax2.scatter(buy_idx,  prices[buy_idx],  marker="^", color="green",
                s=60, zorder=5, label="Buy")
    ax2.scatter(sell_idx, prices[sell_idx], marker="v", color="red",
                s=60, zorder=5, label="Sell")

    ax2.set_title("Stock Price with Agent Actions")
    ax2.set_xlabel("Trading Day")
    ax2.set_ylabel("Price ($)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Plot] Evaluation chart saved → {save_path}")


# ═══════════════════════════════════════════════════════════════════════
# 6. Main
# ═══════════════════════════════════════════════════════════════════════
def main():
    # ── Load data ────────────────────────────────────────────────────
    prices = load_prices(CONFIG["ticker"],
                         CONFIG["start_date"],
                         CONFIG["end_date"])

    # Train / test split
    split       = int(len(prices) * CONFIG["train_ratio"])
    train_prices = prices[:split]
    test_prices  = prices[split:]
    print(f"[Data] Train days: {len(train_prices)} | Test days: {len(test_prices)}")

    # ── Build environment & agent ────────────────────────────────────
    train_env = TradingEnv(train_prices,
                           window=CONFIG["window"],
                           transaction_cost=CONFIG["transaction_cost"])

    test_env  = TradingEnv(test_prices,
                           window=CONFIG["window"],
                           transaction_cost=CONFIG["transaction_cost"])

    agent = DQNAgent(
        state_size    = train_env.state_size,
        action_size   = train_env.action_size,
        lr            = CONFIG["lr"],
        gamma         = CONFIG["gamma"],
        epsilon       = CONFIG["epsilon"],
        epsilon_min   = CONFIG["epsilon_min"],
        epsilon_decay = CONFIG["epsilon_decay"],
        batch_size    = CONFIG["batch_size"],
    )

    # ── Train ────────────────────────────────────────────────────────
    history = train(agent, train_env,
                    episodes=CONFIG["episodes"],
                    target_update=CONFIG["target_update"])
    plot_training(history)

    # ── Evaluate on test set ─────────────────────────────────────────
    print("[Eval] Evaluating on test set …")
    dqn_pnl, win_rate, portfolio_values, actions = evaluate_agent(agent, test_env)
    bnh_pnl   = buy_and_hold(test_prices)
    rand_pnl  = random_agent_pnl(test_env)

    plot_evaluation(portfolio_values, test_prices, actions, bnh_pnl)

    # ── Save model ───────────────────────────────────────────────────
    agent.save("results/dqn_model.pth")
    print("[Save] Model saved → results/dqn_model.pth")

    # ── Print summary ────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("           RESULTS SUMMARY")
    print("=" * 50)
    print(f"  DQN Agent PnL    : ${dqn_pnl:>8.2f}")
    print(f"  Buy & Hold PnL   : ${bnh_pnl:>8.2f}")
    print(f"  Random Agent PnL : ${rand_pnl:>8.2f}")
    print(f"  Win Rate         : {win_rate * 100:>6.1f}%")
    print("=" * 50)

    if dqn_pnl > rand_pnl:
        print("  ✅ DQN outperforms random agent")
    else:
        print("  ⚠️  DQN does not outperform random agent")

    if dqn_pnl > bnh_pnl:
        print("  ✅ DQN outperforms buy-and-hold")
    else:
        print("  ⚠️  Buy-and-hold beats DQN (common in bull markets)")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()