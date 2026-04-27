"""
env.py - Stock Trading Environment
====================================
A simple OpenAI Gym-style trading environment.

State  : [price_change, momentum, volatility, rsi, position]
Actions: 0 = Hold, 1 = Buy, 2 = Sell
Reward : Daily PnL change with transaction cost penalty (our novelty)
"""

import numpy as np


class TradingEnv:
    """
    A simple stock trading environment.

    Research Novelty: We add a transaction cost penalty (0.1% per trade)
    to discourage excessive trading and make the agent learn more
    realistic, cost-aware strategies.
    """

    # Action constants for readability
    HOLD = 0
    BUY  = 1
    SELL = 2

    def __init__(self, prices: np.ndarray, window: int = 10,
                 transaction_cost: float = 0.001):
        """
        Args:
            prices           : 1-D array of closing prices
            window           : lookback window for feature computation
            transaction_cost : fraction of trade value charged per trade (e.g. 0.001 = 0.1%)
        """
        self.prices           = prices.astype(np.float32)
        self.window           = window
        self.transaction_cost = transaction_cost

        # Episode boundaries
        self.start_idx = window          # first valid step
        self.end_idx   = len(prices) - 1

        # Will be set in reset()
        self.current_step   = None
        self.position       = None   # 0 = no stock, 1 = holding stock
        self.buy_price      = None   # price at which we bought
        self.portfolio_value= None
        self.initial_cash   = 10_000.0

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------
    def _compute_state(self) -> np.ndarray:
        """
        Build a 5-element feature vector from recent price history.

        Features:
          1. norm_price_change : today vs yesterday (normalised)
          2. momentum          : price now vs price `window` steps ago (normalised)
          3. volatility        : std of recent returns
          4. rsi               : Relative Strength Index (0–1 scaled)
          5. position          : current position flag (0 or 1)
        """
        i   = self.current_step
        w   = self.window
        seg = self.prices[i - w: i + 1]          # window+1 prices

        # 1. Normalised price change (today vs yesterday)
        norm_change = (seg[-1] - seg[-2]) / (seg[-2] + 1e-8)

        # 2. Momentum (today vs `window` days ago)
        momentum = (seg[-1] - seg[0]) / (seg[0] + 1e-8)

        # 3. Volatility (std of daily returns)
        returns    = np.diff(seg) / (seg[:-1] + 1e-8)
        volatility = float(np.std(returns))

        # 4. RSI – classic 14-period (we use window for simplicity)
        gains  = np.maximum(returns, 0)
        losses = np.maximum(-returns, 0)
        avg_gain = gains.mean()  + 1e-8
        avg_loss = losses.mean() + 1e-8
        rs  = avg_gain / avg_loss
        rsi = 1.0 - 1.0 / (1.0 + rs)   # scaled to [0, 1]

        # 5. Position flag
        pos = float(self.position)

        return np.array([norm_change, momentum, volatility, rsi, pos],
                        dtype=np.float32)

    # ------------------------------------------------------------------
    # Gym-style interface
    # ------------------------------------------------------------------
    def reset(self) -> np.ndarray:
        """Reset environment to the start of an episode."""
        self.current_step    = self.start_idx
        self.position        = 0
        self.buy_price       = 0.0
        self.portfolio_value = self.initial_cash
        return self._compute_state()

    def step(self, action: int):
        """
        Execute one trading step.

        Returns:
            next_state (np.ndarray)
            reward     (float)
            done       (bool)
            info       (dict)
        """
        price = self.prices[self.current_step]
        cost  = 0.0   # transaction cost incurred this step

        # ---- Execute action ----
        if action == self.BUY and self.position == 0:
            # Enter long position
            self.position  = 1
            self.buy_price = price
            cost = self.transaction_cost * price   # pay cost on entry

        elif action == self.SELL and self.position == 1:
            # Close long position
            trade_pnl      = price - self.buy_price
            self.portfolio_value += trade_pnl
            self.position  = 0
            self.buy_price = 0.0
            cost = self.transaction_cost * price   # pay cost on exit

        # ---- Compute reward ----
        # Base reward: change in mark-to-market value
        next_price = self.prices[self.current_step + 1]
        if self.position == 1:
            mark_to_market_change = next_price - price
        else:
            mark_to_market_change = 0.0

        # Novelty: subtract transaction cost from reward to penalise overtrading
        reward = mark_to_market_change - cost

        # ---- Advance step ----
        self.current_step += 1
        done = (self.current_step >= self.end_idx)

        next_state = self._compute_state()
        info = {
            "price"           : price,
            "portfolio_value" : self.portfolio_value,
            "position"        : self.position,
        }

        return next_state, reward, done, info

    @property
    def state_size(self) -> int:
        return 5   # number of features

    @property
    def action_size(self) -> int:
        return 3   # Hold, Buy, Sell