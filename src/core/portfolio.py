"""
Portfolio management and tracking
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd
import numpy as np

from ..greeks.calculator import GreeksCalculator

logger = logging.getLogger(__name__)

class Position:
    """Represents a single position in the portfolio"""
    
    def __init__(self, symbol: str, quantity: float, entry_price: float,
                 position_type: str = "equity", option_details: Dict = None):
        self.symbol = symbol
        self.quantity = quantity
        self.entry_price = entry_price
        self.current_price = entry_price
        self.position_type = position_type  # equity, option
        self.option_details = option_details or {}
        self.entry_time = datetime.now()
        self.last_update = datetime.now()
        
        # Greeks (for options)
        self.greeks = {}
        
    @property
    def market_value(self) -> float:
        """Current market value of the position"""
        if self.position_type == "option":
            return self.quantity * self.current_price * 100  # Option multiplier
        return self.quantity * self.current_price
    
    @property
    def unrealized_pnl(self) -> float:
        """Unrealized P&L of the position"""
        if self.position_type == "option":
            return (self.current_price - self.entry_price) * self.quantity * 100
        return (self.current_price - self.entry_price) * self.quantity
    
    def update_price(self, new_price: float):
        """Update the current price"""
        self.current_price = new_price
        self.last_update = datetime.now()

class Portfolio:
    """Portfolio manager handling positions and metrics"""
    
    def __init__(self, initial_cash: float = 1000000):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_value = initial_cash
        
        # Greeks tracking
        self.greeks_calculator = GreeksCalculator()
        self.portfolio_greeks = {
            'delta': 0.0,
            'gamma': 0.0,
            'theta': 0.0,
            'vega': 0.0,
            'rho': 0.0
        }
        
        logger.info(f"Portfolio initialized with ${initial_cash:,.2f}")
    
    @property
    def total_value(self) -> float:
        """Total portfolio value"""
        positions_value = sum(pos.market_value for pos in self.positions.values())
        return self.cash + positions_value
    
    @property
    def total_return(self) -> float:
        """Total return percentage"""
        return (self.total_value - self.initial_cash) / self.initial_cash
    
    @property
    def sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio (simplified)"""
        # This would need historical returns for proper calculation
        if len(self.closed_positions) < 2:
            return 0.0
        
        returns = [pos.unrealized_pnl / self.initial_cash for pos in self.closed_positions]
        return np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252)
    
    def add_position(self, symbol: str, quantity: float, price: float,
                    position_type: str = "equity", option_details: Dict = None):
        """Add a new position or update existing one"""
        
        cost = quantity * price
        if position_type == "option":
            cost *= 100  # Option multiplier
        
        if cost > self.cash:
            raise ValueError(f"Insufficient cash for position. Need: ${cost:.2f}, Available: ${self.cash:.2f}")
        
        if symbol in self.positions:
            # Update existing position
            existing = self.positions[symbol]
            total_quantity = existing.quantity + quantity
            total_cost = (existing.quantity * existing.entry_price + 
                         quantity * price)
            
            if total_quantity != 0:
                avg_price = total_cost / total_quantity
                existing.quantity = total_quantity
                existing.entry_price = avg_price
            else:
                # Position closed
                self._close_position(symbol)
        else:
            # New position
            self.positions[symbol] = Position(
                symbol, quantity, price, position_type, option_details
            )
        
        self.cash -= cost
        self.total_trades += 1
        
        logger.info(f"Added position: {symbol} {quantity} @ ${price:.2f}")
    
    def update_prices(self, price_data: Dict[str, float]):
        """Update prices for all positions"""
        for symbol, position in self.positions.items():
            if symbol in price_data:
                position.update_price(price_data[symbol])
    
    def update_position(self, fill_event):
        """Update position based on fill event"""
        symbol = fill_event.symbol
        quantity = fill_event.quantity
        price = fill_event.price
        
        self.add_position(symbol, quantity, price)
    
    def _close_position(self, symbol: str):
        """Close a position"""
        if symbol in self.positions:
            position = self.positions.pop(symbol)
            
            # Calculate realized P&L
            realized_pnl = position.unrealized_pnl
            self.total_pnl += realized_pnl
            
            if realized_pnl > 0:
                self.winning_trades += 1
            
            # Add cash back
            self.cash += position.market_value
            
            # Move to closed positions
            self.closed_positions.append(position)
            
            logger.info(f"Closed position: {symbol}, P&L: ${realized_pnl:.2f}")
    
    def calculate_portfolio_greeks(self):
        """Calculate portfolio-level Greeks"""
        total_greeks = {
            'delta': 0.0,
            'gamma': 0.0,
            'theta': 0.0,
            'vega': 0.0,
            'rho': 0.0
        }
        
        for position in self.positions.values():
            if position.position_type == "option":
                # Calculate Greeks for this option position
                greeks = self.greeks_calculator.calculate_greeks(
                    spot_price=position.current_price,
                    strike_price=position.option_details.get('strike', 0),
                    time_to_expiry=position.option_details.get('tte', 0),
                    risk_free_rate=position.option_details.get('rate', 0.02),
                    volatility=position.option_details.get('iv', 0.2),
                    option_type=position.option_details.get('type', 'call')
                )
                
                # Weight by position size
                for greek, value in greeks.items():
                    total_greeks[greek] += value * position.quantity
        
        self.portfolio_greeks = total_greeks
        return total_greeks
    
    def update_metrics(self):
        """Update portfolio performance metrics"""
        current_value = self.total_value
        
        # Update peak value
        if current_value > self.peak_value:
            self.peak_value = current_value
        
        # Calculate drawdown
        drawdown = (self.peak_value - current_value) / self.peak_value
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown
        
        # Update Greeks
        self.calculate_portfolio_greeks()
    
    def get_position_summary(self) -> pd.DataFrame:
        """Get summary of all positions"""
        data = []
        for symbol, position in self.positions.items():
            data.append({
                'Symbol': symbol,
                'Quantity': position.quantity,
                'Entry Price': position.entry_price,
                'Current Price': position.current_price,
                'Market Value': position.market_value,
                'Unrealized P&L': position.unrealized_pnl,
                'Type': position.position_type
            })
        
        return pd.DataFrame(data)
    
    def get_performance_summary(self) -> Dict:
        """Get portfolio performance summary"""
        return {
            'total_value': self.total_value,
            'cash': self.cash,
            'total_return': self.total_return,
            'total_pnl': self.total_pnl,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': self.sharpe_ratio,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': self.winning_trades / max(self.total_trades, 1),
            'portfolio_greeks': self.portfolio_greeks
        }