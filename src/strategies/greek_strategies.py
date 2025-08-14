"""
Greek-based Options Trading Strategies
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any
from datetime import datetime, timedelta

from .base_strategy import BaseStrategy, Signal
from ..core.events import MarketDataEvent
from ..greeks.calculator import GreeksCalculator

class DeltaNeutralStrategy(BaseStrategy):
    """Delta-neutral options strategy"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        
        self.target_delta = self.parameters.get('target_delta', 0.0)
        self.delta_tolerance = self.parameters.get('delta_tolerance', 5.0)
        self.rehedge_threshold = self.parameters.get('rehedge_threshold', 10.0)
        self.underlying_universe = self.parameters.get('underlying_universe', ['SPY'])
        
        self.greeks_calculator = GreeksCalculator()
        self.option_chains = {}
        self.current_positions = {}
        
    async def generate_signals(self, market_data: Dict[str, Any]) -> List[Signal]:
        """Generate delta-neutral signals"""
        signals = []
        
        for underlying in self.underlying_universe:
            # Calculate current portfolio delta for this underlying
            current_delta = self._calculate_portfolio_delta(underlying)
            
            # Check if rehedging is needed
            if abs(current_delta - self.target_delta) > self.rehedge_threshold:
                hedge_signal = self._generate_hedge_signal(underlying, current_delta)
                if hedge_signal:
                    signals.append(hedge_signal)
        
        return signals
    
    def _calculate_portfolio_delta(self, underlying: str) -> float:
        """Calculate current portfolio delta for an underlying"""
        total_delta = 0.0
        
        if underlying in self.current_positions:
            for position in self.current_positions[underlying]:
                if position['type'] == 'option':
                    greeks = self.greeks_calculator.calculate_greeks(
                        spot_price=position['spot_price'],
                        strike_price=position['strike'],
                        time_to_expiry=position['tte'],
                        risk_free_rate=0.02,
                        volatility=position['iv'],
                        option_type=position['option_type']
                    )
                    total_delta += greeks['delta'] * position['quantity']
                elif position['type'] == 'stock':
                    total_delta += position['quantity']
        
        return total_delta
    
    def _generate_hedge_signal(self, underlying: str, current_delta: float) -> Signal:
        """Generate hedging signal to achieve delta neutrality"""
        required_delta_adjustment = self.target_delta - current_delta
        
        # Simple hedging with underlying stock
        if abs(required_delta_adjustment) > self.delta_tolerance:
            signal_type = 'BUY' if required_delta_adjustment > 0 else 'SELL'
            
            return Signal(
                symbol=underlying,
                signal_type=signal_type,
                strength=min(1.0, abs(required_delta_adjustment) / 100),
                confidence=0.9,
                strategy=self.name,
                timestamp=datetime.now(),
                metadata={
                    'delta_adjustment': required_delta_adjustment,
                    'hedge_type': 'stock'
                }
            )
        
        return None
    
    async def on_market_data(self, event: MarketDataEvent):
        """Handle market data updates"""
        # Update spot prices and recalculate Greeks
        for symbol, price in event.data.items():
            if symbol in self.underlying_universe:
                self._update_position_greeks(symbol, price)
    
    async def on_fill(self, fill_event):
        """Handle order fills"""
        symbol = fill_event.symbol
        
        # Update positions
        if symbol not in self.current_positions:
            self.current_positions[symbol] = []
        
        # Add new position or update existing
        self.current_positions[symbol].append({
            'type': 'stock',  # Simplified
            'quantity': fill_event.quantity,
            'price': fill_event.price,
            'timestamp': datetime.now()
        })

class GammaScalpingStrategy(BaseStrategy):
    """Gamma scalping strategy"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        
        self.target_gamma = self.parameters.get('target_gamma', 25.0)
        self.scalping_threshold = self.parameters.get('scalping_threshold', 0.5)
        self.profit_target = self.parameters.get('profit_target', 0.02)
        self.stop_loss = self.parameters.get('stop_loss', 0.05)
        
        self.greeks_calculator = GreeksCalculator()
        self.last_hedge_price = {}
        self.scalping_positions = {}
        
    async def generate_signals(self, market_data: Dict[str, Any]) -> List[Signal]:
        """Generate gamma scalping signals"""
        signals = []
        
        for symbol, current_price in market_data.items():
            if symbol in self.scalping_positions:
                scalp_signals = self._check_scalping_opportunity(symbol, current_price)
                signals.extend(scalp_signals)
        
        return signals
    
    def _check_scalping_opportunity(self, symbol: str, current_price: float) -> List[Signal]:
        """Check for gamma scalping opportunities"""
        signals = []
        
        if symbol not in self.last_hedge_price:
            self.last_hedge_price[symbol] = current_price
            return signals
        
        price_change = current_price - self.last_hedge_price[symbol]
        price_change_pct = price_change / self.last_hedge_price[symbol]
        
        if abs(price_change_pct) > self.scalping_threshold / 100:
            # Calculate required hedge adjustment
            position = self.scalping_positions.get(symbol, {})
            current_gamma = position.get('gamma', 0)
            
            # Hedge delta gained from gamma
            delta_adjustment = current_gamma * price_change
            
            if abs(delta_adjustment) > 5:  # Minimum hedge size
                signal_type = 'SELL' if delta_adjustment > 0 else 'BUY'
                
                signals.append(Signal(
                    symbol=symbol,
                    signal_type=signal_type,
                    strength=min(1.0, abs(delta_adjustment) / 50),
                    confidence=0.85,
                    strategy=self.name,
                    timestamp=datetime.now(),
                    metadata={
                        'gamma_scalp': True,
                        'delta_adjustment': delta_adjustment,
                        'price_change': price_change
                    }
                ))
                
                # Update last hedge price
                self.last_hedge_price[symbol] = current_price
        
        return signals
    
    async def on_market_data(self, event: MarketDataEvent):
        """Handle market data for gamma scalping"""
        for symbol, price in event.data.items():
            if symbol in self.scalping_positions:
                # Update position Greeks
                self._update_gamma_position(symbol, price)
    
    async def on_fill(self, fill_event):
        """Handle fills for gamma scalping"""
        # Update scalping positions
        pass

class ThetaHarvestingStrategy(BaseStrategy):
    """Theta harvesting strategy"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        
        self.min_theta = self.parameters.get('min_theta', -50)
        self.days_to_expiry = self.parameters.get('days_to_expiry', 30)
        self.iv_percentile_threshold = self.parameters.get('implied_vol_percentile', 0.8)
        self.profit_target = self.parameters.get('profit_target', 0.5)
        
        self.greeks_calculator = GreeksCalculator()
        self.theta_positions = {}
        
    async def generate_signals(self, market_data: Dict[str, Any]) -> List[Signal]:
        """Generate theta harvesting signals"""
        signals = []
        
        # Look for high IV options to sell
        for symbol in market_data.keys():
            theta_signals = await self._find_theta_opportunities(symbol)
            signals.extend(theta_signals)
        
        return signals
    
    async def _find_theta_opportunities(self, symbol: str) -> List[Signal]:
        """Find theta harvesting opportunities"""
        signals = []
        
        # Simplified: would need real options chain data
        # For now, generate conceptual signals
        
        if self._is_high_iv_environment(symbol):
            # Sell straddles or strangles for theta decay
            signals.append(Signal(
                symbol=f"{symbol}_STRADDLE",
                signal_type='SELL',
                strength=0.7,
                confidence=0.8,
                strategy=self.name,
                timestamp=datetime.now(),
                metadata={
                    'strategy_type': 'short_straddle',
                    'theta_target': self.min_theta
                }
            ))
        
        return signals
    
    def _is_high_iv_environment(self, symbol: str) -> bool:
        """Check if implied volatility is in high percentile"""
        # Simplified check - would use real IV data
        return True  # Placeholder
    
    async def on_market_data(self, event: MarketDataEvent):
        """Handle market data for theta harvesting"""
        # Update theta positions and check for profit targets
        for symbol in self.theta_positions:
            self._check_profit_target(symbol)
    
    async def on_fill(self, fill_event):
        """Handle fills for theta positions"""
        # Track theta positions
        pass
    
    def _check_profit_target(self, symbol: str):
        """Check if theta position has reached profit target"""
        # Implementation for profit taking
        pass