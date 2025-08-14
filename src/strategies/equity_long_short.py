"""
Multi-Factor Long-Short Equity Strategy
"""

import numpy as np
import pandas as pd
import asyncio
from typing import Dict, List, Any
from datetime import datetime, timedelta

from .base_strategy import BaseStrategy, Signal
from ..core.events import MarketDataEvent
from ..ml.models import MLSignalGenerator

class EquityLongShortStrategy(BaseStrategy):
    """Multi-factor long-short equity strategy with ML enhancement"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        
        # Strategy parameters
        self.lookback_period = self.parameters.get('lookback_period', 252)
        self.rebalance_frequency = self.parameters.get('rebalance_frequency', 'daily')
        self.long_short_ratio = self.parameters.get('long_short_ratio', 1.0)
        self.factors = self.parameters.get('factors', ['momentum', 'value', 'quality'])
        
        # Universe and data
        self.universe = self._get_universe()
        self.price_data = {}
        self.fundamental_data = {}
        
        # ML components
        self.ml_signal_generator = MLSignalGenerator()
        
        # Factor scores
        self.factor_scores = {}
        self.composite_scores = {}
        
    def _get_universe(self) -> List[str]:
        """Get trading universe"""
        # For demo, using a small universe
        return [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',
            'META', 'NVDA', 'JPM', 'JNJ', 'V',
            'PG', 'UNH', 'HD', 'MA', 'DIS'
        ]
    
    async def generate_signals(self, market_data: Dict[str, Any]) -> List[Signal]:
        """Generate long-short equity signals"""
        signals = []
        
        if not self._has_sufficient_data():
            return signals
        
        # Calculate factor scores
        await self._calculate_factor_scores()
        
        # Generate composite scores
        self._calculate_composite_scores()
        
        # Generate ML-enhanced signals
        ml_signals = await self.ml_signal_generator.generate_equity_signals(
            self.price_data, self.factor_scores
        )
        
        # Rank stocks and generate long/short signals
        ranked_stocks = self._rank_stocks()
        
        long_candidates = ranked_stocks[:5]  # Top 5 for long
        short_candidates = ranked_stocks[-5:]  # Bottom 5 for short
        
        # Generate long signals
        for symbol in long_candidates:
            ml_boost = ml_signals.get(symbol, 0.0)
            signal_strength = min(1.0, self.composite_scores[symbol] + ml_boost)
            
            signals.append(Signal(
                symbol=symbol,
                signal_type='BUY',
                strength=signal_strength,
                confidence=0.8,
                strategy=self.name,
                timestamp=datetime.now(),
                metadata={'factor_score': self.composite_scores[symbol]}
            ))
        
        # Generate short signals
        for symbol in short_candidates:
            ml_boost = abs(ml_signals.get(symbol, 0.0))
            signal_strength = min(1.0, abs(self.composite_scores[symbol]) + ml_boost)
            
            signals.append(Signal(
                symbol=symbol,
                signal_type='SELL',
                strength=signal_strength,
                confidence=0.8,
                strategy=self.name,
                timestamp=datetime.now(),
                metadata={'factor_score': self.composite_scores[symbol]}
            ))
        
        self.signals_generated += len(signals)
        return signals
    
    async def _calculate_factor_scores(self):
        """Calculate factor scores for all stocks"""
        self.factor_scores = {}
        
        for symbol in self.universe:
            if symbol not in self.price_data:
                continue
                
            scores = {}
            
            # Momentum factor
            if 'momentum' in self.factors:
                scores['momentum'] = self._calculate_momentum_score(symbol)
            
            # Value factor
            if 'value' in self.factors:
                scores['value'] = self._calculate_value_score(symbol)
            
            # Quality factor
            if 'quality' in self.factors:
                scores['quality'] = self._calculate_quality_score(symbol)
            
            # Volatility factor
            if 'volatility' in self.factors:
                scores['volatility'] = self._calculate_volatility_score(symbol)
            
            self.factor_scores[symbol] = scores
    
    def _calculate_momentum_score(self, symbol: str) -> float:
        """Calculate momentum factor score"""
        if symbol not in self.price_data:
            return 0.0
        
        prices = self.price_data[symbol]
        if len(prices) < 21:
            return 0.0
        
        # 1-month, 3-month, 6-month momentum
        returns_1m = (prices[-1] / prices[-21] - 1) if len(prices) >= 21 else 0
        returns_3m = (prices[-1] / prices[-63] - 1) if len(prices) >= 63 else 0
        returns_6m = (prices[-1] / prices[-126] - 1) if len(prices) >= 126 else 0
        
        # Weighted momentum score
        momentum_score = 0.5 * returns_1m + 0.3 * returns_3m + 0.2 * returns_6m
        
        # Normalize to [-1, 1]
        return np.tanh(momentum_score * 10)
    
    def _calculate_value_score(self, symbol: str) -> float:
        """Calculate value factor score"""
        # Simplified value score (would use fundamental data in production)
        if symbol not in self.price_data:
            return 0.0
        
        prices = self.price_data[symbol]
        if len(prices) < 252:
            return 0.0
        
        # Price relative to 1-year average as proxy for value
        avg_price = np.mean(prices[-252:])
        current_price = prices[-1]
        
        value_score = (avg_price / current_price - 1)
        return np.tanh(value_score * 5)
    
    def _calculate_quality_score(self, symbol: str) -> float:
        """Calculate quality factor score"""
        # Simplified quality score based on price stability
        if symbol not in self.price_data:
            return 0.0
        
        prices = self.price_data[symbol]
        if len(prices) < 63:
            return 0.0
        
        # Lower volatility = higher quality
        returns = np.diff(np.log(prices[-63:]))
        volatility = np.std(returns) * np.sqrt(252)
        
        # Inverse relationship with volatility
        quality_score = 1.0 / (1.0 + volatility)
        return (quality_score - 0.5) * 2  # Normalize to [-1, 1]
    
    def _calculate_volatility_score(self, symbol: str) -> float:
        """Calculate volatility factor score"""
        if symbol not in self.price_data:
            return 0.0
        
        prices = self.price_data[symbol]
        if len(prices) < 21:
            return 0.0
        
        returns = np.diff(np.log(prices[-21:]))
        volatility = np.std(returns) * np.sqrt(252)
        
        # High volatility gets negative score
        return -np.tanh(volatility * 2)
    
    def _calculate_composite_scores(self):
        """Calculate composite factor scores"""
        self.composite_scores = {}
        
        for symbol in self.factor_scores:
            scores = self.factor_scores[symbol]
            
            # Equal weight combination (could be optimized)
            composite = np.mean(list(scores.values()))
            self.composite_scores[symbol] = composite
    
    def _rank_stocks(self) -> List[str]:
        """Rank stocks by composite score"""
        ranked = sorted(
            self.composite_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [symbol for symbol, score in ranked]
    
    def _has_sufficient_data(self) -> bool:
        """Check if we have sufficient data for analysis"""
        return len(self.price_data) > len(self.universe) * 0.8
    
    async def on_market_data(self, event: MarketDataEvent):
        """Handle incoming market data"""
        # Update price data
        for symbol, price in event.data.items():
            if symbol in self.universe:
                if symbol not in self.price_data:
                    self.price_data[symbol] = []
                
                self.price_data[symbol].append(price)
                
                # Keep only lookback period
                if len(self.price_data[symbol]) > self.lookback_period:
                    self.price_data[symbol] = self.price_data[symbol][-self.lookback_period:]
    
    async def on_fill(self, fill_event):
        """Handle order fills"""
        symbol = fill_event.symbol
        quantity = fill_event.quantity
        
        if symbol not in self.positions:
            self.positions[symbol] = 0
        
        self.positions[symbol] += quantity
        
        logger.info(f"Strategy {self.name} fill: {symbol} {quantity}")