"""
Base Strategy Framework
Foundation for all trading strategies
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import numpy as np

from ..core.events import SignalEvent, MarketDataEvent
from ..utils.helpers import load_config

logger = logging.getLogger(__name__)

@dataclass
class Signal:
    """Trading signal representation"""
    symbol: str
    signal_type: str  # 'BUY', 'SELL', 'HOLD'
    strength: float   # Signal strength 0-1
    confidence: float # Confidence level 0-1
    strategy: str     # Strategy name
    timestamp: datetime
    metadata: Dict[str, Any] = None

class BaseStrategy(ABC):
    """Abstract base class for all trading strategies"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.enabled = config.get('enabled', True)
        self.allocation = config.get('allocation', 0.0)
        self.parameters = config.get('parameters', {})
        
        # Strategy state
        self.is_running = False
        self.last_update = None
        self.positions = {}
        self.signals_generated = 0
        self.performance_metrics = {}
        
        logger.info(f"Strategy {self.name} initialized with allocation {self.allocation}")
    
    @abstractmethod
    async def generate_signals(self, market_data: Dict[str, Any]) -> List[Signal]:
        """Generate trading signals based on market data"""
        pass
    
    @abstractmethod
    async def on_market_data(self, event: MarketDataEvent):
        """Handle incoming market data"""
        pass
    
    @abstractmethod
    async def on_fill(self, fill_event):
        """Handle order fills"""
        pass
    
    async def start(self):
        """Start the strategy"""
        if not self.enabled:
            logger.info(f"Strategy {self.name} is disabled")
            return
        
        self.is_running = True
        logger.info(f"Strategy {self.name} started")
    
    async def stop(self):
        """Stop the strategy"""
        self.is_running = False
        logger.info(f"Strategy {self.name} stopped")
    
    async def update(self):
        """Update strategy logic (called periodically)"""
        if not self.is_running:
            return
        
        self.last_update = datetime.now()
    
    def calculate_position_size(self, signal: Signal, portfolio_value: float) -> float:
        """Calculate position size based on signal and risk parameters"""
        base_size = portfolio_value * self.allocation
        size_multiplier = signal.strength * signal.confidence
        
        return base_size * size_multiplier
    
    def update_performance_metrics(self, metrics: Dict[str, float]):
        """Update strategy performance metrics"""
        self.performance_metrics.update(metrics)

class StrategyManager:
    """Manages all trading strategies"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.strategies: Dict[str, BaseStrategy] = {}
        self.strategy_allocations = {}
        
        # Load strategy configurations
        self._load_strategies()
        
        logger.info(f"Strategy Manager initialized with {len(self.strategies)} strategies")
    
    def _load_strategies(self):
        """Load and initialize all strategies"""
        from .equity_long_short import EquityLongShortStrategy
        from .greek_strategies import DeltaNeutralStrategy, GammaScalpingStrategy, ThetaHarvestingStrategy
        from .ml_strategies import MLMomentumStrategy
        
        strategy_configs = load_config("configs/strategies.yaml")
        
        # Initialize strategies based on configuration
        strategy_classes = {
            'equity_long_short': EquityLongShortStrategy,
            'delta_neutral': DeltaNeutralStrategy,
            'gamma_scalping': GammaScalpingStrategy,
            'theta_harvesting': ThetaHarvestingStrategy,
            'ml_momentum': MLMomentumStrategy
        }
        
        for strategy_name, strategy_config in strategy_configs['strategies'].items():
            if strategy_name in strategy_classes and strategy_config.get('enabled', False):
                strategy_class = strategy_classes[strategy_name]
                self.strategies[strategy_name] = strategy_class(strategy_name, strategy_config)
                self.strategy_allocations[strategy_name] = strategy_config.get('allocation', 0.0)
    
    async def start(self):
        """Start all enabled strategies"""
        for strategy in self.strategies.values():
            await strategy.start()
    
    async def stop(self):
        """Stop all strategies"""
        for strategy in self.strategies.values():
            await strategy.stop()
    
    async def update(self):
        """Update all strategies"""
        for strategy in self.strategies.values():
            await strategy.update()
    
    async def on_market_data(self, event: MarketDataEvent):
        """Distribute market data to all strategies"""
        for strategy in self.strategies.values():
            if strategy.is_running:
                await strategy.on_market_data(event)
    
    async def on_fill(self, fill_event):
        """Distribute fill events to all strategies"""
        for strategy in self.strategies.values():
            if strategy.is_running:
                await strategy.on_fill(fill_event)
    
    async def signal_to_orders(self, signal_event: SignalEvent):
        """Convert strategy signals to executable orders"""
        orders = []
        # Implementation depends on order management system
        return orders
    
    async def emergency_stop(self):
        """Emergency stop all strategies"""
        logger.warning("Emergency stop triggered for all strategies")
        for strategy in self.strategies.values():
            await strategy.stop()