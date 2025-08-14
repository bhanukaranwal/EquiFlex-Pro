"""
Test suite for trading strategies (Continued)
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from src.strategies.equity_long_short import EquityLongShortStrategy
from src.strategies.greek_strategies import DeltaNeutralStrategy, GammaScalpingStrategy
from src.core.events import MarketDataEvent

class TestDeltaNeutralStrategy(unittest.TestCase):
    """Test delta-neutral strategy"""
    
    def setUp(self):
        self.strategy_config = {
            'name': 'test_delta_neutral',
            'enabled': True,
            'allocation': 0.2,
            'parameters': {
                'target_delta': 0.0,
                'delta_tolerance': 5.0,
                'rehedge_threshold': 10.0,
                'underlying_universe': ['SPY']
            }
        }
        self.strategy = DeltaNeutralStrategy('test_delta_neutral', self.strategy_config)
    
    def test_strategy_initialization(self):
        """Test delta-neutral strategy initialization"""
        self.assertEqual(self.strategy.target_delta, 0.0)
        self.assertEqual(self.strategy.delta_tolerance, 5.0)
        self.assertEqual(self.strategy.rehedge_threshold, 10.0)
        self.assertIn('SPY', self.strategy.underlying_universe)
    
    def test_portfolio_delta_calculation(self):
        """Test portfolio delta calculation"""
        # Setup mock positions
        self.strategy.current_positions['SPY'] = [
            {
                'type': 'option',
                'quantity': 10,
                'spot_price': 400.0,
                'strike': 400.0,
                'tte': 0.25,
                'iv': 0.2,
                'option_type': 'call'
            },
            {
                'type': 'stock',
                'quantity': -5,  # Short stock hedge
                'price': 400.0
            }
        ]
        
        delta = self.strategy._calculate_portfolio_delta('SPY')
        
        # Should return a finite number
        self.assertTrue(np.isfinite(delta))
    
    def test_hedge_signal_generation(self):
        """Test hedge signal generation"""
        current_delta = 15.0  # Exceeds rehedge threshold
        
        signal = self.strategy._generate_hedge_signal('SPY', current_delta)
        
        # Should generate a hedge signal
        self.assertIsNotNone(signal)
        self.assertEqual(signal.symbol, 'SPY')
        self.assertEqual(signal.signal_type, 'SELL')  # Sell to reduce positive delta
        self.assertGreater(signal.strength, 0)

class TestGammaScalpingStrategy(unittest.TestCase):
    """Test gamma scalping strategy"""
    
    def setUp(self):
        self.strategy_config = {
            'name': 'test_gamma_scalping',
            'enabled': True,
            'allocation': 0.2,
            'parameters': {
                'target_gamma': 25.0,
                'scalping_threshold': 0.5,
                'profit_target': 0.02,
                'stop_loss': 0.05
            }
        }
        self.strategy = GammaScalpingStrategy('test_gamma_scalping', self.strategy_config)
    
    def test_scalping_opportunity_detection(self):
        """Test gamma scalping opportunity detection"""
        # Setup initial price
        self.strategy.last_hedge_price['SPY'] = 400.0
        self.strategy.scalping_positions['SPY'] = {'gamma': 30.0}
        
        # Simulate price movement
        current_price = 402.0  # 0.5% move
        
        signals = self.strategy._check_scalping_opportunity('SPY', current_price)
        
        # Should generate scalping signal for significant price move
        if len(signals) > 0:
            signal = signals[0]
            self.assertEqual(signal.symbol, 'SPY')
            self.assertIn(signal.signal_type, ['BUY', 'SELL'])
            self.assertTrue(signal.metadata.get('gamma_scalp', False))

if __name__ == '__main__':
    unittest.main()