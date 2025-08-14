"""
Test suite for risk management
"""

import unittest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.risk.manager import RiskManager, RiskMetrics, RiskBreach
from src.core.portfolio import Portfolio, Position

class TestRiskManager(unittest.TestCase):
    """Test risk manager"""
    
    def setUp(self):
        self.risk_config = {
            'max_drawdown': 0.15,
            'var_confidence': 0.95,
            'max_position_size': 0.05,
            'max_sector_exposure': 0.3,
            'max_leverage': 2.0
        }
        self.risk_manager = RiskManager(self.risk_config)
        
        # Create test portfolio
        self.portfolio = Portfolio(initial_cash=1000000)
        
        # Add some test positions
        self.portfolio.add_position('AAPL', 100, 150.0)
        self.portfolio.add_position('MSFT', 50, 300.0)
        self.portfolio.add_position('GOOGL', 10, 2500.0)
    
    def test_risk_manager_initialization(self):
        """Test risk manager initialization"""
        self.assertEqual(self.risk_manager.max_drawdown, 0.15)
        self.assertEqual(self.risk_manager.var_confidence, 0.95)
        self.assertEqual(self.risk_manager.max_position_size, 0.05)
    
    def test_risk_metrics_calculation(self):
        """Test risk metrics calculation"""
        # Add some portfolio value history for VaR calculation
        for i in range(50):
            value = 1000000 + np.random.normal(0, 10000)
            self.risk_manager.update_portfolio_history(value)
        
        risk_metrics = self.risk_manager._calculate_risk_metrics(self.portfolio)
        
        # Check that metrics are reasonable
        self.assertIsInstance(risk_metrics, RiskMetrics)
        self.assertGreaterEqual(risk_metrics.var_95, 0)
        self.assertGreaterEqual(risk_metrics.var_99, 0)
        self.assertGreaterEqual(risk_metrics.max_drawdown, 0)
        self.assertLessEqual(risk_metrics.concentration_risk, 1.0)
        self.assertGreaterEqual(risk_metrics.leverage_ratio, 0)
    
    async def test_portfolio_risk_check(self):
        """Test portfolio risk checking"""
        breaches = await self.risk_manager.check_portfolio(self.portfolio)
        
        # Should return a list of breaches
        self.assertIsInstance(breaches, list)
        
        # Each breach should be a RiskBreach object
        for breach in breaches:
            self.assertIsInstance(breach, RiskBreach)
            self.assertIn(breach.severity, ['WARNING', 'CRITICAL'])
    
    def test_position_concentration_check(self):
        """Test position concentration checking"""
        # Add a large position that exceeds limits
        self.portfolio.add_position('TSLA', 1000, 800.0)  # Large position
        
        breaches = self.risk_manager._check_position_concentration(self.portfolio)
        
        # Should detect concentration breach
        self.assertGreater(len(breaches), 0)
        
        # Check that breach is properly formed
        for breach in breaches:
            self.assertEqual(breach.risk_type, 'POSITION_CONCENTRATION')
            self.assertGreater(breach.current_value, breach.limit_value)
    
    def test_scenario_analysis(self):
        """Test stress scenario analysis"""
        # Market crash scenario
        scenario_pnl = self.risk_manager._scenario_analysis(
            self.portfolio, 
            market_shock=-0.20
        )
        
        # Should return negative P&L for market crash
        self.assertLess(scenario_pnl, 0)
        
        # Volatility spike scenario
        vol_scenario_pnl = self.risk_manager._scenario_analysis(
            self.portfolio,
            vol_shock=0.50
        )
        
        # Should return a finite number
        self.assertTrue(np.isfinite(vol_scenario_pnl))
    
    async def test_signal_risk_check(self):
        """Test signal risk checking"""
        from src.core.events import SignalEvent
        from src.strategies.base_strategy import Signal
        
        # Create a test signal
        signal = Signal(
            symbol='AAPL',
            signal_type='BUY',
            strength=0.8,
            confidence=0.9,
            strategy='test_strategy',
            timestamp=datetime.now()
        )
        
        signal_event = Mock()
        signal_event.symbol = signal.symbol
        signal_event.strength = signal.strength
        signal_event.confidence = signal.confidence
        
        # Check if signal passes risk checks
        is_valid = await self.risk_manager.check_signal(signal_event, self.portfolio)
        
        # Should return boolean
        self.assertIsInstance(is_valid, bool)

if __name__ == '__main__':
    unittest.main()