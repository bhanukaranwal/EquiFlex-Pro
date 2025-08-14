"""
Test suite for Greeks calculations
"""

import unittest
import numpy as np
import pytest
from datetime import datetime, timedelta

from src.greeks.calculator import GreeksCalculator, PortfolioGreeksManager

class TestGreeksCalculator(unittest.TestCase):
    """Test Greeks calculator"""
    
    def setUp(self):
        self.calculator = GreeksCalculator()
        
        # Standard test parameters
        self.spot_price = 100.0
        self.strike_price = 100.0
        self.time_to_expiry = 0.25  # 3 months
        self.risk_free_rate = 0.05
        self.volatility = 0.2
    
    def test_call_option_greeks(self):
        """Test call option Greeks calculation"""
        greeks = self.calculator.calculate_greeks(
            spot_price=self.spot_price,
            strike_price=self.strike_price,
            time_to_expiry=self.time_to_expiry,
            risk_free_rate=self.risk_free_rate,
            volatility=self.volatility,
            option_type='call'
        )
        
        # Delta should be positive for call options
        self.assertGreater(greeks['delta'], 0)
        self.assertLess(greeks['delta'], 1)
        
        # Gamma should be positive
        self.assertGreater(greeks['gamma'], 0)
        
        # Theta should be negative for long options
        self.assertLess(greeks['theta'], 0)
        
        # Vega should be positive
        self.assertGreater(greeks['vega'], 0)
        
        # Rho should be positive for call options
        self.assertGreater(greeks['rho'], 0)
    
    def test_put_option_greeks(self):
        """Test put option Greeks calculation"""
        greeks = self.calculator.calculate_greeks(
            spot_price=self.spot_price,
            strike_price=self.strike_price,
            time_to_expiry=self.time_to_expiry,
            risk_free_rate=self.risk_free_rate,
            volatility=self.volatility,
            option_type='put'
        )
        
        # Delta should be negative for put options
        self.assertLess(greeks['delta'], 0)
        self.assertGreater(greeks['delta'], -1)
        
        # Gamma should be positive (same as call)
        self.assertGreater(greeks['gamma'], 0)
        
        # Theta should be negative
        self.assertLess(greeks['theta'], 0)
        
        # Vega should be positive (same as call)
        self.assertGreater(greeks['vega'], 0)
        
        # Rho should be negative for put options
        self.assertLess(greeks['rho'], 0)
    
    def test_at_the_money_delta(self):
        """Test that ATM options have delta around 0.5"""
        greeks = self.calculator.calculate_greeks(
            spot_price=100.0,
            strike_price=100.0,
            time_to_expiry=self.time_to_expiry,
            risk_free_rate=self.risk_free_rate,
            volatility=self.volatility,
            option_type='call'
        )
        
        # ATM call delta should be around 0.5
        self.assertAlmostEqual(greeks['delta'], 0.5, delta=0.1)
    
    def test_time_decay_effect(self):
        """Test that theta increases as expiry approaches"""
        # Calculate Greeks for two different times to expiry
        greeks_long = self.calculator.calculate_greeks(
            spot_price=self.spot_price,
            strike_price=self.strike_price,
            time_to_expiry=0.5,  # 6 months
            risk_free_rate=self.risk_free_rate,
            volatility=self.volatility,
            option_type='call'
        )
        
        greeks_short = self.calculator.calculate_greeks(
            spot_price=self.spot_price,
            strike_price=self.strike_price,
            time_to_expiry=0.1,  # 1 month
            risk_free_rate=self.risk_free_rate,
            volatility=self.volatility,
            option_type='call'
        )
        
        # Shorter time to expiry should have more negative theta (higher time decay)
        self.assertLess(greeks_short['theta'], greeks_long['theta'])
    
    def test_volatility_effect_on_vega(self):
        """Test vega sensitivity to volatility changes"""
        greeks_low_vol = self.calculator.calculate_greeks(
            spot_price=self.spot_price,
            strike_price=self.strike_price,
            time_to_expiry=self.time_to_expiry,
            risk_free_rate=self.risk_free_rate,
            volatility=0.1,  # Low volatility
            option_type='call'
        )
        
        greeks_high_vol = self.calculator.calculate_greeks(
            spot_price=self.spot_price,
            strike_price=self.strike_price,
            time_to_expiry=self.time_to_expiry,
            risk_free_rate=self.risk_free_rate,
            volatility=0.4,  # High volatility
            option_type='call'
        )
        
        # Both should have positive vega
        self.assertGreater(greeks_low_vol['vega'], 0)
        self.assertGreater(greeks_high_vol['vega'], 0)
    
    def test_zero_time_to_expiry(self):
        """Test Greeks for expired options"""
        greeks = self.calculator.calculate_greeks(
            spot_price=self.spot_price,
            strike_price=self.strike_price,
            time_to_expiry=0.0,
            risk_free_rate=self.risk_free_rate,
            volatility=self.volatility,
            option_type='call'
        )
        
        # All Greeks should be zero for expired options
        for greek_name, greek_value in greeks.items():
            self.assertEqual(greek_value, 0.0, f"{greek_name} should be 0 for expired option")
    
    def test_higher_order_greeks(self):
        """Test higher-order Greeks calculation"""
        greeks = self.calculator.calculate_greeks(
            spot_price=self.spot_price,
            strike_price=self.strike_price,
            time_to_expiry=self.time_to_expiry,
            risk_free_rate=self.risk_free_rate,
            volatility=self.volatility,
            option_type='call'
        )
        
        # Check that higher-order Greeks are present
        self.assertIn('vanna', greeks)
        self.assertIn('vomma', greeks)
        self.assertIn('charm', greeks)
        self.assertIn('speed', greeks)
        
        # Check that they are finite numbers
        for greek_name in ['vanna', 'vomma', 'charm', 'speed']:
            self.assertTrue(np.isfinite(greeks[greek_name]), 
                           f"{greek_name} should be finite")

class TestPortfolioGreeksManager(unittest.TestCase):
    """Test portfolio Greeks management"""
    
    def setUp(self):
        self.greek_limits = {
            'delta': 100.0,
            'gamma': 50.0,
            'theta': -200.0,
            'vega': 500.0,
            'rho': 100.0
        }
        self.manager = PortfolioGreeksManager(self.greek_limits)
    
    def test_greek_limits_check(self):
        """Test Greek limits checking"""
        # Portfolio within limits
        portfolio_greeks = {
            'delta': 50.0,
            'gamma': 25.0,
            'theta': -100.0,
            'vega': 250.0,
            'rho': 50.0
        }
        
        breaches = self.manager.check_greek_limits(portfolio_greeks)
        self.assertEqual(len(breaches), 0, "No breaches should be detected")
        
        # Portfolio exceeding limits
        portfolio_greeks_breach = {
            'delta': 150.0,  # Exceeds limit
            'gamma': 75.0,   # Exceeds limit
            'theta': -100.0,
            'vega': 250.0,
            'rho': 50.0
        }
        
        breaches = self.manager.check_greek_limits(portfolio_greeks_breach)
        self.assertGreater(len(breaches), 0, "Breaches should be detected")
        
        # Check that delta and gamma breaches are reported
        breach_text = ' '.join(breaches)
        self.assertIn('DELTA', breach_text)
        self.assertIn('GAMMA', breach_text)

if __name__ == '__main__':
    unittest.main()