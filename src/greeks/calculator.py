"""
Options Greeks Calculator
Comprehensive Greeks calculation using Black-Scholes and numerical methods
"""

import numpy as np
import logging
from scipy.stats import norm
from typing import Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class GreeksResult:
    """Container for Greeks calculation results"""
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    # Higher order Greeks
    vanna: float = 0.0
    vomma: float = 0.0
    charm: float = 0.0
    speed: float = 0.0

class GreeksCalculator:
    """High-precision Greeks calculator"""
    
    def __init__(self):
        self.calculation_method = "black_scholes"
        
    def calculate_greeks(self, spot_price: float, strike_price: float,
                        time_to_expiry: float, risk_free_rate: float,
                        volatility: float, option_type: str = "call",
                        dividend_yield: float = 0.0) -> Dict[str, float]:
        """
        Calculate all Greeks for an option
        
        Args:
            spot_price: Current price of underlying
            strike_price: Strike price of option
            time_to_expiry: Time to expiry in years
            risk_free_rate: Risk-free interest rate
            volatility: Implied volatility
            option_type: 'call' or 'put'
            dividend_yield: Dividend yield of underlying
            
        Returns:
            Dictionary containing all Greeks
        """
        
        if time_to_expiry <= 0:
            return self._zero_greeks()
        
        # Calculate d1 and d2
        d1, d2 = self._calculate_d1_d2(
            spot_price, strike_price, time_to_expiry,
            risk_free_rate, volatility, dividend_yield
        )
        
        # Calculate Greeks
        greeks = {}
        
        if option_type.lower() == "call":
            greeks = self._calculate_call_greeks(
                spot_price, strike_price, time_to_expiry,
                risk_free_rate, volatility, dividend_yield, d1, d2
            )
        else:
            greeks = self._calculate_put_greeks(
                spot_price, strike_price, time_to_expiry,
                risk_free_rate, volatility, dividend_yield, d1, d2
            )
        
        # Add higher-order Greeks
        greeks.update(self._calculate_higher_order_greeks(
            spot_price, strike_price, time_to_expiry,
            risk_free_rate, volatility, dividend_yield, d1, d2, option_type
        ))
        
        return greeks
    
    def _calculate_d1_d2(self, S: float, K: float, T: float, r: float,
                        sigma: float, q: float = 0.0) -> tuple:
        """Calculate d1 and d2 for Black-Scholes formula"""
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return d1, d2
    
    def _calculate_call_greeks(self, S: float, K: float, T: float, r: float,
                              sigma: float, q: float, d1: float, d2: float) -> Dict[str, float]:
        """Calculate Greeks for call option"""
        
        # Standard normal CDF and PDF
        N_d1 = norm.cdf(d1)
        N_d2 = norm.cdf(d2)
        n_d1 = norm.pdf(d1)
        
        # Delta
        delta = np.exp(-q * T) * N_d1
        
        # Gamma
        gamma = (np.exp(-q * T) * n_d1) / (S * sigma * np.sqrt(T))
        
        # Theta
        theta1 = -(S * n_d1 * sigma * np.exp(-q * T)) / (2 * np.sqrt(T))
        theta2 = -r * K * np.exp(-r * T) * N_d2
        theta3 = q * S * np.exp(-q * T) * N_d1
        theta = (theta1 + theta2 + theta3) / 365  # Per day
        
        # Vega
        vega = S * np.exp(-q * T) * n_d1 * np.sqrt(T) / 100  # Per 1% vol change
        
        # Rho
        rho = K * T * np.exp(-r * T) * N_d2 / 100  # Per 1% rate change
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }
    
    def _calculate_put_greeks(self, S: float, K: float, T: float, r: float,
                             sigma: float, q: float, d1: float, d2: float) -> Dict[str, float]:
        """Calculate Greeks for put option"""
        
        # Standard normal CDF and PDF
        N_minus_d1 = norm.cdf(-d1)
        N_minus_d2 = norm.cdf(-d2)
        n_d1 = norm.pdf(d1)
        
        # Delta
        delta = -np.exp(-q * T) * N_minus_d1
        
        # Gamma (same for call and put)
        gamma = (np.exp(-q * T) * n_d1) / (S * sigma * np.sqrt(T))
        
        # Theta
        theta1 = -(S * n_d1 * sigma * np.exp(-q * T)) / (2 * np.sqrt(T))
        theta2 = r * K * np.exp(-r * T) * N_minus_d2
        theta3 = -q * S * np.exp(-q * T) * N_minus_d1
        theta = (theta1 + theta2 + theta3) / 365  # Per day
        
        # Vega (same for call and put)
        vega = S * np.exp(-q * T) * n_d1 * np.sqrt(T) / 100  # Per 1% vol change
        
        # Rho
        rho = -K * T * np.exp(-r * T) * N_minus_d2 / 100  # Per 1% rate change
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }
    
    def _calculate_higher_order_greeks(self, S: float, K: float, T: float, r: float,
                                      sigma: float, q: float, d1: float, d2: float,
                                      option_type: str) -> Dict[str, float]:
        """Calculate higher-order Greeks"""
        
        n_d1 = norm.pdf(d1)
        
        # Vanna (sensitivity of delta to volatility)
        vanna = -np.exp(-q * T) * n_d1 * d2 / sigma / 100
        
        # Vomma (sensitivity of vega to volatility)
        vomma = S * np.exp(-q * T) * n_d1 * np.sqrt(T) * d1 * d2 / sigma / 100
        
        # Charm (sensitivity of delta to time)
        if option_type.lower() == "call":
            charm1 = q * np.exp(-q * T) * norm.cdf(d1)
            charm2 = np.exp(-q * T) * n_d1 * (2 * (r - q) * T - d2 * sigma * np.sqrt(T))
            charm2 /= (2 * T * sigma * np.sqrt(T))
            charm = charm1 - charm2
        else:
            charm1 = -q * np.exp(-q * T) * norm.cdf(-d1)
            charm2 = np.exp(-q * T) * n_d1 * (2 * (r - q) * T - d2 * sigma * np.sqrt(T))
            charm2 /= (2 * T * sigma * np.sqrt(T))
            charm = charm1 - charm2
        
        charm /= 365  # Per day
        
        # Speed (sensitivity of gamma to underlying price)
        speed = -np.exp(-q * T) * n_d1 * (d1 / (sigma * np.sqrt(T)) + 1) / (S**2 * sigma * np.sqrt(T))
        
        return {
            'vanna': vanna,
            'vomma': vomma,
            'charm': charm,
            'speed': speed
        }
    
    def _zero_greeks(self) -> Dict[str, float]:
        """Return zero Greeks for expired options"""
        return {
            'delta': 0.0,
            'gamma': 0.0,
            'theta': 0.0,
            'vega': 0.0,
            'rho': 0.0,
            'vanna': 0.0,
            'vomma': 0.0,
            'charm': 0.0,
            'speed': 0.0
        }
    
    def calculate_implied_volatility(self, market_price: float, spot_price: float,
                                   strike_price: float, time_to_expiry: float,
                                   risk_free_rate: float, option_type: str = "call",
                                   dividend_yield: float = 0.0) -> float:
        """
        Calculate implied volatility using Newton-Raphson method
        """
        from .black_scholes import BlackScholesCalculator
        
        bs_calc = BlackScholesCalculator()
        
        # Initial guess
        sigma = 0.2
        tolerance = 1e-6
        max_iterations = 100
        
        for i in range(max_iterations):
            # Calculate theoretical price and vega
            theoretical_price = bs_calc.calculate_option_price(
                spot_price, strike_price, time_to_expiry,
                risk_free_rate, sigma, option_type, dividend_yield
            )
            
            greeks = self.calculate_greeks(
                spot_price, strike_price, time_to_expiry,
                risk_free_rate, sigma, option_type, dividend_yield
            )
            
            vega = greeks['vega'] * 100  # Convert back to per unit vol
            
            if abs(vega) < 1e-10:
                break
                
            # Newton-Raphson update
            price_diff = theoretical_price - market_price
            sigma_new = sigma - price_diff / vega
            
            if abs(sigma_new - sigma) < tolerance:
                return max(sigma_new, 0.01)  # Ensure positive vol
            
            sigma = max(sigma_new, 0.01)
        
        logger.warning(f"IV calculation did not converge. Final sigma: {sigma}")
        return sigma

class PortfolioGreeksManager:
    """Manage Greeks at portfolio level"""
    
    def __init__(self, greek_limits: Dict[str, float]):
        self.greek_limits = greek_limits
        self.calculator = GreeksCalculator()
        
    def calculate_portfolio_greeks(self, positions: List) -> Dict[str, float]:
        """Calculate total portfolio Greeks"""
        total_greeks = {
            'delta': 0.0,
            'gamma': 0.0,
            'theta': 0.0,
            'vega': 0.0,
            'rho': 0.0
        }
        
        for position in positions:
            if hasattr(position, 'option_details') and position.option_details:
                greeks = self.calculator.calculate_greeks(
                    spot_price=position.current_price,
                    strike_price=position.option_details.get('strike', 0),
                    time_to_expiry=position.option_details.get('tte', 0),
                    risk_free_rate=position.option_details.get('rate', 0.02),
                    volatility=position.option_details.get('iv', 0.2),
                    option_type=position.option_details.get('type', 'call')
                )
                
                # Weight by position size
                for greek in total_greeks:
                    total_greeks[greek] += greeks[greek] * position.quantity
        
        return total_greeks
    
    def check_greek_limits(self, portfolio_greeks: Dict[str, float]) -> List[str]:
        """Check if any Greek limits are breached"""
        breaches = []
        
        for greek, value in portfolio_greeks.items():
            if greek in self.greek_limits:
                limit = self.greek_limits[greek]
                if abs(value) > abs(limit):
                    breaches.append(f"{greek.upper()}: {value:.2f} exceeds limit {limit:.2f}")
        
        return breaches