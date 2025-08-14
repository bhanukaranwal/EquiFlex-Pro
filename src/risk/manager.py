"""
Comprehensive Risk Management System
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass

from ..greeks.calculator import PortfolioGreeksManager
from ..utils.helpers import load_config

logger = logging.getLogger(__name__)

@dataclass
class RiskMetrics:
    """Container for risk metrics"""
    var_95: float
    var_99: float
    cvar_95: float
    max_drawdown: float
    sharpe_ratio: float
    portfolio_beta: float
    concentration_risk: float
    leverage_ratio: float

@dataclass
class RiskBreach:
    """Risk limit breach notification"""
    risk_type: str
    current_value: float
    limit_value: float
    severity: str  # 'WARNING', 'CRITICAL'
    timestamp: datetime
    recommended_action: str

class RiskManager:
    """Comprehensive risk management system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Risk limits
        self.max_drawdown = config.get('max_drawdown', 0.15)
        self.var_confidence = config.get('var_confidence', 0.95)
        self.max_position_size = config.get('max_position_size', 0.05)
        self.max_sector_exposure = config.get('max_sector_exposure', 0.3)
        self.max_leverage = config.get('max_leverage', 2.0)
        
        # Greeks limits
        greek_limits = load_config("configs/config.yaml")['trading']['greek_limits']
        self.portfolio_greeks_manager = PortfolioGreeksManager(greek_limits)
        
        # Historical data for risk calculations
        self.price_history = {}
        self.return_history = {}
        self.portfolio_values = []
        
        # Risk monitoring
        self.risk_breaches = []
        self.last_risk_check = datetime.now()
        
        logger.info("Risk Manager initialized")
    
    async def check_portfolio(self, portfolio) -> List[RiskBreach]:
        """Comprehensive portfolio risk check"""
        breaches = []
        
        # Calculate current risk metrics
        risk_metrics = self._calculate_risk_metrics(portfolio)
        
        # Check drawdown limits
        if risk_metrics.max_drawdown > self.max_drawdown:
            breaches.append(RiskBreach(
                risk_type='MAX_DRAWDOWN',
                current_value=risk_metrics.max_drawdown,
                limit_value=self.max_drawdown,
                severity='CRITICAL',
                timestamp=datetime.now(),
                recommended_action='Reduce position sizes or hedge portfolio'
            ))
        
        # Check VaR limits
        var_limit = portfolio.total_value * 0.05  # 5% of portfolio
        if risk_metrics.var_95 > var_limit:
            breaches.append(RiskBreach(
                risk_type='VAR_95',
                current_value=risk_metrics.var_95,
                limit_value=var_limit,
                severity='WARNING',
                timestamp=datetime.now(),
                recommended_action='Review position sizing and correlations'
            ))
        
        # Check position concentration
        concentration_breaches = self._check_position_concentration(portfolio)
        breaches.extend(concentration_breaches)
        
        # Check Greek limits
        greek_breaches = self._check_greek_limits(portfolio)
        breaches.extend(greek_breaches)
        
        # Check leverage
        if risk_metrics.leverage_ratio > self.max_leverage:
            breaches.append(RiskBreach(
                risk_type='LEVERAGE',
                current_value=risk_metrics.leverage_ratio,
                limit_value=self.max_leverage,
                severity='CRITICAL',
                timestamp=datetime.now(),
                recommended_action='Reduce leveraged positions immediately'
            ))
        
        # Log breaches
        for breach in breaches:
            logger.warning(f"Risk breach: {breach.risk_type} - {breach.current_value:.4f} > {breach.limit_value:.4f}")
        
        self.risk_breaches.extend(breaches)
        self.last_risk_check = datetime.now()
        
        return breaches
    
    async def check_signal(self, signal_event, portfolio) -> bool:
        """Check if a trading signal is within risk limits"""
        symbol = signal_event.symbol
        
        # Check position size limits
        current_position = portfolio.positions.get(symbol, None)
        if current_position:
            current_exposure = abs(current_position.market_value) / portfolio.total_value
            if current_exposure > self.max_position_size:
                logger.warning(f"Signal rejected: Position size limit exceeded for {symbol}")
                return False
        
        # Check if adding position would exceed limits
        signal_size = signal_event.strength * signal_event.confidence * self.max_position_size
        if signal_size > self.max_position_size:
            logger.warning(f"Signal rejected: Signal would exceed position size limit for {symbol}")
            return False
        
        # Check correlation with existing positions
        if not self._check_correlation_limits(symbol, portfolio):
            logger.warning(f"Signal rejected: Correlation limits exceeded for {symbol}")
            return False
        
        return True
    
    def _calculate_risk_metrics(self, portfolio) -> RiskMetrics:
        """Calculate comprehensive risk metrics"""
        
        # Portfolio returns for VaR calculation
        if len(self.portfolio_values) > 30:
            returns = np.diff(self.portfolio_values[-252:]) / self.portfolio_values[-252:-1]
            
            # Value at Risk
            var_95 = np.percentile(returns, 5) * portfolio.total_value
            var_99 = np.percentile(returns, 1) * portfolio.total_value
            
            # Conditional VaR (Expected Shortfall)
            cvar_95 = np.mean(returns[returns <= np.percentile(returns, 5)]) * portfolio.total_value
            
            # Sharpe ratio
            sharpe = np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252)
        else:
            var_95 = var_99 = cvar_95 = sharpe = 0.0
        
        # Maximum drawdown
        max_dd = portfolio.max_drawdown
        
        # Portfolio beta (simplified)
        portfolio_beta = 1.0  # Would calculate against market benchmark
        
        # Concentration risk
        position_weights = [pos.market_value / portfolio.total_value 
                          for pos in portfolio.positions.values()]
        concentration_risk = max(position_weights) if position_weights else 0.0
        
        # Leverage ratio
        gross_exposure = sum(abs(pos.market_value) for pos in portfolio.positions.values())
        leverage_ratio = gross_exposure / portfolio.total_value if portfolio.total_value > 0 else 0.0
        
        return RiskMetrics(
            var_95=abs(var_95),
            var_99=abs(var_99),
            cvar_95=abs(cvar_95),
            max_drawdown=max_dd,
            sharpe_ratio=sharpe,
            portfolio_beta=portfolio_beta,
            concentration_risk=concentration_risk,
            leverage_ratio=leverage_ratio
        )
    
    def _check_position_concentration(self, portfolio) -> List[RiskBreach]:
        """Check position concentration limits"""
        breaches = []
        
        for symbol, position in portfolio.positions.items():
            position_weight = abs(position.market_value) / portfolio.total_value
            
            if position_weight > self.max_position_size:
                breaches.append(RiskBreach(
                    risk_type='POSITION_CONCENTRATION',
                    current_value=position_weight,
                    limit_value=self.max_position_size,
                    severity='WARNING',
                    timestamp=datetime.now(),
                    recommended_action=f'Reduce position size in {symbol}'
                ))
        
        return breaches
    
    def _check_greek_limits(self, portfolio) -> List[RiskBreach]:
        """Check Greek exposure limits"""
        breaches = []
        
        # Calculate portfolio Greeks
        portfolio_greeks = portfolio.calculate_portfolio_greeks()
        
        # Check against limits
        greek_breaches = self.portfolio_greeks_manager.check_greek_limits(portfolio_greeks)
        
        for breach_msg in greek_breaches:
            breaches.append(RiskBreach(
                risk_type='GREEK_EXPOSURE',
                current_value=0.0,  # Parsed from breach_msg
                limit_value=0.0,    # Parsed from breach_msg
                severity='WARNING',
                timestamp=datetime.now(),
                recommended_action=f'Hedge Greek exposure: {breach_msg}'
            ))
        
        return breaches
    
    def _check_correlation_limits(self, symbol: str, portfolio) -> bool:
        """Check correlation limits with existing positions"""
        # Simplified correlation check
        # In production, would use correlation matrix
        
        if len(portfolio.positions) < 5:
            return True
        
        # Count positions in same sector (simplified)
        sector_count = len([pos for pos in portfolio.positions.values() 
                          if self._get_sector(pos.symbol) == self._get_sector(symbol)])
        
        max_sector_positions = len(portfolio.positions) * self.max_sector_exposure
        
        return sector_count < max_sector_positions
    
    def _get_sector(self, symbol: str) -> str:
        """Get sector for symbol (simplified mapping)"""
        tech_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA']
        finance_stocks = ['JPM', 'V', 'MA']
        
        if symbol in tech_stocks:
            return 'TECHNOLOGY'
        elif symbol in finance_stocks:
            return 'FINANCIALS'
        else:
            return 'OTHER'
    
    async def calculate_stress_scenarios(self, portfolio) -> Dict[str, float]:
        """Calculate portfolio performance under stress scenarios"""
        scenarios = {}
        
        # Market crash scenario (-20%)
        scenarios['market_crash'] = self._scenario_analysis(portfolio, market_shock=-0.20)
        
        # Volatility spike scenario (+50% vol)
        scenarios['vol_spike'] = self._scenario_analysis(portfolio, vol_shock=0.50)
        
        # Interest rate shock (+200bps)
        scenarios['rate_shock'] = self._scenario_analysis(portfolio, rate_shock=0.02)
        
        # Sector rotation scenario
        scenarios['sector_rotation'] = self._scenario_analysis(portfolio, sector_rotation=True)
        
        return scenarios
    
    def _scenario_analysis(self, portfolio, market_shock: float = 0.0, 
                          vol_shock: float = 0.0, rate_shock: float = 0.0,
                          sector_rotation: bool = False) -> float:
        """Analyze portfolio under specific scenario"""
        
        total_scenario_pnl = 0.0
        
        for position in portfolio.positions.values():
            if position.position_type == 'equity':
                # Apply market shock
                position_pnl = position.market_value * market_shock
                
                # Apply sector-specific shocks if sector rotation
                if sector_rotation:
                    sector = self._get_sector(position.symbol)
                    if sector == 'TECHNOLOGY':
                        position_pnl *= 1.5  # Tech hit harder
                
            elif position.position_type == 'option':
                # Calculate option PnL under scenario
                # This would use Greeks to estimate PnL changes
                greeks = position.greeks
                
                # Delta impact from price change
                delta_pnl = greeks.get('delta', 0) * position.quantity * market_shock * 100
                
                # Vega impact from vol change
                vega_pnl = greeks.get('vega', 0) * position.quantity * vol_shock
                
                # Rho impact from rate change
                rho_pnl = greeks.get('rho', 0) * position.quantity * rate_shock
                
                position_pnl = delta_pnl + vega_pnl + rho_pnl
            
            total_scenario_pnl += position_pnl
        
        return total_scenario_pnl / portfolio.total_value  # Return as percentage
    
    def update_portfolio_history(self, portfolio_value: float):
        """Update portfolio value history for risk calculations"""
        self.portfolio_values.append(portfolio_value)
        
        # Keep only last 2 years of data
        if len(self.portfolio_values) > 504:  # 2 years of daily data
            self.portfolio_values = self.portfolio_values[-504:]
    
    def get_risk_dashboard(self, portfolio) -> Dict[str, Any]:
        """Get comprehensive risk dashboard data"""
        risk_metrics = self._calculate_risk_metrics(portfolio)
        portfolio_greeks = portfolio.calculate_portfolio_greeks()
        
        return {
            'risk_metrics': {
                'var_95': risk_metrics.var_95,
                'var_99': risk_metrics.var_99,
                'max_drawdown': risk_metrics.max_drawdown,
                'sharpe_ratio': risk_metrics.sharpe_ratio,
                'leverage_ratio': risk_metrics.leverage_ratio
            },
            'greek_exposures': portfolio_greeks,
            'position_concentration': {
                symbol: pos.market_value / portfolio.total_value
                for symbol, pos in portfolio.positions.items()
            },
            'recent_breaches': self.risk_breaches[-10:],  # Last 10 breaches
            'last_update': self.last_risk_check
        }