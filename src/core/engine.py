"""
EquiFlex Pro - Core Trading Engine
Main orchestrator for the trading system
"""

import asyncio
import logging
from typing import Dict, List, Optional
from datetime import datetime, timezone
import signal
import sys

from .portfolio import Portfolio
from .events import EventBus, MarketDataEvent, SignalEvent, OrderEvent
from ..data.ingestion import DataManager
from ..strategies.base_strategy import StrategyManager
from ..risk.manager import RiskManager
from ..execution.broker_interface import BrokerManager
from ..utils.logging import setup_logging
from ..utils.helpers import load_config

logger = logging.getLogger(__name__)

class TradingEngine:
    """Main trading engine orchestrating all components"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize the trading engine"""
        self.config = load_config(config_path)
        self.running = False
        
        # Setup logging
        setup_logging(self.config['logging'])
        
        # Initialize components
        self.event_bus = EventBus()
        self.portfolio = Portfolio(
            initial_cash=self.config['trading']['cash_allocation']
        )
        self.data_manager = DataManager(self.config['data'])
        self.strategy_manager = StrategyManager(self.config['strategies'])
        self.risk_manager = RiskManager(self.config['risk'])
        self.broker_manager = BrokerManager(self.config['brokers'])
        
        # Setup event handlers
        self._setup_event_handlers()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("Trading engine initialized successfully")
    
    def _setup_event_handlers(self):
        """Setup event handlers for different event types"""
        self.event_bus.subscribe('market_data', self._handle_market_data)
        self.event_bus.subscribe('signal', self._handle_signal)
        self.event_bus.subscribe('order', self._handle_order)
        self.event_bus.subscribe('fill', self._handle_fill)
        self.event_bus.subscribe('risk_breach', self._handle_risk_breach)
    
    async def start(self):
        """Start the trading engine"""
        logger.info("Starting EquiFlex Pro Trading Engine...")
        self.running = True
        
        try:
            # Start all components
            await self.data_manager.start()
            await self.broker_manager.connect()
            await self.strategy_manager.start()
            
            logger.info("All systems online. Beginning trading operations...")
            
            # Main trading loop
            await self._main_loop()
            
        except Exception as e:
            logger.error(f"Error in trading engine: {e}")
            await self.stop()
    
    async def stop(self):
        """Stop the trading engine gracefully"""
        logger.info("Stopping trading engine...")
        self.running = False
        
        # Stop all components
        await self.strategy_manager.stop()
        await self.broker_manager.disconnect()
        await self.data_manager.stop()
        
        logger.info("Trading engine stopped successfully")
    
    async def _main_loop(self):
        """Main trading loop"""
        while self.running:
            try:
                # Process events from the event bus
                await self.event_bus.process_events()
                
                # Update portfolio metrics
                self.portfolio.update_metrics()
                
                # Run risk checks
                await self.risk_manager.check_portfolio(self.portfolio)
                
                # Strategy updates
                await self.strategy_manager.update()
                
                # Small delay to prevent excessive CPU usage
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(1)
    
    async def _handle_market_data(self, event: MarketDataEvent):
        """Handle incoming market data"""
        # Update portfolio positions with new prices
        self.portfolio.update_prices(event.data)
        
        # Send to strategies
        await self.strategy_manager.on_market_data(event)
    
    async def _handle_signal(self, event: SignalEvent):
        """Handle trading signals from strategies"""
        # Risk check the signal
        if await self.risk_manager.check_signal(event, self.portfolio):
            # Convert signal to orders
            orders = await self.strategy_manager.signal_to_orders(event)
            
            # Submit orders
            for order in orders:
                await self.broker_manager.submit_order(order)
    
    async def _handle_order(self, event: OrderEvent):
        """Handle order events"""
        logger.info(f"Order event: {event}")
    
    async def _handle_fill(self, event):
        """Handle order fills"""
        # Update portfolio with fill
        self.portfolio.update_position(event)
        
        # Notify strategies
        await self.strategy_manager.on_fill(event)
    
    async def _handle_risk_breach(self, event):
        """Handle risk limit breaches"""
        logger.warning(f"Risk breach detected: {event}")
        
        # Take corrective action
        await self.strategy_manager.emergency_stop()
        await self.broker_manager.cancel_all_orders()
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        asyncio.create_task(self.stop())

# Main entry point
async def main():
    """Main entry point for the trading engine"""
    engine = TradingEngine()
    await engine.start()

if __name__ == "__main__":
    asyncio.run(main())