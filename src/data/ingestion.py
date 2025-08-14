"""
Market Data Ingestion System
Real-time and historical data management
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
import aiohttp
import websockets
import json
from dataclasses import dataclass

from ..core.events import MarketDataEvent, EventBus
from .providers import YFinanceProvider, AlphaVantageProvider, PolygonProvider
from .storage import DataStorage

logger = logging.getLogger(__name__)

@dataclass
class MarketData:
    """Market data container"""
    symbol: str
    price: float
    bid: float
    ask: float
    volume: int
    timestamp: datetime
    data_type: str  # 'quote', 'trade', 'bar'

@dataclass
class OptionsData:
    """Options chain data container"""
    symbol: str
    underlying: str
    strike: float
    expiry: datetime
    option_type: str  # 'call', 'put'
    bid: float
    ask: float
    last: float
    volume: int
    open_interest: int
    implied_vol: float
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    timestamp: datetime

class DataManager:
    """Central data management system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.providers = {}
        self.subscriptions = set()
        self.event_bus = EventBus()
        
        # Data storage
        self.storage = DataStorage(config.get('storage', {}))
        
        # Real-time data feeds
        self.ws_connections = {}
        self.is_running = False
        
        # Data cache
        self.price_cache = {}
        self.options_cache = {}
        
        # Initialize providers
        self._initialize_providers()
        
        logger.info("Data Manager initialized")
    
    def _initialize_providers(self):
        """Initialize data providers"""
        provider_configs = self.config.get('providers', [])
        
        for provider_config in provider_configs:
            if not provider_config.get('enabled', False):
                continue
                
            name = provider_config['name']
            config = provider_config.get('config', {})
            
            if name == 'yfinance':
                self.providers[name] = YFinanceProvider(config)
            elif name == 'alpha_vantage':
                self.providers[name] = AlphaVantageProvider(config)
            elif name == 'polygon':
                self.providers[name] = PolygonProvider(config)
            
            logger.info(f"Initialized data provider: {name}")
    
    async def start(self):
        """Start data ingestion"""
        self.is_running = True
        
        # Start storage
        await self.storage.start()
        
        # Start real-time feeds
        asyncio.create_task(self._run_real_time_feeds())
        
        # Start periodic data updates
        asyncio.create_task(self._run_periodic_updates())
        
        logger.info("Data Manager started")
    
    async def stop(self):
        """Stop data ingestion"""
        self.is_running = False
        
        # Close WebSocket connections
        for ws in self.ws_connections.values():
            await ws.close()
        
        # Stop storage
        await self.storage.stop()
        
        logger.info("Data Manager stopped")
    
    async def subscribe_to_symbols(self, symbols: List[str], data_types: List[str] = None):
        """Subscribe to real-time data for symbols"""
        if data_types is None:
            data_types = ['quotes', 'trades']
        
        for symbol in symbols:
            self.subscriptions.add(symbol)
            
            # Subscribe with each provider
            for provider in self.providers.values():
                if hasattr(provider, 'subscribe_real_time'):
                    await provider.subscribe_real_time(symbol, data_types)
        
        logger.info(f"Subscribed to {len(symbols)} symbols")
    
    async def get_historical_data(self, symbol: str, period: str = "1y", 
                                 interval: str = "1d") -> pd.DataFrame:
        """Get historical price data"""
        
        # Try cache first
        cache_key = f"{symbol}_{period}_{interval}"
        if cache_key in self.price_cache:
            cached_data, timestamp = self.price_cache[cache_key]
            if datetime.now() - timestamp < timedelta(hours=1):
                return cached_data
        
        # Get from provider
        for provider in self.providers.values():
            try:
                data = await provider.get_historical_data(symbol, period, interval)
                if data is not None and not data.empty:
                    # Cache the data
                    self.price_cache[cache_key] = (data, datetime.now())
                    
                    # Store in database
                    await self.storage.store_historical_data(symbol, data)
                    
                    return data
            except Exception as e:
                logger.warning(f"Error getting data from provider: {e}")
                continue
        
        logger.warning(f"Could not get historical data for {symbol}")
        return pd.DataFrame()
    
    async def get_options_chain(self, underlying: str, expiry: str = None) -> List[OptionsData]:
        """Get options chain data"""
        
        cache_key = f"{underlying}_options_{expiry}"
        if cache_key in self.options_cache:
            cached_data, timestamp = self.options_cache[cache_key]
            if datetime.now() - timestamp < timedelta(minutes=5):
                return cached_data
        
        # Get from provider
        for provider in self.providers.values():
            if hasattr(provider, 'get_options_chain'):
                try:
                    options_data = await provider.get_options_chain(underlying, expiry)
                    if options_data:
                        # Cache the data
                        self.options_cache[cache_key] = (options_data, datetime.now())
                        
                        # Store in database
                        await self.storage.store_options_data(options_data)
                        
                        return options_data
                except Exception as e:
                    logger.warning(f"Error getting options data: {e}")
                    continue
        
        logger.warning(f"Could not get options chain for {underlying}")
        return []
    
    async def _run_real_time_feeds(self):
        """Run real-time data feeds"""
        while self.is_running:
            try:
                # Process real-time data from all providers
                for provider_name, provider in self.providers.items():
                    if hasattr(provider, 'get_real_time_data'):
                        data = await provider.get_real_time_data()
                        if data:
                            await self._process_real_time_data(data)
                
                await asyncio.sleep(0.1)  # 100ms polling
                
            except Exception as e:
                logger.error(f"Error in real-time feeds: {e}")
                await asyncio.sleep(1)
    
    async def _process_real_time_data(self, data: List[MarketData]):
        """Process incoming real-time data"""
        price_updates = {}
        
        for market_data in data:
            symbol = market_data.symbol
            
            # Update price cache
            self.price_cache[f"{symbol}_current"] = (market_data.price, datetime.now())
            
            # Prepare for event
            price_updates[symbol] = market_data.price
            
            # Store in database
            await self.storage.store_real_time_data(market_data)
        
        # Publish market data event
        if price_updates:
            event = MarketDataEvent(
                data=price_updates,
                timestamp=datetime.now()
            )
            await self.event_bus.publish('market_data', event)
    
    async def _run_periodic_updates(self):
        """Run periodic data updates"""
        while self.is_running:
            try:
                # Update fundamental data
                await self._update_fundamental_data()
                
                # Update options data
                await self._update_options_data()
                
                # Update economic indicators
                await self._update_economic_data()
                
                # Wait 1 hour
                await asyncio.sleep(3600)
                
            except Exception as e:
                logger.error(f"Error in periodic updates: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    async def _update_fundamental_data(self):
        """Update fundamental data for subscribed symbols"""
        for symbol in self.subscriptions:
            for provider in self.providers.values():
                if hasattr(provider, 'get_fundamental_data'):
                    try:
                        fundamental_data = await provider.get_fundamental_data(symbol)
                        if fundamental_data:
                            await self.storage.store_fundamental_data(symbol, fundamental_data)
                    except Exception as e:
                        logger.warning(f"Error updating fundamental data for {symbol}: {e}")
    
    async def _update_options_data(self):
        """Update options data for major ETFs"""
        major_etfs = ['SPY', 'QQQ', 'IWM', 'GLD', 'TLT']
        
        for etf in major_etfs:
            try:
                options_data = await self.get_options_chain(etf)
                logger.info(f"Updated options chain for {etf}: {len(options_data)} contracts")
            except Exception as e:
                logger.warning(f"Error updating options data for {etf}: {e}")
    
    async def _update_economic_data(self):
        """Update economic indicators"""
        for provider in self.providers.values():
            if hasattr(provider, 'get_economic_data'):
                try:
                    economic_data = await provider.get_economic_data()
                    if economic_data:
                        await self.storage.store_economic_data(economic_data)
                except Exception as e:
                    logger.warning(f"Error updating economic data: {e}")
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price from cache"""
        cache_key = f"{symbol}_current"
        if cache_key in self.price_cache:
            price, timestamp = self.price_cache[cache_key]
            if datetime.now() - timestamp < timedelta(minutes=5):
                return price
        return None
    
    async def get_market_status(self) -> Dict[str, Any]:
        """Get market status information"""
        for provider in self.providers.values():
            if hasattr(provider, 'get_market_status'):
                try:
                    return await provider.get_market_status()
                except Exception as e:
                    logger.warning(f"Error getting market status: {e}")
        
        return {"status": "unknown", "next_open": None, "next_close": None}