"""
Data Provider Implementations
"""

import asyncio
import aiohttp
import pandas as pd
import yfinance as yf
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging

from .ingestion import MarketData, OptionsData

logger = logging.getLogger(__name__)

class YFinanceProvider:
    """Yahoo Finance data provider"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.timeout = config.get('timeout', 30)
        self.session = None
    
    async def get_historical_data(self, symbol: str, period: str = "1y", 
                                 interval: str = "1d") -> pd.DataFrame:
        """Get historical data from Yahoo Finance"""
        try:
            # Use yfinance library
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                logger.warning(f"No data found for {symbol}")
                return pd.DataFrame()
            
            # Standardize column names
            data = data.rename(columns={
                'Open': 'open',
                'High': 'high', 
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            return data
            
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    async def get_options_chain(self, underlying: str, expiry: str = None) -> List[OptionsData]:
        """Get options chain from Yahoo Finance"""
        try:
            ticker = yf.Ticker(underlying)
            
            # Get available expiration dates
            expirations = ticker.options
            if not expirations:
                return []
            
            # Use specified expiry or next available
            target_expiry = expiry if expiry else expirations[0]
            if target_expiry not in expirations:
                target_expiry = expirations[0]
            
            # Get options chain
            option_chain = ticker.option_chain(target_expiry)
            
            options_data = []
            
            # Process calls
            for _, row in option_chain.calls.iterrows():
                options_data.append(OptionsData(
                    symbol=f"{underlying}_{target_expiry}_C_{row['strike']}",
                    underlying=underlying,
                    strike=row['strike'],
                    expiry=datetime.strptime(target_expiry, '%Y-%m-%d'),
                    option_type='call',
                    bid=row.get('bid', 0.0),
                    ask=row.get('ask', 0.0),
                    last=row.get('lastPrice', 0.0),
                    volume=row.get('volume', 0),
                    open_interest=row.get('openInterest', 0),
                    implied_vol=row.get('impliedVolatility', 0.0),
                    delta=0.0,  # Would calculate separately
                    gamma=0.0,
                    theta=0.0,
                    vega=0.0,
                    rho=0.0,
                    timestamp=datetime.now()
                ))
            
            # Process puts
            for _, row in option_chain.puts.iterrows():
                options_data.append(OptionsData(
                    symbol=f"{underlying}_{target_expiry}_P_{row['strike']}",
                    underlying=underlying,
                    strike=row['strike'],
                    expiry=datetime.strptime(target_expiry, '%Y-%m-%d'),
                    option_type='put',
                    bid=row.get('bid', 0.0),
                    ask=row.get('ask', 0.0),
                    last=row.get('lastPrice', 0.0),
                    volume=row.get('volume', 0),
                    open_interest=row.get('openInterest', 0),
                    implied_vol=row.get('impliedVolatility', 0.0),
                    delta=0.0,
                    gamma=0.0,
                    theta=0.0,
                    vega=0.0,
                    rho=0.0,
                    timestamp=datetime.now()
                ))
            
            return options_data
            
        except Exception as e:
            logger.error(f"Error getting options chain for {underlying}: {e}")
            return []
    
    async def get_fundamental_data(self, symbol: str) -> Dict[str, Any]:
        """Get fundamental data"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'symbol': symbol,
                'market_cap': info.get('marketCap'),
                'pe_ratio': info.get('trailingPE'),
                'pb_ratio': info.get('priceToBook'),
                'dividend_yield': info.get('dividendYield'),
                'beta': info.get('beta'),
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh'),
                'fifty_two_week_low': info.get('fiftyTwoWeekLow'),
                'avg_volume': info.get('averageVolume'),
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error getting fundamental data for {symbol}: {e}")
            return {}
    
    async def get_market_status(self) -> Dict[str, Any]:
        """Get market status"""
        try:
            # Use SPY as proxy for market status
            ticker = yf.Ticker("SPY")
            info = ticker.info
            
            return {
                'status': 'open' if info.get('regularMarketPrice') else 'closed',
                'last_price': info.get('regularMarketPrice'),
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error getting market status: {e}")
            return {'status': 'unknown'}

class AlphaVantageProvider:
    """Alpha Vantage data provider"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_key = config.get('api_key')
        self.base_url = "https://www.alphavantage.co/query"
        self.session = None
    
    async def _make_request(self, params: Dict[str, str]) -> Dict[str, Any]:
        """Make API request to Alpha Vantage"""
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        params['apikey'] = self.api_key
        
        try:
            async with self.session.get(self.base_url, params=params) as response:
                data = await response.json()
                return data
        except Exception as e:
            logger.error(f"Alpha Vantage API error: {e}")
            return {}
    
    async def get_historical_data(self, symbol: str, period: str = "1y", 
                                 interval: str = "1d") -> pd.DataFrame:
        """Get historical data from Alpha Vantage"""
        try:
            params = {
                'function': 'TIME_SERIES_DAILY_ADJUSTED',
                'symbol': symbol,
                'outputsize': 'full'
            }
            
            data = await self._make_request(params)
            
            if 'Time Series (Daily)' not in data:
                logger.warning(f"No data found for {symbol}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            time_series = data['Time Series (Daily)']
            df = pd.DataFrame.from_dict(time_series, orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            # Rename columns
            df.columns = ['open', 'high', 'low', 'close', 'adjusted_close', 'volume', 'dividend', 'split']
            df = df.astype(float)
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    async def get_economic_data(self) -> Dict[str, Any]:
        """Get economic indicators"""
        try:
            # Get various economic indicators
            indicators = {}
            
            # GDP
            params = {'function': 'REAL_GDP', 'interval': 'quarterly'}
            gdp_data = await self._make_request(params)
            if 'data' in gdp_data:
                indicators['gdp'] = gdp_data['data'][:4]  # Last 4 quarters
            
            # Unemployment
            params = {'function': 'UNEMPLOYMENT'}
            unemployment_data = await self._make_request(params)
            if 'data' in unemployment_data:
                indicators['unemployment'] = unemployment_data['data'][:12]  # Last 12 months
            
            # Inflation
            params = {'function': 'INFLATION'}
            inflation_data = await self._make_request(params)
            if 'data' in inflation_data:
                indicators['inflation'] = inflation_data['data'][:12]  # Last 12 months
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error getting economic data: {e}")
            return {}

class PolygonProvider:
    """Polygon.io data provider for real-time data"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_key = config.get('api_key')
        self.base_url = "https://api.polygon.io"
        self.ws_url = "wss://socket.polygon.io"
        self.session = None
        self.ws_connection = None
    
    async def subscribe_real_time(self, symbol: str, data_types: List[str]):
        """Subscribe to real-time data"""
        if not self.ws_connection:
            await self._connect_websocket()
        
        # Subscribe to trades and quotes
        subscribe_msg = {
            "action": "subscribe",
            "params": f"T.{symbol},Q.{symbol}"  # Trades and Quotes
        }
        
        await self.ws_connection.send(json.dumps(subscribe_msg))
    
    async def _connect_websocket(self):
        """Connect to Polygon WebSocket"""
        try:
            import websockets
            
            self.ws_connection = await websockets.connect(
                f"{self.ws_url}/stocks?apikey={self.api_key}"
            )
            
            # Authenticate
            auth_msg = {"action": "auth", "params": self.api_key}
            await self.ws_connection.send(json.dumps(auth_msg))
            
            logger.info("Connected to Polygon WebSocket")
            
        except Exception as e:
            logger.error(f"Error connecting to Polygon WebSocket: {e}")
    
    async def get_real_time_data(self) -> List[MarketData]:
        """Get real-time data from WebSocket"""
        if not self.ws_connection:
            return []
        
        try:
            # Check for messages
            message = await asyncio.wait_for(
                self.ws_connection.recv(), timeout=0.1
            )
            
            data = json.loads(message)
            market_data_list = []
            
            for item in data:
                if item.get('ev') == 'T':  # Trade
                    market_data_list.append(MarketData(
                        symbol=item.get('sym'),
                        price=item.get('p'),
                        bid=0.0,
                        ask=0.0,
                        volume=item.get('s', 0),
                        timestamp=datetime.fromtimestamp(item.get('t', 0) / 1000),
                        data_type='trade'
                    ))
                elif item.get('ev') == 'Q':  # Quote
                    market_data_list.append(MarketData(
                        symbol=item.get('sym'),
                        price=(item.get('bp', 0) + item.get('ap', 0)) / 2,
                        bid=item.get('bp', 0),
                        ask=item.get('ap', 0),
                        volume=0,
                        timestamp=datetime.fromtimestamp(item.get('t', 0) / 1000),
                        data_type='quote'
                    ))
            
            return market_data_list
            
        except asyncio.TimeoutError:
            return []
        except Exception as e:
            logger.error(f"Error getting real-time data: {e}")
            return []