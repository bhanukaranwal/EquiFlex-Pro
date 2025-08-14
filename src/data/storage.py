"""
Data Storage and Database Management
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import asyncpg
import json
from dataclasses import asdict

from .ingestion import MarketData, OptionsData

logger = logging.getLogger(__name__)

class DataStorage:
    """Database storage manager for market data and trading records"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.connection_pool = None
        
        # Database configuration
        self.db_config = {
            'host': config.get('host', 'localhost'),
            'port': config.get('port', 5432),
            'database': config.get('database', 'equiflex'),
            'user': config.get('user', 'equiflex_user'),
            'password': config.get('password', 'password'),
            'min_size': 5,
            'max_size': 20
        }
        
    async def start(self):
        """Initialize database connection pool"""
        try:
            self.connection_pool = await asyncpg.create_pool(**self.db_config)
            
            # Create tables if they don't exist
            await self._create_tables()
            
            logger.info("Database connection pool initialized")
            
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
    
    async def stop(self):
        """Close database connection pool"""
        if self.connection_pool:
            await self.connection_pool.close()
            logger.info("Database connection pool closed")
    
    async def _create_tables(self):
        """Create necessary database tables"""
        
        tables_sql = [
            # Market data table
            """
            CREATE TABLE IF NOT EXISTS market_data (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                open_price DECIMAL(10,4),
                high_price DECIMAL(10,4),
                low_price DECIMAL(10,4),
                close_price DECIMAL(10,4),
                volume BIGINT,
                data_type VARCHAR(20) DEFAULT 'daily',
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
            """,
            
            # Options data table
            """
            CREATE TABLE IF NOT EXISTS options_data (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(50) NOT NULL,
                underlying VARCHAR(20) NOT NULL,
                strike_price DECIMAL(10,4) NOT NULL,
                expiry_date DATE NOT NULL,
                option_type VARCHAR(4) NOT NULL,
                bid DECIMAL(10,4),
                ask DECIMAL(10,4),
                last_price DECIMAL(10,4),
                volume INTEGER,
                open_interest INTEGER,
                implied_volatility DECIMAL(8,6),
                delta DECIMAL(8,6),
                gamma DECIMAL(8,6),
                theta DECIMAL(8,6),
                vega DECIMAL(8,6),
                rho DECIMAL(8,6),
                timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
            """,
            
            # Real-time data table
            """
            CREATE TABLE IF NOT EXISTS real_time_data (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                price DECIMAL(10,4) NOT NULL,
                bid DECIMAL(10,4),
                ask DECIMAL(10,4),
                volume INTEGER,
                data_type VARCHAR(20),
                timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
            """,
            
            # Trading orders table
            """
            CREATE TABLE IF NOT EXISTS trading_orders (
                id SERIAL PRIMARY KEY,
                order_id VARCHAR(50) UNIQUE NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                side VARCHAR(10) NOT NULL,
                quantity DECIMAL(15,8) NOT NULL,
                order_type VARCHAR(20) NOT NULL,
                price DECIMAL(10,4),
                stop_price DECIMAL(10,4),
                status VARCHAR(20) NOT NULL,
                filled_quantity DECIMAL(15,8) DEFAULT 0,
                avg_fill_price DECIMAL(10,4),
                broker_order_id VARCHAR(100),
                strategy VARCHAR(50),
                timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
            """,
            
            # Portfolio history table
            """
            CREATE TABLE IF NOT EXISTS portfolio_history (
                id SERIAL PRIMARY KEY,
                total_value DECIMAL(15,2) NOT NULL,
                cash DECIMAL(15,2) NOT NULL,
                positions_value DECIMAL(15,2) NOT NULL,
                total_return DECIMAL(8,6),
                daily_return DECIMAL(8,6),
                max_drawdown DECIMAL(8,6),
                num_positions INTEGER,
                portfolio_greeks JSONB,
                timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
            """,
            
            # Risk metrics table
            """
            CREATE TABLE IF NOT EXISTS risk_metrics (
                id SERIAL PRIMARY KEY,
                var_95 DECIMAL(15,2),
                var_99 DECIMAL(15,2),
                cvar_95 DECIMAL(15,2),
                max_drawdown DECIMAL(8,6),
                sharpe_ratio DECIMAL(8,6),
                leverage_ratio DECIMAL(8,6),
                concentration_risk DECIMAL(8,6),
                risk_breaches JSONB,
                timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
            """,
            
            # Strategy performance table
            """
            CREATE TABLE IF NOT EXISTS strategy_performance (
                id SERIAL PRIMARY KEY,
                strategy_name VARCHAR(50) NOT NULL,
                total_return DECIMAL(8,6),
                sharpe_ratio DECIMAL(8,6),
                max_drawdown DECIMAL(8,6),
                total_trades INTEGER,
                winning_trades INTEGER,
                signals_generated INTEGER,
                allocation DECIMAL(4,3),
                is_running BOOLEAN,
                performance_metrics JSONB,
                timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
            """,
            
            # Fundamental data table
            """
            CREATE TABLE IF NOT EXISTS fundamental_data (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                market_cap BIGINT,
                pe_ratio DECIMAL(8,4),
                pb_ratio DECIMAL(8,4),
                dividend_yield DECIMAL(8,6),
                beta DECIMAL(8,6),
                sector VARCHAR(50),
                industry VARCHAR(100),
                fundamental_metrics JSONB,
                timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
            """
        ]
        
        # Create indexes
        indexes_sql = [
            "CREATE INDEX IF NOT EXISTS idx_market_data_symbol_timestamp ON market_data(symbol, timestamp);",
            "CREATE INDEX IF NOT EXISTS idx_options_data_underlying_expiry ON options_data(underlying, expiry_date);",
            "CREATE INDEX IF NOT EXISTS idx_real_time_data_symbol_timestamp ON real_time_data(symbol, timestamp);",
            "CREATE INDEX IF NOT EXISTS idx_trading_orders_timestamp ON trading_orders(timestamp);",
            "CREATE INDEX IF NOT EXISTS idx_portfolio_history_timestamp ON portfolio_history(timestamp);",
            "CREATE INDEX IF NOT EXISTS idx_strategy_performance_name_timestamp ON strategy_performance(strategy_name, timestamp);"
        ]
        
        async with self.connection_pool.acquire() as connection:
            # Create tables
            for table_sql in tables_sql:
                await connection.execute(table_sql)
            
            # Create indexes
            for index_sql in indexes_sql:
                await connection.execute(index_sql)
        
        logger.info("Database tables and indexes created successfully")
    
    async def store_historical_data(self, symbol: str, data: pd.DataFrame):
        """Store historical market data"""
        if data.empty:
            return
        
        try:
            async with self.connection_pool.acquire() as connection:
                # Prepare data for insertion
                records = []
                for timestamp, row in data.iterrows():
                    records.append((
                        symbol,
                        timestamp,
                        float(row.get('open', 0)),
                        float(row.get('high', 0)),
                        float(row.get('low', 0)),
                        float(row.get('close', 0)),
                        int(row.get('volume', 0)),
                        'daily'
                    ))
                
                # Insert data (on conflict, update)
                await connection.executemany(
                    """
                    INSERT INTO market_data (symbol, timestamp, open_price, high_price, 
                                           low_price, close_price, volume, data_type)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    ON CONFLICT (symbol, timestamp) WHERE data_type = 'daily'
                    DO UPDATE SET 
                        open_price = EXCLUDED.open_price,
                        high_price = EXCLUDED.high_price,
                        low_price = EXCLUDED.low_price,
                        close_price = EXCLUDED.close_price,
                        volume = EXCLUDED.volume
                    """,
                    records
                )
                
            logger.debug(f"Stored {len(records)} historical data points for {symbol}")
            
        except Exception as e:
            logger.error(f"Error storing historical data for {symbol}: {e}")
    
    async def store_real_time_data(self, market_data: MarketData):
        """Store real-time market data"""
        try:
            async with self.connection_pool.acquire() as connection:
                await connection.execute(
                    """
                    INSERT INTO real_time_data (symbol, price, bid, ask, volume, data_type, timestamp)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    """,
                    market_data.symbol,
                    market_data.price,
                    market_data.bid,
                    market_data.ask,
                    market_data.volume,
                    market_data.data_type,
                    market_data.timestamp
                )
                
        except Exception as e:
            logger.error(f"Error storing real-time data: {e}")
    
    async def store_options_data(self, options_data: List[OptionsData]):
        """Store options chain data"""
        if not options_data:
            return
        
        try:
            async with self.connection_pool.acquire() as connection:
                records = []
                for option in options_data:
                    records.append((
                        option.symbol,
                        option.underlying,
                        option.strike,
                        option.expiry.date(),
                        option.option_type,
                        option.bid,
                        option.ask,
                        option.last,
                        option.volume,
                        option.open_interest,
                        option.implied_vol,
                        option.delta,
                        option.gamma,
                        option.theta,
                        option.vega,
                        option.rho,
                        option.timestamp
                    ))
                
                await connection.executemany(
                    """
                    INSERT INTO options_data (symbol, underlying, strike_price, expiry_date,
                                            option_type, bid, ask, last_price, volume,
                                            open_interest, implied_volatility, delta, gamma,
                                            theta, vega, rho, timestamp)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17)
                    ON CONFLICT (symbol, timestamp) 
                    DO UPDATE SET 
                        bid = EXCLUDED.bid,
                        ask = EXCLUDED.ask,
                        last_price = EXCLUDED.last_price,
                        volume = EXCLUDED.volume,
                        open_interest = EXCLUDED.open_interest,
                        implied_volatility = EXCLUDED.implied_volatility,
                        delta = EXCLUDED.delta,
                        gamma = EXCLUDED.gamma,
                        theta = EXCLUDED.theta,
                        vega = EXCLUDED.vega,
                        rho = EXCLUDED.rho
                    """,
                    records
                )
                
            logger.debug(f"Stored {len(records)} options data points")
            
        except Exception as e:
            logger.error(f"Error storing options data: {e}")
    
    async def store_portfolio_snapshot(self, portfolio):
        """Store portfolio snapshot"""
        try:
            async with self.connection_pool.acquire() as connection:
                portfolio_greeks = portfolio.calculate_portfolio_greeks()
                
                await connection.execute(
                    """
                    INSERT INTO portfolio_history (total_value, cash, positions_value,
                                                 total_return, num_positions, portfolio_greeks, timestamp)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    """,
                    portfolio.total_value,
                    portfolio.cash,
                    sum(pos.market_value for pos in portfolio.positions.values()),
                    portfolio.total_return,
                    len(portfolio.positions),
                    json.dumps(portfolio_greeks),
                    datetime.now()
                )
                
        except Exception as e:
            logger.error(f"Error storing portfolio snapshot: {e}")
    
    async def store_order(self, order):
        """Store trading order"""
        try:
            async with self.connection_pool.acquire() as connection:
                await connection.execute(
                    """
                    INSERT INTO trading_orders (order_id, symbol, side, quantity, order_type,
                                              price, stop_price, status, broker_order_id, timestamp)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    ON CONFLICT (order_id)
                    DO UPDATE SET
                        status = EXCLUDED.status,
                        filled_quantity = EXCLUDED.filled_quantity,
                        avg_fill_price = EXCLUDED.avg_fill_price,
                        updated_at = NOW()
                    """,
                    order.id,
                    order.symbol,
                    order.side.value,
                    order.quantity,
                    order.order_type.value,
                    order.price,
                    order.stop_price,
                    order.status.value,
                    order.broker_order_id,
                    order.timestamp
                )
                
        except Exception as e:
            logger.error(f"Error storing order: {e}")
    
    async def get_historical_data(self, symbol: str, start_date: datetime, 
                                 end_date: datetime) -> pd.DataFrame:
        """Retrieve historical data"""
        try:
            async with self.connection_pool.acquire() as connection:
                rows = await connection.fetch(
                    """
                    SELECT timestamp, open_price, high_price, low_price, close_price, volume
                    FROM market_data
                    WHERE symbol = $1 AND timestamp BETWEEN $2 AND $3
                    ORDER BY timestamp
                    """,
                    symbol, start_date, end_date
                )
                
                if not rows:
                    return pd.DataFrame()
                
                # Convert to DataFrame
                data = []
                for row in rows:
                    data.append({
                        'timestamp': row['timestamp'],
                        'open': float(row['open_price']),
                        'high': float(row['high_price']),
                        'low': float(row['low_price']),
                        'close': float(row['close_price']),
                        'volume': int(row['volume'])
                    })
                
                df = pd.DataFrame(data)
                df.set_index('timestamp', inplace=True)
                return df
                
        except Exception as e:
            logger.error(f"Error retrieving historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    async def store_fundamental_data(self, symbol: str, fundamental_data: Dict[str, Any]):
        """Store fundamental data"""
        try:
            async with self.connection_pool.acquire() as connection:
                await connection.execute(
                    """
                    INSERT INTO fundamental_data (symbol, market_cap, pe_ratio, pb_ratio,
                                                dividend_yield, beta, sector, industry,
                                                fundamental_metrics, timestamp)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    """,
                    symbol,
                    fundamental_data.get('market_cap'),
                    fundamental_data.get('pe_ratio'),
                    fundamental_data.get('pb_ratio'),
                    fundamental_data.get('dividend_yield'),
                    fundamental_data.get('beta'),
                    fundamental_data.get('sector'),
                    fundamental_data.get('industry'),
                    json.dumps(fundamental_data),
                    fundamental_data.get('timestamp', datetime.now())
                )
                
        except Exception as e:
            logger.error(f"Error storing fundamental data for {symbol}: {e}")
    
    async def store_economic_data(self, economic_data: Dict[str, Any]):
        """Store economic indicators data"""
        # Implementation for economic data storage
        pass