"""
Alpaca Markets Broker Adapter
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
import alpaca_trade_api as tradeapi
from datetime import datetime

from .broker_interface import BrokerInterface, Order, OrderStatus, OrderSide, OrderType

logger = logging.getLogger(__name__)

class AlpacaAdapter(BrokerInterface):
    """Alpaca Markets broker adapter"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_key = config.get('api_key')
        self.secret_key = config.get('secret_key')
        self.base_url = config.get('base_url', 'https://paper-api.alpaca.markets')
        
        self.api = None
        self.connected = False
        
    async def connect(self) -> bool:
        """Connect to Alpaca"""
        try:
            self.api = tradeapi.REST(
                key_id=self.api_key,
                secret_key=self.secret_key,
                base_url=self.base_url,
                api_version='v2'
            )
            
            # Test connection
            account = self.api.get_account()
            
            self.connected = True
            logger.info(f"Connected to Alpaca - Account: {account.account_number}")
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to Alpaca: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from Alpaca"""
        self.connected = False
        self.api = None
        logger.info("Disconnected from Alpaca")
    
    async def submit_order(self, order: Order) -> str:
        """Submit order to Alpaca"""
        if not self.connected:
            raise Exception("Not connected to Alpaca")
        
        try:
            # Convert order to Alpaca format
            alpaca_order = {
                'symbol': order.symbol,
                'qty': order.quantity,
                'side': self._convert_side(order.side),
                'type': self._convert_order_type(order.order_type),
                'time_in_force': order.time_in_force.upper()
            }
            
            # Add price for limit orders
            if order.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
                alpaca_order['limit_price'] = order.price
            
            # Add stop price for stop orders
            if order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT]:
                alpaca_order['stop_price'] = order.stop_price
            
            # Submit order
            submitted_order = self.api.submit_order(**alpaca_order)
            
            return submitted_order.id
            
        except Exception as e:
            logger.error(f"Error submitting order to Alpaca: {e}")
            raise
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order in Alpaca"""
        if not self.connected:
            return False
        
        try:
            self.api.cancel_order(order_id)
            return True
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False
    
    async def get_order_status(self, order_id: str) -> OrderStatus:
        """Get order status from Alpaca"""
        if not self.connected:
            return OrderStatus.REJECTED
        
        try:
            order = self.api.get_order(order_id)
            return self._convert_status(order.status)
        except Exception as e:
            logger.error(f"Error getting order status {order_id}: {e}")
            return OrderStatus.REJECTED
    
    async def get_positions(self) -> Dict[str, float]:
        """Get current positions from Alpaca"""
        if not self.connected:
            return {}
        
        try:
            positions = self.api.list_positions()
            
            position_dict = {}
            for position in positions:
                position_dict[position.symbol] = float(position.qty)
            
            return position_dict
            
        except Exception as e:
            logger.error(f"Error getting positions from Alpaca: {e}")
            return {}
    
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information from Alpaca"""
        if not self.connected:
            return {}
        
        try:
            account = self.api.get_account()
            
            return {
                'account_number': account.account_number,
                'total_value': float(account.portfolio_value),
                'cash': float(account.cash),
                'buying_power': float(account.buying_power),
                'day_trade_buying_power': float(account.daytrading_buying_power),
                'equity': float(account.equity),
                'last_equity': float(account.last_equity),
                'multiplier': float(account.multiplier),
                'currency': account.currency,
                'status': account.status,
                'pattern_day_trader': account.pattern_day_trader,
                'trade_suspended_by_user': account.trade_suspended_by_user,
                'trading_blocked': account.trading_blocked,
                'transfers_blocked': account.transfers_blocked,
                'account_blocked': account.account_blocked
            }
            
        except Exception as e:
            logger.error(f"Error getting account info from Alpaca: {e}")
            return {}
    
    def _convert_side(self, side: OrderSide) -> str:
        """Convert OrderSide to Alpaca format"""
        mapping = {
            OrderSide.BUY: 'buy',
            OrderSide.SELL: 'sell'
        }
        return mapping.get(side, 'buy')
    
    def _convert_order_type(self, order_type: OrderType) -> str:
        """Convert OrderType to Alpaca format"""
        mapping = {
            OrderType.MARKET: 'market',
            OrderType.LIMIT: 'limit',
            OrderType.STOP: 'stop',
            OrderType.STOP_LIMIT: 'stop_limit',
            OrderType.TRAILING_STOP: 'trailing_stop'
        }
        return mapping.get(order_type, 'market')
    
    def _convert_status(self, alpaca_status: str) -> OrderStatus:
        """Convert Alpaca status to OrderStatus"""
        mapping = {
            'new': OrderStatus.SUBMITTED,
            'partially_filled': OrderStatus.PARTIALLY_FILLED,
            'filled': OrderStatus.FILLED,
            'done_for_day': OrderStatus.CANCELLED,
            'canceled': OrderStatus.CANCELLED,
            'expired': OrderStatus.CANCELLED,
            'replaced': OrderStatus.SUBMITTED,
            'pending_cancel': OrderStatus.SUBMITTED,
            'pending_replace': OrderStatus.SUBMITTED,
            'accepted': OrderStatus.SUBMITTED,
            'pending_new': OrderStatus.PENDING,
            'accepted_for_bidding': OrderStatus.SUBMITTED,
            'stopped': OrderStatus.CANCELLED,
            'rejected': OrderStatus.REJECTED,
            'suspended': OrderStatus.CANCELLED,
            'calculated': OrderStatus.SUBMITTED
        }
        return mapping.get(alpaca_status, OrderStatus.PENDING)