"""
Broker Interface and Order Management
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"
    SHORT = "short"
    COVER = "cover"

class OrderStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

@dataclass
class Order:
    """Order representation"""
    id: str
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "day"
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    timestamp: datetime = None
    broker_order_id: str = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}

@dataclass
class Fill:
    """Fill/execution representation"""
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    timestamp: datetime
    commission: float = 0.0
    fees: float = 0.0

class BrokerInterface(ABC):
    """Abstract broker interface"""
    
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to broker"""
        pass
    
    @abstractmethod
    async def disconnect(self):
        """Disconnect from broker"""
        pass
    
    @abstractmethod
    async def submit_order(self, order: Order) -> str:
        """Submit order to broker"""
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order"""
        pass
    
    @abstractmethod
    async def get_order_status(self, order_id: str) -> OrderStatus:
        """Get order status"""
        pass
    
    @abstractmethod
    async def get_positions(self) -> Dict[str, float]:
        """Get current positions"""
        pass
    
    @abstractmethod
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information"""
        pass

class BrokerManager:
    """Manages multiple broker connections"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.brokers: Dict[str, BrokerInterface] = {}
        self.active_orders: Dict[str, Order] = {}
        self.order_history: List[Order] = []
        self.fill_history: List[Fill] = []
        
        # Initialize brokers
        self._initialize_brokers()
        
        logger.info("Broker Manager initialized")
    
    def _initialize_brokers(self):
        """Initialize broker connections"""
        from .ib_adapter import InteractiveBrokersAdapter
        from .alpaca_adapter import AlpacaAdapter
        
        for broker_name, broker_config in self.config.items():
            if not broker_config.get('enabled', False):
                continue
            
            if broker_name == 'interactive_brokers':
                self.brokers[broker_name] = InteractiveBrokersAdapter(broker_config)
            elif broker_name == 'alpaca':
                self.brokers[broker_name] = AlpacaAdapter(broker_config)
            
            logger.info(f"Initialized broker: {broker_name}")
    
    async def connect(self):
        """Connect to all enabled brokers"""
        for name, broker in self.brokers.items():
            try:
                success = await broker.connect()
                if success:
                    logger.info(f"Connected to {name}")
                else:
                    logger.error(f"Failed to connect to {name}")
            except Exception as e:
                logger.error(f"Error connecting to {name}: {e}")
    
    async def disconnect(self):
        """Disconnect from all brokers"""
        for name, broker in self.brokers.items():
            try:
                await broker.disconnect()
                logger.info(f"Disconnected from {name}")
            except Exception as e:
                logger.error(f"Error disconnecting from {name}: {e}")
    
    async def submit_order(self, order: Order, broker_name: str = None) -> str:
        """Submit order to specified broker or best available"""
        
        if broker_name and broker_name in self.brokers:
            broker = self.brokers[broker_name]
        else:
            # Select best broker for this order
            broker = self._select_best_broker(order)
            if not broker:
                raise Exception("No available broker for order")
        
        try:
            # Submit order
            broker_order_id = await broker.submit_order(order)
            order.broker_order_id = broker_order_id
            order.status = OrderStatus.SUBMITTED
            
            # Track order
            self.active_orders[order.id] = order
            
            logger.info(f"Order submitted: {order.id} -> {broker_order_id}")
            return broker_order_id
            
        except Exception as e:
            order.status = OrderStatus.REJECTED
            logger.error(f"Error submitting order {order.id}: {e}")
            raise
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order"""
        if order_id not in self.active_orders:
            logger.warning(f"Order {order_id} not found in active orders")
            return False
        
        order = self.active_orders[order_id]
        
        # Find broker and cancel
        for broker in self.brokers.values():
            try:
                success = await broker.cancel_order(order.broker_order_id)
                if success:
                    order.status = OrderStatus.CANCELLED
                    self._move_to_history(order_id)
                    logger.info(f"Order cancelled: {order_id}")
                    return True
            except Exception as e:
                logger.error(f"Error cancelling order {order_id}: {e}")
        
        return False
    
    async def cancel_all_orders(self):
        """Cancel all active orders"""
        order_ids = list(self.active_orders.keys())
        
        for order_id in order_ids:
            try:
                await self.cancel_order(order_id)
            except Exception as e:
                logger.error(f"Error cancelling order {order_id}: {e}")
    
    def _select_best_broker(self, order: Order) -> Optional[BrokerInterface]:
        """Select best broker for order based on various criteria"""
        
        # Simplified selection logic
        # In production, would consider:
        # - Liquidity and execution quality
        # - Fees and commissions
        # - Order type support
        # - Market hours
        
        available_brokers = list(self.brokers.values())
        if available_brokers:
            return available_brokers[0]  # Use first available
        
        return None
    
    def _move_to_history(self, order_id: str):
        """Move order from active to history"""
        if order_id in self.active_orders:
            order = self.active_orders.pop(order_id)
            self.order_history.append(order)
    
    async def update_order_status(self):
        """Update status of all active orders"""
        for order_id, order in list(self.active_orders.items()):
            try:
                # Check status with broker
                for broker in self.brokers.values():
                    try:
                        status = await broker.get_order_status(order.broker_order_id)
                        if status != order.status:
                            order.status = status
                            
                            if status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
                                self._move_to_history(order_id)
                            
                            logger.info(f"Order {order_id} status updated: {status}")
                        break
                    except:
                        continue
            except Exception as e:
                logger.error(f"Error updating order status for {order_id}: {e}")
    
    async def get_all_positions(self) -> Dict[str, float]:
        """Get positions from all brokers"""
        all_positions = {}
        
        for name, broker in self.brokers.items():
            try:
                positions = await broker.get_positions()
                for symbol, quantity in positions.items():
                    if symbol in all_positions:
                        all_positions[symbol] += quantity
                    else:
                        all_positions[symbol] = quantity
            except Exception as e:
                logger.error(f"Error getting positions from {name}: {e}")
        
        return all_positions
    
    async def get_account_summary(self) -> Dict[str, Any]:
        """Get account summary from all brokers"""
        summary = {
            'total_value': 0.0,
            'cash': 0.0,
            'buying_power': 0.0,
            'brokers': {}
        }
        
        for name, broker in self.brokers.items():
            try:
                account_info = await broker.get_account_info()
                summary['brokers'][name] = account_info
                
                # Aggregate values
                summary['total_value'] += account_info.get('total_value', 0.0)
                summary['cash'] += account_info.get('cash', 0.0)
                summary['buying_power'] += account_info.get('buying_power', 0.0)
                
            except Exception as e:
                logger.error(f"Error getting account info from {name}: {e}")
        
        return summary