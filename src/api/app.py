"""
FastAPI Web Application and REST API
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from pydantic import BaseModel
import json

from ..core.engine import TradingEngine
from ..core.portfolio import Portfolio
from ..risk.manager import RiskManager
from ..utils.helpers import load_config

logger = logging.getLogger(__name__)

# Pydantic models for API
class OrderRequest(BaseModel):
    symbol: str
    side: str  # 'buy', 'sell'
    quantity: float
    order_type: str = 'market'
    price: Optional[float] = None
    stop_price: Optional[float] = None

class StrategyConfig(BaseModel):
    name: str
    enabled: bool
    allocation: float
    parameters: Dict[str, Any]

class RiskLimits(BaseModel):
    max_drawdown: float
    max_position_size: float
    greek_limits: Dict[str, float]

# FastAPI app
app = FastAPI(
    title="EquiFlex Pro API",
    description="Advanced Long-Short Equity and Options Trading Bot",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Global state
trading_engine: Optional[TradingEngine] = None
websocket_connections: List[WebSocket] = []

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token (simplified)"""
    try:
        # In production, verify with proper secret and validation
        token = credentials.credentials
        payload = jwt.decode(token, "secret_key", algorithms=["HS256"])
        return payload
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.on_event("startup")
async def startup_event():
    """Initialize trading engine on startup"""
    global trading_engine
    
    try:
        trading_engine = TradingEngine()
        # Start engine in background
        asyncio.create_task(trading_engine.start())
        logger.info("Trading engine started successfully")
    except Exception as e:
        logger.error(f"Error starting trading engine: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global trading_engine
    
    if trading_engine:
        await trading_engine.stop()
        logger.info("Trading engine stopped")

# Serve static files
app.mount("/static", StaticFiles(directory="web"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve main dashboard"""
    with open("web/index.html", "r") as f:
        return HTMLResponse(content=f.read(), status_code=200)

# API Routes

@app.get("/api/status")
async def get_system_status():
    """Get system status"""
    global trading_engine
    
    if not trading_engine:
        return {"status": "error", "message": "Trading engine not initialized"}
    
    return {
        "status": "running" if trading_engine.running else "stopped",
        "uptime": (datetime.now() - trading_engine.start_time).total_seconds() if hasattr(trading_engine, 'start_time') else 0,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/portfolio")
async def get_portfolio(user=Depends(verify_token)):
    """Get portfolio information"""
    global trading_engine
    
    if not trading_engine:
        raise HTTPException(status_code=503, detail="Trading engine not available")
    
    portfolio_summary = trading_engine.portfolio.get_performance_summary()
    positions = trading_engine.portfolio.get_position_summary().to_dict('records')
    
    return {
        "summary": portfolio_summary,
        "positions": positions,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/portfolio/greeks")
async def get_portfolio_greeks(user=Depends(verify_token)):
    """Get portfolio Greeks exposure"""
    global trading_engine
    
    if not trading_engine:
        raise HTTPException(status_code=503, detail="Trading engine not available")
    
    greeks = trading_engine.portfolio.calculate_portfolio_greeks()
    
    return {
        "greeks": greeks,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/risk")
async def get_risk_metrics(user=Depends(verify_token)):
    """Get risk metrics and dashboard"""
    global trading_engine
    
    if not trading_engine:
        raise HTTPException(status_code=503, detail="Trading engine not available")
    
    risk_dashboard = trading_engine.risk_manager.get_risk_dashboard(trading_engine.portfolio)
    
    return {
        "risk_data": risk_dashboard,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/strategies")
async def get_strategies(user=Depends(verify_token)):
    """Get strategy status and performance"""
    global trading_engine
    
    if not trading_engine:
        raise HTTPException(status_code=503, detail="Trading engine not available")
    
    strategies_info = {}
    
    for name, strategy in trading_engine.strategy_manager.strategies.items():
        strategies_info[name] = {
            "name": strategy.name,
            "enabled": strategy.enabled,
            "is_running": strategy.is_running,
            "allocation": strategy.allocation,
            "signals_generated": strategy.signals_generated,
            "performance_metrics": strategy.performance_metrics,
            "last_update": strategy.last_update.isoformat() if strategy.last_update else None
        }
    
    return {
        "strategies": strategies_info,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/strategies/{strategy_name}/toggle")
async def toggle_strategy(strategy_name: str, user=Depends(verify_token)):
    """Enable/disable a strategy"""
    global trading_engine
    
    if not trading_engine:
        raise HTTPException(status_code=503, detail="Trading engine not available")
    
    if strategy_name not in trading_engine.strategy_manager.strategies:
        raise HTTPException(status_code=404, detail="Strategy not found")
    
    strategy = trading_engine.strategy_manager.strategies[strategy_name]
    
    if strategy.is_running:
        await strategy.stop()
        action = "stopped"
    else:
        await strategy.start()
        action = "started"
    
    return {
        "strategy": strategy_name,
        "action": action,
        "status": "running" if strategy.is_running else "stopped"
    }

@app.post("/api/orders")
async def submit_order(order_request: OrderRequest, user=Depends(verify_token)):
    """Submit a new order"""
    global trading_engine
    
    if not trading_engine:
        raise HTTPException(status_code=503, detail="Trading engine not available")
    
    try:
        from ..execution.broker_interface import Order, OrderSide, OrderType
        from uuid import uuid4
        
        # Create order object
        order = Order(
            id=str(uuid4()),
            symbol=order_request.symbol,
            side=OrderSide.BUY if order_request.side.lower() == 'buy' else OrderSide.SELL,
            quantity=order_request.quantity,
            order_type=OrderType(order_request.order_type.lower()),
            price=order_request.price,
            stop_price=order_request.stop_price
        )
        
        # Submit order
        broker_order_id = await trading_engine.broker_manager.submit_order(order)
        
        return {
            "order_id": order.id,
            "broker_order_id": broker_order_id,
            "status": "submitted",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/orders")
async def get_orders(user=Depends(verify_token)):
    """Get order history"""
    global trading_engine
    
    if not trading_engine:
        raise HTTPException(status_code=503, detail="Trading engine not available")
    
    active_orders = []
    for order_id, order in trading_engine.broker_manager.active_orders.items():
        active_orders.append({
            "id": order.id,
            "symbol": order.symbol,
            "side": order.side.value,
            "quantity": order.quantity,
            "order_type": order.order_type.value,
            "price": order.price,
            "status": order.status.value,
            "timestamp": order.timestamp.isoformat()
        })
    
    order_history = []
    for order in trading_engine.broker_manager.order_history[-50:]:  # Last 50 orders
        order_history.append({
            "id": order.id,
            "symbol": order.symbol,
            "side": order.side.value,
            "quantity": order.quantity,
            "order_type": order.order_type.value,
            "price": order.price,
            "status": order.status.value,
            "timestamp": order.timestamp.isoformat()
        })
    
    return {
        "active_orders": active_orders,
        "order_history": order_history,
        "timestamp": datetime.now().isoformat()
    }

@app.delete("/api/orders/{order_id}")
async def cancel_order(order_id: str, user=Depends(verify_token)):
    """Cancel an order"""
    global trading_engine
    
    if not trading_engine:
        raise HTTPException(status_code=503, detail="Trading engine not available")
    
    try:
        success = await trading_engine.broker_manager.cancel_order(order_id)
        
        if success:
            return {"order_id": order_id, "status": "cancelled"}
        else:
            raise HTTPException(status_code=400, detail="Failed to cancel order")
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/market-data/{symbol}")
async def get_market_data(symbol: str, period: str = "1d", user=Depends(verify_token)):
    """Get market data for a symbol"""
    global trading_engine
    
    if not trading_engine:
        raise HTTPException(status_code=503, detail="Trading engine not available")
    
    try:
        data = await trading_engine.data_manager.get_historical_data(symbol, period=period)
        
        if data.empty:
            raise HTTPException(status_code=404, detail="No data found")
        
        # Convert to JSON-serializable format
        data_dict = data.reset_index().to_dict('records')
        
        return {
            "symbol": symbol,
            "data": data_dict,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/options/{underlying}")
async def get_options_chain(underlying: str, expiry: Optional[str] = None, user=Depends(verify_token)):
    """Get options chain for underlying"""
    global trading_engine
    
    if not trading_engine:
        raise HTTPException(status_code=503, detail="Trading engine not available")
    
    try:
        options_data = await trading_engine.data_manager.get_options_chain(underlying, expiry)
        
        options_list = []
        for option in options_data:
            options_list.append({
                "symbol": option.symbol,
                "underlying": option.underlying,
                "strike": option.strike,
                "expiry": option.expiry.isoformat(),
                "option_type": option.option_type,
                "bid": option.bid,
                "ask": option.ask,
                "last": option.last,
                "volume": option.volume,
                "open_interest": option.open_interest,
                "implied_vol": option.implied_vol
            })
        
        return {
            "underlying": underlying,
            "options": options_list,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/performance")
async def get_performance_analytics(user=Depends(verify_token)):
    """Get detailed performance analytics"""
    global trading_engine
    
    if not trading_engine:
        raise HTTPException(status_code=503, detail="Trading engine not available")
    
    portfolio = trading_engine.portfolio
    
    # Calculate additional metrics
    performance_data = {
        "total_return": portfolio.total_return,
        "sharpe_ratio": portfolio.sharpe_ratio,
        "max_drawdown": portfolio.max_drawdown,
        "total_trades": portfolio.total_trades,
        "winning_trades": portfolio.winning_trades,
        "win_rate": portfolio.winning_trades / max(portfolio.total_trades, 1),
        "total_pnl": portfolio.total_pnl,
        "current_value": portfolio.total_value,
        "cash": portfolio.cash
    }
    
    # Historical performance (simplified)
    historical_values = getattr(portfolio, 'historical_values', [])
    
    return {
        "performance_metrics": performance_data,
        "historical_values": historical_values,
        "timestamp": datetime.now().isoformat()
    }

# WebSocket endpoint for real-time updates
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await websocket.accept()
    websocket_connections.append(websocket)
    
    try:
        while True:
            # Send periodic updates
            if trading_engine:
                update_data = {
                    "type": "portfolio_update",
                    "data": {
                        "total_value": trading_engine.portfolio.total_value,
                        "total_return": trading_engine.portfolio.total_return,
                        "cash": trading_engine.portfolio.cash,
                        "timestamp": datetime.now().isoformat()
                    }
                }
                
                await websocket.send_text(json.dumps(update_data))
            
            await asyncio.sleep(5)  # Update every 5 seconds
            
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        websocket_connections.remove(websocket)

async def broadcast_update(data: Dict[str, Any]):
    """Broadcast update to all connected WebSocket clients"""
    if websocket_connections:
        message = json.dumps(data)
        for websocket in websocket_connections[:]:  # Copy list to avoid modification issues
            try:
                await websocket.send_text(message)
            except:
                websocket_connections.remove(websocket)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)