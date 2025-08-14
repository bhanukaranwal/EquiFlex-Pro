"""
Machine Learning Models for Trading Signals
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import asyncio

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
import joblib

logger = logging.getLogger(__name__)

class LSTMPricePredictor(nn.Module):
    """LSTM model for price prediction"""
    
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, 
                 output_size: int = 1, dropout: float = 0.2):
        super(LSTMPricePredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM forward pass
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))
        
        # Use the last output
        out = self.dropout(lstm_out[:, -1, :])
        out = self.linear(out)
        
        return out

class VolatilityPredictor(nn.Module):
    """Neural network for volatility prediction"""
    
    def __init__(self, input_size: int, hidden_sizes: List[int] = [64, 32], 
                 output_size: int = 1, dropout: float = 0.3):
        super(VolatilityPredictor, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.BatchNorm1d(hidden_size)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        layers.append(nn.Sigmoid())  # Ensure positive volatility
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class MLSignalGenerator:
    """Machine learning signal generator combining multiple models"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.lookback_window = 60
        self.is_trained = False
        
        # Initialize models
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize ML models"""
        
        # Price prediction models
        self.models['lstm_price'] = LSTMPricePredictor(
            input_size=10,  # Will be set based on features
            hidden_size=128,
            num_layers=2
        )
        
        # Volatility prediction
        self.models['volatility'] = VolatilityPredictor(
            input_size=20,
            hidden_sizes=[64, 32]
        )
        
        # Traditional ML models
        self.models['random_forest'] = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        self.models['gradient_boost'] = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=6,
            random_state=42
        )
        
        # Scalers
        self.scalers['standard'] = StandardScaler()
        self.scalers['minmax'] = MinMaxScaler()
        
        logger.info("ML models initialized")
    
    async def train_models(self, market_data: Dict[str, pd.DataFrame], 
                          target_column: str = 'close'):
        """Train all ML models"""
        
        logger.info("Starting ML model training...")
        
        try:
            # Prepare training data
            X, y, feature_names = self._prepare_training_data(market_data, target_column)
            
            if X.shape[0] < 100:
                logger.warning("Insufficient data for training")
                return False
            
            # Split data for time series
            tscv = TimeSeriesSplit(n_splits=5)
            
            # Train LSTM models
            await self._train_lstm_models(X, y)
            
            # Train traditional ML models
            await self._train_traditional_models(X, y, tscv)
            
            self.feature_columns = feature_names
            self.is_trained = True
            
            logger.info("ML model training completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error training ML models: {e}")
            return False
    
    def _prepare_training_data(self, market_data: Dict[str, pd.DataFrame], 
                              target_column: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare training data with features"""
        
        all_features = []
        all_targets = []
        feature_names = []
        
        for symbol, data in market_data.items():
            if data.empty or len(data) < self.lookback_window + 10:
                continue
            
            # Calculate technical indicators
            features_df = self._calculate_features(data)
            
            if features_df.empty:
                continue
            
            # Create sequences for LSTM
            X_seq, y_seq = self._create_sequences(features_df, target_column)
            
            if len(X_seq) > 0:
                all_features.append(X_seq)
                all_targets.append(y_seq)
                
                if not feature_names:
                    feature_names = features_df.columns.tolist()
        
        if not all_features:
            return np.array([]), np.array([]), []
        
        # Combine all data
        X = np.vstack(all_features)
        y = np.concatenate(all_targets)
        
        return X, y, feature_names
    
    def _calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators and features"""
        
        df = data.copy()
        
        # Price-based features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Moving averages
        for window in [5, 10, 20, 50]:
            df[f'sma_{window}'] = df['close'].rolling(window).mean()
            df[f'ema_{window}'] = df['close'].ewm(span=window).mean()
        
        # Volatility measures
        df['volatility_10'] = df['returns'].rolling(10).std()
        df['volatility_30'] = df['returns'].rolling(30).std()
        
        # Price relatives
        df['price_to_sma_20'] = df['close'] / df['sma_20']
        df['sma_5_to_sma_20'] = df['sma_5'] / df['sma_20']
        
        # Volume features
        if 'volume' in df.columns:
            df['volume_sma_10'] = df['volume'].rolling(10).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma_10']
        
        # Momentum indicators
        df['rsi'] = self._calculate_rsi(df['close'])
        df['macd'], df['macd_signal'] = self._calculate_macd(df['close'])
        
        # Bollinger Bands
        bb_middle = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = bb_middle + (bb_std * 2)
        df['bb_lower'] = bb_middle - (bb_std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / bb_middle
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Remove NaN values
        df = df.dropna()
        
        # Select feature columns (exclude OHLCV)
        feature_cols = [col for col in df.columns 
                       if col not in ['open', 'high', 'low', 'close', 'volume']]
        
        return df[feature_cols]
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, 
                       slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        return macd, macd_signal
    
    def _create_sequences(self, features_df: pd.DataFrame, 
                         target_column: str) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training"""
        
        if len(features_df) < self.lookback_window + 1:
            return np.array([]), np.array([])
        
        # Scale features
        features_scaled = self.scalers['standard'].fit_transform(features_df)
        
        X, y = [], []
        
        for i in range(self.lookback_window, len(features_scaled)):
            X.append(features_scaled[i-self.lookback_window:i])
            
            # Target is next period return
            if i < len(features_df) - 1:
                current_return = features_df.iloc[i]['returns'] if 'returns' in features_df.columns else 0
                y.append(current_return)
        
        return np.array(X), np.array(y)
    
    async def _train_lstm_models(self, X: np.ndarray, y: np.ndarray):
        """Train LSTM models"""
        
        if len(X.shape) != 3:
            logger.warning("Invalid data shape for LSTM training")
            return
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y.reshape(-1, 1))
        
        # Update input size
        self.models['lstm_price'].lstm.input_size = X.shape[2]
        
        # Training parameters
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.models['lstm_price'].parameters(), lr=0.001)
        
        # Training loop
        num_epochs = 50
        batch_size = 32
        
        for epoch in range(num_epochs):
            for i in range(0, len(X_tensor), batch_size):
                batch_X = X_tensor[i:i+batch_size]
                batch_y = y_tensor[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = self.models['lstm_price'](batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"LSTM Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}")
    
    async def _train_traditional_models(self, X: np.ndarray, y: np.ndarray, tscv):
        """Train traditional ML models"""
        
        # Flatten X for traditional ML models
        X_flat = X.reshape(X.shape[0], -1)
        
        # Scale features
        X_scaled = self.scalers['minmax'].fit_transform(X_flat)
        
        # Train Random Forest
        self.models['random_forest'].fit(X_scaled, y)
        
        # Train Gradient Boosting
        self.models['gradient_boost'].fit(X_scaled, y)
        
        logger.info("Traditional ML models trained")
    
    async def generate_equity_signals(self, price_data: Dict[str, List[float]], 
                                    factor_scores: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Generate ML-enhanced equity signals"""
        
        if not self.is_trained:
            logger.warning("Models not trained yet")
            return {}
        
        signals = {}
        
        for symbol, prices in price_data.items():
            if len(prices) < self.lookback_window:
                continue
            
            try:
                # Prepare features for this symbol
                recent_data = pd.DataFrame({'close': prices[-100:]})  # Last 100 points
                features_df = self._calculate_features(recent_data)
                
                if features_df.empty:
                    continue
                
                # Get latest features
                latest_features = features_df.iloc[-self.lookback_window:].values
                
                # LSTM prediction
                lstm_signal = self._predict_with_lstm(latest_features)
                
                # Traditional ML predictions
                ml_signal = self._predict_with_traditional_ml(latest_features.flatten())
                
                # Combine signals
                combined_signal = 0.4 * lstm_signal + 0.3 * ml_signal
                
                # Enhance with factor scores
                if symbol in factor_scores:
                    factor_boost = np.mean(list(factor_scores[symbol].values()))
                    combined_signal += 0.3 * factor_boost
                
                signals[symbol] = np.clip(combined_signal, -1.0, 1.0)
                
            except Exception as e:
                logger.error(f"Error generating signal for {symbol}: {e}")
                continue
        
        return signals
    
    def _predict_with_lstm(self, features: np.ndarray) -> float:
        """Generate prediction using LSTM model"""
        try:
            # Prepare input
            features_scaled = self.scalers['standard'].transform(features)
            X_tensor = torch.FloatTensor(features_scaled).unsqueeze(0)
            
            # Predict
            self.models['lstm_price'].eval()
            with torch.no_grad():
                prediction = self.models['lstm_price'](X_tensor)
            
            return float(prediction.item())
            
        except Exception as e:
            logger.error(f"Error in LSTM prediction: {e}")
            return 0.0
    
    def _predict_with_traditional_ml(self, features: np.ndarray) -> float:
        """Generate prediction using traditional ML models"""
        try:
            # Scale features
            features_scaled = self.scalers['minmax'].transform(features.reshape(1, -1))
            
            # Predictions from both models
            rf_pred = self.models['random_forest'].predict(features_scaled)[0]
            gb_pred = self.models['gradient_boost'].predict(features_scaled)[0]
            
            # Average predictions
            return (rf_pred + gb_pred) / 2.0
            
        except Exception as e:
            logger.error(f"Error in traditional ML prediction: {e}")
            return 0.0
    
    async def predict_volatility(self, price_data: pd.DataFrame) -> float:
        """Predict future volatility"""
        
        if not self.is_trained:
            return 0.2  # Default volatility
        
        try:
            features_df = self._calculate_features(price_data)
            if features_df.empty:
                return 0.2
            
            # Use volatility model
            latest_features = features_df.iloc[-20:].values.flatten()
            features_scaled = self.scalers['standard'].transform(latest_features.reshape(1, -1))
            
            X_tensor = torch.FloatTensor(features_scaled)
            
            self.models['volatility'].eval()
            with torch.no_grad():
                vol_prediction = self.models['volatility'](X_tensor)
            
            return float(vol_prediction.item())
            
        except Exception as e:
            logger.error(f"Error predicting volatility: {e}")
            return 0.2
    
    def save_models(self, model_dir: str = "models/"):
        """Save trained models"""
        import os
        os.makedirs(model_dir, exist_ok=True)
        
        # Save PyTorch models
        torch.save(self.models['lstm_price'].state_dict(), 
                  f"{model_dir}/lstm_price.pth")
        torch.save(self.models['volatility'].state_dict(), 
                  f"{model_dir}/volatility.pth")
        
        # Save sklearn models
        joblib.dump(self.models['random_forest'], 
                   f"{model_dir}/random_forest.pkl")
        joblib.dump(self.models['gradient_boost'], 
                   f"{model_dir}/gradient_boost.pkl")
        
        # Save scalers
        joblib.dump(self.scalers, f"{model_dir}/scalers.pkl")
        
        logger.info(f"Models saved to {model_dir}")
    
    def load_models(self, model_dir: str = "models/"):
        """Load trained models"""
        import os
        
        try:
            # Load PyTorch models
            self.models['lstm_price'].load_state_dict(
                torch.load(f"{model_dir}/lstm_price.pth"))
            self.models['volatility'].load_state_dict(
                torch.load(f"{model_dir}/volatility.pth"))
            
            # Load sklearn models
            self.models['random_forest'] = joblib.load(
                f"{model_dir}/random_forest.pkl")
            self.models['gradient_boost'] = joblib.load(
                f"{model_dir}/gradient_boost.pkl")
            
            # Load scalers
            self.scalers = joblib.load(f"{model_dir}/scalers.pkl")
            
            self.is_trained = True
            logger.info(f"Models loaded from {model_dir}")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")