# robust_model.py - A Model That Actually Learns
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
import logging
from datetime import datetime, timedelta
from typing import Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RobustLSTMModel(nn.Module):
    """
    A more sophisticated LSTM that can actually learn patterns
    """
    def __init__(self, input_size, hidden_size=256, num_layers=3, dropout=0.3):
        super(RobustLSTMModel, self).__init__()
        
        # Bidirectional LSTM for better context understanding
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
        # Attention mechanism (simplified)
        self.attention = nn.Linear(hidden_size * 2, 1)
        
        # Multi-layer prediction head
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 1)
        
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.1)
        
    def forward(self, x):
        # LSTM processing
        lstm_out, _ = self.lstm1(x)  # (batch, seq, hidden*2)
        
        # Simple attention mechanism
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Multi-layer prediction
        out = self.dropout(context)
        out = self.relu(self.bn1(self.fc1(out)))
        out = self.dropout(out)
        out = self.leaky_relu(self.bn2(self.fc2(out)))
        out = self.dropout(out)
        out = self.fc3(out)
        
        return out

class FeatureEngineer:
    """
    Create meaningful features that help the model understand air quality patterns
    """
    @staticmethod
    def create_time_features(df):
        """Create time-based features that actually matter for air quality"""
        df = df.copy()
        
        # Hour of day (pollution patterns change throughout day)
        df['hour'] = df.index.hour
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Day of week (weekday vs weekend patterns)
        df['dayofweek'] = df.index.dayofweek
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
        
        # Month (seasonal patterns)
        df['month'] = df.index.month
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Year trend (long-term environmental changes)
        df['year'] = df.index.year
        df['year_normalized'] = (df['year'] - df['year'].min()) / (df['year'].max() - df['year'].min())
        
        return df
    
    @staticmethod
    def create_lag_features(df, target_col='medicion', lags=[1, 2, 3, 6, 12, 24, 48]):
        """Create lag features - what happened before often predicts what happens next"""
        df = df.copy()
        
        for lag in lags:
            df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
        
        return df
    
    @staticmethod
    def create_rolling_features(df, target_col='medicion', windows=[6, 12, 24, 48]):
        """Rolling statistics help capture trends"""
        df = df.copy()
        
        for window in windows:
            df[f'{target_col}_mean_{window}h'] = df[target_col].rolling(window).mean()
            df[f'{target_col}_std_{window}h'] = df[target_col].rolling(window).std()
            df[f'{target_col}_min_{window}h'] = df[target_col].rolling(window).min()
            df[f'{target_col}_max_{window}h'] = df[target_col].rolling(window).max()
        
        return df

class RobustAirQualityPredictor:
    def __init__(self, sequence_length=48, model_path=None):  # 2 days instead of 7
        self.sequence_length = sequence_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_engineer = FeatureEngineer()
        self.scalers = {}
        self.model = None
        self.feature_columns = None
        
        logger.info(f"Initializing robust predictor on {self.device}")
    
    def _prepare_features(self, df):
        """Create all the features that help the model understand patterns"""
        df = df.copy()
        
        # Basic cleaning
        df = df[df['medicion'] >= 0]  # Remove negative values
        df = df[df['medicion'] <= 1000]  # Remove extreme outliers
        
        # Create time features
        df = self.feature_engineer.create_time_features(df)
        
        # Create lag features (what happened before)
        df = self.feature_engineer.create_lag_features(df)
        
        # Create rolling features (recent trends)
        df = self.feature_engineer.create_rolling_features(df)
        
        # Drop rows with NaN values (from lag/rolling features)
        df = df.dropna()
        
        return df
    
    def _create_sequences(self, df, target_col='medicion'):
        """Create sequences with multiple features"""
        feature_cols = [col for col in df.columns if col != target_col]
        self.feature_columns = feature_cols
        
        X_data = df[feature_cols].values
        y_data = df[target_col].values
        
        # Scale features
        if 'features' not in self.scalers:
            self.scalers['features'] = RobustScaler()
            X_scaled = self.scalers['features'].fit_transform(X_data)
        else:
            X_scaled = self.scalers['features'].transform(X_data)
        
        # Scale target
        if 'target' not in self.scalers:
            self.scalers['target'] = RobustScaler()
            y_scaled = self.scalers['target'].fit_transform(y_data.reshape(-1, 1)).flatten()
        else:
            y_scaled = self.scalers['target'].transform(y_data.reshape(-1, 1)).flatten()
        
        # Create sequences
        X_sequences, y_sequences = [], []
        for i in range(len(X_scaled) - self.sequence_length):
            X_sequences.append(X_scaled[i:i + self.sequence_length])
            y_sequences.append(y_scaled[i + self.sequence_length])
        
        return np.array(X_sequences), np.array(y_sequences)
    
    def train(self, df, validation_split=0.2, epochs=200, batch_size=64):
        """Train the model with proper validation"""
        logger.info("Preparing features for training...")
        
        # Prepare features
        df_features = self._prepare_features(df)
        logger.info(f"Created {len(df_features.columns)} features from raw data")
        
        # Create sequences
        X, y = self._create_sequences(df_features)
        logger.info(f"Created {len(X)} training sequences")
        
        # Train/validation split (time-based)
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Convert to tensors
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.FloatTensor(y_train).to(self.device)
        X_val = torch.FloatTensor(X_val).to(self.device)
        y_val = torch.FloatTensor(y_val).to(self.device)
        
        # Initialize model
        input_size = X.shape[2]  # Number of features
        self.model = RobustLSTMModel(input_size=input_size).to(self.device)
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = self.model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val).squeeze()
                val_loss = criterion(val_outputs, y_val).item()
            
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), f'best_model_checkpoint.pth')
            else:
                patience_counter += 1
            
            if epoch % 10 == 0:
                logger.info(f'Epoch {epoch}: Train Loss: {train_loss/len(X_train)*batch_size:.6f}, Val Loss: {val_loss:.6f}')
            
            if patience_counter >= 20:
                logger.info(f'Early stopping at epoch {epoch}')
                break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_model_checkpoint.pth'))
        logger.info("Training completed!")
    
    def predict(self, df, prediction_window):
        """Make actual predictions that aren't just 4.13"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Prepare features
        df_features = self._prepare_features(df)
        
        # Get recent data for prediction
        recent_data = df_features.tail(self.sequence_length)
        
        if len(recent_data) < self.sequence_length:
            raise ValueError(f"Need at least {self.sequence_length} data points")
        
        # Prepare sequence
        X_data = recent_data[self.feature_columns].values
        X_scaled = self.scalers['features'].transform(X_data)
        
        predictions = []
        current_sequence = torch.FloatTensor(X_scaled).unsqueeze(0).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            for step in range(prediction_window):
                # Get prediction
                output = self.model(current_sequence).cpu().numpy()[0]
                predictions.append(output)
                
                # Update sequence (this is the tricky part - we need to forecast features too)
                # For simplicity, we'll use a naive approach for other features
                new_features = X_scaled[-1].copy()
                new_features[0] = output  # Assuming first feature is the target
                
                # Update time features for next step
                current_time = recent_data.index[-1] + timedelta(hours=step+1)
                new_features[1] = np.sin(2 * np.pi * current_time.hour / 24)  # hour_sin
                new_features[2] = np.cos(2 * np.pi * current_time.hour / 24)  # hour_cos
                
                # Shift sequence
                new_sequence = torch.cat([
                    current_sequence[:, 1:, :],
                    torch.FloatTensor(new_features).unsqueeze(0).unsqueeze(0).to(self.device)
                ], dim=1)
                current_sequence = new_sequence
        
        # Inverse transform predictions
        predictions_array = np.array(predictions).reshape(-1, 1)
        predictions_rescaled = self.scalers['target'].inverse_transform(predictions_array).flatten()
        
        # Generate timestamps
        timestamps = []
        last_time = df.index[-1]
        for i in range(1, prediction_window + 1):
            future_time = last_time + timedelta(hours=i)
            timestamps.append(future_time.isoformat())
        
        return predictions_rescaled.tolist(), timestamps
    
    def get_model_info(self):
        total_params = sum(p.numel() for p in self.model.parameters()) if self.model else 0
        return {
            "sequence_length": self.sequence_length,
            "device": str(self.device),
            "total_parameters": total_params,
            "model_type": "Robust_LSTM_with_Features",
            "features_count": len(self.feature_columns) if self.feature_columns else 0
        }

# API Interface
predictor = None

def get_predictor():
    global predictor
    if predictor is None:
        predictor = RobustAirQualityPredictor()
        logger.info("Robust LSTM Predictor initialized")
    return predictor

def train_model_for_station_element(df):
    
    predictor = get_predictor()
    predictor.train(df)

    return predictor
