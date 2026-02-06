import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns   
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor


def decompose_timeseries(df, column='Precipitation', period=12):
    """Decompose time series into trend, seasonal, residual."""
    decomposition = seasonal_decompose(df[column], model='additive', period=period)
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 10))
    
    df[column].plot(ax=axes[0], title='Original')
    decomposition.trend.plot(ax=axes[1], title='Trend')
    decomposition.seasonal.plot(ax=axes[2], title='Seasonality')
    decomposition.resid.plot(ax=axes[3], title='Residual')
    
    plt.tight_layout()
    plt.savefig('images/decomposition.png')
    plt.show()
    
    return decomposition













def create_season_dummies(df):
   
    df['is_winter'] = df['Month'].isin([12, 1, 2]).astype(int)
    df['is_spring'] = df['Month'].isin([3, 4, 5]).astype(int)
    df['is_summer'] = df['Month'].isin([6, 7, 8]).astype(int)
    df['is_fall'] = df['Month'].isin([9, 10, 11]).astype(int)
    df['is_monsoon'] = df['Month'].isin([6, 7, 8, 9]).astype(int)
    return df


def create_adatimeobjhect(df):
    df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
    df = df.set_index('Date').sort_index()
    df.drop(columns=['Year', 'Month', 'Day'], inplace=True) 
    return df   

def reorder_columns(df, target_col='Precipitation'):
    # Move target column to the end
    cols = [col for col in df.columns if col != target_col]
    cols.append(target_col)
    return df[cols]

def add_log_transformation(df):
    """Add log-transformed Precipitation column."""
    df['Precipitation_log'] = np.log1p(df['Precipitation'])  # log(1 + x)
    return df

def plot_precipitation_comparison(df):
    """Compare original and log-transformed precipitation distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Original distribution
    axes[0].hist(df['Precipitation'], bins=50, edgecolor='black')
    axes[0].set_title('Original Precipitation Distribution')
    axes[0].set_xlabel('Precipitation (mm)')
    axes[0].set_ylabel('Frequency')
    
    # Log-transformed distribution
    axes[1].hist(df['Precipitation_log'], bins=50, edgecolor='black', color='orange')
    axes[1].set_title('Log-Transformed Precipitation Distribution')
    axes[1].set_xlabel('log(1 + Precipitation)')
    axes[1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('images/precipitation_comparison.png')

def plot_correlation_heatmap(df):
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('images/correlation_heatmapOfRainfallCSV.png')  # Save the plot as an image file
    plt.show()



def lag_features(df, target_col='Precipitation', lags=[1, 7, 30, 90]):
    
    for lag in lags:
        df[f'{target_col}_lag{lag}'] = df[target_col].shift(lag)
    
    # Drop NaN rows only once at the end
    df = df.dropna()
    return df

def plot_arima_sarima_results(y_test, y_pred_arima, y_pred_sarima):
    """Plot ARIMA and SARIMA predictions vs actual."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # ARIMA plot
    axes[0].plot(y_test.index, y_test.values, label='Actual', linewidth=2, marker='o')
    axes[0].plot(y_test.index, y_pred_arima, label='ARIMA Forecast', linestyle='--', linewidth=2, marker='x', color='red')
    axes[0].set_title('ARIMA Forecast vs Actual')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Precipitation (mm)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # SARIMA plot
    axes[1].plot(y_test.index, y_test.values, label='Actual', linewidth=2, marker='o')
    axes[1].plot(y_test.index, y_pred_sarima, label='SARIMA Forecast', linestyle='--', linewidth=2, marker='x', color='orange')
    axes[1].set_title('SARIMA Forecast vs Actual')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Precipitation (mm)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('images/arima_sarima_comparison.png')
    plt.show()






# Main execution
df = pd.read_csv('rainfall/Rainfall_data.csv')
df = create_season_dummies(df)
df = create_adatimeobjhect(df)
df = reorder_columns(df, target_col='Precipitation')
df = add_log_transformation(df)
df = lag_features(df, target_col='Precipitation', lags=[1, 7, 30, 90])

def prepare_data(df, target_col='Precipitation', test_size=0.2):
    """Split data chronologically and prepare X, y."""
    
    # Separate features (X) and target (y)
    X = df.drop(columns=[target_col, 'Precipitation_log'])  # Drop target and log version
    y = df[target_col]
    
    # Chronological split (80/20)
    split_idx = int(len(df) * (1 - test_size))
    
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    
    print(f"Train set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

# Usage
X_train, X_test, y_train, y_test = prepare_data(df, target_col='Precipitation', test_size=0.2)

def fit_arima_model(y_train, y_test, order=(1, 1, 1)):
    """Fit ARIMA model."""
    model = ARIMA(y_train, order=order)
    model_fit = model.fit()
    
    y_pred = model_fit.forecast(steps=len(y_test))
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"ARIMA{order} - MAE: {mae:.2f}, RMSE: {rmse:.2f}")
    return model_fit, y_pred

def fit_sarima_model(y_train, y_test, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
    """Fit SARIMA model (Seasonal ARIMA)."""
    model = SARIMAX(y_train, order=order, seasonal_order=seasonal_order)
    model_fit = model.fit(disp=False)
    
    y_pred = model_fit.forecast(steps=len(y_test))
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"SARIMA{order}x{seasonal_order} - MAE: {mae:.2f}, RMSE: {rmse:.2f}")
    return model_fit, y_pred
X_train, X_test, y_train, y_test = prepare_data(df, target_col='Precipitation', test_size=0.2)


def train_xgboost_model(X_train, X_test, y_train, y_test):
    """Train XGBoost model and evaluate."""
    
    model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Metrics
    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2_test = r2_score(y_test, y_pred_test)
    
    print(f"XGBoost Test MAE: {mae_test:.2f}")
    print(f"XGBoost Test RMSE: {rmse_test:.2f}")
    print(f"XGBoost Test R²: {r2_test:.4f}")
    
    return model, y_pred_train, y_pred_test

def plot_xgboost_results(y_train, y_test, y_pred_train, y_pred_test, model, X_train):
    """Plot XGBoost predictions and feature importance."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Predictions vs Actual
    axes[0].plot(y_test.index, y_test.values, label='Actual', linewidth=2, marker='o')
    axes[0].plot(y_test.index, y_pred_test, label='XGBoost Forecast', linestyle='--', linewidth=2, marker='x', color='green')
    axes[0].set_title('XGBoost Forecast vs Actual (Test Set)')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Precipitation (mm)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Feature Importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    axes[1].barh(feature_importance['feature'], feature_importance['importance'], color='skyblue')
    axes[1].set_xlabel('Importance')
    axes[1].set_title('Feature Importance')
    axes[1].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('images/xgboost_results.png')
    plt.show()

def plot_model_comparison(metrics_dict):
    """Plot comparison of MAE and RMSE for all models."""
    models = list(metrics_dict.keys())
    mae_values = [metrics_dict[m]['MAE'] for m in models]
    rmse_values = [metrics_dict[m]['RMSE'] for m in models]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # MAE comparison
    axes[0].bar(models, mae_values, color=['blue', 'orange', 'green'])
    axes[0].set_title('Mean Absolute Error (MAE) Comparison')
    axes[0].set_ylabel('MAE')
    axes[0].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(mae_values):
        axes[0].text(i, v + 5, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # RMSE comparison
    axes[1].bar(models, rmse_values, color=['blue', 'orange', 'green'])
    axes[1].set_title('Root Mean Squared Error (RMSE) Comparison')
    axes[1].set_ylabel('RMSE')
    axes[1].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(rmse_values):
        axes[1].text(i, v + 10, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('images/model_comparison.png')
    plt.show()

def scale_features(X_train, X_test):
    """Scale features using StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame to preserve column names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    print("Features scaled using StandardScaler")
    return X_train_scaled, X_test_scaled, scaler

# Train all models
print("\nTraining ARIMA...")
arima_model, y_pred_arima = fit_arima_model(y_train, y_test, order=(1, 1, 1))

print("\nTraining SARIMA...")
sarima_model, y_pred_sarima = fit_sarima_model(y_train, y_test, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))

# Scale features for XGBoost
X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

print("\nTraining XGBoost (with scaled features)...")
xgb_model, y_pred_train_xgb, y_pred_test_xgb = train_xgboost_model(X_train_scaled, X_test_scaled, y_train, y_test)

# Collect metrics
metrics = {
    'ARIMA': {
        'MAE': mean_absolute_error(y_test, y_pred_arima),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_arima))
    },
    'SARIMA': {
        'MAE': mean_absolute_error(y_test, y_pred_sarima),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_sarima))
    },
    'XGBoost': {
        'MAE': mean_absolute_error(y_test, y_pred_test_xgb),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_test_xgb))
    }
}

# Plot all comparisons
plot_arima_sarima_results(y_test, y_pred_arima, y_pred_sarima)
plot_xgboost_results(y_train, y_test, y_pred_train_xgb, y_pred_test_xgb, xgb_model, X_train_scaled)
plot_model_comparison(metrics)