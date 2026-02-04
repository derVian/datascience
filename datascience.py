import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# This will hide only the convergence warnings
warnings.simplefilter('ignore', ConvergenceWarning)


def read_air_passengers_file(file_path='AirPassengers.csv'):
    df = pd.read_csv(file_path)
    df[['Year','Month']] = df['Month'].str.split('-', expand=True).astype(int)
    df = df[['Year', 'Month', '#Passengers']]
    return df

df = read_air_passengers_file()
print("Data types:")
print(df.dtypes)
print("\nNull counts:")
print(df.isnull().sum())
print(f"\n{df.duplicated().sum()} duplicate values found.\n")
print(df.head())

# Initial Data Visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Time series plot
axes[0, 0].plot(df.index, df['#Passengers'], linewidth=2, color='blue')
axes[0, 0].set_title('Air Passengers Over Time', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Month Index')
axes[0, 0].set_ylabel('Number of Passengers')
axes[0, 0].grid(True, alpha=0.3)

# Distribution plot
axes[0, 1].hist(df['#Passengers'], bins=20, color='green', edgecolor='black', alpha=0.7)
axes[0, 1].set_title('Distribution of Passengers', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Number of Passengers')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Box plot by year
df.boxplot(column='#Passengers', by='Year', ax=axes[1, 0])
axes[1, 0].set_title('Passengers by Year', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Year')
axes[1, 0].set_ylabel('Number of Passengers')
plt.sca(axes[1, 0])
plt.xticks(rotation=45)

# Seasonal pattern scatter plot
scatter = axes[1, 1].scatter(df['Month'], df['#Passengers'], 
                             c=df['Year'], cmap='viridis', s=50, alpha=0.6)
axes[1, 1].set_title('Seasonal Pattern (Month vs Passengers)', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Month')
axes[1, 1].set_ylabel('Number of Passengers')
cbar = plt.colorbar(scatter, ax=axes[1, 1])
cbar.set_label('Year')

plt.tight_layout()
plt.show()

# Summary statistics
print("\nData Summary:")
print(df.describe())

# Feature engineering: Create lag features and seasonal indicators
df['lag_1'] = df['#Passengers'].shift(1)
df['lag_12'] = df['#Passengers'].shift(12)
df['year_summer_impact'] = df['Year'] * ((df['Month'] >= 6) & (df['Month'] <= 8)).astype(int)

# Create binary columns for seasons using efficient boolean masking
df['is_winter'] = ((df['Month'] == 12) | (df['Month'] < 3)).astype(int)
df['is_spring'] = ((df['Month'] >= 3) & (df['Month'] < 6)).astype(int)
df['is_summer'] = ((df['Month'] >= 6) & (df['Month'] < 9)).astype(int)
df['is_autumn'] = ((df['Month'] >= 9) & (df['Month'] < 12)).astype(int)

# Reorder and clean columns
df = df[['Year', 'Month', 'is_winter', 'is_spring', 'is_summer', 'is_autumn', 
          'year_summer_impact', 'lag_1', 'lag_12', '#Passengers']]
df = df.dropna()

# Convert lag features to int, then apply log transformation
df[['lag_1', 'lag_12']] = df[['lag_1', 'lag_12']].astype(int)
print(f"\nData types:\n{df.dtypes}")

# Reset index and apply log transformation
df = df.reset_index(drop=True)
log_features = ['#Passengers', 'lag_1', 'lag_12']
df[log_features] = np.log(df[log_features])

# Add trend feature
df['trend'] = np.arange(len(df))
print(f"\nLast 30 rows:\n{df.tail(30)}")



# Correlation heatmap
print("\nGenerating visualizations...")
corr_fig, corr_ax = plt.subplots(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=corr_ax)
corr_ax.set_title('Feature Correlation Matrix', fontsize=12, fontweight='bold')








# ==============================================================================
# DATA PREPARATION
# ==============================================================================

# Chronological Split (No shuffling)
split_point = int(len(df) * 0.8)
train, test = df.iloc[:split_point], df.iloc[split_point:]

# Feature scaling
df_features = ['lag_1', 'lag_12', 'trend', '#Passengers']
scaler = MinMaxScaler()
train[df_features] = scaler.fit_transform(train[df_features])
test[df_features] = scaler.transform(test[df_features])

# Separate Features (X) and Target (y)
X_train = train.drop('#Passengers', axis=1)
y_train = train['#Passengers']
X_test = test.drop('#Passengers', axis=1)
y_test = test['#Passengers']


# ==============================================================================
# MODEL 1: RANDOM FOREST
# ==============================================================================

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred_log = rf_model.predict(X_test)
rf_test_actual = np.exp(y_test)
rf_pred_actual = np.exp(rf_pred_log)
rf_mape = mean_absolute_percentage_error(rf_test_actual, rf_pred_actual)
print(f"Random Forest Error (MAPE): {rf_mape:.2%}")


# ==============================================================================
# MODEL 2: XGBOOST
# ==============================================================================

xgb_model = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=5,
    objective='reg:squarederror',
    random_state=42
)
xgb_model.fit(X_train, y_train)
xgb_pred_scaled = xgb_model.predict(X_test)
xgb_mape = mean_absolute_percentage_error(y_test, xgb_pred_scaled)
print(f"XGBoost Model Error (MAPE): {xgb_mape:.2%}")


# ==============================================================================
# MODEL 3: ARIMA
# ==============================================================================



history_arima = list(train['#Passengers'])
arima_predictions = []

# ARIMA parameters (p, d, q)
arima_order = (1, 1, 1)

for t in range(len(test)):
    try:
        arima_model = ARIMA(history_arima, order=arima_order)
        arima_fit = arima_model.fit()
        yhat = arima_fit.forecast()[0]
        arima_predictions.append(yhat)
        history_arima.append(test['#Passengers'].iloc[t])
    except:
        # Skip problematic predictions
        arima_predictions.append(np.nan)
        history_arima.append(test['#Passengers'].iloc[t])

# Convert back from log scale
arima_test_orig = np.exp(test['#Passengers'].values)
arima_pred_orig = np.exp(arima_predictions)

# Calculate ARIMA MAPE (excluding NaN values)
valid_mask_arima = ~np.isnan(arima_pred_orig)
if valid_mask_arima.sum() > 0:
    arima_mape = mean_absolute_percentage_error(arima_test_orig[valid_mask_arima], np.array(arima_pred_orig)[valid_mask_arima])
else:
    arima_mape = np.nan
print(f"ARIMA Model Error (MAPE): {arima_mape:.2%}")


# ==============================================================================
# MODEL 4: SARIMA
# ==============================================================================

history = list(train['#Passengers'])
sarima_predictions = []

# Pre-compile model specification to avoid repeated parsing
model_spec = {'order': (1, 1, 1), 'seasonal_order': (1, 1, 0, 12), 'disp': False, 'maxiter': 200}

for t in range(len(test)):
    try:
        model_fit = SARIMAX(history, **model_spec).fit()
    except:
        # Fallback to simpler method if main method fails
        try:
            model_fit = SARIMAX(history, **model_spec).fit(method='nm')
        except:
            # Skip problematic predictions
            sarima_predictions.append(np.nan)
            history.append(test['#Passengers'].iloc[t])
            continue
    
    yhat = model_fit.forecast()[0]
    sarima_predictions.append(yhat)
    history.append(test['#Passengers'].iloc[t])

# Convert back from log scale
sarima_test_orig = np.exp(test['#Passengers'].values)
sarima_pred_orig = np.exp(sarima_predictions)

# Calculate SARIMA MAPE (excluding NaN values)
valid_mask = ~np.isnan(sarima_pred_orig)
if valid_mask.sum() > 0:
    sarima_mape = mean_absolute_percentage_error(sarima_test_orig[valid_mask], np.array(sarima_pred_orig)[valid_mask])
else:
    sarima_mape = np.nan
print(f"SARIMA Model Error (MAPE): {sarima_mape:.2%}")


# ==============================================================================
# VISUALIZATIONS
# ==============================================================================

# Create a figure with 9 subplots (4 model comparisons + 5 analysis plots)
fig = plt.figure(figsize=(20, 18))

# Random Forest Plot
ax1 = plt.subplot(3, 3, 1)
ax1.plot(test.index, rf_test_actual, label='Actual Passengers', color='blue', linewidth=2)
ax1.plot(test.index, rf_pred_actual, label='Random Forest Prediction', color='red', linestyle='--')
ax1.scatter(test.index, rf_pred_actual, color='red', marker='x', s=40, label='RF Prediction (x)')
ax1.set_title('Random Forest: Actual vs Predicted', fontsize=12, fontweight='bold')
ax1.set_xlabel('Time Index')
ax1.set_ylabel('Passengers')
ax1.legend()
ax1.grid(True, alpha=0.3)

# XGBoost Plot
ax2 = plt.subplot(3, 3, 2)
ax2.scatter(y_test.index, y_test, color='blue', label='Actual (Log Scale)', alpha=0.7)
ax2.scatter(y_test.index, xgb_pred_scaled, color='red', label='XGBoost Prediction', marker='x')
ax2.plot(y_test.index, y_test, color='blue', linewidth=2)
ax2.plot(y_test.index, xgb_pred_scaled, color='red', linewidth=2)
ax2.set_title('XGBoost: Actual vs Predicted', fontsize=12, fontweight='bold')
ax2.set_xlabel('Index')
ax2.set_ylabel('Passengers (Log Scale)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# ARIMA Plot
ax3 = plt.subplot(3, 3, 3)
ax3.plot(test.index, arima_test_orig, color='blue', marker='o', linestyle='-', linewidth=1, alpha=0.3, label='Actual Path')
ax3.scatter(test.index, arima_test_orig, color='blue', label='Actual Passengers')
ax3.plot(test.index, arima_pred_orig, color='green', marker='x', linestyle='--', linewidth=1, alpha=0.4, label='ARIMA Path')
ax3.scatter(test.index, arima_pred_orig, color='green', marker='x', label='ARIMA Prediction')
ax3.set_title('ARIMA: Real Passenger Counts', fontsize=12, fontweight='bold')
ax3.set_xlabel('Time Index')
ax3.set_ylabel('Total Passengers')
ax3.legend()
ax3.grid(True, alpha=0.3)

# SARIMA Plot
ax4 = plt.subplot(3, 3, 4)
ax4.plot(test.index, sarima_test_orig, color='blue', marker='o', linestyle='-', linewidth=1, alpha=0.3, label='Actual Path')
ax4.scatter(test.index, sarima_test_orig, color='blue', label='Actual Passengers')
ax4.plot(test.index, sarima_pred_orig, color='red', marker='x', linestyle='--', linewidth=1, alpha=0.4, label='SARIMA Path')
ax4.scatter(test.index, sarima_pred_orig, color='red', marker='x', label='SARIMA Prediction')
ax4.set_title('SARIMA: Real Passenger Counts', fontsize=12, fontweight='bold')
ax4.set_xlabel('Time Index')
ax4.set_ylabel('Total Passengers')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Model Performance Comparison
ax5 = plt.subplot(3, 3, 5)
models = ['Random Forest', 'XGBoost', 'ARIMA', 'SARIMA']
mape_scores = [rf_mape * 100, xgb_mape * 100, arima_mape * 100 if not np.isnan(arima_mape) else 0, sarima_mape * 100 if not np.isnan(sarima_mape) else 0]
colors = ['#1f77b4', '#ff7f0e', '#9467bd', '#2ca02c']
bars = ax5.bar(models, mape_scores, color=colors, alpha=0.7, edgecolor='black')
ax5.set_title('Model Performance (MAPE)', fontsize=12, fontweight='bold')
ax5.set_ylabel('MAPE (%)')
ax5.set_ylim(0, max(mape_scores) * 1.2 if max(mape_scores) > 0 else 10)
for bar, score in zip(bars, mape_scores):
    if score > 0:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.2f}%', ha='center', va='bottom', fontweight='bold')
ax5.grid(True, alpha=0.3, axis='y')

# Feature Correlation Heatmap
ax6 = plt.subplot(3, 3, 6)
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap="coolwarm", ax=ax6, cbar_kws={'shrink': 0.8})
ax6.set_title('Feature Correlation Matrix', fontsize=12, fontweight='bold')

# Residuals Analysis for Random Forest
ax7 = plt.subplot(3, 3, 7)
rf_residuals = rf_test_actual - rf_pred_actual
rf_residuals_pct = ((rf_test_actual - rf_pred_actual) / rf_test_actual) * 100
rf_within_pct = (np.abs(rf_residuals_pct) <= 10).mean() * 100
ax7.hist(rf_residuals, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
ax7.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
ax7.set_title('Random Forest Residuals Distribution', fontsize=12, fontweight='bold')
ax7.set_xlabel('Residuals')
ax7.set_ylabel('Frequency')
ax7.legend()
ax7.text(0.02, 0.95,
    f'{rf_within_pct:.1f}% of predictions within ±10% of actual',
    transform=ax7.transAxes, va='top', fontsize=9)
ax7.grid(True, alpha=0.3, axis='y')

# Residuals Analysis for ARIMA
ax8 = plt.subplot(3, 3, 8)
arima_residuals = arima_test_orig - arima_pred_orig
arima_residuals_pct = ((arima_test_orig - arima_pred_orig) / arima_test_orig) * 100
arima_within_pct = (np.abs(arima_residuals_pct) <= 10).mean() * 100
ax8.hist(arima_residuals, bins=20, color='purple', edgecolor='black', alpha=0.7)
ax8.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
ax8.set_title('ARIMA Residuals Distribution', fontsize=12, fontweight='bold')
ax8.set_xlabel('Residuals')
ax8.set_ylabel('Frequency')
ax8.legend()
ax8.text(0.02, 0.95,
    f'{arima_within_pct:.1f}% of predictions within ±10% of actual',
    transform=ax8.transAxes, va='top', fontsize=9)
ax8.grid(True, alpha=0.3, axis='y')

# Residuals Analysis for SARIMA
ax9 = plt.subplot(3, 3, 9)
sarima_residuals = sarima_test_orig - sarima_pred_orig
sarima_residuals_pct = ((sarima_test_orig - sarima_pred_orig) / sarima_test_orig) * 100
sarima_within_pct = (np.abs(sarima_residuals_pct) <= 10).mean() * 100
ax9.hist(sarima_residuals, bins=20, color='green', edgecolor='black', alpha=0.7)
ax9.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
ax9.set_title('SARIMA Residuals Distribution', fontsize=12, fontweight='bold')
ax9.set_xlabel('Residuals')
ax9.set_ylabel('Frequency')
ax9.legend()
ax9.text(0.02, 0.95,
    f'{sarima_within_pct:.1f}% of predictions within ±10% of actual',
    transform=ax9.transAxes, va='top', fontsize=9)
ax9.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()
plt.show()