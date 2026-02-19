import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings('ignore')

def load_data(filepath):
    """Load solar data from CSV file."""
    df = pd.read_csv(filepath, skiprows=2)
    return df

def preprocess_data(df):
    """Preprocess solar data: create DateTime index and handle Fill Flag."""
    df['DateTime'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']], errors='coerce')
    return df

def filter_ghi_data(df, threshold=5):
    """Filter rows where GHI >= threshold and reset index."""
    initial_rows = len(df)
    df = df[df['GHI'] >= threshold].copy()
    df.reset_index(drop=True, inplace=True)
    filtered_rows = len(df)
    print(f"Filtered {initial_rows - filtered_rows} rows where GHI < {threshold}")
    print(f"Remaining rows: {filtered_rows}")
    return df

def add_season_column(df):
    """Add a season column based on month."""
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        elif month in [9, 10, 11]:
            return 'Fall'
        else:
            return 'Unknown'
    
    df['Season'] = df['Month'].apply(get_season)
    
    print("\nSeason distribution:")
    print(df['Season'].value_counts().sort_index())
    
    return df

def plot_ghi_distribution_by_season(df, save_path='./resultpngs/ghi_distribution_by_season.png'):
    """Create and save histograms of GHI distribution for each season."""
    seasons = ['Winter', 'Spring', 'Summer', 'Fall']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, season in enumerate(seasons):
        season_data = df[df['Season'] == season]['GHI']
        
        if len(season_data) > 0:
            axes[idx].hist(season_data, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
            axes[idx].set_xlabel('GHI (W/m²)', fontsize=10)
            axes[idx].set_ylabel('Frequency', fontsize=10)
            axes[idx].set_title(f'{season} - GHI Distribution', fontsize=12, fontweight='bold')
            axes[idx].grid(axis='y', alpha=0.3)
            
            # Add statistics
            mean_val = season_data.mean()
            median_val = season_data.median()
            axes[idx].axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
            axes[idx].axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
            axes[idx].legend(fontsize=8)
    
    plt.suptitle('GHI Distribution by Season', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nGHI Statistics by Season:")
    for season in seasons:
        season_data = df[df['Season'] == season]['GHI']
        if len(season_data) > 0:
            print(f"\n{season}:")
            print(f"  Count: {len(season_data)}")
            print(f"  Mean: {season_data.mean():.2f}")
            print(f"  Median: {season_data.median():.2f}")
            print(f"  Std Dev: {season_data.std():.2f}")
            print(f"  Min: {season_data.min():.2f}")
            print(f"  Max: {season_data.max():.2f}")

def plot_ghi_boxplot_by_season(df, save_path='./resultpngs/ghi_boxplot_by_season.png'):
    """Create and save box plot of GHI by season."""
    plt.figure(figsize=(12, 6))
    
    # Create box plot
    seasons_order = ['Winter', 'Spring', 'Summer', 'Fall']
    ax = sns.boxplot(data=df, x='Season', y='GHI', order=seasons_order, palette='Set2')
    
    plt.xlabel('Season', fontsize=12, fontweight='bold')
    plt.ylabel('GHI (W/m^2)', fontsize=12, fontweight='bold')
    plt.title('GHI Distribution by Season (Box Plot)', fontsize=14, fontweight='bold', pad=15)
    plt.grid(axis='y', alpha=0.3)

    # Annotate quartile stats on the plot
    y_max = df['GHI'].max() if len(df) > 0 else 0
    y_offset = max(y_max * 0.02, 1)
    for i, season in enumerate(seasons_order):
        season_data = df[df['Season'] == season]['GHI']
        if len(season_data) > 0:
            q1 = season_data.quantile(0.25)
            q2 = season_data.quantile(0.50)
            q3 = season_data.quantile(0.75)
            iqr = q3 - q1
            label = f"Q1:{q1:.1f}\nQ2:{q2:.1f}\nQ3:{q3:.1f}\nIQR:{iqr:.1f}"
            ax.text(i, q3 + y_offset, label, ha='center', va='bottom', fontsize=8, color='black')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nBox Plot Statistics:")
    for season in seasons_order:
        season_data = df[df['Season'] == season]['GHI']
        if len(season_data) > 0:
            q1 = season_data.quantile(0.25)
            q2 = season_data.quantile(0.50)
            q3 = season_data.quantile(0.75)
            iqr = q3 - q1
            print(f"\n{season}:")
            print(f"  Q1 (25%): {q1:.2f}")
            print(f"  Q2 (Median): {q2:.2f}")
            print(f"  Q3 (75%): {q3:.2f}")
            print(f"  IQR: {iqr:.2f}")

def plot_temperature_vs_ghi(df, save_path='./resultpngs/temperature_vs_ghi.png'):
    """Create scatter plot of Temperature vs GHI and analyze their relationship."""
    required_cols = {'Temperature', 'GHI'}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        print(f"Skipping Temperature vs GHI plot. Missing columns: {sorted(missing_cols)}")
        return

    valid_mask = df['Temperature'].notna() & df['GHI'].notna()
    if valid_mask.sum() < 2:
        print("Skipping Temperature vs GHI plot. Not enough valid data points.")
        return

    temp_vals = df.loc[valid_mask, 'Temperature']
    ghi_vals = df.loc[valid_mask, 'GHI']

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Scatter plot
    ax1.scatter(temp_vals, ghi_vals, alpha=0.3, s=10, c='steelblue', edgecolor='none')
    ax1.set_xlabel('Temperature (deg C)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('GHI (W/m^2)', fontsize=12, fontweight='bold')
    ax1.set_title('Temperature vs GHI Relationship', fontsize=14, fontweight='bold', pad=15)
    ax1.grid(alpha=0.3)

    # Add trend line
    z = np.polyfit(temp_vals, ghi_vals, 1)
    p = np.poly1d(z)
    temp_sorted = np.sort(temp_vals)
    ax1.plot(temp_sorted, p(temp_sorted), "r--", linewidth=2, label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
    ax1.legend()

    # Scatter plot colored by season (if available)
    seasons_order = ['Winter', 'Spring', 'Summer', 'Fall']
    colors = {'Winter': 'blue', 'Spring': 'green', 'Summer': 'red', 'Fall': 'orange'}
    if 'Season' in df.columns:
        for season in seasons_order:
            season_data = df[df['Season'] == season]
            ax2.scatter(season_data['Temperature'], season_data['GHI'],
                        alpha=0.4, s=15, c=colors[season], label=season, edgecolor='none')
        ax2.legend()
    else:
        ax2.scatter(temp_vals, ghi_vals, alpha=0.3, s=10, c='steelblue', edgecolor='none')

    ax2.set_xlabel('Temperature (deg C)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('GHI (W/m^2)', fontsize=12, fontweight='bold')
    ax2.set_title('Temperature vs GHI by Season', fontsize=14, fontweight='bold', pad=15)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    # Calculate correlation
    correlation = temp_vals.corr(ghi_vals)

    print("\n" + "=" * 80)
    print("TEMPERATURE vs GHI RELATIONSHIP ANALYSIS")
    print("=" * 80)
    print(f"\nPearson Correlation Coefficient: {correlation:.4f}")

    # Interpret correlation
    if correlation > 0.7:
        strength = "strong positive"
    elif correlation > 0.4:
        strength = "moderate positive"
    elif correlation > 0.2:
        strength = "weak positive"
    elif correlation > -0.2:
        strength = "very weak or no"
    elif correlation > -0.4:
        strength = "weak negative"
    elif correlation > -0.7:
        strength = "moderate negative"
    else:
        strength = "strong negative"

    print(f"Correlation Strength: {strength.upper()}")

    print("\nINTERPRETATION:")
    print("-" * 80)

    if correlation > 0.4:
        print("OK: POSITIVE correlation between Temperature and GHI.")
        print("OK: As GHI increases, temperature tends to increase.")
        print("OK: This is expected (more sunlight -> more heating).")
        print("\nKey Insights:")
        print("  - Higher GHI values align with warmer temperatures")
        print("  - Solar radiation is a primary driver of surface temperature")
        print("  - The relationship varies by season (see seasonal plot)")
        print("  - Maximum GHI occurs during summer with highest temperatures")
    elif correlation < -0.2:
        print("OK: NEGATIVE correlation between Temperature and GHI.")
        print("OK: This is unusual and may indicate:")
        print("  - Cloud cover reducing GHI while trapping heat")
        print("  - Time of day effects (evening warmth vs. decreasing sunlight)")
    else:
        print("OK: WEAK or NO linear correlation between Temperature and GHI.")
        print("OK: This may be due to:")
        print("  - Other factors affecting temperature (wind, humidity, time lag)")
        print("  - Thermal inertia (temperature responds slowly to radiation changes)")

    # Seasonal analysis
    if 'Season' in df.columns:
        print("\n" + "-" * 80)
        print("SEASONAL CORRELATION ANALYSIS:")
        print("-" * 80)
        for season in seasons_order:
            season_data = df[df['Season'] == season]
            if len(season_data) > 0:
                season_corr = season_data['Temperature'].corr(season_data['GHI'])
                print(f"{season:8s}: Correlation = {season_corr:.4f}")

    print("=" * 80 + "\n")

def plot_ghi_distribution(df, save_path='./resultpngs/ghi_distribution_after_filtering_zeros.png'):
    """Create and save histogram of GHI distribution."""
    plt.figure(figsize=(10, 6))
    plt.hist(df['GHI'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.xlabel('GHI (W/m²)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Global Horizontal Irradiance (GHI) after filtering out zero values', fontsize=14, pad=15)
    plt.grid(axis='y', alpha=0.3)
    
    # Add statistics to the plot
    mean_val = df['GHI'].mean()
    median_val = df['GHI'].median()
    plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
    plt.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nGHI Distribution Statistics:")
    print(f"  Mean: {mean_val:.2f}")
    print(f"  Median: {median_val:.2f}")
    print(f"  Std Dev: {df['GHI'].std():.2f}")
    print(f"  Min: {df['GHI'].min():.2f}")
    print(f"  Max: {df['GHI'].max():.2f}")

def create_correlation_heatmap(df, save_path='./resultpngs/correlation_heatmap.png'):
    """Create and save correlation heatmap for numeric features."""
    numeric_df = df.drop(columns=['Fill Flag','Year', 'Month', 'Day', 'Hour', 'Minute'], errors='ignore')
    numeric_df = numeric_df.select_dtypes(include=[np.number])
    
    correlation_matrix = numeric_df.corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Heatmap of Solar Data Features', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(correlation_matrix)
    return correlation_matrix

def normalize_data(df):
    """Apply normalization methods to GHI column."""
    df['GHI_MinMax'] = (df['GHI'] - df['GHI'].min()) / (df['GHI'].max() - df['GHI'].min())
    df['GHI_ZScore'] = (df['GHI'] - df['GHI'].mean()) / df['GHI'].std()
    return df

def perform_adf_test(series, name):
    """Perform Augmented Dickey-Fuller test for stationarity."""
    result = adfuller(series.dropna())
    print(f"\n{name} - Augmented Dickey-Fuller Test Results:")
    print(f"  ADF Statistic: {result[0]:.6f}")
    print(f"  p-value: {result[1]:.6f}")
    print(f"  Critical Values:")
    for key, value in result[4].items():
        print(f"    {key}: {value:.3f}")
    
    if result[1] < 0.05:
        print(f"  >>> Result: STATIONARY (p-value < 0.05)")
    else:
        print(f"  >>> Result: NON-STATIONARY (p-value >= 0.05)")
    
    return {
        'Column': name,
        'ADF Statistic': result[0],
        'p-value': result[1],
        'Stationary': 'Yes' if result[1] < 0.05 else 'No'
    }

def create_arima_dataframe(df):
    
    # Select required columns
    required_cols = ['DateTime', 'Temperature', 'GHI']
    
    # Check if all required columns exist
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"WARNING: Missing columns: {missing_cols}")
        print("Available columns:", df.columns.tolist())
        return None
    
    # Create new dataframe with selected attributes
    df_arima = df[required_cols].copy()
    
    # Remove rows with missing values
    initial_rows = len(df_arima)
    df_arima = df_arima.dropna()
    removed_rows = initial_rows - len(df_arima)
    
    # Sort by DateTime
    df_arima = df_arima.sort_values('DateTime').reset_index(drop=True)
    
    # Print summary
    print("\n" + "="*80)
    print("ARIMA TRAINING DATAFRAME CREATED")
    print("="*80)
    print(f"Total rows: {len(df_arima)}")
    print(f"Rows removed (missing values): {removed_rows}")
    print(f"\nDataframe shape: {df_arima.shape}")
    print(f"\nDataframe information:")
    print(df_arima.info())
    print(f"\nFirst 5 rows:")
    print(df_arima.head())
    print(f"\nLast 5 rows:")
    print(df_arima.tail())
    print(f"\nStatistical Summary:")
    print(df_arima.describe())
    print("="*80 + "\n")
    
    return df_arima

def plot_timeseries_with_seasonality(df, save_path='./resultpngs/timeseries_seasonality.png'):
    """Create a time-series plot showing seasonality of GHI data."""
    # Sort by DateTime to ensure proper time-series ordering
    df_sorted = df.sort_values('DateTime').reset_index(drop=True)
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    # Plot 1: Full time-series with season coloring
    seasons_order = ['Winter', 'Spring', 'Summer', 'Fall']
    colors = {'Winter': '#1f77b4', 'Spring': '#2ca02c', 'Summer': '#ff7f0e', 'Fall': '#d62728'}
    
    for season in seasons_order:
        season_data = df_sorted[df_sorted['Season'] == season]
        axes[0].scatter(season_data['DateTime'], season_data['GHI'], 
                       alpha=0.4, s=5, c=colors[season], label=season, edgecolor='none')
    
    axes[0].set_xlabel('DateTime', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('GHI (W/m²)', fontsize=12, fontweight='bold')
    axes[0].set_title('Time-Series of Solar Irradiance (GHI) with Seasonal Coloring', 
                     fontsize=14, fontweight='bold', pad=15)
    axes[0].legend(loc='upper right', fontsize=11)
    axes[0].grid(True, alpha=0.3, linestyle='--')
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot 2: Daily average GHI with seasonality (to see cleaner pattern)
    daily_avg = df_sorted.groupby(df_sorted['DateTime'].dt.date).agg({'GHI': 'mean'}).reset_index()
    daily_avg.columns = ['Date', 'Mean_GHI']
    daily_avg['Date'] = pd.to_datetime(daily_avg['Date'])
    
    # Add season column to daily data
    daily_avg['Month'] = daily_avg['Date'].dt.month
    daily_avg['Season'] = daily_avg['Month'].apply(lambda m: 
        'Winter' if m in [12, 1, 2] else
        'Spring' if m in [3, 4, 5] else
        'Summer' if m in [6, 7, 8] else 'Fall')
    
    for season in seasons_order:
        season_daily = daily_avg[daily_avg['Season'] == season]
        axes[1].plot(season_daily['Date'], season_daily['Mean_GHI'], 
                    marker='o', markersize=4, linewidth=1.5, color=colors[season], 
                    label=season, alpha=0.8)
    
    axes[1].set_xlabel('DateTime', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Daily Average GHI (W/m²)', fontsize=12, fontweight='bold')
    axes[1].set_title('Daily Average Trend - Seasonal Patterns Visualization', 
                     fontsize=14, fontweight='bold', pad=15)
    axes[1].legend(loc='upper right', fontsize=11)
    axes[1].grid(True, alpha=0.3, linestyle='--')
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print seasonality statistics
    print("\n" + "="*80)
    print("TIME-SERIES SEASONALITY ANALYSIS")
    print("="*80)
    
    print("\nSeasonal GHI Statistics:")
    print("-"*80)
    for season in seasons_order:
        season_data = df_sorted[df_sorted['Season'] == season]['GHI']
        print(f"\n{season}:")
        print(f"  Mean GHI: {season_data.mean():.2f} W/m²")
        print(f"  Median GHI: {season_data.median():.2f} W/m²")
        print(f"  Std Dev: {season_data.std():.2f} W/m²")
        print(f"  Min: {season_data.min():.2f} W/m²")
        print(f"  Max: {season_data.max():.2f} W/m²")
        print(f"  Total Data Points: {len(season_data)}")
    
    # Calculate seasonal pattern strength
    overall_mean = df_sorted['GHI'].mean()
    seasonal_means = [df_sorted[df_sorted['Season'] == s]['GHI'].mean() for s in seasons_order]
    seasonal_variance = np.var(seasonal_means)
    
    print("\n" + "-"*80)
    print("Seasonality Strength:")
    print(f"  Overall Mean GHI: {overall_mean:.2f} W/m²")
    print(f"  Seasonal Variance: {seasonal_variance:.2f}")
    print(f"  Seasonal Range: {max(seasonal_means) - min(seasonal_means):.2f} W/m²")
    print(f"  Peak Season: {seasons_order[np.argmax(seasonal_means)]} ({max(seasonal_means):.2f} W/m²)")
    print(f"  Lowest Season: {seasons_order[np.argmin(seasonal_means)]} ({min(seasonal_means):.2f} W/m²)")
    
    print("="*80 + "\n")
    
    return daily_avg

def train_arima_model(df_arima, order=(1, 0, 0), target_col='GHI', test_size=0.2):
    """
    Train an ARIMA model on the provided dataframe with train/test split.
    
    Parameters:
    -----------
    df_arima : pd.DataFrame
        Dataframe with DateTime and target column (GHI)
    order : tuple
        ARIMA order (p, d, q) - defaults to (1, 0, 0)
    target_col : str
        Name of the column to forecast (default: 'GHI')
    test_size : float
        Proportion of data to use for testing (default: 0.2 = 20%)
    
    Returns:
    --------
    model : ARIMA fitted model (trained on train set)
    results : ARIMA fit results object
    metrics : dict with RMSE, MAE, and test set predictions
    """
    
    if df_arima is None or len(df_arima) == 0:
        return None, None, None
    
    # Split data into train and test sets
    split_idx = int(len(df_arima) * (1 - test_size))
    train_data = df_arima[target_col][:split_idx]
    test_data = df_arima[target_col][split_idx:]
    
    # Train ARIMA model on training set
    try:
        arima_model = ARIMA(train_data, order=order)
        arima_results = arima_model.fit()
        
        # Make predictions on test set
        predictions = arima_results.get_forecast(steps=len(test_data)).predicted_mean
        
        # Calculate metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        rmse = np.sqrt(mean_squared_error(test_data, predictions))
        mae = mean_absolute_error(test_data, predictions)
        
        # Calculate correlation
        correlation = np.corrcoef(test_data, predictions)[0, 1]
        
        metrics = {
            'test_rmse': rmse,
            'test_mae': mae,
            'train_rmse': np.sqrt(arima_results.mse),
            'aic': arima_results.aic,
            'bic': arima_results.bic,
            'correlation': correlation,
            'actual': test_data.values,
            'predicted': predictions.values,
            'test_indices': test_data.index
        }
        
        return arima_model, arima_results, metrics
        
    except Exception as e:
        return None, None, None

def train_sarima_model(
    df_arima,
    order,
    seasonal_order,
    target_col='GHI',
    test_size=0.2,
):
    """
    Train a SARIMAX model on the provided dataframe with train/test split.
    
    Parameters:
    -----------
    df_arima : pd.DataFrame
        Dataframe with DateTime and target column
    order : tuple
        ARIMA order (p, d, q)
    seasonal_order : tuple
        Seasonal order (P, D, Q, s) for 9 samples/day
    target_col : str
        Name of the column to forecast (default: 'GHI')
    test_size : float
        Proportion of data to use for testing (default: 0.2 = 20%)
    
    Returns:
    --------
    model : SARIMAX fitted model (trained on train set)
    results : SARIMAX fit results object
    metrics : dict with RMSE, MAE, and test set predictions
    """
    
    if df_arima is None or len(df_arima) == 0:
        return None, None, None
    if order is None or seasonal_order is None:
        return None, None, None
    
    # Split data into train and test sets
    split_idx = int(len(df_arima) * (1 - test_size))
    train_data = df_arima[target_col][:split_idx]
    test_data = df_arima[target_col][split_idx:]
    
    # Train SARIMA model on training set
    try:
        # Use improved SARIMA order: (1,1,1)x(1,1,1,9) - better than original (1,0,0)x(1,0,1,9)
        sarima_model = SARIMAX(
            train_data,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        sarima_results = sarima_model.fit(disp=False)
        
        # Make predictions on test set
        predictions = sarima_results.get_forecast(steps=len(test_data)).predicted_mean
        
        # Calculate metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        rmse = np.sqrt(mean_squared_error(test_data, predictions))
        mae = mean_absolute_error(test_data, predictions)
        
        # Calculate correlation
        correlation = np.corrcoef(test_data, predictions)[0, 1]
        
        metrics = {
            'test_rmse': rmse,
            'test_mae': mae,
            'train_rmse': np.sqrt(sarima_results.mse),
            'aic': sarima_results.aic,
            'bic': sarima_results.bic,
            'correlation': correlation,
            'actual': test_data.values,
            'predicted': predictions.values,
            'test_indices': test_data.index
        }
        
        return sarima_model, sarima_results, metrics
        
    except Exception as e:
        return None, None, None

def train_sarimax_model(
    df_arima,
    order,
    seasonal_order,
    target_col='GHI',
    exog_cols=None,
    test_size=0.2,
):
    """
    Train a SARIMAX model on the provided dataframe with train/test split.
    
    Parameters:
    -----------
    df_arima : pd.DataFrame
        Dataframe with DateTime, target, and exogenous columns
    order : tuple
        ARIMA order (p, d, q)
    seasonal_order : tuple
        Seasonal order (P, D, Q, s) for 9 samples/day
    target_col : str
        Name of the column to forecast (default: 'GHI')
    exog_cols : list[str] | None
        Exogenous columns to include (default: ['Temperature'])
    test_size : float
        Proportion of data to use for testing (default: 0.2 = 20%)
    
    Returns:
    --------
    model : SARIMAX fitted model (trained on train set)
    results : SARIMAX fit results object
    metrics : dict with RMSE, MAE, and test set predictions
    """

    if df_arima is None or len(df_arima) == 0:
        return None, None, None
    if order is None or seasonal_order is None:
        return None, None, None

    if exog_cols is None:
        exog_cols = ['Temperature']

    missing_exog = [col for col in exog_cols if col not in df_arima.columns]
    if missing_exog:
        return None, None, None

    # Split data into train and test sets
    split_idx = int(len(df_arima) * (1 - test_size))
    train_data = df_arima[target_col][:split_idx]
    test_data = df_arima[target_col][split_idx:]
    train_exog = df_arima[exog_cols][:split_idx]
    test_exog = df_arima[exog_cols][split_idx:]

    # Train SARIMAX model on training set
    try:
        sarimax_model = SARIMAX(
            train_data,
            exog=train_exog,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        sarimax_results = sarimax_model.fit(disp=False)

        # Make predictions on test set
        predictions = sarimax_results.get_forecast(
            steps=len(test_data),
            exog=test_exog,
        ).predicted_mean

        # Calculate metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        rmse = np.sqrt(mean_squared_error(test_data, predictions))
        mae = mean_absolute_error(test_data, predictions)

        # Calculate correlation
        correlation = np.corrcoef(test_data, predictions)[0, 1]

        metrics = {
            'test_rmse': rmse,
            'test_mae': mae,
            'train_rmse': np.sqrt(sarimax_results.mse),
            'aic': sarimax_results.aic,
            'bic': sarimax_results.bic,
            'correlation': correlation,
            'actual': test_data.values,
            'predicted': predictions.values,
            'test_indices': test_data.index
        }

        return sarimax_model, sarimax_results, metrics

    except Exception as e:
        return None, None, None

def compare_arima_sarima(arima_metrics, sarima_metrics, sarimax_metrics=None):
    if arima_metrics is None or sarima_metrics is None:
        print("ERROR: One or both metric dictionaries are None. Cannot compare.")
        return

    models = ['ARIMA', 'SARIMA']
    training_rmse = [arima_metrics['train_rmse'], sarima_metrics['train_rmse']]
    test_rmse = [arima_metrics['test_rmse'], sarima_metrics['test_rmse']]
    test_mae = [arima_metrics['test_mae'], sarima_metrics['test_mae']]
    test_corr = [arima_metrics['correlation'], sarima_metrics['correlation']]
    aic_vals = [arima_metrics['aic'], sarima_metrics['aic']]
    bic_vals = [arima_metrics['bic'], sarima_metrics['bic']]

    if sarimax_metrics is not None:
        models.append('SARIMAX')
        training_rmse.append(sarimax_metrics['train_rmse'])
        test_rmse.append(sarimax_metrics['test_rmse'])
        test_mae.append(sarimax_metrics['test_mae'])
        test_corr.append(sarimax_metrics['correlation'])
        aic_vals.append(sarimax_metrics['aic'])
        bic_vals.append(sarimax_metrics['bic'])

    comparison_dict = {
        'Model': models,
        'Training RMSE': training_rmse,
        'Test RMSE': test_rmse,
        'Test MAE': test_mae,
        'Test Correlation': test_corr,
        'AIC': aic_vals,
        'BIC': bic_vals
    }

    comparison_df = pd.DataFrame(comparison_dict)
    print("\n" + comparison_df.to_string(index=False))

def plot_actual_vs_prediction(arima_metrics, sarima_metrics, sarimax_metrics=None, save_path='./resultpngs/actual_vs_prediction.png'):
    if arima_metrics is None or sarima_metrics is None:
        return

    actual_values = arima_metrics['actual']
    plot_items = [
        ('ARIMA', arima_metrics['predicted'], '#1f77b4'),
        ('SARIMA', sarima_metrics['predicted'], '#ff7f0e'),
    ]
    if sarimax_metrics is not None:
        plot_items.append(('SARIMAX', sarimax_metrics['predicted'], '#2ca02c'))

    fig, axes = plt.subplots(1, len(plot_items), figsize=(8 * len(plot_items), 7))
    if len(plot_items) == 1:
        axes = [axes]

    min_val = min(actual_values.min(), min([p.min() for _, p, _ in plot_items]))
    max_val = max(actual_values.max(), max([p.max() for _, p, _ in plot_items]))

    for ax, (name, pred, color) in zip(axes, plot_items):
        ax.scatter(actual_values, pred, alpha=0.4, s=20, color=color, edgecolor='none', label='Predictions')
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction (y=x)')
        ax.set_xlabel('Actual GHI (W/m²)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Predicted GHI (W/m²)', fontsize=12, fontweight='bold')
        ax.set_title(f'{name}: Actual vs Prediction', fontsize=13, fontweight='bold', pad=15)
        ax.legend(fontsize=11, loc='upper left')
        ax.grid(True, alpha=0.3)

        mape = np.mean(np.abs((actual_values - pred) / actual_values)) * 100
        mae = np.mean(np.abs(actual_values - pred))
        ax.text(0.98, 0.02, f'MAPE: {mape:.2f}%\nMAE: {mae:.2f}',
                transform=ax.transAxes, fontsize=10, verticalalignment='bottom',
                horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.suptitle('Actual Data vs Model Predictions (Test Set Scatter Plot)', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
   
def plot_residuals(arima_metrics, sarima_metrics, sarimax_metrics=None, save_path='./resultpngs/residuals_plot.png'):
    if arima_metrics is None or sarima_metrics is None:
        return

    residuals = {
        'ARIMA': (arima_metrics['actual'] - arima_metrics['predicted'], '#1f77b4'),
        'SARIMA': (sarima_metrics['actual'] - sarima_metrics['predicted'], '#ff7f0e'),
    }
    if sarimax_metrics is not None:
        residuals['SARIMAX'] = (sarimax_metrics['actual'] - sarimax_metrics['predicted'], '#2ca02c')

    n_models = len(residuals)
    fig, axes = plt.subplots(n_models, 2, figsize=(16, 5 * n_models))
    if n_models == 1:
        axes = [axes]

    for row_idx, (name, (resid, color)) in enumerate(residuals.items()):
        ax_time = axes[row_idx][0]
        ax_hist = axes[row_idx][1]
        ax_time.plot(resid, label=f'{name} Residuals', color=color, linewidth=0.8, alpha=0.8)
        ax_time.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax_time.fill_between(range(len(resid)), resid, alpha=0.3, color=color)
        ax_time.set_ylabel('Residuals (W/m²)', fontsize=11, fontweight='bold')
        ax_time.set_xlabel('Test Sample Index', fontsize=11, fontweight='bold')
        ax_time.set_title(f'{name}: Test Set Residuals Over Time', fontsize=12, fontweight='bold')
        ax_time.legend(fontsize=10)
        ax_time.grid(True, alpha=0.3)

        ax_hist.hist(resid, bins=50, color=color, edgecolor='black', alpha=0.7, label=f'{name} Residuals')
        ax_hist.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Mean=0')
        ax_hist.axvline(x=resid.mean(), color='green', linestyle='--', linewidth=2, label=f'Actual Mean={resid.mean():.2f}')
        ax_hist.set_xlabel('Residual Value (W/m²)', fontsize=11, fontweight='bold')
        ax_hist.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax_hist.set_title(f'{name}: Test Set Residuals Distribution', fontsize=12, fontweight='bold')
        ax_hist.legend(fontsize=9)
        ax_hist.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Residual Analysis (Test Set)', fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_detailed_comparison(arima_metrics, sarima_metrics, sarimax_metrics=None, save_path='./resultpngs/detailed_comparison.png'):
    if arima_metrics is None or sarima_metrics is None:
        return

    # Get predictions and actual values
    arima_pred = arima_metrics['predicted']
    sarima_pred = sarima_metrics['predicted']
    actual_values = arima_metrics['actual']
    sarimax_pred = sarimax_metrics['predicted'] if sarimax_metrics is not None else None

    # Calculate metrics
    arima_rmse = arima_metrics['test_rmse']
    sarima_rmse = sarima_metrics['test_rmse']
    arima_mae = arima_metrics['test_mae']
    sarima_mae = sarima_metrics['test_mae']
    sarimax_rmse = sarimax_metrics['test_rmse'] if sarimax_metrics is not None else None
    sarimax_mae = sarimax_metrics['test_mae'] if sarimax_metrics is not None else None

    # Create figure
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.1, wspace=0.1)

    # Plot 1: RMSE Comparison (Bar Chart)
    ax1 = fig.add_subplot(gs[0, 0])
    models = ['ARIMA', 'SARIMA']
    rmse_values = [arima_rmse, sarima_rmse]
    colors = ['#1f77b4', '#ff7f0e']
    if sarimax_metrics is not None:
        models.append('SARIMAX')
        rmse_values.append(sarimax_rmse)
        colors.append('#2ca02c')
    bars1 = ax1.bar(models, rmse_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5, width=0.8)

    # Add value labels on bars
    for bar, val in zip(bars1, rmse_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax1.set_ylabel('RMSE (W/m²)', fontsize=12, fontweight='bold')
    ax1.set_title('RMSE Comparison', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # Plot 2: MAE Comparison (Bar Chart)
    ax2 = fig.add_subplot(gs[0, 1])
    mae_values = [arima_mae, sarima_mae]
    if sarimax_metrics is not None:
        mae_values.append(sarimax_mae)
    bars2 = ax2.bar(models, mae_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5, width=0.8)

    # Add value labels on bars
    for bar, val in zip(bars2, mae_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax2.set_ylabel('MAE (W/m²)', fontsize=12, fontweight='bold')
    ax2.set_title('MAE Comparison', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    # Plot 3: Predictions vs Actual (Line Plot - First 300 Points)
    ax3 = fig.add_subplot(gs[1, 0])
    n_points = min(300, len(actual_values))
    x_points = np.arange(n_points)

    ax3.plot(x_points, actual_values[:n_points], label='Actual', color='pink', linewidth=5, marker='x', markersize=3, alpha=0.7)
    ax3.plot(x_points, arima_pred[:n_points], label='ARIMA', color="#ecca09", linewidth=3, linestyle='--', alpha=0.7)
    ax3.plot(x_points, sarima_pred[:n_points], label='SARIMA', color="#21b5fa", linewidth=1.5, linestyle='--', alpha=0.7)
    if sarimax_pred is not None:
        ax3.plot(x_points, sarimax_pred[:n_points], label='SARIMAX', color="#2ca02c", linewidth=1.5, linestyle='--', alpha=0.7)

    ax3.set_xlabel('Time Index (First 300 Points)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('GHI (W/m²)', fontsize=11, fontweight='bold')
    ax3.set_title('Predictions vs Actual Values', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10, loc='upper left')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Summary Box with Key Metrics
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')

    arima_corr = arima_metrics['correlation']
    sarima_corr = sarima_metrics['correlation']
    sarimax_corr = sarimax_metrics['correlation'] if sarimax_metrics is not None else None

    rmse_best = {'ARIMA': arima_rmse, 'SARIMA': sarima_rmse}
    mae_best = {'ARIMA': arima_mae, 'SARIMA': sarima_mae}
    corr_best = {'ARIMA': arima_corr, 'SARIMA': sarima_corr}
    if sarimax_metrics is not None and sarimax_corr is not None:
        rmse_best['SARIMAX'] = sarimax_rmse
        mae_best['SARIMAX'] = sarimax_mae
        corr_best['SARIMAX'] = sarimax_corr

    best_rmse = min(rmse_best, key=rmse_best.get)
    best_mae = min(mae_best, key=mae_best.get)
    best_corr = max(corr_best, key=corr_best.get)

    summary_lines = [
        "SUMMARY - BEST MODELS BY METRIC",
        "",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "",
        "RMSE (Root Mean Squared Error)",
        f"  Best: {best_rmse}",
        f"  ARIMA:  {arima_rmse:.4f}",
        f"  SARIMA: {sarima_rmse:.4f}",
    ]
    if sarimax_metrics is not None:
        summary_lines.append(f"  SARIMAX: {sarimax_rmse:.4f}")

    summary_lines += [
        "",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "",
        "MAE (Mean Absolute Error)",
        f"  Best: {best_mae}",
        f"  ARIMA:  {arima_mae:.4f}",
        f"  SARIMA: {sarima_mae:.4f}",
    ]
    if sarimax_metrics is not None:
        summary_lines.append(f"  SARIMAX: {sarimax_mae:.4f}")

    summary_lines += [
        "",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━__",
        "",
        "Correlation (Actual vs Prediction)",
        f"  Best: {best_corr}",
        f"  ARIMA:  {arima_corr:.4f}",
        f"  SARIMA: {sarima_corr:.4f}",
    ]
    if sarimax_metrics is not None and sarimax_corr is not None:
        summary_lines.append(f"  SARIMAX: {sarimax_corr:.4f}")

    summary_lines += [
        "",
        " ━━ ━━ ━━ ━━ ━━ ━━ ━━ ━━ ━━",
        f"Overall Data Points: {len(actual_values):,}",
    ]

    summary_text = "\n".join(summary_lines)

    ax4.text(0.1, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9, pad=1))

    plt.suptitle('ARIMA vs SARIMA vs SARIMAX: Comprehensive Comparison', fontsize=16, fontweight='bold', y=0.995)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\nDetailed comparison plot saved to: {save_path}")

def main():
    """Main function to execute solar data analysis workflow."""
    
    # Step 1: Load data
    df = load_data('../solar2022.csv')
    
    # Step 2: Preprocess data
    df = preprocess_data(df)
    
    # Step 3: Filter GHI data (remove rows where GHI < 5)
    df = filter_ghi_data(df, threshold=5)
    
    # Step 4: Add season column based on month
    df = add_season_column(df)
    
    # # Step 5: Plot time-series with seasonality
    plot_timeseries_with_seasonality(df)
    
    # # Step 6: Plot GHI distribution by season
    plot_ghi_distribution_by_season(df)
    
    # # Step 7: Plot GHI box plot by season
    plot_ghi_boxplot_by_season(df)
    
    # # Step 8: Plot Temperature vs GHI and explain relationship
    plot_temperature_vs_ghi(df)
    
    # #print(df[['DateTime', 'GHI', 'Month', 'Season']].head(25))

    # # Step 9: Plot overall GHI distribution
    plot_ghi_distribution(df)
    
    # # Step 10: Create correlation heatmap
    create_correlation_heatmap(df)
   
    print("\nSTATIONARITY TESTS")
    print("="*80)
    adf_results = []
    for col in ['GHI']:
        adf_results.append(perform_adf_test(df[col], col))
    print("\n" + "="*80)
    print("\nSUMMARY COMPARISON:")
    summary_df = pd.DataFrame(adf_results)
    print(summary_df.to_string(index=False))
  
    

    
    # Step 13: Create ARIMA training dataframe with DateTime, Temperature, and GHI
    df_arimasarima = create_arima_dataframe(df)
      # Step 11: Normalize data (commented out)
    df_arima = normalize_data(df_arimasarima)
        # Step 12: Perform stationarity tests (commented out)
    print("\nSTATIONARITY TESTS")
    print("="*80)
    adf_results = []
    for col in ['GHI', 'GHI_MinMax', 'GHI_ZScore']:
        adf_results.append(perform_adf_test(df_arima[col], col))
    print("\n" + "="*80)
    print("\nSUMMARY COMPARISON:")
    summary_df = pd.DataFrame(adf_results)
    print(summary_df.to_string(index=False))
    # print(df_arima.head())
    # Step 14: Train ARIMA model on GHI
    if df_arima is not None:
        arima_model, arima_results, arima_metrics = train_arima_model(df_arima, order=(3, 0, 3 ), target_col='GHI_MinMax')
        
        # Step 15: Train SARIMA model on GHI
        sarima_model, sarima_results, sarima_metrics = train_sarima_model(
            df_arima,
            order=(3, 0, 3),
            seasonal_order=(3, 0, 3, 9),
            target_col='GHI_MinMax',
        )

        # Step 16: Train SARIMAX model on GHI with Temperature as exogenous
        sarimax_model, sarimax_results, sarimax_metrics = train_sarimax_model(
            df_arima,
            order=(3, 0, 3),
            seasonal_order=(3, 0, 3, 9),
            target_col='GHI_MinMax',
            exog_cols=['Temperature'],
        )
        
        # Step 17: Compare ARIMA, SARIMA, and SARIMAX
        if arima_metrics is not None and sarima_metrics is not None:
            compare_arima_sarima(arima_metrics, sarima_metrics, sarimax_metrics)
            
            # Step 18: Plot actual vs prediction scatter plots
            plot_actual_vs_prediction(arima_metrics, sarima_metrics, sarimax_metrics)
            
            # Step 19: Plot residuals for both models
            plot_residuals(arima_metrics, sarima_metrics, sarimax_metrics)
            
            # Step 20: Create detailed comparison plot
            plot_detailed_comparison(arima_metrics, sarima_metrics, sarimax_metrics)
    else:
        print("ERROR: Failed to create ARIMA dataframe. Cannot train models.")




if __name__ == "__main__":
    main()


