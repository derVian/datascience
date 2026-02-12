
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
file_path = './solar2022.csv'  # Use ORIGINAL file, not cleaned file
def read_solar_data(file_path):
    try:
        # Skip first 2 rows (metadata) and use row 3 as header
        df = pd.read_csv(file_path, skiprows=2) 
        return df
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def drop_columns_with_high_nulls(df, threshold=0.8):

    null_percentage = df.isnull().sum() / len(df)
    columns_to_drop = null_percentage[null_percentage > threshold].index.tolist()
    
    if columns_to_drop:
        print(f"Dropping columns with >80% null values: {columns_to_drop}")
        df = df.drop(columns=columns_to_drop)
    else:
        print("No columns with >80% null values found.")

    return df


def create_datetime_column(df):
    """
    Create a single DateTime column from Year, Month, Day, Hour, Minute columns
    and drop the original separate columns.
    """
    print("Creating DateTime column from Year, Month, Day, Hour, Minute...")
    # Create full DateTime column with date and time
    df['DateTime'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']])
    # Drop original columns
    df = df.drop(columns=['Year', 'Month', 'Day', 'Hour', 'Minute'])
    # Move DateTime to first column
    cols = ['DateTime'] + [col for col in df.columns if col != 'DateTime']
    df = df[cols]
    print(f"DateTime column created and moved to first position. Reduced from 5 columns to 1.")
    return df


def analyze_column_variance(df):
    """
    Analyze variance and uniqueness of columns to identify redundant columns.
    """
    print("\n" + "="*70)
    print("VARIANCE AND REDUNDANCY ANALYSIS")
    print("="*70)
    
    # Select only numeric columns for variance analysis
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    print(f"\nAnalyzing {len(numeric_cols)} numeric columns...\n")
    
    # Calculate variance and unique values
    variance_data = []
    for col in numeric_cols:
        var = df[col].var()
        unique_count = df[col].nunique()
        unique_pct = (unique_count / len(df)) * 100
        variance_data.append({
            'Column': col,
            'Variance': var,
            'Unique Values': unique_count,
            'Unique %': unique_pct
        })
    
    # Create dataframe and sort by variance
    var_df = pd.DataFrame(variance_data).sort_values('Variance')
    
    print(var_df.to_string(index=False))
    
    # Identify potentially redundant columns
    print("\n" + "-"*70)
    print("REDUNDANCY INDICATORS:")
    print("-"*70)
    
    constant_cols = var_df[var_df['Variance'] == 0]['Column'].tolist()
    low_var_cols = var_df[(var_df['Variance'] > 0) & (var_df['Variance'] < 0.01)]['Column'].tolist()
    
    if constant_cols:
        print(f"\n⚠ CONSTANT columns (variance = 0): {constant_cols}")
        print("  → These columns have the same value in all rows - highly redundant!")
    else:
        print("\n✓ No constant columns found.")
    
    if low_var_cols:
        print(f"\n⚠ LOW VARIANCE columns (< 0.01): {low_var_cols}")
        print("  → These columns have very little variation - potentially redundant.")
    else:
        print("✓ No low variance columns found.")
    
    print("\n" + "="*70 + "\n")
    
    return var_df


def analyze_column_correlation(df, low_var_columns=None):
    """
    Analyze correlation between columns to identify highly correlated (redundant) columns.
    """
    print("\n" + "="*70)
    print("CORRELATION ANALYSIS")
    print("="*70)
    
    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    corr_matrix = df[numeric_cols].corr()
    
    # Find highly correlated pairs
    print("\nHighly correlated column pairs (correlation > 0.95):\n")
    
    high_corr_found = False
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_value = corr_matrix.iloc[i, j]
            if abs(corr_value) > 0.95:
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                print(f"  {col1} <---> {col2}: {corr_value:.4f}")
                high_corr_found = True
    
    if not high_corr_found:
        print("  ✓ No highly correlated pairs found!")
    
    # Analyze low variance columns specifically
    if low_var_columns:
        print("\n" + "-"*70)
        print(f"CORRELATION of LOW VARIANCE columns: {low_var_columns}")
        print("-"*70)
        
        for col in low_var_columns:
            if col in corr_matrix.columns:
                correlations = corr_matrix[col].abs().sort_values(ascending=False)[1:6]
                print(f"\n{col} - Top correlations:")
                for corr_col, corr_val in correlations.items():
                    print(f"  {corr_col}: {corr_val:.4f}")
    
    print("\n" + "="*70 + "\n")
    
    return corr_matrix


def check_column_correlation(df, column_name):
    """
    Check correlation of a specific column with all other numeric columns.
    Shows the pattern and strength of relationships.
    """
    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    if column_name not in numeric_cols:
        print(f"Error: '{column_name}' is not a numeric column or doesn't exist.")
        print(f"Available columns: {list(numeric_cols)}")
        return None
    
    # Get correlation matrix
    corr_matrix = df[numeric_cols].corr()
    
    # Get correlations for the specified column
    column_corr = corr_matrix[column_name].sort_values(ascending=False)
    
    print("\n" + "="*70)
    print(f"CORRELATION ANALYSIS FOR: {column_name}")
    print("="*70)
    print(f"\nCorrelation with all numeric columns (sorted by strength):\n")
    
    for idx, (col, corr_val) in enumerate(column_corr.items(), 1):
        # Create a visual bar
        bar_length = int(abs(corr_val) * 30)
        bar = "█" * bar_length
        direction = "+" if corr_val >= 0 else "-"
        
        print(f"{idx:2d}. {col:25s} {direction} {abs(corr_val):.4f} {bar}")
    
    print("\n" + "="*70 + "\n")
    
    # Interpretation
    print("INTERPRETATION:")
    print("-"*70)
    strong_pos = column_corr[(column_corr > 0.7) & (column_corr < 1.0)]
    strong_neg = column_corr[(column_corr < -0.7) & (column_corr > -1.0)]
    
    if len(strong_pos) > 0:
        print(f"\n✓ Strong positive correlation (> 0.7):")
        for col, val in strong_pos.items():
            print(f"  {col}: {val:.4f}")
    
    if len(strong_neg) > 0:
        print(f"\n✗ Strong negative correlation (< -0.7):")
        for col, val in strong_neg.items():
            print(f"  {col}: {val:.4f}")
    
    if len(strong_pos) == 0 and len(strong_neg) == 0:
        print("\nNo strong correlations found (> 0.7 or < -0.7)")
    
    print("="*70 + "\n")
    
    return column_corr


def drop_low_variance_columns(df, low_var_columns):
    """
    Remove low variance columns that are redundant.
    """
    print(f"\nRemoving low variance columns: {low_var_columns}")
    df = df.drop(columns=low_var_columns)
    print(f"Dropped {len(low_var_columns)} columns. Remaining: {len(df.columns)}")
    return df


def filter_and_drop_fill_flag(df):
    """
    Remove rows with Fill Flag > 0 (data quality issues)
    and drop the Fill Flag column.
    """
    if 'Fill Flag' in df.columns:
        initial_rows = len(df)
        # Keep only rows where Fill Flag == 0 (good quality data)
        df = df[df['Fill Flag'] == 0]
        rows_removed = initial_rows - len(df)
        
        print(f"\nFiltering by Fill Flag...")
        print(f"  Rows removed (Fill Flag != 0): {rows_removed}")
        print(f"  Rows remaining: {len(df)}")
        
        # Drop the Fill Flag column (no longer needed)
        df = df.drop(columns=['Fill Flag'])
        print(f"  Fill Flag column dropped")
    else:
        print("\n'Fill Flag' column not found.")
    
    return df


def detect_outliers_iqr(df, display_details=True):
    """
    Detect outliers using Interquartile Range (IQR) method.
    
    How it works:
    1. Calculate Q1 (25th percentile) and Q3 (75th percentile)
    2. Calculate IQR = Q3 - Q1 (middle 50% of data)
    3. Lower Bound = Q1 - 1.5 * IQR
    4. Upper Bound = Q3 + 1.5 * IQR
    5. Any value outside bounds = OUTLIER
    
    Parameters:
    - df: DataFrame to analyze
    - display_details: If True, show detailed statistics
    
    Returns:
    - Dictionary with outlier statistics for each column
    """
    print("\n" + "="*70)
    print("STEP 1: OUTLIER DETECTION (IQR Method)")
    print("="*70)
    
    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    outlier_info = {}
    
    for col in numeric_cols:
        # Calculate Q1, Q3, IQR
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # Calculate bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Find outliers
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_count = len(outliers)
        outlier_pct = (outlier_count / len(df)) * 100 if len(df) > 0 else 0
        
        # Store info
        outlier_info[col] = {
            'count': outlier_count,
            'percentage': outlier_pct,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'min': df[col].min(),
            'max': df[col].max()
        }
        
        # Display details
        if display_details and outlier_count > 0:
            print(f"\n📊 {col}:")
            print(f"   Q1: {Q1:.2f}")
            print(f"   Q3: {Q3:.2f}")
            print(f"   IQR: {IQR:.2f}")
            print(f"   Valid range: [{lower_bound:.2f}, {upper_bound:.2f}]")
            print(f"   ⚠ Outliers: {outlier_count} rows ({outlier_pct:.2f}%)")
            print(f"   Min value: {df[col].min():.2f}")
            print(f"   Max value: {df[col].max():.2f}")
    
    # Summary
    total_outliers = sum([info['count'] for info in outlier_info.values()])
    cols_with_outliers = sum([1 for info in outlier_info.values() if info['count'] > 0])
    
    print("\n" + "-"*70)
    print(f"SUMMARY: {total_outliers} total outliers found in {cols_with_outliers} columns")
    print("="*70 + "\n")
    
    return outlier_info


def visualize_outliers(df, outlier_info, num_to_show=10):
    """
    Display actual outlier rows from the dataframe.
    
    Parameters:
    - df: DataFrame
    - outlier_info: Dictionary returned from detect_outliers_iqr()
    - num_to_show: How many outlier examples to display per column
    """
    print("\n" + "="*70)
    print("VISUALIZING OUTLIERS - Sample Rows")
    print("="*70)
    
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    for col in numeric_cols:
        info = outlier_info.get(col, {})
        outlier_count = info.get('count', 0)
        
        if outlier_count == 0:
            continue
        
        lower_bound = info.get('lower_bound', 0)
        upper_bound = info.get('upper_bound', 0)
        
        # Find outliers for this column
        outlier_rows = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        
        print(f"\n{'='*70}")
        print(f"📍 Column: {col}")
        print(f"   Total outliers: {len(outlier_rows)} | Valid range: [{lower_bound:.2f}, {upper_bound:.2f}]")
        print(f"{'='*70}")
        
        # Show first few outliers
        display_count = min(num_to_show, len(outlier_rows))
        print(f"\nShowing first {display_count} outliers:\n")
        
        # Select relevant columns to display
        cols_to_show = ['DateTime', col] if 'DateTime' in df.columns else [col]
        outlier_display = outlier_rows[cols_to_show].head(display_count)
        
        print(outlier_display.to_string())
        print()


def show_outlier_statistics(df, outlier_info):
    """
    Show summary statistics for columns with outliers.
    """
    print("\n" + "="*70)
    print("OUTLIER STATISTICS SUMMARY")
    print("="*70 + "\n")
    
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    summary_data = []
    for col in numeric_cols:
        info = outlier_info.get(col, {})
        count = info.get('count', 0)
        
        if count > 0:
            summary_data.append({
                'Column': col,
                'Outlier Count': count,
                'Percentage': f"{info.get('percentage', 0):.2f}%",
                'Min': f"{info.get('min', 0):.2f}",
                'Max': f"{info.get('max', 0):.2f}",
                'Valid Range': f"[{info.get('lower_bound', 0):.2f}, {info.get('upper_bound', 0):.2f}]"
            })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
    else:
        print("No outliers found!")
    
    print("\n" + "="*70 + "\n")


def validate_solar_domain(df):
    """
    STEP 2: Domain-Specific Validation for Solar Data
    
    Check for physically impossible values:
    1. DHI > Clearsky DHI (impossible - actual can't exceed theoretical clear sky)
    2. GHI > Clearsky GHI (impossible)
    3. DNI > Clearsky DNI (impossible)
    4. Negative solar radiation values (impossible - can't have negative light)
    5. Extreme temperature values (likely measurement errors)
    
    Returns:
    - Dictionary with validation results
    - Dataframe of invalid rows (if any)
    """
    print("\n" + "="*70)
    print("STEP 2: DOMAIN VALIDATION (Solar-Specific Rules)")
    print("="*70)
    
    validation_results = {}
    invalid_rows_all = pd.DataFrame()
    total_issues = 0
    
    # Get column names (handling both proper headers and numeric column names)
    cols = df.columns.tolist()
    
    # 1. Check DHI vs Clearsky DHI
    if 'DHI' in cols and 'Clearsky DHI' in cols:
        invalid_dhi = df[df['DHI'] > df['Clearsky DHI']]
        count = len(invalid_dhi)
        total_issues += count
        validation_results['DHI > Clearsky DHI'] = count
        if count > 0:
            print(f"\n❌ DHI > Clearsky DHI: {count} rows ({count/len(df)*100:.2f}%)")
            print(f"   Example: DHI max={df['DHI'].max():.2f}, Clearsky DHI max={df['Clearsky DHI'].max():.2f}")
            invalid_rows_all = pd.concat([invalid_rows_all, invalid_dhi])
    
    # 2. Check GHI vs Clearsky GHI
    if 'GHI' in cols and 'Clearsky GHI' in cols:
        invalid_ghi = df[df['GHI'] > df['Clearsky GHI']]
        count = len(invalid_ghi)
        total_issues += count
        validation_results['GHI > Clearsky GHI'] = count
        if count > 0:
            print(f"\n❌ GHI > Clearsky GHI: {count} rows ({count/len(df)*100:.2f}%)")
            print(f"   Example: GHI max={df['GHI'].max():.2f}, Clearsky GHI max={df['Clearsky GHI'].max():.2f}")
            invalid_rows_all = pd.concat([invalid_rows_all, invalid_ghi])
    
    # 3. Check DNI vs Clearsky DNI
    if 'DNI' in cols and 'Clearsky DNI' in cols:
        invalid_dni = df[df['DNI'] > df['Clearsky DNI']]
        count = len(invalid_dni)
        total_issues += count
        validation_results['DNI > Clearsky DNI'] = count
        if count > 0:
            print(f"\n❌ DNI > Clearsky DNI: {count} rows ({count/len(df)*100:.2f}%)")
            print(f"   Example: DNI max={df['DNI'].max():.2f}, Clearsky DNI max={df['Clearsky DNI'].max():.2f}")
            invalid_rows_all = pd.concat([invalid_rows_all, invalid_dni])
    
    # 4. Check for negative solar radiation values
    radiation_cols = [col for col in cols if any(x in str(col).upper() for x in ['DHI', 'GHI', 'DNI'])]
    for col in radiation_cols:
        negative = df[df[col] < 0]
        count = len(negative)
        if count > 0:
            total_issues += count
            validation_results[f'Negative {col}'] = count
            print(f"\n❌ Negative {col}: {count} rows ({count/len(df)*100:.2f}%)")
            print(f"   Min value: {df[col].min():.2f}")
            invalid_rows_all = pd.concat([invalid_rows_all, negative])
    
    # 5. Check for extreme temperature values
    if 'Temperature' in cols:
        temp_col = 'Temperature'
        # Reasonable range: -50°C to 60°C
        extreme_temp = df[(df[temp_col] < -50) | (df[temp_col] > 60)]
        count = len(extreme_temp)
        if count > 0:
            total_issues += count
            validation_results['Extreme Temperature'] = count
            print(f"\n⚠ Extreme Temperature (< -50°C or > 60°C): {count} rows ({count/len(df)*100:.2f}%)")
            print(f"   Range: {df[temp_col].min():.2f}°C to {df[temp_col].max():.2f}°C")
            invalid_rows_all = pd.concat([invalid_rows_all, extreme_temp])
    
    # Summary
    print("\n" + "-"*70)
    if total_issues == 0:
        print("✅ All data passed domain validation! No physically impossible values found.")
    else:
        print(f"⚠ TOTAL ISSUES FOUND: {total_issues} problematic rows")
        print("\nSummary of issues:")
        for issue, count in validation_results.items():
            if count > 0:
                print(f"   - {issue}: {count} rows")
    
    print("="*70 + "\n")
    
    # Remove duplicates from invalid_rows (same row may violate multiple rules)
    if len(invalid_rows_all) > 0:
        invalid_rows_all = invalid_rows_all.drop_duplicates()
    
    return validation_results, invalid_rows_all


def remove_invalid_rows(df, invalid_rows):
    """
    Remove rows that failed domain validation.
    
    Parameters:
    - df: Original dataframe
    - invalid_rows: Dataframe of invalid rows to remove
    
    Returns:
    - Cleaned dataframe
    """
    if len(invalid_rows) == 0:
        print("✓ No invalid rows to remove.")
        return df
    
    initial_count = len(df)
    # Remove invalid rows by index
    df_cleaned = df.drop(invalid_rows.index).reset_index(drop=True)
    removed_count = initial_count - len(df_cleaned)
    
    print(f"\n🗑️ Removed {removed_count} invalid rows")
    print(f"   Before: {initial_count} rows")
    print(f"   After: {len(df_cleaned)} rows")
    print(f"   Data loss: {removed_count/initial_count*100:.2f}%\n")
    
    return df_cleaned


def encode_categorical_variables(df):
    """
    STEP 3: Encode Categorical Variables using One-Hot Encoding
    
    What is One-Hot Encoding?
    - Converts categorical columns into multiple binary (0/1) columns
    - Each unique category value gets its own column
    
    Example:
        Cloud Type = [0, 1, 2, 3, ...]
        
        Becomes:
        Cloud_Type_0, Cloud_Type_1, Cloud_Type_2, Cloud_Type_3, ...
        (each column is 0 or 1)
    
    Why?
    - Machine learning algorithms need numeric data
    - Cloud Type values (0-10) are labels, not quantities
    - One-hot encoding prevents model from assuming "5" is somehow "bigger" than "2"
    
    Returns:
    - DataFrame with categorical variables encoded
    """
    print("\n" + "="*70)
    print("STEP 3: CATEGORICAL VARIABLE ENCODING")
    print("="*70)
    
    initial_columns = len(df.columns)
    
    # Identify potential categorical columns
    # Cloud Type is the main categorical variable in solar data
    categorical_cols = []
    
    # Check for Cloud Type column
    cloud_type_col = None
    for col in df.columns:
        if 'cloud' in str(col).lower() and 'type' in str(col).lower():
            cloud_type_col = col
            categorical_cols.append(col)
            break
    
    if not categorical_cols:
        print("\n✓ No categorical variables found to encode.")
        print("  (All columns appear to be continuous numeric)")
        print("="*70 + "\n")
        return df
    
    print(f"\nFound {len(categorical_cols)} categorical column(s) to encode:")
    for col in categorical_cols:
        unique_values = df[col].nunique()
        print(f"   - {col}: {unique_values} unique values")
        print(f"     Values: {sorted(df[col].unique())}")
    
    print("\n" + "-"*70)
    print("ENCODING PROCESS:")
    print("-"*70)
    
    # Encode Cloud Type
    if cloud_type_col:
        print(f"\n📋 Encoding '{cloud_type_col}'...")
        
        # Create dummy variables (One-Hot Encoding)
        cloud_dummies = pd.get_dummies(df[cloud_type_col], prefix='Cloud_Type', dtype=int)
        
        print(f"   Original: 1 column ({cloud_type_col})")
        print(f"   Encoded:  {len(cloud_dummies.columns)} binary columns")
        print(f"   New columns: {list(cloud_dummies.columns)}")
        
        # Add dummy columns to dataframe
        df = pd.concat([df, cloud_dummies], axis=1)
        
        # Drop original categorical column
        df = df.drop(columns=[cloud_type_col])
        print(f"   ✓ Dropped original '{cloud_type_col}' column")
    
    final_columns = len(df.columns)
    column_change = final_columns - initial_columns
    
    print("\n" + "-"*70)
    print("SUMMARY:")
    print("-"*70)
    print(f"   Columns before: {initial_columns}")
    print(f"   Columns after:  {final_columns}")
    print(f"   Change:         +{column_change} columns")
    print("="*70 + "\n")
    
    return df


def create_time_series_features(df):
    """
    Create time series features for forecasting models (especially XGBoost).
    
    Features created:
    1. Clearness Index - ratio of actual to clear sky values
    2. Lag features - past values (1h, 3h, 6h, 24h)
    3. Rolling statistics - moving averages and std dev
    4. Time-based features - hour, day, month, day of year, season
    5. Rate of change - differences between current and past values
    
    Parameters:
    - df: Dataframe with DateTime column and solar measurements
    
    Returns:
    - DataFrame with additional time series features
    """
    print("\n" + "="*70)
    print("TIME SERIES FEATURE ENGINEERING")
    print("="*70)
    
    initial_cols = len(df.columns)
    
    # Ensure DateTime is set as index for time series operations
    if 'DateTime' in df.columns:
        df = df.set_index('DateTime')
        print("\n✓ DateTime set as index")
    
    # 1. CLEARNESS INDEX (Cloud Cover Proxy)
    print("\n📊 Creating Clearness Index features...")
    if 'GHI' in df.columns and 'Clearsky GHI' in df.columns:
        df['Clearness_Index_GHI'] = df['GHI'] / (df['Clearsky GHI'] + 1e-6)  # avoid division by zero
        df['Clearness_Index_GHI'] = df['Clearness_Index_GHI'].clip(0, 1.2)  # cap at reasonable values
        print("   + Clearness_Index_GHI (GHI / Clearsky GHI)")
    
    if 'DNI' in df.columns and 'Clearsky DNI' in df.columns:
        df['Clearness_Index_DNI'] = df['DNI'] / (df['Clearsky DNI'] + 1e-6)
        df['Clearness_Index_DNI'] = df['Clearness_Index_DNI'].clip(0, 1.2)
        print("   + Clearness_Index_DNI (DNI / Clearsky DNI)")
    
    if 'DHI' in df.columns and 'Clearsky DHI' in df.columns:
        df['Clearness_Index_DHI'] = df['DHI'] / (df['Clearsky DHI'] + 1e-6)
        df['Clearness_Index_DHI'] = df['Clearness_Index_DHI'].clip(0, 3.0)  # DHI can exceed clearsky
        print("   + Clearness_Index_DHI (DHI / Clearsky DHI)")
    
    # 2. TIME-BASED FEATURES
    print("\n📅 Creating time-based features...")
    df['Hour'] = df.index.hour
    df['Day'] = df.index.day
    df['Month'] = df.index.month
    df['DayOfYear'] = df.index.dayofyear
    df['Quarter'] = df.index.quarter
    df['IsWeekend'] = (df.index.dayofweek >= 5).astype(int)
    print("   + Hour, Day, Month, DayOfYear, Quarter, IsWeekend")
    
    # Cyclical encoding for Hour and Month (better for ML models)
    df['Hour_Sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['Hour_Cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
    df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    print("   + Cyclical encoding (Hour_Sin/Cos, Month_Sin/Cos)")
    
    # 3. LAG FEATURES (Past Values)
    print("\n⏰ Creating lag features...")
    lag_columns = ['GHI', 'DNI', 'DHI', 'Temperature', 'Relative Humidity']
    lag_periods = [1, 3, 6, 24]  # 1h, 3h, 6h, 24h ago
    
    lag_count = 0
    for col in lag_columns:
        if col in df.columns:
            for lag in lag_periods:
                df[f'{col}_lag_{lag}h'] = df[col].shift(lag)
                lag_count += 1
    print(f"   + Created {lag_count} lag features ({lag_periods}h shifts)")
    
    # 4. ROLLING STATISTICS (Moving Averages & Std Dev)
    print("\n📈 Creating rolling statistics...")
    rolling_windows = [3, 6, 24]  # 3h, 6h, 24h windows
    rolling_cols = ['GHI', 'DNI', 'Temperature']
    
    rolling_count = 0
    for col in rolling_cols:
        if col in df.columns:
            for window in rolling_windows:
                df[f'{col}_rolling_mean_{window}h'] = df[col].rolling(window=window).mean()
                df[f'{col}_rolling_std_{window}h'] = df[col].rolling(window=window).std()
                rolling_count += 2
    print(f"   + Created {rolling_count} rolling features (mean & std)")
    
    # 5. RATE OF CHANGE (Differences)
    print("\n📉 Creating rate of change features...")
    diff_cols = ['GHI', 'DNI', 'Temperature', 'Wind Speed']
    diff_count = 0
    for col in diff_cols:
        if col in df.columns:
            df[f'{col}_diff_1h'] = df[col].diff(1)  # change from 1h ago
            df[f'{col}_diff_24h'] = df[col].diff(24)  # change from 24h ago (same time yesterday)
            diff_count += 2
    print(f"   + Created {diff_count} rate of change features")
    
    # Reset index to keep DateTime as column
    df = df.reset_index()
    
    final_cols = len(df.columns)
    added_features = final_cols - initial_cols
    
    print("\n" + "-"*70)
    print("SUMMARY:")
    print("-"*70)
    print(f"   Initial columns: {initial_cols}")
    print(f"   Final columns:   {final_cols}")
    print(f"   Features added:  {added_features}")
    print(f"\n   ⚠ Note: First 24 rows will have NaN values due to lags/rolling")
    print(f"   You may want to drop these rows before modeling")
    print("="*70 + "\n")
    
    return df


def visualize_ghi_distribution(df, save_path='./images/solar/ghi_distribution.png'):
    """
    Visualize the distribution of GHI (Global Horizontal Irradiance).
    
    Creates a comprehensive plot with:
    1. Histogram with KDE
    2. Box plot
    3. QQ plot (normality test)
    4. Statistical summary
    
    Parameters:
    - df: Dataframe containing GHI column
    - save_path: Path to save the visualization
    """
    print("\n" + "="*70)
    print("GHI DISTRIBUTION ANALYSIS")
    print("="*70)
    
    if 'GHI' not in df.columns:
        print("❌ GHI column not found in dataframe!")
        return
    
    ghi = df['GHI'].dropna()
    
    # Statistical summary
    print("\n📊 STATISTICAL SUMMARY:")
    print("-"*70)
    print(f"Count:       {len(ghi):,}")
    print(f"Mean:        {ghi.mean():.2f} W/m²")
    print(f"Median:      {ghi.median():.2f} W/m²")
    print(f"Std Dev:     {ghi.std():.2f} W/m²")
    print(f"Min:         {ghi.min():.2f} W/m²")
    print(f"Max:         {ghi.max():.2f} W/m²")
    print(f"\nQuartiles:")
    print(f"  Q1 (25%):  {ghi.quantile(0.25):.2f} W/m²")
    print(f"  Q2 (50%):  {ghi.quantile(0.50):.2f} W/m²")
    print(f"  Q3 (75%):  {ghi.quantile(0.75):.2f} W/m²")
    print(f"  IQR:       {ghi.quantile(0.75) - ghi.quantile(0.25):.2f} W/m²")
    
    # Skewness and Kurtosis
    from scipy import stats
    skewness = stats.skew(ghi)
    kurtosis = stats.kurtosis(ghi)
    print(f"\nSkewness:    {skewness:.3f}", end="")
    if abs(skewness) < 0.5:
        print(" (fairly symmetric)")
    elif skewness > 0:
        print(" (right-skewed - tail on right)")
    else:
        print(" (left-skewed - tail on left)")
    
    print(f"Kurtosis:    {kurtosis:.3f}", end="")
    if abs(kurtosis) < 0.5:
        print(" (normal-like tails)")
    elif kurtosis > 0:
        print(" (heavy tails - more outliers)")
    else:
        print(" (light tails - fewer outliers)")
    
    # Nighttime vs Daytime
    nighttime = (ghi == 0).sum()
    daytime = (ghi > 0).sum()
    print(f"\n🌙 Nighttime (GHI = 0):  {nighttime:,} samples ({nighttime/len(ghi)*100:.1f}%)")
    print(f"☀️  Daytime (GHI > 0):    {daytime:,} samples ({daytime/len(ghi)*100:.1f}%)")
    
    # Create visualization
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Histogram with KDE (all data)
    ax1 = plt.subplot(2, 3, 1)
    ghi.hist(bins=100, alpha=0.7, color='orange', edgecolor='black')
    ax1.set_xlabel('GHI (W/m²)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('GHI Distribution - All Data', fontsize=14, fontweight='bold')
    ax1.axvline(ghi.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean = {ghi.mean():.1f}')
    ax1.axvline(ghi.median(), color='green', linestyle='--', linewidth=2, label=f'Median = {ghi.median():.1f}')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 2. Histogram (daytime only - GHI > 0)
    ax2 = plt.subplot(2, 3, 2)
    ghi_daytime = ghi[ghi > 0]
    ghi_daytime.hist(bins=50, alpha=0.7, color='gold', edgecolor='black')
    ax2.set_xlabel('GHI (W/m²)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('GHI Distribution - Daytime Only (GHI > 0)', fontsize=14, fontweight='bold')
    ax2.axvline(ghi_daytime.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean = {ghi_daytime.mean():.1f}')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # 3. Box plot
    ax3 = plt.subplot(2, 3, 3)
    box = ax3.boxplot(ghi, vert=True, patch_artist=True, widths=0.5)
    box['boxes'][0].set_facecolor('lightblue')
    box['boxes'][0].set_alpha(0.7)
    ax3.set_ylabel('GHI (W/m²)', fontsize=12)
    ax3.set_title('GHI Box Plot', fontsize=14, fontweight='bold')
    ax3.set_xticklabels(['GHI'])
    ax3.grid(alpha=0.3, axis='y')
    
    # 4. KDE Plot
    ax4 = plt.subplot(2, 3, 4)
    ghi.plot.kde(color='blue', linewidth=2)
    ax4.fill_between(ghi.plot.kde().get_lines()[0].get_data()[0], 
                      ghi.plot.kde().get_lines()[0].get_data()[1], 
                      alpha=0.3, color='blue')
    ax4.set_xlabel('GHI (W/m²)', fontsize=12)
    ax4.set_ylabel('Density', fontsize=12)
    ax4.set_title('GHI Kernel Density Estimate', fontsize=14, fontweight='bold')
    ax4.grid(alpha=0.3)
    
    # 5. Q-Q Plot (normality test)
    ax5 = plt.subplot(2, 3, 5)
    stats.probplot(ghi, dist="norm", plot=ax5)
    ax5.set_title('Q-Q Plot (Normality Test)', fontsize=14, fontweight='bold')
    ax5.grid(alpha=0.3)
    
    # 6. Cumulative Distribution
    ax6 = plt.subplot(2, 3, 6)
    sorted_ghi = np.sort(ghi)
    cumulative = np.arange(1, len(sorted_ghi) + 1) / len(sorted_ghi)
    ax6.plot(sorted_ghi, cumulative, linewidth=2, color='purple')
    ax6.set_xlabel('GHI (W/m²)', fontsize=12)
    ax6.set_ylabel('Cumulative Probability', fontsize=12)
    ax6.set_title('Cumulative Distribution Function', fontsize=14, fontweight='bold')
    ax6.grid(alpha=0.3)
    ax6.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Median')
    ax6.legend()
    
    plt.tight_layout()
    
    # Save figure
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ GHI distribution plots saved to: {save_path}")
    plt.close()
    
    print("="*70 + "\n")


def create_correlation_matrix(df, figsize=(20, 16), save_path='./images/solar/correlation_matrix.png'):
    """
    Create and analyze a comprehensive correlation matrix for the solar dataset.
    
    Parameters:
    - df: Cleaned dataframe
    - figsize: Size of the figure (width, height)
    - save_path: Path to save the correlation heatmap
    
    Returns:
    - Correlation matrix dataframe
    """
    print("\n" + "="*70)
    print("CORRELATION MATRIX ANALYSIS")
    print("="*70)
    
    # Select only numeric columns (exclude DateTime)
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    print(f"\nAnalyzing correlations between {len(numeric_cols)} numeric features...")
    
    # Calculate correlation matrix
    corr_matrix = df[numeric_cols].corr()
    
    # Create figure with larger size for better readability
    plt.figure(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(corr_matrix, 
                annot=True,  # Show correlation values
                fmt='.2f',  # Format numbers to 2 decimal places
                annot_kws={'size': 6},  # Small font size for readability
                cmap='coolwarm',  # Red = positive, Blue = negative
                center=0,
                vmin=-1, vmax=1,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"})
    
    plt.title('Solar Data Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    
    # Save the figure
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Correlation heatmap saved to: {save_path}")
    plt.close()
    
    # ANALYZE CORRELATIONS
    print("\n" + "="*70)
    print("CORRELATION INSIGHTS")
    print("="*70)
    
    # Find strong positive correlations (> 0.8, excluding self-correlation)
    print("\n🔵 STRONG POSITIVE CORRELATIONS (> 0.8):")
    print("-"*70)
    strong_pos_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if corr_val > 0.8:
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                strong_pos_pairs.append((col1, col2, corr_val))
    
    if strong_pos_pairs:
        # Sort by correlation value
        strong_pos_pairs.sort(key=lambda x: x[2], reverse=True)
        for col1, col2, corr_val in strong_pos_pairs:
            print(f"   {col1:30s} <--> {col2:30s}  |  r = {corr_val:.4f}")
        print(f"\n   Total pairs: {len(strong_pos_pairs)}")
    else:
        print("   None found.")
    
    # Find strong negative correlations (< -0.7)
    print("\n🔴 STRONG NEGATIVE CORRELATIONS (< -0.7):")
    print("-"*70)
    strong_neg_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if corr_val < -0.7:
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                strong_neg_pairs.append((col1, col2, corr_val))
    
    if strong_neg_pairs:
        strong_neg_pairs.sort(key=lambda x: x[2])
        for col1, col2, corr_val in strong_neg_pairs:
            print(f"   {col1:30s} <--> {col2:30s}  |  r = {corr_val:.4f}")
        print(f"\n   Total pairs: {len(strong_neg_pairs)}")
    else:
        print("   None found.")
    
    # Identify most correlated features for key solar variables
    print("\n📊 KEY VARIABLE CORRELATIONS:")
    print("-"*70)
    
    key_vars = ['GHI', 'DNI', 'DHI', 'Temperature', 'Wind Speed', 'Relative Humidity']
    for var in key_vars:
        if var in corr_matrix.columns:
            print(f"\n{var}:")
            # Get top 5 correlations (excluding self)
            top_corr = corr_matrix[var].abs().sort_values(ascending=False)[1:6]
            for corr_col, corr_val in top_corr.items():
                direction = "+" if corr_matrix[var][corr_col] > 0 else "-"
                print(f"   {direction} {corr_col:25s}  |  r = {corr_matrix[var][corr_col]:+.4f}")
    
    # Summary statistics
    print("\n" + "="*70)
    print("CORRELATION SUMMARY STATISTICS:")
    print("="*70)
    
    # Get all correlation values (upper triangle only, excluding diagonal)
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    correlations = upper_triangle.stack()
    
    print(f"\nTotal correlation pairs analyzed: {len(correlations)}")
    print(f"Mean absolute correlation: {correlations.abs().mean():.4f}")
    print(f"Median correlation: {correlations.median():.4f}")
    print(f"Max correlation: {correlations.max():.4f}")
    print(f"Min correlation: {correlations.min():.4f}")
    
    # Distribution of correlations
    very_strong = (correlations.abs() > 0.8).sum()
    strong = ((correlations.abs() > 0.6) & (correlations.abs() <= 0.8)).sum()
    moderate = ((correlations.abs() > 0.4) & (correlations.abs() <= 0.6)).sum()
    weak = (correlations.abs() <= 0.4).sum()
    
    print(f"\nCorrelation strength distribution:")
    print(f"   Very Strong (|r| > 0.8):  {very_strong:4d} pairs ({very_strong/len(correlations)*100:5.1f}%)")
    print(f"   Strong (0.6 < |r| ≤ 0.8): {strong:4d} pairs ({strong/len(correlations)*100:5.1f}%)")
    print(f"   Moderate (0.4 < |r| ≤ 0.6): {moderate:4d} pairs ({moderate/len(correlations)*100:5.1f}%)")
    print(f"   Weak (|r| ≤ 0.4):         {weak:4d} pairs ({weak/len(correlations)*100:5.1f}%)")
    
    print("\n" + "="*70 + "\n")
    
    return corr_matrix


def save_cleaned_data(df, output_path='solar_final_cleaned.csv'):
    """
    STEP 4: Save the cleaned and preprocessed dataframe to CSV
    
    Parameters:
    - df: Cleaned dataframe
    - output_path: Path where to save the file
    """
    print("\n" + "="*70)
    print("STEP 4: SAVING CLEANED DATA")
    print("="*70)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    print(f"\n✅ Data saved successfully!")
    print(f"   File: {output_path}")
    print(f"   Shape: {df.shape}")
    print(f"   Rows: {len(df):,}")
    print(f"   Columns: {len(df.columns)}")
    print(f"   File size: ~{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    
    print("\n📊 Final preprocessing summary:")
    print(f"   - DateTime column created")
    print(f"   - Low variance columns removed")
    print(f"   - Bad quality rows filtered")
    print(f"   - Cloud Type encoded (10 binary columns)")
    print(f"   - Ready for machine learning!")
    
    print("="*70 + "\n")


def separate_by_seasons(df):
    """
    Separate the dataset into seasons based on month.
    
    Season Definitions (Northern Hemisphere):
    - Winter: December, January, February (12, 1, 2)
    - Spring: March, April, May (3, 4, 5)
    - Summer: June, July, August (6, 7, 8)
    - Fall: September, October, November (9, 10, 11)
    
    Parameters:
    - df: DataFrame with 'DateTime' column
    
    Returns:
    - Dictionary with seasonal DataFrames
    """
    print("\n" + "="*70)
    print("SEASONAL DATA SEPARATION")
    print("="*70)
    
    # Ensure DateTime is in datetime format
    if 'DateTime' in df.columns:
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        df['Month'] = df['DateTime'].dt.month
    else:
        print("❌ ERROR: 'DateTime' column not found!")
        return None
    
    # Define seasons
    seasons = {
        'Winter': [12, 1, 2],      # Dec, Jan, Feb
        'Spring': [3, 4, 5],       # Mar, Apr, May
        'Summer': [6, 7, 8],       # Jun, Jul, Aug
        'Fall': [9, 10, 11]        # Sep, Oct, Nov
    }
    
    season_data = {}
    
    print(f"\n📅 DATASET DATE RANGE:")
    print(f"   Start: {df['DateTime'].min()}")
    print(f"   End:   {df['DateTime'].max()}")
    print(f"   Total: {len(df):,} samples\n")
    
    print("="*70)
    print("📊 SEASONAL BREAKDOWN:")
    print("="*70)
    
    for season_name, months in seasons.items():
        # Filter data by months
        season_df = df[df['Month'].isin(months)].copy()
        season_data[season_name] = season_df
        
        if len(season_df) > 0:
            season_start = season_df['DateTime'].min()
            season_end = season_df['DateTime'].max()
            season_days = (season_end - season_start).days
            
            print(f"\n🌍 {season_name.upper()}")
            print(f"   Months: {months}")
            print(f"   Date Range: {season_start.strftime('%Y-%m-%d')} to {season_end.strftime('%Y-%m-%d')}")
            print(f"   Duration: {season_days} days")
            print(f"   Samples: {len(season_df):,}")
            print(f"   Percentage: {len(season_df)/len(df)*100:.1f}%")
            
            # Seasonal statistics
            if 'GHI' in season_df.columns:
                print(f"   GHI Stats:")
                print(f"      Mean: {season_df['GHI'].mean():.2f} W/m²")
                print(f"      Median: {season_df['GHI'].median():.2f} W/m²")
                print(f"      Min: {season_df['GHI'].min():.2f} W/m²")
                print(f"      Max: {season_df['GHI'].max():.2f} W/m²")
                print(f"      Std Dev: {season_df['GHI'].std():.2f} W/m²")
            
            if 'Temperature' in season_df.columns:
                print(f"   Temperature Stats:")
                print(f"      Mean: {season_df['Temperature'].mean():.2f} °C")
                print(f"      Min: {season_df['Temperature'].min():.2f} °C")
                print(f"      Max: {season_df['Temperature'].max():.2f} °C")
        else:
            print(f"\n🌍 {season_name.upper()}: No data found")
    
    # Drop the added Month column (not needed in final data)
    df = df.drop(columns=['Month'])
    
    print("\n" + "="*70 + "\n")
    
    return season_data, df


def visualize_seasonal_comparison(season_data, save_path='./images/solar/seasonal_comparison.png'):
    """
    Create visualization comparing GHI patterns across seasons.
    
    Includes:
    1. GHI distribution by season
    2. Average hourly pattern by season
    3. Temperature comparison
    4. Monthly statistics
    """
    print("\n" + "="*70)
    print("SEASONAL ANALYSIS VISUALIZATION")
    print("="*70)
    
    if not season_data or 'GHI' not in next(iter(season_data.values())).columns:
        print("❌ No GHI data found in seasonal data!")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    colors = {'Winter': '#4169E1', 'Spring': '#32CD32', 'Summer': '#FFD700', 'Fall': '#FF8C00'}
    seasons_order = ['Winter', 'Spring', 'Summer', 'Fall']
    
    # 1. GHI Distribution by Season
    ax1 = axes[0, 0]
    for season in seasons_order:
        if season in season_data:
            season_df = season_data[season]
            ax1.hist(season_df['GHI'], bins=50, alpha=0.6, label=season, color=colors[season], edgecolor='black')
    ax1.set_xlabel('GHI (W/m²)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax1.set_title('GHI Distribution by Season', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    
    # 2. Box plot of GHI by season
    ax2 = axes[0, 1]
    ghi_by_season = [season_data[season]['GHI'].values for season in seasons_order if season in season_data]
    bp = ax2.boxplot(ghi_by_season, labels=seasons_order, patch_artist=True)
    for patch, season in zip(bp['boxes'], seasons_order):
        patch.set_facecolor(colors[season])
        patch.set_alpha(0.7)
    ax2.set_ylabel('GHI (W/m²)', fontsize=12, fontweight='bold')
    ax2.set_title('GHI Range by Season (Box Plot)', fontsize=13, fontweight='bold')
    ax2.grid(alpha=0.3, axis='y')
    
    # 3. Mean GHI by season
    ax3 = axes[1, 0]
    season_means = [season_data[season]['GHI'].mean() for season in seasons_order if season in season_data]
    season_stds = [season_data[season]['GHI'].std() for season in seasons_order if season in season_data]
    bars = ax3.bar(seasons_order, season_means, yerr=season_stds, capsize=5, 
                   color=[colors[s] for s in seasons_order], alpha=0.7, edgecolor='black', linewidth=1.5)
    ax3.set_ylabel('Mean GHI (W/m²)', fontsize=12, fontweight='bold')
    ax3.set_title('Average GHI by Season (with Std Dev)', fontsize=13, fontweight='bold')
    ax3.grid(alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, mean, std in zip(bars, season_means, season_stds):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.0f}±{std:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 4. Temperature comparison by season
    ax4 = axes[1, 1]
    if 'Temperature' in season_data[seasons_order[0]].columns:
        temp_by_season = [season_data[season]['Temperature'].values for season in seasons_order if season in season_data]
        bp_temp = ax4.boxplot(temp_by_season, labels=seasons_order, patch_artist=True)
        for patch, season in zip(bp_temp['boxes'], seasons_order):
            patch.set_facecolor(colors[season])
            patch.set_alpha(0.7)
        ax4.set_ylabel('Temperature (°C)', fontsize=12, fontweight='bold')
        ax4.set_title('Temperature by Season', fontsize=13, fontweight='bold')
        ax4.grid(alpha=0.3, axis='y')
    else:
        ax4.text(0.5, 0.5, 'Temperature data not available', ha='center', va='center', fontsize=12)
    
    plt.suptitle('Seasonal Analysis: Solar Radiation & Temperature Patterns', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save figure
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Seasonal comparison visualization saved to: {save_path}")
    plt.close()
    
    print("="*70)


def calculate_seasonal_ghi_summary(season_data):
    """
    Calculate comprehensive GHI statistics for each season.
    
    Returns detailed summary with:
    - Average (mean) GHI
    - Peak GHI (maximum)
    - Minimum GHI (when > 0)
    - Daytime vs Nighttime breakdown
    - Production potential
    - Seasonal comparison insights
    """
    print("\n" + "="*70)
    print("SEASONAL GHI ANALYSIS SUMMARY")
    print("="*70)
    
    seasons_order = ['Winter', 'Spring', 'Summer', 'Fall']
    summary_stats = {}
    
    print("\n📊 DETAILED SEASONAL GHI STATISTICS:\n")
    
    for season_name in seasons_order:
        if season_name not in season_data:
            continue
            
        season_df = season_data[season_name]
        ghi = season_df['GHI']
        
        # Calculate comprehensive statistics
        stats = {
            'season': season_name,
            'samples': len(ghi),
            'mean': ghi.mean(),
            'median': ghi.median(),
            'std': ghi.std(),
            'min': ghi.min(),
            'max': ghi.max(),
            'q25': ghi.quantile(0.25),
            'q75': ghi.quantile(0.75),
            'iqr': ghi.quantile(0.75) - ghi.quantile(0.25),
            'nighttime': (ghi == 0).sum(),
            'daytime': (ghi > 0).sum(),
            'nighttime_pct': (ghi == 0).sum() / len(ghi) * 100,
            'daytime_pct': (ghi > 0).sum() / len(ghi) * 100,
        }
        
        summary_stats[season_name] = stats
        
        # Print detailed breakdown
        print(f"{'='*70}")
        print(f"🌍 {season_name.upper()}")
        print(f"{'='*70}")
        print(f"\n  📈 RADIATION STATISTICS:")
        print(f"     Average (Mean):     {stats['mean']:.2f} W/m²")
        print(f"     Median:             {stats['median']:.2f} W/m²")
        print(f"     Std Deviation:      {stats['std']:.2f} W/m²")
        print(f"     Range:              {stats['min']:.2f} - {stats['max']:.2f} W/m²")
        print(f"     IQR (25%-75%):      {stats['q25']:.2f} - {stats['q75']:.2f} W/m²")
        
        print(f"\n  🌤️ DAY/NIGHT BREAKDOWN:")
        print(f"     Nighttime (GHI=0):  {stats['nighttime']:,} samples ({stats['nighttime_pct']:.1f}%)")
        print(f"     Daytime (GHI>0):    {stats['daytime']:,} samples ({stats['daytime_pct']:.1f}%)")
        
        # Daytime-only statistics
        ghi_daytime = ghi[ghi > 0]
        if len(ghi_daytime) > 0:
            print(f"\n  ☀️ DAYTIME-ONLY STATISTICS (GHI > 0):")
            print(f"     Mean:               {ghi_daytime.mean():.2f} W/m²")
            print(f"     Median:             {ghi_daytime.median():.2f} W/m²")
            print(f"     Std Dev:            {ghi_daytime.std():.2f} W/m²")
            print(f"     Min (daytime):      {ghi_daytime.min():.2f} W/m²")
            print(f"     Max (daytime):      {ghi_daytime.max():.2f} W/m²")
        
        # Production tier analysis
        print(f"\n  ⚡ PRODUCTION TIER ANALYSIS:")
        low_prod = (ghi > 0) & (ghi <= 250)
        medium_prod = (ghi > 250) & (ghi <= 700)
        high_prod = (ghi > 700)
        
        print(f"     Low Production (0-250 W/m²):     {low_prod.sum():,} samples ({low_prod.sum()/len(ghi)*100:.1f}%)")
        print(f"     Medium Production (250-700 W/m²): {medium_prod.sum():,} samples ({medium_prod.sum()/len(ghi)*100:.1f}%)")
        print(f"     High Production (>700 W/m²):     {high_prod.sum():,} samples ({high_prod.sum()/len(ghi)*100:.1f}%)")
        
        # Peak hours
        peak_threshold = ghi.quantile(0.75)
        peak_hours = (ghi > peak_threshold).sum()
        print(f"\n  🔥 PEAK PRODUCTION HOURS:")
        print(f"     Top 25% threshold:  > {peak_threshold:.2f} W/m²")
        print(f"     Peak hours:         {peak_hours:,} ({peak_hours/len(ghi)*100:.1f}%)")
    
    # Cross-seasonal comparison
    print(f"\n{'='*70}")
    print("📊 CROSS-SEASONAL COMPARISON:")
    print(f"{'='*70}")
    
    # Find best and worst seasons
    if summary_stats:
        seasons_for_comparison = [s for s in seasons_order if s in summary_stats]
        means = [(s, summary_stats[s]['mean']) for s in seasons_for_comparison]
        means_sorted = sorted(means, key=lambda x: x[1], reverse=True)
        
        print(f"\n🏆 RANKING BY AVERAGE GHI:")
        for rank, (season, mean) in enumerate(means_sorted, 1):
            print(f"   {rank}. {season:10s}: {mean:7.2f} W/m²")
        
        # Calculate differences
        best_season, best_mean = means_sorted[0]
        worst_season, worst_mean = means_sorted[-1]
        difference = best_mean - worst_mean
        percent_diff = (difference / worst_mean) * 100
        
        print(f"\n📈 SEASONAL VARIATION:")
        print(f"   Best:   {best_season} ({best_mean:.2f} W/m²)")
        print(f"   Worst:  {worst_season} ({worst_mean:.2f} W/m²)")
        print(f"   Difference: {difference:.2f} W/m² ({percent_diff:.1f}% variation)")
        
        # Overall average
        overall_mean = np.mean([summary_stats[s]['mean'] for s in seasons_for_comparison])
        print(f"\n   Overall Annual Average: {overall_mean:.2f} W/m²")
    
    # Key insights
    print(f"\n{'='*70}")
    print("💡 KEY INSIGHTS & FINDINGS:")
    print(f"{'='*70}")
    
    if summary_stats and 'Summer' in summary_stats and 'Winter' in summary_stats:
        summer_mean = summary_stats['Summer']['mean']
        winter_mean = summary_stats['Winter']['mean']
        
        print(f"\n1. ☀️ SUMMER vs ❄️ WINTER:")
        print(f"   • Summer has {summer_mean/winter_mean:.1f}x higher GHI than Winter")
        print(f"   • Summer avg: {summer_mean:.2f} W/m² vs Winter avg: {winter_mean:.2f} W/m²")
        print(f"   • Best for solar generation: SUMMER")
        
        print(f"\n2. 📈 SEASONAL PATTERNS:")
        print(f"   • Summer: Peak generation potential")
        print(f"   • Winter: Lowest generation (shorter days, lower sun angle)")
        print(f"   • Transition seasons (Spring/Fall): Moderate generation")
        
        print(f"\n3. 💼 PRACTICAL IMPLICATIONS:")
        print(f"   • Build solar capacity based on SUMMER production")
        print(f"   • Winter requires battery storage or grid backup")
        print(f"   • Expect {percent_diff:.0f}% variation throughout year")
        
        print(f"\n4. 📊 FORECASTING STRATEGY:")
        print(f"   • Train separate seasonal models for better accuracy")
        print(f"   • Or use XGBoost with seasonal features")
        print(f"   • Account for ~{percent_diff:.0f}% seasonal variation in predictions")
    
    print(f"\n{'='*70}\n")
    
    return summary_stats


def analyze_hourly_sunlight(df, save_path='./images/solar/hourly_sunlight_pattern.png'):
    """
    Analyze GHI by hour of day to understand daily sunlight pattern.
    
    Creates visualization showing:
    1. Average GHI at each hour
    2. Peak sunlight hours
    3. Sunrise/sunset timing
    4. Variability by hour
    
    Parameters:
    - df: DataFrame with 'DateTime' and 'GHI' columns
    - save_path: Path to save the visualization
    """
    print("\n" + "="*70)
    print("HOURLY SUNLIGHT ANALYSIS")
    print("="*70)
    
    # Extract hour from DateTime
    if 'DateTime' in df.columns:
        df['Hour'] = pd.to_datetime(df['DateTime']).dt.hour
    elif 'Hour' not in df.columns:
        print("❌ ERROR: 'DateTime' or 'Hour' column not found!")
        return
    
    # Group by hour and calculate statistics
    hourly_stats = df.groupby('Hour')['GHI'].agg([
        ('mean', 'mean'),
        ('median', 'median'),
        ('std', 'std'),
        ('min', 'min'),
        ('max', 'max'),
        ('count', 'count')
    ]).reset_index()
    
    # Calculate confidence interval (95%)
    hourly_stats['ci_lower'] = hourly_stats['mean'] - 1.96 * (hourly_stats['std'] / np.sqrt(hourly_stats['count']))
    hourly_stats['ci_upper'] = hourly_stats['mean'] + 1.96 * (hourly_stats['std'] / np.sqrt(hourly_stats['count']))
    hourly_stats['ci_lower'] = hourly_stats['ci_lower'].clip(lower=0)
    
    # Display hourly summary
    print("\n📊 HOURLY GHI STATISTICS (All hours: 0-23):\n")
    print(f"{'Hour':<6} {'Mean GHI':<12} {'Median':<12} {'Std Dev':<12} {'Min':<10} {'Max':<10} {'Samples':<10}")
    print("="*72)
    
    for idx, row in hourly_stats.iterrows():
        hour = int(row['Hour'])
        mean_ghi = row['mean']
        median_ghi = row['median']
        std_ghi = row['std']
        min_ghi = row['min']
        max_ghi = row['max']
        count = int(row['count'])
        
        # Hour labels
        if hour == 0:
            hour_label = "00:00"
        elif hour == 12:
            hour_label = "12:00"
        else:
            hour_label = f"{hour:02d}:00"
        
        print(f"{hour_label:<6} {mean_ghi:>10.2f}  {median_ghi:>10.2f}  {std_ghi:>10.2f}  {min_ghi:>8.2f}  {max_ghi:>8.2f}  {count:>8d}")
    
    print("="*72)
    
    # Key statistics
    print("\n🔍 KEY HOURLY INSIGHTS:")
    print("-"*70)
    
    # Peak hour
    peak_hour_idx = hourly_stats['mean'].idxmax()
    peak_hour = int(hourly_stats.loc[peak_hour_idx, 'Hour'])
    peak_ghi = hourly_stats.loc[peak_hour_idx, 'mean']
    print(f"\n  ⭐ PEAK SUNLIGHT HOUR: {peak_hour:02d}:00 ({peak_ghi:.2f} W/m²)")
    
    # Sunrise and sunset (when GHI becomes > 0 and = 0)
    daytime_hours = hourly_stats[hourly_stats['mean'] > 0]
    if len(daytime_hours) > 0:
        sunrise_hour = int(daytime_hours['Hour'].min())
        sunset_hour = int(daytime_hours['Hour'].max())
        daylight_hours = sunset_hour - sunrise_hour + 1
        
        print(f"\n  🌅 SUNRISE: ~{sunrise_hour:02d}:00")
        print(f"  🌇 SUNSET:  ~{sunset_hour:02d}:00")
        print(f"  ☀️  DAYLIGHT DURATION: {daylight_hours} hours")
    
    # Production tiers by hour
    print(f"\n  ⚡ PRODUCTION BY HOUR:")
    high_prod = (hourly_stats['mean'] > 700).sum()
    medium_prod = ((hourly_stats['mean'] > 250) & (hourly_stats['mean'] <= 700)).sum()
    low_prod = ((hourly_stats['mean'] > 0) & (hourly_stats['mean'] <= 250)).sum()
    night = (hourly_stats['mean'] == 0).sum()
    
    print(f"     High output (>700 W/m²):     {high_prod} hours")
    print(f"     Medium output (250-700 W/m²): {medium_prod} hours")
    print(f"     Low output (0-250 W/m²):     {low_prod} hours")
    print(f"     Nighttime (0 W/m²):          {night} hours")
    
    # Create visualization
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    # Plot 1: Average GHI by hour with confidence interval
    ax1 = axes[0]
    hours = hourly_stats['Hour'].values
    means = hourly_stats['mean'].values
    stds = hourly_stats['std'].values
    ci_lower = hourly_stats['ci_lower'].values
    ci_upper = hourly_stats['ci_upper'].values
    
    # Fill confidence interval
    ax1.fill_between(hours, ci_lower, ci_upper, alpha=0.2, color='blue', label='95% Confidence Interval')
    
    # Plot mean line
    ax1.plot(hours, means, color='darkblue', linewidth=3, marker='o', markersize=8, label='Average GHI')
    
    # Highlight peak hour
    ax1.scatter([peak_hour], [peak_ghi], color='red', s=200, zorder=5, label=f'Peak Hour ({peak_hour:02d}:00)', marker='*')
    
    # Shade day/night regions
    ax1.axvspan(0, sunrise_hour-0.5, alpha=0.1, color='gray', label='Nighttime')
    ax1.axvspan(sunset_hour+0.5, 23.5, alpha=0.1, color='gray')
    
    ax1.set_xlabel('Hour of Day', fontsize=13, fontweight='bold')
    ax1.set_ylabel('GHI (W/m²)', fontsize=13, fontweight='bold')
    ax1.set_title('Average Global Horizontal Irradiance by Hour of Day', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(0, 24))
    ax1.set_xticklabels([f'{h:02d}:00' for h in range(24)], rotation=45, ha='right')
    ax1.grid(alpha=0.3, linestyle='--')
    ax1.legend(fontsize=11, loc='upper left')
    ax1.set_ylim(bottom=0)
    
    # Plot 2: Variability (Std Dev) by hour
    ax2 = axes[1]
    colors_bar = ['red' if std > 100 else 'orange' if std > 50 else 'green' for std in stds]
    bars = ax2.bar(hours, stds, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=1)
    
    ax2.set_xlabel('Hour of Day', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Standard Deviation (W/m²)', fontsize=13, fontweight='bold')
    ax2.set_title('GHI Variability by Hour (Standard Deviation)', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(0, 24))
    ax2.set_xticklabels([f'{h:02d}:00' for h in range(24)], rotation=45, ha='right')
    ax2.grid(alpha=0.3, axis='y', linestyle='--')
    
    # Shade day/night in second plot
    ax2.axvspan(0, sunrise_hour-0.5, alpha=0.1, color='gray')
    ax2.axvspan(sunset_hour+0.5, 23.5, alpha=0.1, color='gray')
    
    # Add legend for colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', alpha=0.7, label='High Variability (>100 W/m²)'),
        Patch(facecolor='orange', alpha=0.7, label='Moderate Variability (50-100 W/m²)'),
        Patch(facecolor='green', alpha=0.7, label='Low Variability (<50 W/m²)')
    ]
    ax2.legend(handles=legend_elements, fontsize=11, loc='upper left')
    
    plt.tight_layout()
    
    # Save figure
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Hourly sunlight pattern visualization saved to: {save_path}")
    plt.close()
    
    print("="*70 + "\n")
    
    # EXPLANATION PARAGRAPH
    print("📝 HOURLY SUNLIGHT PATTERN EXPLANATION:")
    print("="*70)
    explanation = f"""
The hourly Global Horizontal Irradiance (GHI) pattern exhibits a clear and predictable 
bell-curve distribution, characterizing the daily solar cycle at this location. The plot 
shows virtually zero radiation during nighttime hours ({sunrise_hour:02d}:00-{sunrise_hour-1:02d}:00 and 
{sunset_hour+1:02d}:00-23:00), which is expected as the sun is below the horizon. Radiation 
begins increasing after sunrise around {sunrise_hour:02d}:00 and peaks at {peak_hour:02d}:00 with an 
average of {peak_ghi:.1f} W/m², representing when the sun reaches its highest point in the sky 
(solar noon). This peak timing aligns with solar geometry—the sun is highest above the 
horizon at solar noon, direct radiation is maximum, and there is minimal atmospheric path 
length for light to travel through. After peak hours, GHI decreases symmetrically toward 
sunset around {sunset_hour:02d}:00 as the sun's zenith angle increases. The {daylight_hours}-hour daylight 
period ({sunrise_hour:02d}:00-{sunset_hour:02d}:00) represents the productive window for solar generation. 
The standard deviation pattern shows HIGH VARIABILITY during mid-day hours (indicated by 
red bars), which reflects the impact of transient cloud cover during peak production times, 
while morning and evening show lower variability (green) due to consistently low sun angles. 
This pattern is fundamental to solar forecasting and system design—peak generation windows 
are short (~4-6 peak hours), making energy storage critical for handling the daily mismatch 
between supply and demand, and high mid-day variability requires accurate cloud prediction 
for grid stability.
"""
    print(explanation)
    print("="*70 + "\n")


def analyze_temperature_ghi_relationship(df, save_path='./images/solar/temperature_ghi_relationship.png'):
    """
    Analyze and visualize the relationship between Temperature and GHI.
    
    Creates comprehensive visualization showing:
    1. Scatter plot of Temperature vs GHI
    2. Hexbin density plot
    3. Correlation analysis
    4. Hourly pattern relationships
    5. Seasonal relationships
    
    Parameters:
    - df: DataFrame with 'Temperature' and 'GHI' columns
    - save_path: Path to save the visualization
    """
    print("\n" + "="*70)
    print("TEMPERATURE vs GHI RELATIONSHIP ANALYSIS")
    print("="*70)
    
    # Check for required columns
    if 'Temperature' not in df.columns or 'GHI' not in df.columns:
        print("❌ ERROR: 'Temperature' or 'GHI' column not found!")
        return
    
    # Extract required data
    temp = df['Temperature'].dropna()
    ghi = df.loc[temp.index, 'GHI'].dropna()
    
    # Align indices
    common_idx = temp.index.intersection(ghi.index)
    temp = temp[common_idx]
    ghi = ghi[common_idx]
    
    print(f"\n✓ Analyzing {len(temp)} samples")
    
    # 1. CORRELATION ANALYSIS
    correlation = np.corrcoef(temp, ghi)[0, 1]
    
    from scipy.stats import pearsonr, spearmanr
    pearson_corr, pearson_pval = pearsonr(temp, ghi)
    spearman_corr, spearman_pval = spearmanr(temp, ghi)
    
    print("\n📊 CORRELATION ANALYSIS:")
    print("-"*70)
    print(f"  Pearson Correlation:  {pearson_corr:+.4f} (p-value: {pearson_pval:.2e})")
    print(f"  Spearman Correlation: {spearman_corr:+.4f} (p-value: {spearman_pval:.2e})")
    
    # Interpretation
    if abs(pearson_corr) < 0.3:
        corr_strength = "WEAK"
        interpretation = "Temperature has minimal direct impact on GHI"
    elif abs(pearson_corr) < 0.7:
        corr_strength = "MODERATE"
        interpretation = "Temperature shows moderate relationship with GHI"
    else:
        corr_strength = "STRONG"
        interpretation = "Temperature strongly predicts GHI variations"
    
    print(f"\n  💡 Interpretation: {corr_strength} correlation - {interpretation}")
    
    # 2. BASIC STATISTICS
    print("\n📈 STATISTICAL SUMMARY:")
    print("-"*70)
    print(f"\n  TEMPERATURE:")
    print(f"    Mean:     {temp.mean():.2f} °C")
    print(f"    Median:   {temp.median():.2f} °C")
    print(f"    Std Dev:  {temp.std():.2f} °C")
    print(f"    Range:    {temp.min():.2f} - {temp.max():.2f} °C")
    
    print(f"\n  GHI:")
    print(f"    Mean:     {ghi.mean():.2f} W/m²")
    print(f"    Median:   {ghi.median():.2f} W/m²")
    print(f"    Std Dev:  {ghi.std():.2f} W/m²")
    print(f"    Range:    {ghi.min():.2f} - {ghi.max():.2f} W/m²")
    
    # 3. CONDITIONAL ANALYSIS
    print("\n🌡️ GHI BY TEMPERATURE RANGES:")
    print("-"*70)
    
    temp_ranges = [
        ("Cold", temp.min(), 0),
        ("Cool", 0, 10),
        ("Moderate", 10, 20),
        ("Warm", 20, 30),
        ("Hot", 30, temp.max())
    ]
    
    for range_name, range_min, range_max in temp_ranges:
        mask = (temp >= range_min) & (temp < range_max)
        if mask.sum() > 0:
            ghi_in_range = ghi[mask]
            print(f"\n  {range_name} ({range_min}°C - {range_max}°C):")
            print(f"    Samples:  {mask.sum():,}")
            print(f"    Avg GHI:  {ghi_in_range.mean():.2f} W/m²")
            print(f"    Max GHI:  {ghi_in_range.max():.2f} W/m²")
    
    # 4. CREATE VISUALIZATION
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Plot 1: Scatter plot with regression line
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.scatter(temp, ghi, alpha=0.3, s=20, color='blue', edgecolors='none')
    
    # Add regression line
    z = np.polyfit(temp, ghi, 1)
    p = np.poly1d(z)
    temp_sorted = np.sort(temp)
    ax1.plot(temp_sorted, p(temp_sorted), "r-", linewidth=3, label=f'Linear fit: y={z[0]:.2f}x+{z[1]:.2f}')
    
    ax1.set_xlabel('Temperature (°C)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('GHI (W/m²)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Temperature vs GHI (Pearson r = {pearson_corr:+.3f})', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)
    
    # Plot 2: Hexbin density plot
    ax2 = fig.add_subplot(gs[0, 2])
    hexbin = ax2.hexbin(temp, ghi, gridsize=20, cmap='YlOrRd', mincnt=1)
    ax2.set_xlabel('Temperature (°C)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('GHI (W/m²)', fontsize=12, fontweight='bold')
    ax2.set_title('Density: Temperature vs GHI', fontsize=13, fontweight='bold')
    cbar = plt.colorbar(hexbin, ax=ax2)
    cbar.set_label('Count', fontsize=10, fontweight='bold')
    
    # Plot 3: Distribution comparison
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.hist(temp, bins=50, alpha=0.7, color='hotpink', edgecolor='black')
    ax3.set_xlabel('Temperature (°C)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax3.set_title('Temperature Distribution', fontsize=13, fontweight='bold')
    ax3.axvline(temp.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean = {temp.mean():.1f}°C')
    ax3.legend(fontsize=10)
    ax3.grid(alpha=0.3, axis='y')
    
    # Plot 4: GHI distribution
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.hist(ghi, bins=50, alpha=0.7, color='gold', edgecolor='black')
    ax4.set_xlabel('GHI (W/m²)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax4.set_title('GHI Distribution', fontsize=13, fontweight='bold')
    ax4.axvline(ghi.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean = {ghi.mean():.1f} W/m²')
    ax4.legend(fontsize=10)
    ax4.grid(alpha=0.3, axis='y')
    
    # Plot 5: Box plot by temperature ranges
    ax5 = fig.add_subplot(gs[1, 2])
    temp_labels = []
    ghi_by_temp = []
    
    for range_name, range_min, range_max in temp_ranges:
        mask = (temp >= range_min) & (temp < range_max)
        if mask.sum() > 0:
            temp_labels.append(f"{range_name}\n({range_min}°C-{range_max}°C)")
            ghi_by_temp.append(ghi[mask].values)
    
    bp = ax5.boxplot(ghi_by_temp, labels=temp_labels, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    
    ax5.set_ylabel('GHI (W/m²)', fontsize=12, fontweight='bold')
    ax5.set_title('GHI by Temperature Ranges', fontsize=13, fontweight='bold')
    ax5.grid(alpha=0.3, axis='y')
    
    plt.suptitle('Temperature vs Global Horizontal Irradiance (GHI) Analysis',
                 fontsize=14, fontweight='bold', y=0.995)
    
    # Save figure
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Temperature-GHI relationship plot saved to: {save_path}")
    plt.close()
    
    print("="*70)
    
    # 5. DETAILED EXPLANATION
    print("\n📝 RELATIONSHIP EXPLANATION:")
    print("="*70)
    
    explanation = f"""
The relationship between Temperature and Global Horizontal Irradiance (GHI) shows a 
{corr_strength} POSITIVE correlation (r = {pearson_corr:+.3f}), with {interpretation.lower()}. 
This pattern arises from the fundamental physics of solar radiation: temperature and GHI 
are not directly causally linked, but rather are both consequences of the same underlying 
factor—cloud cover and atmospheric clarity. During clear-sky days with minimal clouds, 
solar radiation (GHI) reaches maximum values, which simultaneously heats the atmosphere 
and surface, resulting in higher temperatures. Conversely, on cloudy days with thick cloud 
cover blocking incoming radiation, both GHI and temperature are reduced. The moderate-to-weak 
correlation occurs because temperature is affected by multiple factors beyond solar radiation, 
including time of day (diurnal cycle), seasonal variation, wind speed, humidity, and surface 
properties that regulate how efficiently solar energy converts to heat. Additionally, there 
is a time lag effect: ground temperature gradually increases throughout the morning as solar 
radiation accumulates, reaching peak temperatures in early afternoon (around 14:00-16:00), 
while GHI peaks earlier near solar noon (12:00-13:00) when the sun's elevation angle is 
maximum. This lag means that at any given hour, temperature doesn't perfectly reflect 
instantaneous GHI values. For solar forecasting, this relationship suggests that temperature 
alone is an IMPERFECT predictor of solar irradiance—atmospheric clearness (cloud optical 
depth), aerosol concentration, and humidity are better direct predictors. However, combining 
temperature with other features (humidity, pressure, hour of day, season) in machine learning 
models (like XGBoost) can capture these indirect relationships and improve forecasting accuracy.
"""
    
    print(explanation)
    print("="*70 + "\n")


def normalize_data_minmax(df, exclude_cols=None):
    """
    MinMax Normalization (Scaling to [0, 1] range)
    
    FORMULA: X_normalized = (X - X_min) / (X_max - X_min)
    
    CHARACTERISTICS:
    - Rescales data to [0, 1] range
    - Preserves the original distribution shape
    - Sensitive to outliers (outliers become 0 or 1)
    - Useful when you need bounded values
    - Works well with algorithms expecting data in fixed range (Neural Networks)
    
    PROS:
    ✓ Data range predictable [0, 1]
    ✓ Easy to interpret
    ✓ Preserves relationships between data points
    
    CONS:
    ✗ Sensitive to outliers since max/min are used
    ✗ Changes if new extreme values are added later
    ✗ Not suitable for data with heavy-tailed distributions
    """
    from sklearn.preprocessing import MinMaxScaler
    
    if exclude_cols is None:
        exclude_cols = ['DateTime']
    
    df_normalized = df.copy()
    numeric_cols = df_normalized.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    scaler = MinMaxScaler()
    df_normalized[numeric_cols] = scaler.fit_transform(df_normalized[numeric_cols])
    
    return df_normalized, numeric_cols, 'MinMax'


def normalize_data_zscore(df, exclude_cols=None):
    """
    Z-Score Normalization (Standardization)
    
    FORMULA: X_normalized = (X - μ) / σ
    where μ = mean, σ = standard deviation
    
    CHARACTERISTICS:
    - Centers data around 0 with std dev = 1
    - Data follows standard normal distribution N(0, 1)
    - NOT bounded - values can be outside [-3, 3] for extreme outliers
    - Less sensitive to outliers than MinMax
    - Useful for statistical methods assuming normality
    
    PROS:
    ✓ Less sensitive to outliers
    ✓ Centers data (mean=0)
    ✓ Standard deviation = 1 (interpretable)
    ✓ Better for statistical models (linear regression, logistic regression)
    ✓ Works with unbounded data
    
    CONS:
    ✗ Not bound to specific range
    ✗ Extreme outliers can produce very large values
    ✗ Assumes data is normally distributed
    """
    from sklearn.preprocessing import StandardScaler
    
    if exclude_cols is None:
        exclude_cols = ['DateTime']
    
    df_normalized = df.copy()
    numeric_cols = df_normalized.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    scaler = StandardScaler()
    df_normalized[numeric_cols] = scaler.fit_transform(df_normalized[numeric_cols])
    
    return df_normalized, numeric_cols, 'ZScore'


def visualize_normalization_comparison(df, df_minmax, df_zscore, numeric_cols):
    """
    Create detailed visualization comparing original data with two normalization techniques
    """
    # Select sample columns for visualization (max 6 columns)
    sample_cols = numeric_cols[:6] if len(numeric_cols) > 6 else numeric_cols
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, len(sample_cols), hspace=0.4, wspace=0.3)
    
    for idx, col in enumerate(sample_cols):
        # Original Data
        ax1 = fig.add_subplot(gs[0, idx])
        ax1.hist(df[col], bins=50, color='blue', alpha=0.7, edgecolor='black')
        ax1.set_title(f'{col}\n(Original Data)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Frequency', fontsize=10)
        if idx == 0:
            ax1.text(-0.3, 1.15, 'ORIGINAL', transform=ax1.transAxes, 
                    fontsize=12, fontweight='bold', color='blue')
        
        # MinMax Normalization
        ax2 = fig.add_subplot(gs[1, idx])
        ax2.hist(df_minmax[col], bins=50, color='green', alpha=0.7, edgecolor='black')
        ax2.set_title(f'{col}\n(MinMax: [0,1])', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Frequency', fontsize=10)
        ax2.set_xlim(-0.1, 1.1)
        if idx == 0:
            ax2.text(-0.3, 1.15, 'MINMAX', transform=ax2.transAxes, 
                    fontsize=12, fontweight='bold', color='green')
        
        # Z-Score Normalization
        ax3 = fig.add_subplot(gs[2, idx])
        ax3.hist(df_zscore[col], bins=50, color='red', alpha=0.7, edgecolor='black')
        ax3.set_title(f'{col}\n(Z-Score: μ=0, σ=1)', fontsize=11, fontweight='bold')
        ax3.set_xlabel('Value', fontsize=10)
        ax3.set_ylabel('Frequency', fontsize=10)
        if idx == 0:
            ax3.text(-0.3, 1.15, 'Z-SCORE', transform=ax3.transAxes, 
                    fontsize=12, fontweight='bold', color='red')
    
    plt.suptitle('Data Normalization Comparison: Original vs MinMax vs Z-Score', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    import os
    os.makedirs('../images/solar/', exist_ok=True)
    save_path = '../images/solar/normalization_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Normalization comparison plot saved to: {save_path}")
    plt.close()


def create_normalization_summary_report(df, df_minmax, df_zscore, numeric_cols):
    """
    Create detailed statistical comparison report of normalization techniques
    """
    print("\n" + "="*80)
    print("DATA NORMALIZATION ANALYSIS & COMPARISON")
    print("="*80)
    
    # Select sample columns for detailed report
    sample_cols = numeric_cols[:6] if len(numeric_cols) > 6 else numeric_cols
    
    for col in sample_cols:
        print(f"\n📊 COLUMN: {col}")
        print("-"*80)
        
        print(f"\n🔵 ORIGINAL DATA:")
        print(f"  Mean:          {df[col].mean():>12.4f}")
        print(f"  Std Dev:       {df[col].std():>12.4f}")
        print(f"  Min:           {df[col].min():>12.4f}")
        print(f"  Max:           {df[col].max():>12.4f}")
        print(f"  Range:         {df[col].max() - df[col].min():>12.4f}")
        print(f"  Skewness:      {df[col].skew():>12.4f}")
        print(f"  Kurtosis:      {df[col].kurtosis():>12.4f}")
        
        print(f"\n🟢 MINMAX NORMALIZED (Range: [0, 1]):")
        print(f"  Mean:          {df_minmax[col].mean():>12.4f}")
        print(f"  Std Dev:       {df_minmax[col].std():>12.4f}")
        print(f"  Min:           {df_minmax[col].min():>12.4f}")
        print(f"  Max:           {df_minmax[col].max():>12.4f}")
        print(f"  Range:         {df_minmax[col].max() - df_minmax[col].min():>12.4f}")
        print(f"  Skewness:      {df_minmax[col].skew():>12.4f} (✓ Unchanged - distribution shape preserved)")
        print(f"  Kurtosis:      {df_minmax[col].kurtosis():>12.4f} (✓ Unchanged)")
        
        print(f"\n🔴 Z-SCORE NORMALIZED (Mean: 0, Std Dev: 1):")
        print(f"  Mean:          {df_zscore[col].mean():>12.4f} (✓ ≈ 0)")
        print(f"  Std Dev:       {df_zscore[col].std():>12.4f} (✓ ≈ 1)")
        print(f"  Min:           {df_zscore[col].min():>12.4f}")
        print(f"  Max:           {df_zscore[col].max():>12.4f}")
        print(f"  Range:         {df_zscore[col].max() - df_zscore[col].min():>12.4f}")
        print(f"  Skewness:      {df_zscore[col].skew():>12.4f} (✓ Unchanged - distribution shape preserved)")
        print(f"  Kurtosis:      {df_zscore[col].kurtosis():>12.4f} (✓ Unchanged)")
    
    # FINAL RECOMMENDATIONS
    print("\n\n" + "="*80)
    print("💡 NORMALIZATION METHOD RECOMMENDATIONS")
    print("="*80)
    
    print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│ USE MINMAX NORMALIZATION WHEN:                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│ ✓ Building Neural Networks (bounded [0,1] input expected)                   │
│ ✓ Using algorithms sensitive to feature magnitude (KNN, K-means)            │
│ ✓ Input units need to be interpretable [0 to 1]                            │
│ ✓ No significant outliers in your dataset                                   │
│ ✓ All features have similar importance in the output                        │
│                                                                             │
│ EFFECT: Compresses all values to [0,1] range while preserving relationships │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ USE Z-SCORE NORMALIZATION WHEN:                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│ ✓ Using statistical models (Linear/Logistic Regression, SVM)               │
│ ✓ Data is normally distributed or assumptions of normality apply            │
│ ✓ Dataset contains outliers (less sensitive than MinMax)                    │
│ ✓ Using algorithms assuming zero mean & unit variance                      │
│ ✓ Interpretability: values >3 or <-3 are statistical outliers              │
│                                                                             │
│ EFFECT: Centers data (μ=0) and scales by std dev (σ=1) - data unbounded   │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ FOR SOLAR FORECASTING (This Project):                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│ 🎯 RECOMMENDATION: Z-SCORE NORMALIZATION                                    │
│                                                                             │
│ RATIONALE:                                                                  │
│ • Solar data has natural outliers (extreme weather, sensor errors)         │
│ • Robust statistical models benefit from zero-centered data                │
│ • Bimodal distribution (day/night) better handled by Z-score              │
│ • Machine learning models (XGBoost, SVM) often perform better with Z-norm  │
│ • Interpretable: values beyond ±3 σ are statistically significant anomalies│
│                                                                             │
│ SECONDARY: MinMax if using Neural Networks for deep learning               │
└─────────────────────────────────────────────────────────────────────────────┘
""")
    
    print("\n✅ NORMALIZATION ANALYSIS COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    df = read_solar_data(file_path)
    if df is not None:
        print(f"Initial shape: {df.shape}\n")
        
        # PREPROCESSING PIPELINE
        df = drop_columns_with_high_nulls(df)
        df = create_datetime_column(df)
        
        # Analyze variance and redundancy
        variance_analysis = analyze_column_variance(df)
        
        # Get low variance columns
        low_var_cols = variance_analysis[(variance_analysis['Variance'] > 0) & (variance_analysis['Variance'] < 0.01)]['Column'].tolist()
        
        # Analyze correlation
        if low_var_cols:
            corr_matrix = analyze_column_correlation(df, low_var_cols)
            
            # Remove low variance columns
            df = drop_low_variance_columns(df, low_var_cols)
        
        # Filter by Fill Flag and drop the column
        df = filter_and_drop_fill_flag(df)
        
        # STEP 1: Outlier Detection (Statistical)
        outlier_info = detect_outliers_iqr(df, display_details=True)
        
        # VISUALIZE: Show outlier rows
        visualize_outliers(df, outlier_info, num_to_show=5)
        
        # SUMMARY: Show outlier statistics
        show_outlier_statistics(df, outlier_info)
        
        # STEP 2: Domain Validation (Solar-Specific)
        validation_results, invalid_rows = validate_solar_domain(df)
        
        # OPTIONAL: Remove invalid rows (uncomment to enable)
        # df = remove_invalid_rows(df, invalid_rows)
        
        # STEP 3: Encode Categorical Variables
        df = encode_categorical_variables(df)
        
        # TIME SERIES FEATURE ENGINEERING
        df = create_time_series_features(df)
        
        # Drop rows with NaN values created by lag/rolling features
        print(f"Removing rows with NaN values from lag/rolling features...")
        initial_rows = len(df)
        df = df.dropna()
        removed_rows = initial_rows - len(df)
        print(f"   Removed {removed_rows} rows with NaN values")
        print(f"   Remaining: {len(df)} rows\n")
        
        # GHI DISTRIBUTION ANALYSIS
        visualize_ghi_distribution(df)
        
        # CORRELATION ANALYSIS: Create and analyze correlation matrix
        corr_matrix = create_correlation_matrix(df)
        
        # STEP 4: Save the cleaned data
        save_cleaned_data(df)
        
        # SEASONAL SEPARATION & ANALYSIS
        season_data, df = separate_by_seasons(df)
        visualize_seasonal_comparison(season_data)
        seasonal_summary = calculate_seasonal_ghi_summary(season_data)
        
        # HOURLY SUNLIGHT ANALYSIS
        analyze_hourly_sunlight(df)
        
        # TEMPERATURE vs GHI RELATIONSHIP ANALYSIS
        analyze_temperature_ghi_relationship(df)
        
        # DATA NORMALIZATION ANALYSIS (Task 9)
        print("\n" + "="*80)
        print("TASK 9: DATA NORMALIZATION USING TWO TECHNIQUES")
        print("="*80)
        
        # Apply MinMax Normalization
        df_minmax, numeric_cols, _ = normalize_data_minmax(df)
        
        # Apply Z-Score Normalization
        df_zscore, numeric_cols, _ = normalize_data_zscore(df)
        
        # Create comprehensive visualization
        visualize_normalization_comparison(df, df_minmax, df_zscore, numeric_cols)
        
        # Create detailed statistical report
        create_normalization_summary_report(df, df_minmax, df_zscore, numeric_cols)
        
        # Save normalized datasets
        print("\n📁 SAVING NORMALIZED DATASETS...")
        print("-"*80)
        
        try:
            df_minmax.to_csv('solar_normalized_minmax.csv', index=False)
            print(f"✅ MinMax normalized data saved to: solar_normalized_minmax.csv")
            print(f"   Shape: {df_minmax.shape}")
        except Exception as e:
            print(f"❌ Error saving MinMax data: {e}")
        
        try:
            df_zscore.to_csv('solar_normalized_zscore.csv', index=False)
            print(f"✅ Z-Score normalized data saved to: solar_normalized_zscore.csv")
            print(f"   Shape: {df_zscore.shape}")
        except Exception as e:
            print(f"❌ Error saving Z-Score data: {e}")
        
        print(f"\n✓ Final dataset shape: {df.shape}")
        print(f"✓ Final column count: {len(df.columns)}")
        print(f"\nColumn names:")
        print(df.columns.tolist())
        print(f"\nFirst 5 columns: {df.columns[:5].tolist()}")
        print(f"Last 5 columns: {df.columns[-5:].tolist()}")
        print(df.head())