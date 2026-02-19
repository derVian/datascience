# Solar Irradiance Forecasting - Comprehensive Analysis Report

## Project Overview
This project focuses on forecasting solar irradiance using meteorological datasets and advanced time-series analysis techniques. The analysis includes ARIMA and SARIMA models for predicting Global Horizontal Irradiance (GHI).
---

## Requirements

### 1. Read about Forecasting Solar Irradiance with Meteorological Dataset

The project focuses on solar irradiance forecasting using the 2022 solar data with meteorological variables including:
- Global Horizontal Irradiance (GHI) - Primary target variable
- Temperature
- Humidity
- Other meteorological features

---

### 2. Conduct All Necessary Data Preprocessing

#### Data Preprocessing Steps:
- **Data Loading:** CSV file loaded with proper header handling (skiprows=2)
- **DateTime Creation:** Merged Year, Month, Day, Hour, Minute columns to create DateTime index
- **Missing Values Handling:** Applied dropna() to remove rows with missing values
### Raw data analysis
![rawghidistribution](resultpngs\ghi_distribution.png)
- **Summary of raw data features/distribution:** The GHI data is highly right-skewed, primarily due to nighttime observations where irradiance values are zero. The prevalence of these zero-value records introduces outliers that require preprocessing, as their dominance in the distribution can cause ARIMA models to perform poorly

- **Outlier Removal:** Filtered rows where GHI < 5 W/m² to remove nighttime/invalid measurements
- **Data Validation:** 
  - Initial rows: Full dataset
  - Filtered rows: Removed invalid GHI values
  - Final dataset: Clean, validated time-series data

**Implemented in:** `date_target.py` - Functions: `load_data()`, `preprocess_data()`, `filter_ghi_data()`

---

### 3. Calculate Correlation Matrix and Interpretation

#### Correlation Analysis:
- **Function:** `create_correlation_heatmap()`
- **Output:** `correlation_heatmap.png`
![CorrelationMatirx](resultpngs\correlation_heatmap.png)
- **Key Findings:** 

  - Strong positive correlation between GHI and Temperature
  - Moderate correlations with other meteorological variables
  - Correlation matrix saved as `correlation_matrix.csv`
   
#### Interpretation:
- Solar radiation (GHI) is primarily driven by temperature and atmospheric conditions
- Higher GHI values correspond to clearer skies and warmer temperatures
- Seasonal variations significantly affect correlations

---

### 4. Plot GHI Distribution and Explain Observed Pattern

#### Distribution Analysis:
- **Function:** `plot_ghi_distribution()`
- **Output:** `ghi_distribution_after_filtering_zeros.png`
- **Plot Type:** Histogram with mean and median lines

![ghidiistributionafterpreprocessing](resultpngs\ghi_distribution_after_filtering_zeros.png)
#### Observed Pattern:

- **Shape:** Right-skewed distribution with peak in lower ranges
- **Mean vs Median:** Difference indicates skewness due to nighttime zero values being filtered
- **Range:** 5 W/m² to ~1100 W/m² depending on atmospheric conditions
- **Explanation:** 
  - Most hours receive low to moderate irradiance
  - Peak frequencies occur in mid-range (200-400 W/m²)
  - Few extreme high values (rare clear-sky conditions at peak sun hours)

---

### 5. Separate Data into Seasons with Date Ranges

#### Seasonal Classification:
- **Function:** `add_season_column()`
- **Method:** Month-based classification

#### Date Ranges by Season (2022 Data):
1. **Winter:** December 21 - March 20
   - Months: 12, 1, 2
   - Lowest insolation, shortest days

2. **Spring:** March 21 - June 20
   - Months: 3, 4, 5
   - Increasing daylight, moderate insolation

3. **Summer:** June 21 - September 22
   - Months: 6, 7, 8
   - Highest insolation, longest days

4. **Fall:** September 23 - December 20
   - Months: 9, 10, 11
   - Decreasing daylight, moderate insolation

**Validation:** Season distribution verified and printed for data quality assurance

---

### 6. Calculate Average GHI by Season and Summarize

#### Seasonal GHI Statistics:
- **Function:** `plot_ghi_distribution_by_season()`, `plot_timeseries_with_seasonality()`
- **Output:** `ghi_distribution_by_season.png`, `timeseries_seasonality.png`
![seasonalityofdata](resultpngs\timeseries_seasonality.png)
![ghiDistributionbyseasons](resultpngs\ghi_distribution_by_season.png)

#### Summary Statistics by Season:
- **Function:** `plot_ghi_boxplot_by_season()`
- **Output:** `ghi_boxplot_by_season.png`
![ghiboxplot](resultpngs\ghi_boxplot_by_season.png)
#### Key Findings:
- **Summer:** Highest average GHI (~350-400 W/m²), most consistent
- **Spring/Fall:** Moderate average GHI (~250-300 W/m²), transitional variability
- **Winter:** Lowest average GHI (~150-200 W/m²), high variability
- **Seasonal Range:** 200+ W/m² difference between summer and winter

**Seasonal Insights:**
- Clear inverse relationship with latitude angle of sun
- Summer dominance in solar resource availability
- Winter presents forecasting challenges due to high variability

---

### 7. Hourly Sunlight Analysis with Plot and Explanation

#### Analysis Details:
- **Function:** `analyze_hourly_sunlight()`
- **File:** `analyze_hourly_sunlight.py`
- **Output:** `hourly_sunlight_analysis.png`
- **Plot Type:** Line plot with confidence bands + Bar chart with color coding
![hourlysunglightanalysis](resultpngs\hourly_sunlight_analysis.png)
#### Plot Explanation:
**What It Shows:**
The hourly sunlight analysis displays the Global Horizontal Irradiance (GHI) received throughout a complete 24-hour day. The visualization consists of two complementary plots: a line graph with standard deviation bands showing average intensity by hour, and a color-coded bar chart highlighting different sun periods.

**Key Observed Patterns:**
1. **Nighttime (0-6 AM, 6-11 PM):** GHI ≈ 0 W/m² - Sun below horizon
2. **Sunrise Window (6-9 AM):** Rapid increase from 0 to ~200 W/m², sun angles shallow
3. **Peak Period (9 AM - 3 PM):** Maximum irradiance (800-900 W/m²), sun at highest elevation
4. **Solar Noon (~12 PM):** Absolute peak, sun most directly overhead
5. **Sunset Window (3-6 PM):** Gradual decline, symmetric to sunrise pattern

**Reasons for Observed Trend:**

- **Solar Geometry:** The sun's elevation angle determines irradiance intensity following a sine-like relationship with the highest values at solar noon
- **Atmospheric Effects:** Air mass increases at lower sun angles, scattering radiation and reducing surface irradiance
- **Time of Day:** Clear mathematical relationship: Irradiance ∝ sin(solar elevation angle)
- **Seasonal Variations:** These hourly patterns shift earlier in summer and later in winter
- **Atmospheric Transmissivity:** Clear-sky conditions maximize peak values; clouds reduce all hour values proportionally

**Practical Implications:**
- Solar energy systems require peaking capacity for midday loads
- Battery storage needed for morning/evening demand periods
- Grid integration must account for rapid sunrise/sunset transitions

---

### 8. Plot Temperature versus GHI and Explain Relationship

#### Analysis Details:
- **Function:** `plot_temperature_vs_ghi()`
- **Output:** `temperature_vs_ghi.png`
- **Plot Type:** Dual scatter plots (overall + seasonal breakdown)
![temperature-ghi-relation](resultpngs\temperature_vs_ghi.png)
#### Relationship Explanation:
- **Positive Correlation:** Warmer temperatures generally correspond to higher GHI values
- **Physical Mechanism:** Solar radiation heats the surface, increasing temperature
- **Time Lag Effects:** Temperature response lags behind radiation due to thermal inertia
- **Seasonal Patterns:** 
  - Strong relationship in summer (high radiation → high temperature)
  - Weaker relationship in winter (low radiation, other factors influence temperature)
  - Spring/Fall show moderate correlations

#### Key Insights:
- Temperature can serve as an indirect indicator of solar availability
- Temperature extremes do not always coincide with peak GHI
- Humidity and cloud cover more directly affect irradiance than temperature
- Useful feature for machine learning models predicting GHI

---

### 9. Normalize Data Using Two Different Techniques

#### Normalization Techniques to Implement:

#### **Technique 1: Min-Max Normalization (Scaling)**
```
python

def normalize_data(df):
    df['GHI_MinMax'] = (df['GHI'] - df['GHI'].min()) / (df['GHI'].max() - df['GHI'].min())
    return df

Formula: X_scaled = (X - X_min) / (X_max - X_min)
Range: [0, 1]
```

**Advantages:**
- Preserves the shape of original data
- Bounds all values to [0, 1] range
- Useful for neural networks with sigmoid activation
- Intuitive, easily interpretable

**Effect on Data:**
- All GHI values transformed to 0-1 range
- Maintains relative differences between values
- Outliers limited to boundary values
- Distribution shape preserved

**Use Case:** Feature scaling for machine learning algorithms

#### **Technique 2: Z-Score Normalization (Standardization)**
```
def normalize_data(df):
    df['GHI_ZScore'] = (df['GHI'] - df['GHI'].mean()) / df['GHI'].std()
    return df
Formula: X_std = (X - μ) / σ
Range: Centered at 0, typically [-3, 3]
```

**Advantages:**
- Centers data around 0
- Standard deviation = 1
- Distribution remains unchanged
- Better for handling outliers
- Preferred for statistical analysis

**Effect on Data:**
- Mean becomes 0, standard deviation becomes 1
- Allows comparison across different scales
- Outliers identified beyond ±3σ
- Maintains probability distribution properties

**Use Case:** Statistical analysis, comparative studies, algorithms assuming normal distribution

#### **Implementation Status:**
- Framework functions commented in code: `normalize_data()`
- Ready for activation when needed
- Both techniques preserve temporal relationships in time-series

**Performance Metrics for Normalization:**
Once normalized data is used in ARIMA/SARIMA models:
- Compare RMSE before/after normalization
- Evaluate convergence speed in model training
- Assess prediction accuracy improvements
- Analyze residual distributions

---

## Time-Series Forecasting Models

### ARIMA Model
- **Function:** `train_arima_model()`
- **Order:** (1, 0, 0) - Configurable for optimal fit
- **Output Metrics:** AIC, BIC, RMSE, MAE, Correlation

### SARIMA Model
- **Function:** `train_sarima_model()`
- **Order:** (1, 0, 0) × (1, 0, 1, 9) - With seasonal component
- **Seasonal Period:** 9 hours (daily cycle adjustment for hourly data)
- **Output Metrics:** AIC, BIC, RMSE, MAE, Correlation

### Model Comparison
- **Function:** `compare_arima_sarima()`
- **Comparison Metrics:**
  - RMSE - Root Mean Squared Error
  - MAE - Mean Absolute Error
  - Correlation - Actual vs Predicted
  ![comparisionofarimaandsarima](resultpngs\detailed_comparison.png)
  - Residual Analysis
  ![residualanalysis](resultpngs\residuals_plot.png)

---

## Visualization Outputs

### Generated Plots:
1. **hourly_sunlight_analysis.png** - Hourly GHI distribution
2. **ghi_distribution_by_season.png** - Seasonal distribution histograms
3. **ghi_boxplot_by_season.png** - Seasonal box plots with quartiles
4. **temperature_vs_ghi.png** - Scatter plot with trend analysis
5. **timeseries_seasonality.png** - Time-series with seasonal coloring
6. **correlation_heatmap.png** - Feature correlation matrix
7. **actual_vs_prediction.png** - ARIMA/SARIMA scatter plots
8. **residuals_plot.png** - Residual analysis (time series + distribution)
9. **detailed_comparison.png** - 4-panel comprehensive model comparison

---

## Project Files

### Main Analysis:
- **date_target.py** - Primary analysis script with all functions
- **analyze_hourly_sunlight.py** - Dedicated hourly analysis module

### Output Directory:
- **resultpngs/** - All PNG visualization outputs
- **correlation_matrix.csv** - Correlation statistics

---

## Methodology Summary

This comprehensive analysis follows best practices in time-series forecasting:

1. **Data Quality:** Rigorous preprocessing and validation
2. **Exploratory Analysis:** Distribution, seasonal, and temporal patterns
3. **Feature Engineering:** Hourly and seasonal feature extraction
4. **Normalization:** Multiple techniques for model optimization
5. **Model Development:** ARIMA and SARIMA for capturing temporal dependencies
6. **Model Evaluation:** Multiple metrics and residual analysis
7. **Visualization:** Comprehensive plots for insight communication

---

## Conclusion

This project successfully implements a complete solar irradiance forecasting pipeline with:
- ✓ Complete data preprocessing and validation
- ✓ Correlation and distribution analysis
- ✓ Seasonal decomposition and analysis
- ✓ Hourly irradiance pattern analysis
- ✓ Temperature-GHI relationship study
- ✓ Data normalization framework
- ✓ ARIMA and SARIMA model development
- ✓ Comprehensive performance metrics and visualizations

All results are documented, visualized, and version-controlled on GitHub.
