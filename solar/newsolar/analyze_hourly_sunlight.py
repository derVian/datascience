import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyze_hourly_sunlight(df, save_path='./resultpngs/hourly_sunlight_analysis.png'):
    """
    Determine the amount of sunlight received at each hour of the day.
    Creates a plot and provides explanation of the hourly sunlight distribution pattern.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe containing 'Hour' and 'GHI' columns
    save_path : str
        Path to save the output plot
    """
    
    # Group by hour and calculate mean GHI
    hourly_ghi = df.groupby('Hour')['GHI'].agg(['mean', 'std', 'count']).reset_index()
    hourly_ghi.columns = ['Hour', 'Mean_GHI', 'Std_GHI', 'Count']
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Line plot with confidence band
    ax1.plot(hourly_ghi['Hour'], hourly_ghi['Mean_GHI'], 
             marker='o', linewidth=2.5, markersize=8, color='#FF8C00', label='Mean GHI')
    ax1.fill_between(hourly_ghi['Hour'], 
                      hourly_ghi['Mean_GHI'] - hourly_ghi['Std_GHI'], 
                      hourly_ghi['Mean_GHI'] + hourly_ghi['Std_GHI'], 
                      alpha=0.3, color='#FF8C00', label='±1 Std Dev')
    
    ax1.set_xlabel('Hour of Day (24-hour format)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Global Horizontal Irradiance (W/m²)', fontsize=12, fontweight='bold')
    ax1.set_title('Mean Sunlight Intensity by Hour of Day', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xticks(range(0, 24, 2))
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=11, loc='upper left')
    ax1.set_xlim(-0.5, 23.5)
    
    # Plot 2: Bar plot
    colors = ['#87CEEB' if x < 6 or x > 18 else '#FF8C00' if x >= 9 and x <= 15 else '#FFD700' 
              for x in hourly_ghi['Hour']]
    ax2.bar(hourly_ghi['Hour'], hourly_ghi['Mean_GHI'], color=colors, edgecolor='black', alpha=0.8)
    
    ax2.set_xlabel('Hour of Day (24-hour format)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Global Horizontal Irradiance (W/m²)', fontsize=12, fontweight='bold')
    ax2.set_title('Hourly Sunlight Distribution', fontsize=14, fontweight='bold', pad=15)
    ax2.set_xticks(range(0, 24, 2))
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax2.set_xlim(-0.5, 23.5)
    
    # Add legend for colors
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#87CEEB', edgecolor='black', label='Night (0-6, 18-23)'),
                       Patch(facecolor='#FFD700', edgecolor='black', label='Morning/Evening (6-9, 15-18)'),
                       Patch(facecolor='#FF8C00', edgecolor='black', label='Peak Sun (9-15)')]
    ax2.legend(handles=legend_elements, fontsize=10, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print detailed statistics
    print("\n" + "="*80)
    print("HOURLY SUNLIGHT ANALYSIS - GHI BY HOUR OF DAY")
    print("="*80)
    print("\nDetailed Statistics:")
    print("-"*80)
    print(f"{'Hour':>6} | {'Mean GHI':>10} | {'Std Dev':>10} | {'Min':>10} | {'Max':>10} | {'Count':>8}")
    print("-"*80)
    
    for _, row in hourly_ghi.iterrows():
        hour_data = df[df['Hour'] == row['Hour']]['GHI']
        print(f"{int(row['Hour']):>6} | {row['Mean_GHI']:>10.2f} | {row['Std_GHI']:>10.2f} | "
              f"{hour_data.min():>10.2f} | {hour_data.max():>10.2f} | {int(row['Count']):>8}")
    
    print("-"*80)
    
    # Calculate peak and minimum hours
    peak_hour = hourly_ghi.loc[hourly_ghi['Mean_GHI'].idxmax()]
    min_hour = hourly_ghi.loc[hourly_ghi['Mean_GHI'].idxmin()]
    
    print(f"\nPeak Sunlight Hour: {int(peak_hour['Hour']):02d}:00 (GHI: {peak_hour['Mean_GHI']:.2f} W/m²)")
    print(f"Minimum Sunlight Hour: {int(min_hour['Hour']):02d}:00 (GHI: {min_hour['Mean_GHI']:.2f} W/m²)")
    
    # Calculate sunrise/sunset approximately (when GHI > 10 W/m²)
    sunrise_hours = hourly_ghi[hourly_ghi['Mean_GHI'] > 10]['Hour']
    if len(sunrise_hours) > 0:
        sunrise = sunrise_hours.min()
        sunset = sunrise_hours.max()
        daylight_hours = sunset - sunrise
        print(f"Approximate Sunrise: {int(sunrise):02d}:00")
        print(f"Approximate Sunset: {int(sunset):02d}:00")
        print(f"Effective Daylight Hours: {daylight_hours:.1f} hours")
    
    print("="*80)
    
    # Generate explanation
    explanation = generate_hourly_sunlight_explanation(hourly_ghi, peak_hour, min_hour)
    
    return hourly_ghi, explanation


def generate_hourly_sunlight_explanation(hourly_ghi, peak_hour, min_hour):
    """
    Generate a detailed explanation paragraph of the hourly sunlight pattern.
    """
    
    explanation = f"""
EXPLANATION OF HOURLY SUNLIGHT DISTRIBUTION:

The figure displays the Global Horizontal Irradiance (GHI) received throughout a 24-hour day, 
revealing a distinct and expected solar radiation pattern driven by Earth's rotation and the sun's 
apparent motion across the sky. The analysis shows that solar irradiance is virtually absent during 
nighttime hours (0-6 and 18-23), with GHI values near zero as the sun is below the horizon. Starting 
around 6-7 AM, solar radiation begins to increase as the sun rises above the horizon, initially at 
shallow angles that spread energy over larger surface areas. Solar irradiance steadily increases through 
the morning hours, reaching its maximum intensity during the midday period (approximately {int(peak_hour['Hour']):02d}:00 hours), 
when the sun is at its highest position in the sky and achieving the most direct perpendicular angle 
relative to the Earth's surface. This peak of {peak_hour['Mean_GHI']:.2f} W/m² represents optimal conditions for solar 
energy collection. After solar noon, irradiance gradually decreases through the afternoon and evening 
hours as the sun descends toward the horizon, returning to minimal values by sunset (around 18-19 hours). 
The symmetric bell-curve pattern, slightly asymmetric due to geographic location and seasonal variations, 
is fundamental to solar resource assessment and directly determines the available solar energy for 
electricity generation, with implications for grid management, storage requirements, and overall renewable 
energy system design. The observed trend reflects the sine-like relationship between the sun's elevation 
angle and surface irradiance, governed by solar geometry principles and atmospheric conditions that may 
further attenuate the radiation on cloudy days.
"""
    
    print("\n" + explanation)
    
    return explanation


# Example usage:
if __name__ == "__main__":
    import sys
    sys.path.append('..')
    from date_target import load_data, preprocess_data, filter_ghi_data, add_season_column
    
    # Load and preprocess data
    df = load_data('../solar2022.csv')
    df = preprocess_data(df)
    df = filter_ghi_data(df, threshold=5)
    df = add_season_column(df)
    
    # Analyze hourly sunlight
    hourly_stats, explanation = analyze_hourly_sunlight(df)
