# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 21:49:40 2024

@author: Steve
"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_excel('Fig.3-new.xlsx',sheet_name='s1')

df_melted = df.melt(id_vars=['horizon', 'mae'], var_name='season', value_name='error')

# Consolidate each season across years by extracting the season name
df_melted['season'] = df_melted['season'].apply(lambda x: x.split('_')[1])

# Define colors for forecast and simulation
colors = {
    'forecast': 'black',     # Color for forecast
    'simulate': 'red'  # Color for simulation
}

# Plot each horizon (T+1, T+3, T+6) in separate violin plots
fig, axes = plt.subplots(2, 1
                         , figsize=(3, 10), sharey=True, dpi=300)

# Font sizes
axis_label_fontsize = 14
title_fontsize = 16

# Plot T+1 Horizon
sns.violinplot(ax=axes[0], x='season', y='error', hue='mae', data=df_melted[df_melted['horizon'] == 'T+1'], inner=None, split=True)
# Adding "virtual lines" for the median, Q1, and Q3 within each violin plot with different colors
for season in df_melted['season'].unique():
    subset = df_melted[(df_melted['horizon'] == 'T+1') & (df_melted['season'] == season)]
    quantiles = subset.groupby('mae')['error'].quantile([0.25, 0.5, 0.75]).unstack()
    for mae, values in quantiles.iterrows():
        color = colors[mae]  # Use color based on the 'mae' type
        # Plot median with color
        axes[0].plot([season], [values[0.5]], 'o', color=color, markersize=5)  # Dot for median
        # Plot Q1 and Q3 with horizontal line in color
        axes[0].plot([season, season], [values[0.25], values[0.75]], '-', color=color, lw=2)  # Line for Q1 to Q3

axes[0].set_title("T+1", fontsize=title_fontsize)
axes[0].set_ylim(0, 1)
axes[0].set_ylabel("Absolute Error(kPa)", fontsize=axis_label_fontsize)
axes[0].legend([],[], frameon=False)  # Hide legend for middle plot
axes[0].set_xlabel('') 
axes[0].set_xticklabels([])  

# Plot T+6 Horizon
sns.violinplot(ax=axes[1], x='season', y='error', hue='mae', data=df_melted[df_melted['horizon'] == 'T+6'], inner=None, split=True)

for season in df_melted['season'].unique():
    subset = df_melted[(df_melted['horizon'] == 'T+6') & (df_melted['season'] == season)]
    quantiles = subset.groupby('mae')['error'].quantile([0.25, 0.5, 0.75]).unstack()
    for mae, values in quantiles.iterrows():
        color = colors[mae]  # Use color based on the 'mae' type
        # Plot median with color
        axes[1].plot([season], [values[0.5]], 'o', color=color, markersize=5)  # Dot for median
        # Plot Q1 and Q3 with horizontal line in color
        axes[1].plot([season, season], [values[0.25], values[0.75]], '-', color=color, lw=2)  # Line for Q1 to Q3
        
axes[1].set_title("T+6", fontsize=title_fontsize)
axes[1].set_ylim(0, 1)
axes[1].set_xlabel("Season", fontsize=axis_label_fontsize)
axes[1].set_ylabel("Absolute Error(kPa)", fontsize=axis_label_fontsize)
axes[1].legend([],[], frameon=False)  # Hide legend for last plot
axes[1].tick_params(axis='x', rotation=45, labelsize=axis_label_fontsize)

# Display the plot
plt.tight_layout()
plt.show()
