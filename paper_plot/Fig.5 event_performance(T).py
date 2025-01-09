# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 19:36:42 2024

@author: Steve
"""

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Load the dataset
data = pd.read_excel('Fig.5 event_performance(T).xlsx',sheet_name='2022-08-22 11-all')

# Calculate the average error across all T+ values for each station (if needed)
data['error_avg'] = data[['T+1', 'T+2', 'T+3', 'T+4', 'T+5', 'T+6']].mean(axis=1)

# Sort values by station id to ensure consistent order
station_error_sorted = data.sort_values(by='station id').reset_index(drop=True)

#%%

# Define colors for each subcatchment category
category_colors = {
    'S1': 'orange',    # Color for S1
    'S2': 'blue',   # Color for S2
    'S3': 'green'      # Color for S3
}

# Set up a single-row figure with three columns for T+1, T+3, and T+6 with specified error range
fig, axs = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={'projection': 'polar'}, dpi=300)
fig.patch.set_facecolor('white')

# Titles and error columns for T+1, T+3, and T+6 plots
error_columns = ['T+1', 'T+3', 'T+6']
titles = ['T+1', 'T+3', 'T+6']

# Define angles for each station based on sorted station id order
angles = np.linspace(0, 2 * np.pi, len(station_error_sorted), endpoint=False).tolist()

# Loop through each selected time point and plot in single row layout with fixed error range 0 to 5
for ax, error_col, title in zip(axs, error_columns, titles):
    for i in range(len(station_error_sorted)):
        angle = angles[i]
        error_value = station_error_sorted[error_col].iloc[i]

        # Set color based on error intensity with reversed colormap, using fixed range 0 to 5
        
        color = 'Red'
        # color = plt.cm.Reds(error_value / station_error_sorted[error_col].max())
        ax.bar(angle, error_value, color=color, width=0.1, edgecolor='None', alpha=1)
    
    # ax.grid(False)
    # Set the radial limit to 5 to make sure all plots use the same scale
    ax.set_ylim(0, 3)
    ax.set_xticks([])
    # Set station ID labels with specific colors based on subcatchment category
    station_labels = []
    for i, label in enumerate(station_error_sorted['station id']):
        subcatchment = station_error_sorted['subcatchment'].iloc[i]
        label_color = category_colors.get(subcatchment, 'black')  # Use category color or default to black if missing
        station_labels.append(ax.text(angles[i], 3.5, str(label), color=label_color, fontsize=12, fontweight='bold', ha='center'))

    # Set title with increased padding
    ax.set_title(title, va='bottom', fontsize=16, fontweight='bold', pad=35)

# Add a single color legend (color bar) under the figure with range from 0 to 5
sm = plt.cm.ScalarMappable(cmap='Reds', norm=plt.Normalize(vmin=0, vmax=5))
cbar = plt.colorbar(sm, ax=axs, orientation='horizontal', pad=0.2, aspect=40)
cbar.set_label('Prediction Error Intensity (0 to 5)', fontsize=12)

# Display the plot
plt.show()