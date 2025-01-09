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
df = pd.read_excel('Fig.5 event_performance(RH).xlsx',sheet_name='2022-05-14 23-all')

# Calculate the absolute errors for each time step
for t in range(1, 7):  # T+1 to T+6
    df[f"T+{t}_error"] = abs(df[f"obs_T+{t}"] - df[f"pred_T+{t}"])
    
# Group by subcatchment and calculate the mean error for each time step
mean_errors = df.groupby("subcatchment")[[f"T+{t}_error" for t in range(1, 7)]].mean()

# Rename columns for clarity
mean_errors.columns = [f"T+{t}" for t in range(1, 7)]

#%%

import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(subplot_kw=dict(projection='polar'), figsize=(8, 8),dpi=300)
time_steps = [f"T+{t}" for t in range(1, 7)]
colors = ["orange", "blue", "green"]
subcatchments = ["S1", "S2", "S3"]

width = 2 * np.pi / (len(time_steps) * len(subcatchments))  # Narrower width to fit all bars

# Plot each time step for each subcatchment, grouped together
for idx, subcatchment in enumerate(subcatchments):
    values = mean_errors.loc[subcatchment].values
    theta = np.linspace(0.0, 2 * np.pi, len(values), endpoint=False)
    
    # Adjust theta for each subcatchment to position bars together by time step
    adjusted_theta = [angle + idx * width for angle in theta]
    
    # Plot bars for the subcatchment with adjusted angles
    bars = ax.bar(adjusted_theta, values, width=width, color=colors[idx], alpha=1, edgecolor="black", label=subcatchment)

# Add labels and legend
ax.set_xticks(theta)
ax.set_xticklabels(["t+1", "t+2", "t+3", "t+4", "t+5", "t+6"], fontsize=20, fontweight='bold')
# Move x-axis labels further from the center
for label in ax.get_xticklabels():
    label.set_y(-0.1)  # Adjust this value to move the labels further out

# ax.set_yticks(range(0, int(mean_errors.values.max()) + 2))
ax.set_yticks(range(0, 8))

ax.yaxis.set_tick_params(labelsize=14)  # Increase y-axis font size
# ax.set_title("MAE for Subcatchments S1, S2, and S3", fontsize=16, pad=40, fontweight='bold')
# ax.legend(loc='lower center', bbox_to_anchor=(0.5,-0.2), ncol=3, frameon=False, fontsize=14)

plt.show()