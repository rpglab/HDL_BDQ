import numpy as np
import matplotlib.pyplot as plt
import re

# Function to extract data from a .dat file
def extract_data(file_name):
    with open(file_name, 'r') as file:
        lines = file.readlines()
        
        period_data = []
        start_collecting = False
        for line in lines:
            if start_collecting and line.strip() == ";":
                break
            if start_collecting:
                parts = re.split(r'\s+', line.strip())
                period_data.append([float(parts[1]), float(parts[3]), float(parts[4])])
            if 'param: PERIOD:' in line:
                start_collecting = True
        
    period_data = np.array(period_data)
    time_total_pd = period_data[:, 0]
    pv_ava = period_data[:, 1] * 100  # Scale solar data
    wt_ava = period_data[:, 2] * 2    # Scale wind data
    return time_total_pd, pv_ava, wt_ava  # Time_TotalPd, PV_Ava, WT_Ava

# List of .dat files representing each month
dat_files = [
    "January.dat", "February.dat", "March.dat", "April.dat",
    "May.dat", "June.dat", "July.dat", "August.dat",
    "September.dat", "October.dat", "November.dat", "December.dat"
]

# Initialize a figure and axis for the plot
fig, ax = plt.subplots(4, 3, figsize=(18, 16))
ax = ax.ravel()  # Flatten the 2D array of axes to 1D

# Plot each month's data
for i, dat_file in enumerate(dat_files):
    load, solar, wind = extract_data(dat_file)
    time_steps = np.arange(1, len(load) + 1)
    
    ax[i].plot(time_steps, load, label="Load (kW)", color='blue')
    ax[i].plot(time_steps, solar, label="Solar (kW)", color='orange')
    ax[i].plot(time_steps, wind, label="Wind (kW)", color='green')
    
    ax[i].set_title(dat_file.split('.')[0])
    ax[i].set_xlabel('Hour')
    ax[i].set_ylabel('Power')
    ax[i].legend()
    ax[i].grid(True)

# Adjust layout for better visualization
plt.tight_layout()

# Save the plot as a high-resolution image
plt.savefig('monthly_data_plot.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()