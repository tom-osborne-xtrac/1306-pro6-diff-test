import tkinter as tk
from tkinter import filedialog
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq



# Define filePath
# ---------------------- #
root = tk.Tk()
root.withdraw()
filePath = filedialog.askopenfilename()
fileName = os.path.basename(filePath)
fileDir = os.path.dirname(filePath)
outputFileName = fileDir.split('/')[-1]
outputFile = f'{fileDir}\\{outputFileName}.png'

print("Location:", filePath)
print("File: ", fileName)
print("Dir:", fileDir)

raw_data = pd.read_csv(filePath, sep=",", header=[3, 4], encoding='ISO-8859-1')
print(raw_data.head())

filter_ = np.argwhere(np.array(raw_data['OP Speed 1']) < 5).flatten().tolist()
# filter_ = raw_data[raw_data['OP Speed 1'] > 10].index
raw_data.drop(filter_, inplace=True)
raw_data.reset_index(drop=True, inplace=True)
# print(filter_.head())

# print(filtered_data.head())

fig, ax = plt.subplots(3, figsize=(16, 9))

ax[0].plot(
    raw_data['Event Time'],
    raw_data['OP Speed 1'],
    color="green",
    label="LH OP Speed [rpm]",
    marker=None
)
ax[0].plot(
    raw_data['Event Time'],
    raw_data['OP Speed 2'],
    color="blue",
    label="RH OP Speed [rpm]",
    marker=None
)
ax[0].plot(
    raw_data['Event Time'],
    raw_data['1306 Oil In - TCM_SumpoilTBas'],
    color="magenta",
    label="Oil Temperature [degC]",
    marker=None
)
ax[0].set_title("Shaft Speeds", loc='left')
ax[0].grid()
ax[0].legend(loc=4)
# ax[0].set_xlim([0, max_data_EventTime])
ax[0].set_xlabel("Time [s]")
ax[0].set_ylim([0, 100])

ax[1].plot(
    raw_data['Event Time'],
    raw_data['TCM MaiShaftNBas'],
    color="red",
    label="Mainshaft Speed [rpm]",
    marker=None
)
ax[1].set_title("Mainshaft Speed", loc='left')
ax[1].grid()
ax[1].legend(loc=4)
# ax[0].set_xlim([0, max_data_EventTime])
ax[1].set_xlabel("Time [s]")
ax[1].set_ylim([0, 300])

ax[2].plot(
    raw_data['Event Time'],
    raw_data['OP Torque 1'],
    color="purple",
    label="LH OP Torque 1 [Nm]",
    marker=None
)
ax[2].plot(
    raw_data['Event Time'],
    raw_data['OP Torque 2'],
    color="orange",
    label="RH OP Torque 2 [Nm]",
    marker=None
)

ax[2].set_title("Output Torque", loc='left')
ax[2].grid()
ax[2].legend(loc=4)
# ax[0].set_xlim([0, max_data_EventTime])
ax[2].set_xlabel("Time [s]")
# ax[1].set_ylim([0, 300])

#
# ax[3].plot(
#     raw_data['Event Time'],
#     raw_data['OP Torque 1'],
#     color="red",
#     label="LH OP Torque [Nm]",
#     marker=None
# )
# ax[2].plot(
#     raw_data['Event Time'],
#     raw_data['OP Torque 2'],
#     color="orange",
#     label="RH OP Torque [Nm]",
#     marker=None
# )
#
# ax[2].set_title("Output Torque", loc='left')
# ax[1].grid()
# ax[1].legend(loc=2)
# # ax[0].set_xlim([0, max_data_EventTime])
# ax[1].set_xlabel("Time [s]")
# # ax[1].set_ylim([0, 300])

fig.suptitle(f'{fileDir}', fontsize=10)
plt.subplots_adjust(left=0.05, bottom=0.07, right=0.965, top=0.9, wspace=0.2, hspace=0.4)
plt.savefig(outputFile, format='png', bbox_inches='tight', dpi=150)

# Number of sample points
N = len(raw_data['OP Torque 1'])
# sample spacing
T = 1.0 / 160.0
# x = np.linspace(0.0, N*T, N, endpoint=False)
x = np.linspace(0.0, N*T, N, endpoint=False)
y1 = np.array(raw_data['OP Torque 1']).flatten().tolist()
yf1 = fft(y1)

y2 = np.array(raw_data['OP Torque 2']).flatten().tolist()
yf2 = fft(y2)
xf = fftfreq(N, T)[:N//2]

yf3 = [a - b for a, b in zip(yf1, yf2)]

# ------------
# 1Hz plot
# ------------
fig2, ax2 = plt.subplots(3)
plt.subplots_adjust(left=0.05, bottom=0.06, right=0.97, top=0.9, wspace=0.2, hspace=0.4)
ax2[0].plot(
    xf,
    2.0/N * np.abs(yf1[0:N//2]),
    color="purple",
    label="FFT",
    marker=None
)
ax2[1].plot(
    xf,
    2.0/N * np.abs(yf2[0:N//2]),
    color="orange",
    label="FFT",
    marker=None
)
ax2[2].plot(
    raw_data['Event Time'],
    raw_data['OP Torque 1'],
    color="purple",
    label="LH OP Torque 1 [Nm]",
    marker=None
)
ax2[2].plot(
    raw_data['Event Time'],
    raw_data['OP Torque 2'],
    color="orange",
    label="RH OP Torque 2 [Nm]",
    marker=None
)
ax2[2].set_title("Output Torque", loc='left')
ax2[2].grid()
ax2[2].legend(loc=4)
# ax[0].set_xlim([0, max_data_EventTime])
ax2[2].set_xlabel("Time [s]")

# x coordinates for the lines
xcoords = [0.16, 0.33]
# colors for the lines
colors = ['r', 'k']

for xc, c in zip(xcoords,colors):
    ax2[0].axvline(x=xc, label='line at x = {}'.format(xc), c=c)
    trans = ax2[0].get_xaxis_transform()
    ax2[0].text(xc, .5, xc, transform=trans)
    ax2[1].axvline(x=xc, label='line at x = {}'.format(xc), c=c)
    trans = ax2[1].get_xaxis_transform()
    ax2[1].text(xc, .5, xc, transform=trans)

# Major ticks every 20, minor ticks every 5
major_ticks = np.arange(0, 1.1, 0.1)
minor_ticks = np.arange(0, 1.1, 0.05)

ax2[0].set_xticks(major_ticks)
ax2[0].set_xticks(minor_ticks, minor=True)


# And a corresponding grid
ax2[0].grid(which='both')

# Or if you want different settings for the grids:
ax2[0].grid(which='minor', alpha=0.2)
ax2[0].grid(which='major', alpha=0.5)


ax2[1].set_xticks(major_ticks)
ax2[1].set_xticks(minor_ticks, minor=True)

# And a corresponding grid
ax2[1].grid(which='both')

# Or if you want different settings for the grids:
ax2[1].grid(which='minor', alpha=0.2)
ax2[1].grid(which='major', alpha=0.5)


ax2[0].set_xlabel("Frequency [Hz]")
ax2[1].set_xlabel("Frequency [Hz]")

ax2[0].set_xlim([0, 1])
ax2[0].set_ylim([0, 1])
ax2[1].set_xlim([0, 1])
ax2[1].set_ylim([0, 1])
ax2[0].set_title("LH Output Torque FFT", loc='left')
ax2[1].set_title("RH Output Torque FFT", loc='left')

fig2.suptitle(f'{fileDir}', fontsize=10)


# ------------
# 10 Hz plot
# ------------
fig3, ax3 = plt.subplots(3)
plt.subplots_adjust(left=0.05, bottom=0.06, right=0.97, top=0.9, wspace=0.2, hspace=0.4)
ax3[0].plot(
    xf,
    2.0/N * np.abs(yf1[0:N//2]),
    color="purple",
    label="FFT",
    marker=None
)
ax3[1].plot(
    xf,
    2.0/N * np.abs(yf2[0:N//2]),
    color="orange",
    label="FFT",
    marker=None
)
ax3[2].plot(
    raw_data['Event Time'],
    raw_data['OP Torque 1'],
    color="purple",
    label="LH OP Torque 1 [Nm]",
    marker=None
)
ax3[2].plot(
    raw_data['Event Time'],
    raw_data['OP Torque 2'],
    color="orange",
    label="RH OP Torque 2 [Nm]",
    marker=None
)
ax3[2].set_title("Output Torque", loc='left')
ax3[2].grid()
ax3[2].legend(loc=4)
# ax[0].set_xlim([0, max_data_EventTime])
ax3[2].set_xlabel("Time [s]")

# x coordinates for the lines
xcoords = [2.32, 5.0]
# colors for the lines
colors = ['r', 'k']

for xc, c in zip(xcoords,colors):
    ax3[0].axvline(x=xc, label='line at x = {}'.format(xc), c=c, alpha=0.5)
    trans = ax3[0].get_xaxis_transform()
    ax3[0].text(xc, .5, xc, transform=trans)
    ax3[1].axvline(x=xc, label='line at x = {}'.format(xc), c=c, alpha=0.5)
    trans = ax3[1].get_xaxis_transform()
    ax3[1].text(xc, .5, xc, transform=trans)

# Major ticks every 20, minor ticks every 5
major_ticks = np.arange(0, 11, 1)
minor_ticks = np.arange(0, 11, 0.5)

ax3[0].set_xticks(major_ticks)
ax3[0].set_xticks(minor_ticks, minor=True)


# And a corresponding grid
ax3[0].grid(which='both')

# Or if you want different settings for the grids:
ax3[0].grid(which='minor', alpha=0.2)
ax3[0].grid(which='major', alpha=0.5)


ax3[1].set_xticks(major_ticks)
ax3[1].set_xticks(minor_ticks, minor=True)

# And a corresponding grid
ax3[1].grid(which='both')

# Or if you want different settings for the grids:
ax3[1].grid(which='minor', alpha=0.2)
ax3[1].grid(which='major', alpha=0.5)


ax3[0].set_xlabel("Frequency [Hz]")
ax3[1].set_xlabel("Frequency [Hz]")

ax3[0].set_xlim([0, 10])
ax3[0].set_ylim([0, 1])
ax3[1].set_xlim([0, 10])
ax3[1].set_ylim([0, 1])
ax3[0].set_title("LH Output Torque FFT", loc='left')
ax3[1].set_title("RH Output Torque FFT", loc='left')

fig3.suptitle(f'{fileDir}', fontsize=10)


plt.show()