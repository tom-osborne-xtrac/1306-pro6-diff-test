import tkinter as tk
from tkinter import filedialog
import os
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq

config = {
    "save_plot": True,
    "show_IPTrq": True,
    "show_FreqPlots": False
}


def get_data(path=''):
    """
    Defines the file path for analysis.
    File path can be passed as variable or left blank.
    If left blank, the function will open a file dialog
    allowing user to select a file
    """
    if path == '':
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename()
    else:
        file_path = path
    file_name = os.path.basename(file_path)
    file_dir = os.path.dirname(file_path)
    file_name = file_dir.split('/')[-1]

    print("#---------------------------------#")
    print("# File: ", file_name)
    print("#---------------------------------#")
    print("Location:", file_path)
    print("#---------------------------------#")

    df = pd.read_csv(file_path, sep=",", header=[3, 4], encoding='ISO-8859-1')
    return df, file_path, file_dir, file_name


def set_start(data):
    """
    Finds the start of the EoL test run and removes data beforehand.
    Adjusts 'Event Time' channel to 0s at start
    """
    # File 1
    EoL_start_1 = np.argwhere(np.array(data['OP Speed 1']) < 5).flatten().tolist()
    EoL_start_1 = [i for i in EoL_start_1 if i != 0]

    EoL_start_2 = np.argwhere(np.array(data['Event Time']) < 300).flatten().tolist()
    EoL_start_2 = [i for i in EoL_start_2 if i != 0]

    EoL_start_3 = np.argwhere(np.array(data['[V9] Pri. GBox Oil Temp']) > 20).flatten().tolist()
    EoL_start_3 = [i for i in EoL_start_3 if i != 0]

    EoL_start = sorted(list(set(EoL_start_1).intersection(EoL_start_2, EoL_start_3)))[-1]

    data = data.loc[EoL_start:len(data)].copy()
    data.reset_index(drop=True, inplace=True)
    data["Event Time"] = data.loc[:, "Event Time"] - data.loc[0, "Event Time"]
    return data


def calc_sample_rate(data):
    """
    Calculates the sample rate from a given dataset
    """
    return round(1 / data['Event Time'].diff().mean().values[0], 1)


def set_axis(plots, axis, label, start, end, major, minor=None):
    """
    Function for setting plot label, axis major
    and minor ticks and formats the gridlines
    """
    for plot in plots:
        if major:
            major_ticks = np.arange(start, end + 1, major)
            if axis == 'x':
                plot.set_xlabel(label)
                plot.set_xlim([start, end])
                plot.set_xticks(major_ticks)
            elif axis == 'y':
                plot.set_ylabel(label)
                plot.set_ylim([start, end])
                plot.set_yticks(major_ticks)

        if minor:
            minor_ticks = np.arange(start, end + 1, minor)
            if axis == 'x':
                plot.set_xticks(minor_ticks, minor=True)
            elif axis == 'y':
                plot.set_yticks(minor_ticks, minor=True)

        if major and minor:
            plot.grid(which='both')
        plot.grid(which='minor', alpha=0.4)
        plot.grid(which='major', alpha=0.8)


def plot_df(df, chart, x_axis, y_axis, label, colour, marker=''):
    print(f'Plotting {y_axis}')
    chart.plot(
        df[x_axis],
        df[y_axis],
        color=colour,
        label=f'{label}',
        marker=marker,
        markersize=3
    )


def plot_series(chart, x_axis, y_axis, label, colour, marker=''):
    print(f'Plotting {y_axis}')
    chart.plot(
        x_axis,
        y_axis,
        color=colour,
        label=f'{label}',
        marker=marker,
        markersize=3
    )


def add_vert_lines(chart, xcoords, colors):
    for xc, c in zip(xcoords, colors):
        chart.axvline(x=xc, label='line at x = {}'.format(xc), c=c)
        trans = chart.get_xaxis_transform()
        chart.text(xc, .5, xc, transform=trans)
        # ax2[1].axvline(x=xc, label='line at x = {}'.format(xc), c=c)
        # trans = ax2[1].get_xaxis_transform()
        # ax2[1].text(xc, .5, xc, transform=trans)


def prepare_fft(data, channel, sample_rate):
    N = len(data[channel])
    T = 1.0 / sample_rate
    x = fftfreq(N, T)[:N//2]
    y = fft(np.array(data[channel]).flatten().tolist())
    y = 2.0/N * np.abs(y[0:N//2])
    return x, y


def calc_difference(data_a, data_b):
    return [a - b for a, b in zip(data_a, data_b)]


rdata, fpath, fdir, fname = get_data()
outputFile = f'{fdir}/{fname}.jpg'
print(f'OUTPUT FILE: {outputFile}')

filter_ = np.argwhere(np.array(rdata['OP Speed 1']) < 2).flatten().tolist()
# rdata = set_start(rdata)

rdata.drop(filter_, inplace=True)
rdata.reset_index(drop=True, inplace=True)

x_major = 50
x_minor = 10
time_min = rdata['Event Time'].min(numeric_only=True)
time_max = rdata['Event Time'].max(numeric_only=True)
startx = math.floor(time_min / x_major) * x_major
endx = math.ceil(time_max / x_major) * x_major

# Plot Settings
plt.rcParams['lines.linewidth'] = 0.7
figsize = (16, 9)

# Figure 1 - Time domain
fig, ax = plt.subplots(3, figsize=figsize)
fig.suptitle(f'{fname}', fontsize=10)
plt.subplots_adjust(
    left=0.05,
    bottom=0.07,
    right=0.955,
    top=0.9,
    wspace=0.2,
    hspace=0.4
)
set_axis(ax, 'x', 'Time [s]', startx, endx, x_major, x_minor)

plot_df(rdata, ax[0], 'Event Time', 'OP Speed 1', 'LH OP Speed [rpm]', 'green')
plot_df(rdata, ax[0], 'Event Time', 'OP Speed 2', 'RH OP Speed [rpm]', 'blue')
plot_df(rdata, ax[0], 'Event Time', '1306 Oil In - TCM_SumpoilTBas', 'Oil Temp [degC]', 'magenta')
set_axis([ax[0]], 'y', 'Speed [rpm] / Temp [degC]', 0, 100, 10, 10)
ax[0].set_title("Shaft Speeds", loc='left')
ax[0].legend(loc=2)

plot_df(rdata, ax[1], 'Event Time', 'TCM MaiShaftNBas', 'Mainshaft Speed [rpm]', 'red')
set_axis([ax[1]], 'y', 'Speed [rpm]', 0, 500, 100, 50)
ax[1].set_title("Mainshaft Speed", loc='left')
ax[1].legend(loc=2)

if config["show_IPTrq"]:
    ax2 = ax[0].twinx()
    plot_df(rdata, ax2, 'Event Time', 'IP Torque 1', 'Input Torque [Nm]', 'darkcyan')
    set_axis([ax2], 'y', 'Torque [Nm]', 0, 100, 20, 10)
    ax2.legend(loc=1)


plot_df(rdata, ax[2], 'Event Time', 'OP Torque 1', 'LH OP Torque 1 [Nm]', 'purple')
plot_df(rdata, ax[2], 'Event Time', 'OP Torque 2', 'RH OP Torque 2 [Nm]', 'orange')
set_axis([ax[2]], 'y', 'Torque [Nm]', -200, 200, 50, 10)
ax[2].set_title("Output Torque", loc='left')
ax[2].legend(loc=4)

if config["save_plot"]:
    plt.savefig(outputFile, format='png', bbox_inches='tight', dpi=150)


if config["show_FreqPlots"]:
    #  Figure 2 - FFT Plots
    # Number of sample points
    sr = calc_sample_rate(rdata)

    trq1_fft_x, trq1_fft_y = prepare_fft(rdata, 'OP Torque 1', sr)
    trq2_fft_x, trq2_fft_y = prepare_fft(rdata, 'OP Torque 2', sr)

    # N = len(rdata['OP Torque 1'])
    # # sample spacing
    # T = 1.0 / 160.0
    # # x = np.linspace(0.0, N*T, N, endpoint=False)
    # x = np.linspace(0.0, N*T, N, endpoint=False)
    # y1 = np.array(rdata['OP Torque 1']).flatten().tolist()
    # yf1 = fft(y1)

    # y2 = np.array(rdata['OP Torque 2']).flatten().tolist()
    # yf2 = fft(y2)
    # xf = fftfreq(N, T)[:N//2]

    # yf3 = [a - b for a, b in zip(yf1, yf2)]

    # ------------
    # 1Hz plot
    # ------------
    fig2, ax2 = plt.subplots(3)
    plt.subplots_adjust(
        left=0.05,
        bottom=0.06,
        right=0.97,
        top=0.9,
        wspace=0.2,
        hspace=0.4
    )

    plot_series(ax2[0], trq1_fft_x, trq1_fft_y, 'FFT - LH OP Torque 1', 'purple')
    plot_series(ax2[1], trq2_fft_x, trq2_fft_y, 'FFT - RH OP Torque 2', 'orange')
    set_axis([ax2[0], ax2[1]], 'x', 'Frequency [Hz]', 0, 1, 0.1, 0.02)
    set_axis([ax2[0], ax2[1]], 'y', 'Frequency [Hz]', 0, 1, 0.2, 0.1)

    plot_df(rdata, ax2[2], 'Event Time', 'OP Torque 1', 'LH OP Torque 1 [Nm]', 'purple')
    plot_df(rdata, ax2[2], 'Event Time', 'OP Torque 2', 'RH OP Torque 2 [Nm]', 'orange')
    set_axis([ax2[2]], 'x', 'Time [s]', startx, endx, x_major, x_minor)
    set_axis([ax2[2]], 'y', 'Torque [Nm]', -80, 0, 10, 5)

    # ax2[0].plot(
    #     trq1_fft_x,
    #     trq1_fft_y,
    #     color="purple",
    #     label="FFT",
    #     marker=None
    # )
    # ax2[1].plot(
    #     trq2_fft_x,
    #     trq2_fft_y,
    #     color="orange",
    #     label="FFT",
    #     marker=None
    # )
    # ax2[2].plot(
    #     rdata['Event Time'],
    #     rdata['OP Torque 1'],
    #     color="purple",
    #     label="LH OP Torque 1 [Nm]",
    #     marker=None
    # )
    # ax2[2].plot(
    #     rdata['Event Time'],
    #     rdata['OP Torque 2'],
    #     color="orange",
    #     label="RH OP Torque 2 [Nm]",
    #     marker=None
    # )
    ax2[2].set_title("Output Torque", loc='left')
    ax2[2].legend(loc=4)

    # x coordinates for the lines
    xcoords = [0.16, 0.33]
    # colors for the lines
    colors = ['r', 'k']

    add_vert_lines(ax2[0], xcoords, colors)

    # for xc, c in zip(xcoords, colors):
    #     ax2[0].axvline(x=xc, label='line at x = {}'.format(xc), c=c)
    #     trans = ax2[0].get_xaxis_transform()
    #     ax2[0].text(xc, .5, xc, transform=trans)
    #     ax2[1].axvline(x=xc, label='line at x = {}'.format(xc), c=c)
    #     trans = ax2[1].get_xaxis_transform()
    #     ax2[1].text(xc, .5, xc, transform=trans)

    # # Major ticks every 20, minor ticks every 5
    # major_ticks = np.arange(0, 1.1, 0.1)
    # minor_ticks = np.arange(0, 1.1, 0.05)

    # ax2[0].set_xticks(major_ticks)
    # ax2[0].set_xticks(minor_ticks, minor=True)

    # # And a corresponding grid
    # ax2[0].grid(which='both')

    # # Or if you want different settings for the grids:
    # ax2[0].grid(which='minor', alpha=0.2)
    # ax2[0].grid(which='major', alpha=0.5)

    # ax2[1].set_xticks(major_ticks)
    # ax2[1].set_xticks(minor_ticks, minor=True)

    # # And a corresponding grid
    # ax2[1].grid(which='both')

    # # Or if you want different settings for the grids:
    # ax2[1].grid(which='minor', alpha=0.2)
    # ax2[1].grid(which='major', alpha=0.5)

    # ax2[0].set_xlabel("Frequency [Hz]")
    # ax2[1].set_xlabel("Frequency [Hz]")

    # ax2[0].set_xlim([0, 1])
    # ax2[0].set_ylim([0, 1])
    # ax2[1].set_xlim([0, 1])
    # ax2[1].set_ylim([0, 1])
    # ax2[0].set_title("LH Output Torque FFT", loc='left')
    # ax2[1].set_title("RH Output Torque FFT", loc='left')

    fig2.suptitle(f'{fname}', fontsize=10)

    # ------------
    # 10 Hz plot
    # ------------
    fig3, ax3 = plt.subplots(3)
    plt.subplots_adjust(
        left=0.05,
        bottom=0.06,
        right=0.97,
        top=0.9,
        wspace=0.2,
        hspace=0.4
    )
    ax3[0].plot(
        trq1_fft_x,
        trq1_fft_y,
        color="purple",
        label="FFT",
        marker=None
    )
    ax3[1].plot(
        trq2_fft_x,
        trq2_fft_y,
        color="orange",
        label="FFT",
        marker=None
    )
    ax3[2].plot(
        rdata['Event Time'],
        rdata['OP Torque 1'],
        color="purple",
        label="LH OP Torque 1 [Nm]",
        marker=None
    )
    ax3[2].plot(
        rdata['Event Time'],
        rdata['OP Torque 2'],
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

    for xc, c in zip(xcoords, colors):
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

    fig3.suptitle(f'{fdir}', fontsize=10)


plt.show()
