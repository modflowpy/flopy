import json
import sys
from pathlib import Path
from pprint import pprint

import matplotlib.dates as dates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm
from matplotlib.lines import Line2D

indir = Path(sys.argv[1])
outdir = Path(sys.argv[2])
json_paths = list(Path(indir).rglob("*.json"))

print(f"Found {len(json_paths)} JSON files")


def get_benchmarks(paths):
    benchmarks = []
    num_benchmarks = 0

    for path in paths:
        with open(path) as file:
            jsn = json.load(file)
            system = jsn["machine_info"]["system"]
            python = jsn["machine_info"]["python_version"]
            if len(python.split(".")) == 3:
                python = python.rpartition(".")[0]
            tstamp = jsn["datetime"]
            bmarks = jsn["benchmarks"]
            for benchmark in bmarks:
                num_benchmarks += 1
                fullname = benchmark["fullname"]
                included = ["min", "mean"]
                for stat, value in benchmark["stats"].items():
                    if stat not in included:
                        continue
                    benchmarks.append(
                        {
                            "system": system,
                            "python": python,
                            "time": tstamp,
                            "case": fullname,
                            "stat": stat,
                            "value": value,
                        }
                    )

    print("Found", num_benchmarks, "benchmarks")
    return benchmarks


# create data frame and save to CSV
benchmarks_df = pd.DataFrame(get_benchmarks(json_paths))
benchmarks_df["time"] = pd.to_datetime(benchmarks_df["time"])
benchmarks_df.to_csv(str(outdir / "benchmarks.csv"), index=False)


def matplotlib_plot(stats):
    nstats = len(stats)
    fig, axs = plt.subplots(nstats, 1, sharex=True)

    # color-code according to python version
    pythons = np.unique(benchmarks_df["python"])
    colors = dict(zip(pythons, cm.jet(np.linspace(0, 1, len(pythons)))))

    # markers according to system
    systems = np.unique(benchmarks_df["system"])
    markers = dict(zip(systems, ["x", "o", "s"]))  # osx, linux, windows
    benchmarks_df["marker"] = benchmarks_df["system"].apply(lambda x: markers[x])

    for i, (stat_name, stat_group) in enumerate(stats):
        stat_df = pd.DataFrame(stat_group)
        ax = axs[i] if nstats > 1 else axs
        ax.set_title(stat_name)
        ax.tick_params(axis="x", rotation=45)
        ax.xaxis.set_major_locator(dates.DayLocator(interval=1))
        ax.xaxis.set_major_formatter(dates.DateFormatter("\n%m-%d-%Y"))

        for si, system in enumerate(systems):
            ssub = stat_df[stat_df["system"] == system]
            marker = markers[system]
            for pi, python in enumerate(pythons):
                psub = ssub[ssub["python"] == python]
                color = colors[python]
                ax.scatter(psub["time"], psub["value"], color=color, marker=marker)
                ax.plot(psub["time"], psub["value"], linestyle="dotted", color=color)

    # configure legend
    patches = []
    for system in systems:
        for python in pythons:
            patches.append(
                Line2D(
                    [0],
                    [0],
                    color=colors[python],
                    marker=markers[system],
                    label=f"{system} Python{python}",
                )
            )
    leg = plt.legend(
        handles=patches,
        loc="upper left",
        ncol=3,
        bbox_to_anchor=(0, 0),
        framealpha=0.5,
        bbox_transform=ax.transAxes,
    )
    for lh in leg.legendHandles:
        lh.set_alpha(0.5)

    fig.suptitle(case_name)
    plt.ylabel("ms")

    fig.tight_layout()
    fig.set_size_inches(8, 8)

    return fig


def seaborn_plot(stats):
    nstats = len(stats)
    fig, axs = plt.subplots(nstats, 1, sharex=True)

    for i, (stat_name, stat_group) in enumerate(stats):
        stat_df = pd.DataFrame(stat_group)
        ax = axs[i] if nstats > 1 else axs
        ax.tick_params(axis="x", rotation=45)

        sp = sns.scatterplot(
            x="time",
            y="value",
            style="system",
            hue="python",
            data=stat_df,
            ax=ax,
            palette="YlOrBr",
        )
        sp.set(xlabel=None)
        ax.set_title(stat_name)
        ax.get_legend().remove()
        ax.set_ylabel("ms")

    fig.suptitle(case_name)
    fig.tight_layout()

    plt.subplots_adjust(left=0.3)
    plt.legend(loc="lower left", framealpha=0.3, bbox_to_anchor=(-0.45, -0.6))

    return fig


# create and save plots
cases = benchmarks_df.groupby("case")
for case_name, case in cases:
    stats = pd.DataFrame(case).groupby("stat")
    case_name = str(case_name).replace("/", "_").replace(":", "_")

    fig = seaborn_plot(stats)
    plt.savefig(str(outdir / f"{case_name}.png"))
