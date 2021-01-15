import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

SHOW_PLOT = False
DAS_PATH = "/var/scratch/lvs215/processed-surf-dataset/"
SAVEFIG_PATH= "/home/cmt2002/rack-barplots/"
FIGNAME = "rack_violinplots"

def get_rack_nodes(df):
    rack_nodes = {}

    for node in df.columns:
        rack = node.split("n")[0]
        if rack not in rack_nodes:
            rack_nodes[rack] = set()

        rack_nodes[rack].add(node)

    return rack_nodes

def get_custom_values(df):
    values = np.array([])
    for column in df.columns:
        arr = df[column].values
        mask = (np.isnan(arr) | (arr < 0))

        arr = arr[~mask]  # Filter out NaN values and less than 0
        values = np.append(values, arr)

    return values


def rack_violinplot(ax, df_covid, df_non_covid, subtitle, ylabel):
    rack_nodes = get_rack_nodes(df_covid) # To get the rack nodes
    rack_values = list()
    rack_names = list()
    violin_width = 0.8
    
    for rack, columns in rack_nodes.items():
        arr_covid = get_custom_values(df_covid[list(columns)])
        arr_non_covid = get_custom_values(df_non_covid[list(columns)])
        rack_values.append(arr_covid)
        rack_values.append(arr_non_covid)
        rack_names.append(rack)
        
    sns.violinplot(data=rack_values, ax=ax, cut=0, width=violin_width, palette=['lightcoral', 'steelblue'] * (int(len(rack_values)/2)))
    ax.set_ylabel(ylabel,fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=14)
    ax.set_xticks([i + 0.5 for i in range(0, len(rack_values), 2)])
    ax.set_xlabel(subtitle, fontsize=14)

    # For load1, depict the values that exceed 100 load
    if ylabel == "Load1":
        ax.set_ylim(0, 100)
        for index, val in enumerate(rack_values):
            max_val = np.amax(val)
            if max_val > 100:
                ax.text(x=index-0.2, y=102.2, s=str(int(max_val)), fontsize=22, color="black", va="center")

    else:
        ax.set_ylim(0, )

    ax.set_xticklabels(
        rack_names,
        ha='center', fontsize=16
    )
    for i in range(0, len(rack_values), 2):
        ax.axvline(i + 1.5, lw=2, ls='dashed')

def rack_analysis_violinplot(df_dic, ax, ylabel):
    rack_violinplot(
        ax=ax,
        df_covid=df_dic["covid"],
        df_non_covid=df_dic["non_covid"],
        subtitle=None,
        ylabel=ylabel)
    ax.axvline(x=9.5, c="green", lw=1.5)

def covid_non_covid(df):
    if df.index.dtype == "int64":
        df.index = pd.to_datetime(df.index, unit='s')

    covid_df = df.loc['2020-02-27 00:00:00':, :]
    non_covid_df = df.loc[: '2020-02-26 23:59:45', :]
    covid_df.reset_index()
    non_covid_df.reset_index()
    return covid_df, non_covid_df

df_free = pd.read_parquet(DAS_PATH + "node_memory_MemFree")
df_total = pd.read_parquet(DAS_PATH + "node_memory_MemTotal")
df_ram_covid, df_ram_non_covid = covid_non_covid(100 * (1 - (df_free / df_total)))
df_load_covid, df_load_non_covid = covid_non_covid(pd.read_parquet(DAS_PATH + "node_load1"))
df_power_covid, df_power_non_covid = covid_non_covid(pd.read_parquet(DAS_PATH + "surfsara_power_usage"))
df_temp_covid, df_temp_non_covid = covid_non_covid(pd.read_parquet(DAS_PATH + "surfsara_ambient_temp"))


_, (ax_power, ax_load, ax_temp, ax_ram) = plt.subplots(4, 1, figsize=(11, 8), constrained_layout=True, sharex=True)
rack_analysis_violinplot(
    df_dic={"covid": df_power_covid, "non_covid": df_power_non_covid},
    ax=ax_power,
    ylabel="Power consumption [W]")
rack_analysis_violinplot(
    df_dic={"covid": df_temp_covid, "non_covid": df_temp_non_covid},
    ax=ax_temp,
    ylabel="Temperature [C]")
rack_analysis_violinplot(
    df_dic={"covid": df_load_covid, "non_covid": df_load_non_covid},
    ax=ax_load)
rack_analysis_violinplot(
    df_dic={"covid": df_ram_covid, "non_covid": df_ram_non_covid},
    ax=ax_ram,
    ylabel="RAM utilization [%]")

ax_ram.set_xlabel("Racks")

# Depict legend on top of the first plot
lightcoral_patch = mpatches.Patch(color='lightcoral', label='covid (left)')
steelblue_patch = mpatches.Patch(color='steelblue', label='non-covid (right)')
ax_power.legend(handles=[lightcoral_patch, steelblue_patch], loc="center", bbox_to_anchor=(0.5, 1.13), fontsize=14,
          ncol=2)

plt.savefig((SAVEFIG_PATH + FIGNAME + ".pdf"), dpi=100)
if SHOW_PLOT:
    plt.show()
plt.pause(0.0001)


print("DONE!")
