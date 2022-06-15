# %%
from toolbox import get_pred_energy_diff_data
from constants import plots_dir
from matplotlib import pyplot as plt
import argparse

# Parse arguments
parser = argparse.ArgumentParser(description='Scatter plotterer')
parser.add_argument("run_id", type=str ,help="the id of the run to be analyzed, eg '3.2' for run3.2")

args = parser.parse_args()
run_id = args.run_id

# Save the run name
run_name = f"run{run_id}"

plots_dir = f"{plots_dir}/{run_id}"
# %%
energy_difference_data, shower_energy_log10_predict, shower_energy_log10 = get_pred_energy_diff_data(run_name, True)

# %%
log_E_string = r"$\log_{10}\:E$"

xmin = min(shower_energy_log10_predict)
xmax = max(shower_energy_log10_predict)
ymin = min(shower_energy_log10)
ymax = max(shower_energy_log10)

fig  = plt.figure()
ax = fig.gca()
ax.plot(shower_energy_log10_predict, shower_energy_log10, '.', markersize=0.1)
ax.plot([min(xmin, ymin), max(xmax, ymax)], [min(xmin, ymin), max(xmax, ymax)], 'k--')

ax.set_title(f"Scatter plot for {run_name}")
ax.set_xlabel(f"predicted {log_E_string}")
ax.set_ylabel(f"true {log_E_string}")

ax.set_xlim(16, 19.5)
ax.set_ylim(16, 19.5)
# ax.set_xlim(xmin - 0.5, xmax + 0.5)
# ax.set_ylim(ymin - 0.5, ymax + 0.5)

fig.tight_layout()

fig.savefig(f'{plots_dir}/scatter_{run_name}.png')
