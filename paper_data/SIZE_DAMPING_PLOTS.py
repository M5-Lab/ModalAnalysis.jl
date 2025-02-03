import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib
matplotlib.use('tkagg')

pots = ["LJ" , "SW"]

temps = {"SW": [100,1300], "LJ": [10,80]}
sizes = {"SW" : [2,3,4,5,6], "LJ": [3,4,5,6]}
damps = {"SW" : [10,32,100,316,1000], "LJ": [10,32,100,316,1000]}

dir_path = os.path.dirname(os.path.realpath(__file__))

plt.figure(figsize=(9,6.75), dpi = 300)

colors = ["#d43136", "#53b06a", "#4589c4", "#f29b44"]

# SIZE EFFECTS PLOT
idx = 0
for i, pot in enumerate(pots):

    data = np.load(os.path.join(dir_path, f"DATA/Studies/{pot}_SIZE_EFFECTS_DATA.npz"))
    
    mean_cv = data["mean_cv"]
    se = data["se"]

    for j, T in enumerate(temps[pot]):
        plt.errorbar(sizes[pot], mean_cv[j], se[j], label = f"{pot} {T} K",
                      fmt = 'o', markersize = 25, capsize = 5, color = colors[idx])
        idx += 1

plt.xticks([2,3,4,5,6], fontsize = 22)
plt.yticks([1.3,1.4,1.5,1.6,1.7], fontsize = 22)
plt.xlabel("Number of Unit Cells Per Dimension", fontsize = 22)
plt.ylabel("$C_{V,U}$ / (N $k_B$)", fontsize = 22)
plt.legend(loc='upper left', fontsize = 22, ncol = 2)
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.xlim([1.5,6.5])
plt.ylim([1.25,1.9])

plt.savefig(os.path.join(dir_path, "OTHER_PLOTS/SIZE_EFFECTS_PLOT.pdf"), bbox_inches='tight')
plt.savefig(os.path.join(dir_path, "OTHER_PLOTS/SIZE_EFFECTS_PLOT.png"), bbox_inches='tight')


# DAMPING EFFECTS PLOT

plt.figure(figsize=(9,6.75), dpi = 300)
idx = 0
for i, pot in enumerate(pots):
    
    data = np.load(os.path.join(dir_path, f"DATA/Studies/{pot}_DAMPING_STUDY_DATA.npz"))
    
    mean_cv = data["mean_cv"]
    se = data["se"]

    for j, T in enumerate(temps[pot]):
        plt.errorbar(damps[pot], mean_cv[j], se[j], label = f"{pot} {T} K", 
                     fmt = 'o', markersize = 25, capsize = 5, color = colors[idx])
        idx += 1

plt.xscale("log")
plt.xticks([10,100,1000], fontsize = 22)
plt.yticks([1.3,1.4,1.5,1.6,1.7], fontsize = 22)
plt.xlabel("Damping Time Scale Prefactor", fontsize = 22)
plt.ylabel("$C_{V,U}$ / (N $k_B$)", fontsize = 22)
plt.legend(loc='upper left', fontsize = 22, ncol = 2)
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.xlim([5,2000])
plt.ylim([1.25,1.9])

plt.savefig(os.path.join(dir_path, "OTHER_PLOTS/DAMPING_EFFECTS_PLOT.pdf"), bbox_inches='tight')
plt.savefig(os.path.join(dir_path, "OTHER_PLOTS/DAMPING_EFFECTS_PLOT.png"), bbox_inches='tight')
