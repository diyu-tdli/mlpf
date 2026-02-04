import numpy as np
import matplotlib.pyplot as plt

seed=939

# Load the dictionary
data = np.load(f"reco_CLD_Hss_output_seed_{seed}.npy", allow_pickle=True).item()

# Inspect the keys to know what’s inside
print("Keys in the loaded dictionary:", data.keys())

mindR = data["mindR"][data["mindR"] <= 99]
mindR_c = data["mindR_c"][data["mindR_c"] <= 99]
energy = data["energy"]
mindR_elow = data["mindR"][(data["mindR"] <= 99) & (data["energy"]<5)]
mindR_ehigh = data["mindR"][(data["mindR"] <= 99) & (data["energy"]>10)]

charged_enegry = data["energy"][(data["pdg"]==321) | (data["pdg"]==-321) | (data["pdg"]==211) | (data["pdg"]==-211) | (data["pdg"]==2212)]

mindR_elow_c = data["mindR_c"][(data["mindR_c"] <= 99) & (charged_enegry<5) ]
mindR_ehigh_c = data["mindR_c"][(data["mindR_c"] <= 99) & (charged_enegry > 10) ]

fig1 = plt.figure()
plt.hist(mindR, bins=100, range=(0, 0.3), edgecolor='black', label='inclusive')
#plt.xscale('log')
#plt.xlim(0, 0.5)
plt.xlabel(' min dR')
plt.ylabel('#')
plt.title('min dR  in H->ss, about 3500 events')
plt.hist(mindR_c, bins=100, range=(0, 0.3), edgecolor='black', label='charged particles (K+-, pi+-, p)')
plt.legend()
plt.savefig(f"plots/deltaR_seed_{seed}.pdf")

fig2 = plt.figure()
plt.hist(energy, bins=50, range=(0, 100), edgecolor='black', label='inclusive')
plt.xlabel('E [GeV]')
plt.ylabel('#')
plt.title('H->ss')
plt.savefig(f"plots/energy_{seed}.pdf")

fig3 = plt.figure()
plt.hist(mindR_elow, bins=50, range=(0, 0.3), edgecolor='black', label=' E [0-5GeV]')
plt.hist(mindR_ehigh, bins=50, range=(0, 0.3), edgecolor='black', label='E > 10 GeV')
plt.xlabel('mindR')
plt.ylabel('#')
plt.title('H->ss')
plt.legend()
plt.savefig(f"plots/deltaR_energycomp_{seed}.pdf")

fig4 = plt.figure()
plt.hist(mindR_elow_c, bins=50, range=(0, 0.3), edgecolor='black', label='charged particles, E [0-5GeV]')
plt.hist(mindR_ehigh_c, bins=50, range=(0, 0.3), edgecolor='black', label='charged particles, E > 10 GeV')
plt.xlabel('mindR')
plt.ylabel('#')
plt.title('H->ss')
plt.legend()
plt.savefig(f"plots/deltaR_energycomp_c_{seed}.pdf")