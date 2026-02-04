from podio import root_io
import edm4hep
import sys
import ROOT
from ROOT import TFile, TTree
import numpy as np 
from array import array
import math
import  glob 
import os 
import torch
import matplotlib.pyplot as plt


def cdist_wrap_angles(x1, x2=None, p=2, angle_indices=None):
    """
    Compute pairwise distances with wrap-around for multiple angular coordinates.
    
    Parameters
    ----------
    x1 : torch.Tensor, shape (N, D)
    x2 : torch.Tensor, shape (M, D) or None (defaults to x1)
    p  : float, norm degree (default 2)
    angle_indices : list or tuple of int
        Indices of dimensions that are angles to be wrapped in [-pi, pi].
        If None, no wrapping is done.
    """
    if x2 is None:
        x2 = x1

    diff = x1[:, None, :] - x2[None, :, :]

    if angle_indices is not None:
        for idx in angle_indices:
            diff[..., idx] = (diff[..., idx] + math.pi) % (2 * math.pi) - math.pi

    if p == 2:
        dist = torch.sqrt((diff ** 2).sum(dim=-1))
    else:
        dist = (diff.abs() ** p).sum(dim=-1) ** (1 / p)

    return dist


list_overlay = []
for i in range(1,20):
    list_overlay.append("/eos/experiment/fcc/users/m/mgarciam/mlpf/condor/Hss_2025_08_06_key4hep_20250529_CLD_r20250526/"+str(i)+"/out_reco_edm4hep_REC.edm4hep.root")
    #list_overlay.append("/eos/experiment/fcc/ee/datasets/DC_tracking/Pythia/scratch/Zcard_CLD_background_IDEA_o1_v03_v4/"+str(i)+"/out_sim_edm4hep_base.root")

# seed = 2
# file_directory = f"/eos/experiment/fcc/users/m/mgarciam/mlpf/condor/gun_2025_08_06_key4hep_20250529_CLD_r20250526/{seed}"
# file_name_template = "out_reco_edm4hep_REC.edm4hep.root"
# list_overlay = glob.glob(os.path.join(file_directory, file_name_template))
dic = {}
dic["delta_MC"] = [] #min dR between charged particles in the event
np.save(f"reco_CLD_Hss_output_seed_1_20.npy", dic)
total_time = 0 
for path_number,path in enumerate(list_overlay):
    rootfile = path
    reader = root_io.Reader(rootfile)
    metadata = reader.get("metadata")[0]
    print("path_number", path_number)
    counter = 0
    for event in reader.get("events"):
        counter +=1
        if counter % 100 == 0:
            print(counter)
        
        dic = np.load(f"reco_CLD_Hss_output_seed_1_20.npy", allow_pickle=True).item()
        list_index = []
        seen = []
        count_hits = []
        list_pdg = []
        
        hcal_hits = event.get("ECalBarrelCollectionContributions")
        for num_hit, hcal_hit in enumerate(hcal_hits):
            mcParticle = hcal_hit.PDG()
            hcalhit_index = hcal_hit.getParticle().getObjectID().index
            list_index.append(hcalhit_index)
        unique_mcs = np.unique(np.array(list_index))
        
        # print(unique_mcs.shape, np.array(count_hits).shape)
       
        MCparticles = event.get("MCParticles")
        list_eta = []
        list_phi = []
      
      

        for i in range(0, len(unique_mcs)):
            mc_index = unique_mcs[i]
            mcParticle = MCparticles[int(mc_index)]
            x_vertex = mcParticle.getVertex().x
            y_vertex = mcParticle.getVertex().y
            z_vertex = mcParticle.getVertex().z
            x_end = mcParticle.getEndpoint().x
            y_end = mcParticle.getEndpoint().y
            z_end = mcParticle.getEndpoint().z

            theta = np.arccos(z_end / np.sqrt(x_end ** 2 + y_end ** 2 + z_end ** 2))
            eta = -np.log(np.tan(theta / 2))
            phi = np.arctan2(y_end, x_end)
            
            momentum = mcParticle.getMomentum()
            energy = mcParticle.getEnergy()
            p = math.sqrt(momentum.x**2 + momentum.y**2)
            list_eta.append(eta)
            list_phi.append(phi)

            #find index for charged particles
            # if mcParticle.getPDG() in [-321,321,211,-211,2212]:
            #     charged_indices.append(i)

        eta_MCs = torch.tensor(np.array(list_eta))
        phi_MCs = torch.tensor(np.array(list_phi))
        
        
     
        x1 = torch.cat((eta_MCs.view(-1, 1), phi_MCs.view(-1, 1)), dim=1)
        distance_matrix = cdist_wrap_angles(x1, x1, p=2)
        shape_d = distance_matrix.shape[0]
        values, _ = torch.sort(distance_matrix, dim=1)
       
        delta_MC = values[:, 1]
        dic["delta_MC"]=np.append(dic["delta_MC"], delta_MC.numpy())
        np.save(f"reco_CLD_Hss_output_seed_1_20.npy", dic)
        
# dic = np.load(f"reco_CLD_gun_output_seed_2.npy", allow_pickle=True).item()
# dic_Hss = np.load(f"reco_CLD_Hss_output_seed_2.npy", allow_pickle=True).item()
# bins = np.linspace(0, 2, 1000)

# plt.figure(figsize=(6, 4))
# seaborn.histplot(dic["delta_MC"], bins=bins, edgecolor='black', alpha=0.8,stat="percent",label="gun", color='skyblue')
# seaborn.histplot(dic_Hss["delta_MC"], bins=bins, edgecolor='black', alpha=0.8,stat="percent", label="Hss", color='orange')
# plt.xlabel(r"Minimum $\Delta R$")
# plt.ylabel("Number of particles")
# plt.title("Distribution of Minimum $\Delta R$")
# plt.grid(True, linestyle='--', alpha=0.5)
# plt.xlim([0,0.3])
# plt.legend()

# print("Saved deltaR_min_distribution.png and deltaR_min_distribution.pdf")

# plt.figure(figsize=(6, 4))
# # Avoid zeros for log scale
# deltaR_min = dic["delta_MC"]
# deltaR_min = deltaR_min[deltaR_min > 0]
# # Log-spaced bins
# bins = np.logspace(np.log10(deltaR_min.min()), np.log10(deltaR_min.max()), 30)
# seaborn.histplot(dic["delta_MC"][dic["delta_MC"]>0], bins=bins,  edgecolor='black', alpha=0.8,stat="percent",label="gun", color='skyblue')
# seaborn.histplot(dic_Hss["delta_MC"][dic_Hss["delta_MC"]>0], bins=bins,  edgecolor='black', alpha=0.8,stat="percent", label="Hss", color='orange')
# plt.xlabel(r"Minimum $\Delta R$")
# plt.ylabel("Number of particles")
# plt.title("Distribution of Minimum $\Delta R$")
# plt.xscale('log')
# plt.legend()
# plt.grid(True, linestyle='--', alpha=0.5)

# # Save to file
# plt.tight_layout()
# plt.savefig("deltaR_min_distribution_log_gun.png", dpi=300)
# plt.close()

# print("Saved deltaR_min_distribution.png and deltaR_min_distribution.pdf")


