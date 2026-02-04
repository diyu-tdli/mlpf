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
list_overlay = []
# for i in range(1,500):
#     list_overlay.append("/eos/experiment/fcc/ee/datasets/DC_tracking/Pythia/IDEA_background_only_IDEA_o1_v03_v4/out_sim_edm4hep_background_"+str(i)+".root")
#     #list_overlay.append("/eos/experiment/fcc/ee/datasets/DC_tracking/Pythia/scratch/Zcard_CLD_background_IDEA_o1_v03_v4/"+str(i)+"/out_sim_edm4hep_base.root")
    
file_directory = "/afs/cern.ch/work/l/lherrman/public/MLPF/mlpf/create_files/938"
file_name_template = "out_sim_edm4hep.root"
list_overlay = glob.glob(os.path.join(file_directory, file_name_template))
dic = {}
dic["R"] = []
dic["pt"] = []
dic["px"] = []
dic["py"] = []
dic["pz"] = []
dic["x"] = []
dic["y"] = []
dic["z"] = []
dic["gens"] = []
dic["count_hits"] = []
dic["pdg"] = []
np.save("sim_output_CLD_Hss.npy", dic)
total_time = 0 
for path_number,path in enumerate(list_overlay):
    rootfile = path
    reader = root_io.Reader(rootfile)
    metadata = reader.get("metadata")[0]
    print("path_number", path_number)
    for event in reader.get("events"):
        
        dic = np.load("sim_output_CLD_Hss.npy", allow_pickle=True).item()
        list_index = []
        seen = []
        count_hits = []
        list_pdg = []
        ecal_hits = event.get("ECalBarrelCollectionContributions")
        for num_hit, ecal_hit in enumerate(ecal_hits):
            mcParticleID = ecal_hit.PDG()
            particle = ecal_hit.getParticle()
            index_mc = particle.index_mc()
            print("hit from MCParticle")
            print(mcParticleID)
            print(index_mc)
            # if mcParticle.getPDG() == 2112:
                # print("is produced by secondary", dc_hit.isProducedBySecondary())
            # if (not dc_hit.isProducedBySecondary()) and (mcParticle.getPDG()== 2112):
            #index_mc = mcParticle.getObjectID().index
            list_index.append(index_mc)
            #check if the hit was produced by a particle already seen (in list_index)
            if index_mc not in seen:
                #add 1 to the count of hits at this index
                count_hits.append(1)
                seen.append(index_mc)
            else:
                #find the index of the hit in seen
                index = seen.index(index_mc)
                #add 1 to the count of hits at this index
                count_hits[index] += 1
        unique_mcs = np.unique(np.array(list_index))
        # print(unique_mcs.shape, np.array(count_hits).shape)
        MCparticles = event.get("MCParticles")
        list_R = []
        list_p = []
        list_px = []
        list_py = []
        list_pz = []
        list_x = []
        list_y = []
        list_z = []
        list_gen_status = []
        for i in range(0, len(unique_mcs)):
            mc_index = unique_mcs[i]
            mcParticle = MCparticles[int(mc_index)]
            x_vertex = mcParticle.getVertex().x
            y_vertex = mcParticle.getVertex().y
            z_vertex = mcParticle.getVertex().z
            vertex_R = math.sqrt(mcParticle.getVertex().x ** 2 + mcParticle.getVertex().y ** 2)* 1e-03
            list_R.append(vertex_R)
            momentum = mcParticle.getMomentum()
            p = math.sqrt(momentum.x**2 + momentum.y**2)
            list_p.append(p)
            list_px.append(momentum.x)
            list_py.append(momentum.y)
            list_pz.append(momentum.z)
            list_x.append(mcParticle.getVertex().x)
            list_y.append(mcParticle.getVertex().y)
            list_z.append(mcParticle.getVertex().z)
            list_pdg.append(mcParticle.getPDG())
            gen_status = mcParticle.getGeneratorStatus()
            list_gen_status.append(gen_status)

        dic["R"]=np.append(dic["R"], np.array(list_R))
        dic["pdg"]=np.append(dic["pdg"], np.array(list_pdg))
        dic["pt"]=np.append(dic["pt"], np.array(list_p))
        dic["px"]=np.append(dic["px"], np.array(list_px))
        dic["py"]=np.append(dic["py"], np.array(list_py))
        dic["pz"]=np.append(dic["pz"], np.array(list_pz))
        dic["x"]=np.append(dic["x"], np.array(list_x))
        dic["y"]=np.append(dic["y"], np.array(list_y))
        dic["z"]=np.append(dic["z"], np.array(list_z))
        dic["count_hits"]=np.append(dic["count_hits"], np.array(count_hits))
        dic["gens"]=np.append(dic["gens"], np.array(list_gen_status))
        np.save("background_particles_IDEA_o1_v03_v8_v1.npy", dic)

# eos_base_file = "/eos/experiment/fcc/ee/datasets/DC_tracking/Pythia/scratch/Zcard_CLD_background/4/out_sim_edm4hep_base.root"

# reader = root_io.Reader(eos_base_file)
# metadata = reader.get("metadata")[0]
# list_times = []
# for event in reader.get("events"):
#     dc_hits = event.get("CDCHHits")
#     for num_hit, dc_hit in enumerate(dc_hits):
#         time = dc_hit.getTime()
#         if time<400:
#             list_times.append(time)

# dic["base"] =  list_times

# np.save("background_particles.npy", dic)