import numpy as np 
from podio import root_io
import edm4hep
import math 

input_file="/eos/experiment/fcc/users/m/mgarciam/mlpf/condor/test/7/out_reco_edm4hep_REC.edm4hep.root"

reader = root_io.Reader(input_file)
track_indes_MCS = []
dic_events = {}
electrons_with_tracks_E = []
electrons_with_tracks_phi = []
electrons_with_no_tracks_E = []
electrons_with_no_tracks_phi = []
for i, event in enumerate(reader.get("events")):
    print("i", i)
    genparts = "MCParticles"
    mc_sitracks_link = "SiTracksMCTruthLink"
    mc_cluster_link = "ClusterMCTruthLink"
    gen_part_coll = event.get(genparts)
    mc_sitracks_link_coll = event.get(mc_sitracks_link)
    mc_clusters_link_coll = event.get(mc_cluster_link)
    particles_with_tracks = []
    particles_with_cluster= []
    pdgs = []
    energies= []
    angles = []
    gens = []
    for link_track in mc_sitracks_link_coll:
        mc_particle = link_track.getTo()
        index_MC = mc_particle.getObjectID().index
        particles_with_tracks.append(index_MC)
    for link_cluster in mc_clusters_link_coll:
        mc_particle = link_cluster.getTo()
        index_MC = mc_particle.getObjectID().index
        particles_with_cluster.append(index_MC)
    for mc in gen_part_coll:
        pdg = mc.getPDG()
        energy = mc.getEnergy()
        phi = mc.getEnergy()
        momentum = mc.getMomentum()
        genstat = mc.getGeneratorStatus()
        p = math.sqrt(momentum.x**2 + momentum.y**2 + momentum.z**2)
        theta = math.acos(momentum.z / p)
        phi = math.atan2(momentum.y, momentum.x)
        pdgs.append(pdg)
        energies.append(energy)
        angles.append(theta)
        gens.append(genstat)
    particles_with_tracks = np.unique(particles_with_tracks)
    particles_with_cluster = np.unique(particles_with_cluster)
    electron_mask =  (np.abs(np.array(pdgs))==11)*(np.array(gens)==1)
    electron_indices = np.where(electron_mask)[0]
    energies = np.array(energies)
    angles = np.array(angles)
    if len(electron_indices)>0:
        electrons_with_tracks = np.intersect1d(electron_indices, particles_with_tracks)
        electrons_with_clusters = np.intersect1d(electron_indices, particles_with_cluster)
        electrons_with_both = np.intersect1d(electrons_with_tracks, electrons_with_clusters)
        electrons_with_clusters_no_tracks = np.setdiff1d(electrons_with_clusters, electrons_with_tracks)
        if len(electrons_with_both)>0:
            electrons_with_both = np.array(electrons_with_both, dtype=int)
            electrons_with_tracks_E.append(energies[electrons_with_both])
            electrons_with_tracks_phi.append(angles[electrons_with_both])
        if len(electrons_with_clusters_no_tracks)>0:
            electrons_with_clusters_no_tracks = np.array(electrons_with_clusters_no_tracks, dtype=int)
            electrons_with_no_tracks_E.append(energies[electrons_with_clusters_no_tracks])
            electrons_with_no_tracks_phi.append(angles[electrons_with_clusters_no_tracks])


dic_events = {}
dic_events["electrons_with_tracks_E"]=electrons_with_tracks_E
dic_events["electrons_with_tracks_phi"]=electrons_with_tracks_phi
dic_events["electrons_with_no_tracks_E"]=electrons_with_no_tracks_E
dic_events["electrons_with_no_tracks_phi"]=electrons_with_no_tracks_phi


        
        
       
                    
np.save('data_numhits_gen1.npy', dic_events, allow_pickle=True)
