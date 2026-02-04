import numpy as np 
from podio import root_io
import edm4hep
# data = np.load("/eos/experiment/fcc/users/m/mgarciam/mlpf/condor/test/3/data.npy", allow_pickle=True).item()
# # print(data)
# energy_track = np.concatenate(data['energy_track'])
# energy_no_track = np.concatenate(data["energy_no_track"])

# print(len(energy_track), len(energy_no_track))
# # print(energy_no_track[energy_no_track>1])

# print(data["energy_no_track"])
input_file="/eos/experiment/fcc/users/m/mgarciam/mlpf/condor/Z_ss_CLD_o2_v05/10016/out_reco_edm4hep_REC.edm4hep.root"

reader = root_io.Reader(input_file)


track_indes_MCS = []
dic_events = {}
for i, event in enumerate(reader.get("events")):
        if i ==20+15:
            genparts = "MCParticles"
        
            gen_part_coll = event.get(genparts)

            for id, part in enumerate(gen_part_coll):
                 print(id, part.getPDG(), part.getGeneratorStatus(),part.getEndpoint().x ,part.getEndpoint().y,part.getEndpoint().z)
            parents = gen_part_coll[9].getParents()
            daugthers = gen_part_coll[9].getDaughters()
            for p in parents:
                print("p",p.getObjectID().index)
            for d in daugthers:
                print("d",d.getObjectID().index)
            sim_track = "SiTracksMCTruthLink"
            tracks_coll_link = event.get(sim_track)
            for link in tracks_coll_link:
                 print(link.getTo().getObjectID().index)

            sim_track = ["VertexBarrelCollection", "VertexEndcapCollection", "InnerTrackerBarrelCollection", "InnerTrackerEndcapCollection", "OuterTrackerBarrelCollection", "OuterTrackerEndcapCollection"]
            for col in sim_track:
                print("_____",col)
                tracks_coll_link = event.get(col)
                list_ = []
                for link in tracks_coll_link:
                    list_.append(link.getParticle().getObjectID().index)
                print("particle that left tracker hits", np.unique(list_))
                for mc in np.unique(list_):
                    print(mc,np.sum(list_==mc))
            # vertex= gen_part_coll[60].getVertex()
            # vertex = np.array([vertex.x,vertex.y,vertex.z]).reshape(1,3)
            # # print(vertex)
            # # print(isProducedInCalo(vertex))
            # for parent in parents:
            #     print(parent.getObjectID().index)
            # for col in calohit_collections_relation:
            #     relation_collection = event.get(col)
            #     for  hit_link in relation_collection:
            #         hit_sim = hit_link.getTo()
    
            #         particle = hit_sim.getParticle()
            #         object_ID_particle = particle.getObjectID().index
            #         track_indes_MCS.append(object_ID_particle)


            # unique_mcs = np.unique(np.array(track_indes_MCS))
            # dic_events[str(i) + "MCS"] = []
            # dic_events[str(i) + "number_hits"] = []
            # for unique_mc in unique_mcs:
            #     dic_events[str(i) + "MCS"].append(unique_mc)
            #     dic_events[str(i) + "number_hits"].append(np.sum(np.array(track_indes_MCS)==i))
