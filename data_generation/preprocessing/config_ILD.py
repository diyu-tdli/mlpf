
import os
import numpy as np
import awkward
import uproot
import vector
import tqdm
from scipy.sparse import coo_matrix
from preprocessing.utils import Geometry, correct_link, particle_feature_order, Names_Collections, PandoraPFO_feature_order,  particle_feature_order, PandoraPFO_feature_order, track_feature_order, hit_feature_order

def create_name_coll():
    NAMES_COL = Names_Collections()

    # Configure all Collection names
    # NOTE: Should be in the configuration file
    NAMES_COL.MC_PARTICLE_COL = "MCParticles"
    NAMES_COL.PANDORA_PFO_COL = "PandoraPFOs"
    NAMES_COL.TRACKS_COL = "MarlinTrkTracks"
    NAMES_COL.CLUSTERS_COL = "PandoraClusters"
    NAMES_COL.CALOHIT_TO_MC_LINK_COL = "CalohitMCTruthLink"
    NAMES_COL.TRACK_TO_MC_LINK_COL = "MCTruthMarlinTrkTracksLink"
    NAMES_COL.CALO_HIT_COLS = [
        "EcalBarrelCollectionRec",
        "EcalEndcapRingCollectionRec",
        "EcalEndcapsCollectionRec",
        "HcalBarrelCollectionRec",
        "HcalEndcapRingCollectionRec",
        "HcalEndcapsCollectionRec",
        "MUON",
        # "LCAL",
        # "LHCAL",
        # "BCAL",
    ]

geometry = Geometry(1804.8, 8,  2411.8, 2)
# Configure detector geometry parameters
# https://github.com/key4hep/k4geo/tree/a473d3fd3d7fb182530636f64d033f277d1c185d/ILD/compact/ILD_common_v02
#BARREL_RADIUS = 1804.8   top_TPC_outer_radius + Ecal_Tpc_gap = 1769.8*mm + 35*mm
#ENDCAP_Z = 2411.8        TPC_Ecal_Hcal_barrel_halfZ + 61.8*mm = 2350.0*mm + 61.8*mm




hit_feature_order = [
    "elemtype",
    "et",
    "eta",
    "sin_phi",
    "cos_phi",
    "energy",
    "position.x",
    "position.y",
    "position.z",
    "time",
    "time_10ps",
    "time_50ps",
    "time_100ps",
    "time_1000ps",
    "subdetector",
    "type",
]















