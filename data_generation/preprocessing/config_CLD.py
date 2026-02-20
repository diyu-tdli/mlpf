
from preprocessing.utils import Geometry, Names_Collections

track_coll = "SiTracks_Refitted"
mc_coll = "MCParticles"

def create_name_coll(truth_tracking):
    NAMES_COL = Names_Collections()

    NAMES_COL.MC_PARTICLE_COL = "MCParticles"
    NAMES_COL.PANDORA_PFO_COL = "PandoraPFOs"
    NAMES_COL.TRACKS_COL = "SiTracks_Refitted"
    NAMES_COL.CLUSTERS_COL = "PandoraClusters"
    NAMES_COL.CALOHIT_TO_MC_LINK_COL = "CalohitMCTruthLink"
    if truth_tracking:
        NAMES_COL.TRACK_TO_MC_LINK_COL = "SiTracks_Refitted_Relation"
    else:
        NAMES_COL.TRACK_TO_MC_LINK_COL = "SiTracksMCTruthLink"
    NAMES_COL.CALO_HIT_COLS = [
        "ECALBarrel",
        "ECALEndcap",
        "HCALBarrel",
        "HCALEndcap",
        "HCALOther",
        "MUON",
    ]
    return NAMES_COL


geometry = Geometry(BarrelRadius=2150, NBarrelSides=12, EndCapZ=2307, B=2)
