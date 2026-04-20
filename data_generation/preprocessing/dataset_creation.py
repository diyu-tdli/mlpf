import os
import numpy as np
import awkward
import uproot
import tqdm
from preprocessing.utils import get_feature_matrix, sanitize, build_dummy_array
from preprocessing.utils import track_feature_order, particle_feature_order, PandoraPFO_feature_order
from preprocessing.utils import  get_genparticles_and_adjacencies
import time 




def process_one_file(fn, ofn, args):
    if args.ILD:
        from preprocessing.config_ILD import  create_name_coll, geometry, hit_feature_order
    elif args.ARC:
        from preprocessing.config_CLD_ARC import  create_name_coll, geometry
        from preprocessing.utils import hit_feature_order
    elif args.ALLEGRO:
        from preprocessing.config_ALLEGRO import  create_name_coll, geometry
        from preprocessing.utils import hit_feature_order
    else:
        from preprocessing.config_CLD import   create_name_coll, geometry
        from preprocessing.utils import hit_feature_order
    # output exists, do not recreate
    # if os.path.isfile(ofn):
    #     return
    # print(fn)

    fi = uproot.open(fn)

    arrs = fi["events"]
    collectionIDs = {
        k: v
        for k, v in zip(
            fi.get("podio_metadata").arrays("events___CollectionTypeInfo.name")["events___CollectionTypeInfo.name"][0],
            fi.get("podio_metadata").arrays("events___CollectionTypeInfo.collectionID")["events___CollectionTypeInfo.collectionID"][0]
        )
    }
    NAMES_COL = create_name_coll(args.truth)

    prop_data = arrs.arrays(
        [
            NAMES_COL.MC_PARTICLE_COL,
            f"{NAMES_COL.MC_PARTICLE_COL}.PDG",
            f"{NAMES_COL.MC_PARTICLE_COL}.momentum.x",
            f"{NAMES_COL.MC_PARTICLE_COL}.momentum.y",
            f"{NAMES_COL.MC_PARTICLE_COL}.momentum.z",
            f"{NAMES_COL.MC_PARTICLE_COL}.mass",
            f"{NAMES_COL.MC_PARTICLE_COL}.charge",
            f"{NAMES_COL.MC_PARTICLE_COL}.generatorStatus",
            f"{NAMES_COL.MC_PARTICLE_COL}.simulatorStatus",
            f"{NAMES_COL.MC_PARTICLE_COL}.daughters_begin",
            f"{NAMES_COL.MC_PARTICLE_COL}.daughters_end",
            f"{NAMES_COL.MC_PARTICLE_COL}.daughters_end",
            f"_{NAMES_COL.MC_PARTICLE_COL}_daughters/_{NAMES_COL.MC_PARTICLE_COL}_daughters.index",  # similar to "MCParticles#1.index" in clic
            f"_{NAMES_COL.MC_PARTICLE_COL}_parents/_{NAMES_COL.MC_PARTICLE_COL}_parents.index",  # similar to "MCParticles#1.index" in clic
            NAMES_COL.TRACKS_COL,
            f"_{NAMES_COL.TRACKS_COL}_trackStates",
        ]
    )
    if args.dataset:
        pandora_data = arrs.arrays(
            [
                f"_{NAMES_COL.PANDORA_PFO_COL}_tracks/_{NAMES_COL.PANDORA_PFO_COL}_tracks.index",
                f"{NAMES_COL.CLUSTERS_COL}",
                f"_{NAMES_COL.CLUSTERS_COL}_hits/_{NAMES_COL.CLUSTERS_COL}_hits.index",
                f"_{NAMES_COL.CLUSTERS_COL}_hits/_{NAMES_COL.CLUSTERS_COL}_hits.collectionID",
                f"{NAMES_COL.PANDORA_PFO_COL}",
                f"_{NAMES_COL.PANDORA_PFO_COL}_clusters/_{NAMES_COL.PANDORA_PFO_COL}_clusters.index",
                # "SiTracks_Refitted_dQdx",
            ]
        )
    else:
        pandora_data = []

    calohit_links = arrs.arrays(
        [
            f"{NAMES_COL.CALOHIT_TO_MC_LINK_COL}.weight",
            f"_{NAMES_COL.CALOHIT_TO_MC_LINK_COL}_to/_{NAMES_COL.CALOHIT_TO_MC_LINK_COL}_to.collectionID",
            f"_{NAMES_COL.CALOHIT_TO_MC_LINK_COL}_to/_{NAMES_COL.CALOHIT_TO_MC_LINK_COL}_to.index",
            f"_{NAMES_COL.CALOHIT_TO_MC_LINK_COL}_from/_{NAMES_COL.CALOHIT_TO_MC_LINK_COL}_from.collectionID",
            f"_{NAMES_COL.CALOHIT_TO_MC_LINK_COL}_from/_{NAMES_COL.CALOHIT_TO_MC_LINK_COL}_from.index",
        ]
    )
    
    sitrack_links = arrs.arrays(
        [
            f"{NAMES_COL.TRACK_TO_MC_LINK_COL}.weight",
            f"_{NAMES_COL.TRACK_TO_MC_LINK_COL}_to/_{NAMES_COL.TRACK_TO_MC_LINK_COL}_to.collectionID",
            f"_{NAMES_COL.TRACK_TO_MC_LINK_COL}_to/_{NAMES_COL.TRACK_TO_MC_LINK_COL}_to.index",
            f"_{NAMES_COL.TRACK_TO_MC_LINK_COL}_from/_{NAMES_COL.TRACK_TO_MC_LINK_COL}_from.collectionID",
            f"_{NAMES_COL.TRACK_TO_MC_LINK_COL}_from/_{NAMES_COL.TRACK_TO_MC_LINK_COL}_from.index",
        ]
    )
    hit_data = {}
    for CALO_HIT_COL in NAMES_COL.CALO_HIT_COLS:
        hit_data[CALO_HIT_COL] = arrs[CALO_HIT_COL].array() 

    ret = []
    for iev in tqdm.tqdm(range(arrs.num_entries), total=arrs.num_entries):
        # get the genparticles and the links between genparticles and tracks/clusters
        # if iev==71:
        gpdata  = get_genparticles_and_adjacencies( prop_data, hit_data, pandora_data, calohit_links, sitrack_links, iev, collectionIDs,NAMES_COL, geometry, args)


        n_tracks = len(gpdata.track_features["type"])
        n_hits = len(gpdata.hit_features["type"])
        n_gps = len(gpdata.gen_features_target["PDG"])
        print("hits={} tracks={} gps={}".format(n_hits, n_tracks, n_gps))

        track_to_gp = gpdata.track_to_gp
        hit_to_gp = gpdata.hit_to_gp


        X_track = get_feature_matrix(gpdata.track_features, track_feature_order)
        X_hit = get_feature_matrix(gpdata.hit_features, hit_feature_order)
        X_target = get_feature_matrix(gpdata.gen_features_target, particle_feature_order)
        # X_gen = get_feature_matrix(gpdata.gen_features_true, particle_feature_order)
        if args.dataset:
            X_pandora = get_feature_matrix(gpdata.pandora_features, PandoraPFO_feature_order)
        ytarget_track = track_to_gp
        ytarget_hit = hit_to_gp
    #     ycand_track = rps_track
    #     ycand_hit = rps_hit

        sanitize(X_track)
        sanitize(X_hit)
        sanitize(X_target)
        if len(ytarget_track)>0:
            sanitize(ytarget_track)
        sanitize(ytarget_hit)
        if args.dataset:
            sanitize(X_pandora) 
            sanitize(gpdata.pfo_to_calohit)
            sanitize(gpdata.pfo_to_track)

        this_ev = {
            "X_track": X_track,
            "X_hit": X_hit,
            "X_gen": X_target, 
            # "X_gen_true": X_gen, 
            "ygen_track": ytarget_track,
            "ygen_hit": ytarget_hit,
            "ygen_hit_calom": gpdata.gp_to_calohit_beforecalomother,
        }
        if args.dataset:
            this_ev["X_pandora"] = X_pandora
            this_ev["pfo_calohit"] = gpdata.pfo_to_calohit
            this_ev["pfo_track"] = gpdata.pfo_to_track
            

        this_ev = awkward.Record(this_ev)
        ret.append(this_ev)
        # i = i +§
        if args.ALLEGRO:
            total_events = arrs.num_entries
            if (iev + 1) % args.chunk_size == 0 or iev == total_events - 1:
                if len(ret) > 0:
                    ret_chunk = {k: awkward.from_iter([r[k] for r in ret]) for k in ret[0].fields}
                    for k in ret_chunk.keys():
                        if len(awkward.flatten(ret_chunk[k])) == 0:
                            ret_chunk[k] = build_dummy_array(len(ret_chunk[k]), np.float32)
                    ret_chunk = awkward.Record(ret_chunk)

                    ofn = f"{ofn}_{chunk_index}.parquet"
                    awkward.to_parquet(ret_chunk, ofn, compression="snappy")
                    print(f"✅ Saved {ofn} ({len(ret)} events)")
                    ret = []  # clear buffer
                    chunk_index += 1
    
    if not args.ALLEGRO: 
        ret = {k: awkward.from_iter([r[k] for r in ret]) for k in ret[0].fields}
        for k in ret.keys():
            if len(awkward.flatten(ret[k])) == 0:
                ret[k] = build_dummy_array(len(ret[k]), np.float32)
        ret = awkward.Record(ret)
        awkward.to_parquet(ret, ofn, compression="snappy")
    # np.save('data.npy', dic, allow_pickle=True)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Input file ROOT file", required=True)
    parser.add_argument("--outpath", type=str, default="raw", help="output path")
    parser.add_argument("--dataset", action="store_true", default=False, help="is dataset for eval")
    parser.add_argument("--pandora", action="store_true", default=False,
                        help="include Pandora PFO features in output parquet (sets --dataset implicitly)")
    parser.add_argument("--truth", action="store_true", default=False, help="do tracks come from gen")
    parser.add_argument("--ILD", action="store_true", default=False, help="use ILD data")
    parser.add_argument("--ARC", action="store_true", default=False, help="use ARC data")
    parser.add_argument("--ALLEGRO", action="store_true", default=False, help="use ALLEGRO data")
    parser.add_argument("--chunk_size", type=int, default=100, help="Events per output file")
    args = parser.parse_args()
    if args.pandora:
        args.dataset = True
    return args


def process(args):
    infile = args.input
    outfile = os.path.join(args.outpath, os.path.basename(infile).split(".")[0] + ".parquet")
    tic = time.time()

    process_one_file(infile, outfile, args)
    toc = time.time()
    print("Processing time: ", toc - tic)


if __name__ == "__main__":
    args = parse_args()
    process(args)
