__doc__ = """
Dataset Module Utilities - mostly for handling files and datasets
"""
import glob
import os
import random

from meshparty import trimesh_io


# Datasets -----------------------
SVEN_BASE = "seungmount/research/svenmd"
NICK_BASE = "seungmount/research/Nick/"
BOTH_BASE = "seungmount/research/nick_and_sven"
DATASET_DIRS = {
    "orig_full_cells": [f"{SVEN_BASE}/pointnet_axoness_gt_180223/"],

    "soma_vs_rest": [f"{SVEN_BASE}/pointnet_soma_masked_180401"],

    "orphans": [f"{SVEN_BASE}/pointnet_orphan_axons_gt_180308/",
                f"{SVEN_BASE}/pointnet_orphan_dendrites_gt_180308/"],

    "orphans2": [f"{NICK_BASE}/pointnet/orphan_dataset/train_val_axons",
                 f"{NICK_BASE}/pointnet/orphan_dataset/train_val_dends/"],

    "orphan_axons": [f"{SVEN_BASE}/pointnet_orphan_axons_gt_180308/"],

    "chandelier_axons": [f"{SVEN_BASE}/InhAnalysis/meshes_Chandelier/"],

    "orphan_axons_refined": [(f"{SVEN_BASE}"
                              "/pointnet_orphan_axons_gt_180308_refined/")],

    "pinky100_orphan_dends": [(f"{BOTH_BASE}/data/180920_orphan_dends/")],

    "pinky100_v50": [(f"{SVEN_BASE}/MeshData/pinky100_meshes/")],

    "fish_axons": [(f"{SVEN_BASE}/1902_axon_mesh_gt_fish/")],
    "pinky100_orphan_axons": [(f"{SVEN_BASE}/1902_axon_mesh_gt/")],
    "pinky100_baskets": [(f"{SVEN_BASE}/190426_basket_axons/")],
    "pinky100_baskets_small": [(f"{SVEN_BASE}/1902_axon_mesh_gt_baskets_small/")],
    "pinky100_pycs": [(f"{SVEN_BASE}/1902_mesh_gt_pycs/")],
    "pinky100_pycs_small": [(f"{SVEN_BASE}/1902_mesh_gt_pycs_small/")],
    "pinky40_orphan_axons": [(f"{SVEN_BASE}/InhAnalysis/meshes/pinky40/put_axon/")],
    # "pinky100_orphan_axons": [(f"meshes_put_axon/")],

    "fish_refined": [f"{SVEN_BASE}/180831_meshes_ashwin_refined/"],

    "full_cells_unrefined": [(f"{SVEN_BASE}"
                              "/pointnet_full_semantic_labels"
                              "_masked_180401")],

    "full_cells_refined": [(f"{SVEN_BASE}"
                            "/pointnet_full_semantic_labels"
                            "_masked_180401_refined/")],

    "full_cells_pinky40_clean": [(f"{SVEN_BASE}/InhAnalysis/meshes/pinky40/full_cells/")],

    "pinky100_orphan_dend_features": [(f"{BOTH_BASE}"
                                       "/nick_archive/p100_dend_outer"
                                       "/inference/proj32/")],

    "pinky100_orphan_dend_features_32": [(f"{BOTH_BASE}"
                                         "/nick_archive/p100_dend_outer_32"
                                         "/inference/")],

    "default": [f"{SVEN_BASE}/pointnet_axoness_gt_rfc_based_masked_180322/",
                f"{SVEN_BASE}/pointnet_orphan_axons_gt_180308/",
                f"{SVEN_BASE}/pointnet_orphan_dendrites_gt_180308/"]
}
# --------------------------------


def fetch_dset_dirs(dset_name=None):
    """
    Finds the global pathname to a list of directories which represent a
    dataset by name.
    """
    assert (dset_name is None) or (dset_name in DATASET_DIRS), "invalid name"

    dset_name = "default" if dset_name is None else dset_name

    home = os.path.expanduser("~")

    return list(os.path.join(home, d) for d in DATASET_DIRS[dset_name])


def files_from_dir(dirname, exts=["obj", "h5"]):
    """
    Searches a directory for a set of extensions and returns the files
    matching those extensions, sorted by basename
    """
    filenames = list()
    for ext in exts:
        ext_expr = os.path.join(dirname, f"*.{ext}")
        filenames.extend(glob.glob(ext_expr))

    return sorted(filenames, key=os.path.basename)


def split_files(filenames, train_split=0.8,
                val_split=0.1, test_split=0.1, seed=None):

    if seed is not None:
        random.seed(seed)

    # Normalizing splits for arbitrary values
    total = train_split + val_split + test_split

    train_split = train_split / total
    val_split = val_split / total
    test_split = test_split / total

    n_train = round(train_split * len(filenames))
    n_val = round(val_split * len(filenames))

    permutation = random.sample(filenames, len(filenames))

    train_files = permutation[:n_train]
    val_files = permutation[n_train:(n_train+n_val)]
    test_files = permutation[(n_train+n_val):]

    return train_files, val_files, test_files


# Helper functions for testing (e.g. sample.py)
def pull_n_samples(dset, n):
    """Pulls n random samples from a dataset object"""
    return list(dset[i] for i in random.sample(range(len(dset)), n))


def save_samples(samples, output_prefix="sample"):
    """Saves a list of samples to ply files (with h5 labels)"""

    for (i, vertices) in enumerate(samples):
        vertex_fname = "{pref}{i}_vertices.ply".format(pref=output_prefix, i=i)
        if os.path.dirname(vertex_fname) == "":
            vertex_fname = "./" + vertex_fname
        trimesh_io.Mesh.write_vertices_ply(None, vertex_fname, coords=vertices)
