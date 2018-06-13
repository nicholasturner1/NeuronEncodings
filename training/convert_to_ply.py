#!/usr/bin/env python3
__doc__ = """
Read a coordinate vertex h5 file
Normalize the coordinates to the unit sphere
Save the results as a ply file
"""

import os

import plyfile, h5py
import numpy as np
import glob


def main(fnames, n=4000):

    for fname in fnames:
        print("Converting file {}...".format(fname))
        vertex_coords = read_vertices_h5(fname)
        labels = read_vertices_h5(fname, "labels")

        normed_coords = normalize_coords(vertex_coords)
        sampled_coords, labels = sample_coords(normed_coords, labels=labels, n=n)

        output_fname = convert_fname(fname, n=n)
        write_lbld_vertices(sampled_coords, labels, output_fname)


def read_vertices_h5(fname, dset="vertices"):
    """Reading vertices from an hdf5"""
    assert os.path.isfile(fname), "{} isn't a file".format(fname)

    with h5py.File(fname) as f:
        return f[dset].value


def normalize_coords(coords):
    """Normalizing each cell's coordinates to the unit sphere"""
    #Centering
    mean_coord = np.mean(coords,0)
    coords = coords - mean_coord

    #Scaling
    max_norm = np.max(np.sqrt(np.sum(np.square(coords),1)))
    coords /= max_norm

    return coords


def sample_coords(coords, labels=None, n=2048):
    """Samples n rows from coords"""
    if coords.shape[0] < n:
        sample_indices = np.arange(coords.shape[0])
    else:
        sample_indices = np.random.choice(range(coords.shape[0]), n, replace=False)

    if labels is None:
        return coords[sample_indices,:]
    else:
        return coords[sample_indices,:], labels[sample_indices]


def convert_fname(fname, n=None):
    """Replaces extension with .ply"""
    # if n is None:
    root, ext = os.path.splitext(fname)
    return "{}.ply".format(root)
    # else:
        # return os.path.dirname(fname) + "/plys_%d/" % n + os.path.basename(fname)[:-3] + ".ply"


def write_vertices_ply(coords, fname):
    """Writing vertex coordinates as a .ply file using plyfile"""
    print("Writing to file {}...".format(fname))

    tweaked_array = np.array(list(zip(coords[:,0],coords[:,1],coords[:,2])),
                             dtype=[('x','f4'),('y','f4'),('z','f4')])

    vertex_element = plyfile.PlyElement.describe(tweaked_array, "vertex")

    if not os.path.exists(os.path.dirname(fname)):
        os.makedirs(os.path.dirname(fname))

    plyfile.PlyData([vertex_element]).write(fname)


def write_lbld_vertices(coords, labels, fname):
    """Writing vertex coordinates as a .ply file using plyfile"""

    R = ((labels == 0)*255).astype("uint8")
    G = ((labels == 1)*255).astype("uint8")
    B = ((labels == 2)*255).astype("uint8")

    tweaked_array = np.array(list(zip(coords[:,0],coords[:,1],coords[:,2],R,G,B)),
                             dtype=[('x','f4'),('y','f4'),('z','f4'),
                                    ('red','u1'),('green','u1'),('blue','u1')])

    vertex_element = plyfile.PlyElement.describe(tweaked_array, "vertex")

    if not os.path.exists(os.path.dirname(fname)):
        os.makedirs(os.path.dirname(fname))

    plyfile.PlyData([vertex_element]).write(fname)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="quick ply file converter")

    parser.add_argument("input_fname")
    parser.add_argument("-n", type=int, default=2048)

    args = parser.parse_args()

    main([args.input_fname], args.n)
    #paths = glob.glob("/usr/people/svenmd/seungmount/research/svenmd/pointnet_orphan_dendrites_gt_180308/*.h5")[:1000]
    
    #for n in [1000, 5000, 10000, 20000, 30000]:
    #    main(paths, n=n)
    
    #paths = glob.glob("/usr/people/svenmd/seungmount/research/svenmd/pointnet_orphan_axons_gt_180308/*.h5")[:1000]
    
    #for n in [1000, 5000, 10000, 20000, 30000]:
    #    main(paths, n=n)
