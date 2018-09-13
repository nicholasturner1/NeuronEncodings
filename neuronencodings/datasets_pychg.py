from __future__ import print_function
import torch.utils.data as data

import numpy as np
import random
import glob
import h5py
import os
import collections
import torch
import transforms3d
import time
import enum
import networkx as nx
from scipy import spatial

import convert_to_ply as ply

from meshparty import mesh_io


class Phase(enum.Enum):
    TRAIN = 1   # Training
    VAL = 2     # Validation / Test
    FULL = 3    # Full dataset


class CellDataset(data.Dataset):
    def __init__(self, gt_dirs, n_points=2500, phase=1, random_seed=0,
                 train_test_split_ratio=.8, batch_size=1, local_env=True,
                 apply_jitter=False,
                 apply_rotation=False, apply_scaling=False,
                 apply_movement=False, apply_chopping=False,
                 n_max_meshes_per_dataset=None):
        self._gt_dirs = gt_dirs
        self._gt_files = None
        self._n_points = n_points
        self._phase = Phase(phase)
        self._local_env = local_env
        self._apply_jitter = apply_jitter
        self._apply_rotation = apply_rotation
        self._apply_scaling = apply_scaling
        self._apply_movement = apply_movement
        self._apply_chopping = apply_chopping
        self._random_seed = random_seed
        self._train_test_split_ratio = train_test_split_ratio
        self._batch_size = batch_size
        self._n_max_meshes_per_dataset = n_max_meshes_per_dataset
        self.meshmeta = mesh_io.MeshMeta()

        self._valid_files = []
        self._train_files = []
        self._test_files = []
        self._val_files = []

        self._num_classes = 0
        self._n_instances_per_class = collections.Counter()

        np.random.seed(random_seed)
        self.pick_files()

    @property
    def gt_dirs(self):
        return self._gt_dirs

    @property
    def n_points(self):
        return self._n_points

    @property
    def is_train(self):
        return self._is_train

    @property
    def local_env(self):
        return self._local_env

    @property
    def apply_jitter(self):
        return self._apply_jitter

    @property
    def apply_rotation(self):
        return self._apply_rotation

    @property
    def apply_scaling(self):
        return self._apply_scaling

    @property
    def apply_movement(self):
        return self._apply_movement

    @property
    def apply_chopping(self):
        return self._apply_chopping

    @property
    def random_seed(self):
        return self._random_seed

    @property
    def train_test_split_ratio(self):
        return self._train_test_split_ratio

    @property
    def train_ids(self):
        return list(map(self.read_soma_id, self._train_files))

    @property
    def test_ids(self):
        return list(map(self.read_soma_id, self._val_files))

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def n_max_meshes_per_dataset(self):
        return self._n_max_meshes_per_dataset

    @property
    def cache_data(self):
        return self._cache_data

    @property
    def gt_files(self):
        if self._gt_files is not None:
            return self._gt_files

        self._gt_files = []
        for gt_dir in self.gt_dirs:
            dir_paths = sorted(glob.glob(gt_dir + "/*.*"),
                               key=os.path.basename)

            if self.n_max_meshes_per_dataset is not None:
                dir_paths = dir_paths[: self.n_max_meshes_per_dataset]

            self._gt_files += dir_paths

        return self._gt_files

    @property
    def gt_labels_key(self):
        return "labels"

    @property
    def gt_vertices_key(self):
        return "vertices"

    @property
    def gt_faces_key(self):
        return "faces"

    @property
    def gt_normals_key(self):
        return "normals"

    @property
    def gt_skel_nodes_key(self):
        return "skeleton_nodes"

    @property
    def gt_skel_edges_key(self):
        return "skeleton_edges"

    def __len__(self):
        if self._phase == Phase.TRAIN:
            return len(self._train_files)
        elif self._phase == Phase.VAL:
            return len(self._val_files)
        else:   # default phase == Phase.FULL
            return len(self._valid_files)

    def __getitem__(self, index):
        if self._phase == Phase.TRAIN:
            mesh_fname = self._train_files[index]
        elif self._phase == Phase.VAL:
            mesh_fname = self._val_files[index]
        else:   # default phase == Phase.FULL
            mesh_fname = self._valid_files[index]

        mesh = self.meshmeta.mesh(mesh_fname)

        if self.local_env:
            vertices, _ = mesh.get_local_view(self.n_points, pc_align=True,
                                              method="kdtree")
        else:
            vertices = mesh.vertices[np.random.choice(
                np.arange(len(mesh.vertices), dtype=np.int), self.n_points,
                replace=True)]

        if len(vertices) < self.n_points:
            vertices = vertices[np.random.choice(np.arange(len(vertices), dtype=np.int), self.n_points, replace=True)]

        # Normalize to unit sphere and mean zero
        vertices -= np.min(vertices, axis=0)[None]
        vertices /= np.max(np.linalg.norm(vertices, axis=1))
        
        # Apply rotation and jitter
        if self.apply_rotation:
            vertices = rotate_point_cloud(vertices)
        if self.apply_jitter:
            vertices = jitter_point_cloud(vertices)
        if self.apply_scaling:
            vertices = scale_point_cloud(vertices)
        if self.apply_movement:
            vertices = move_point_cloud(vertices)

        vertices = torch.from_numpy(vertices.astype(np.float32))

        return vertices


    def pick_files(self):
        """
        Picks which files can be used for neuronencodings, and splits them
        into neuronencodings & test
        """

        self._valid_files = self.gt_files
        self._train_files = self.gt_files
        self._val_files = self.gt_files

def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction

        initial code : https://github.com/charlesq34/pointnet

        Input:
          Nx3 array, original batch of point clouds
        Return:
          Nx3 array, rotated batch of point clouds
    """
    euler_angles = np.random.rand(3) * 2 * np.pi
    R = transforms3d.euler.euler2mat(euler_angles[0],
                                     euler_angles[1],
                                     euler_angles[2],
                                     "sxyz").astype(np.float32)
    rotated_data = np.dot(batch_data, R.T)

    return rotated_data


def jitter_point_cloud(batch_data, sigma=0.0001, clip=0.01):
    """ Randomly jitter points. jittering is per point.

        initial code : https://github.com/charlesq34/pointnet

        Input:
          Nx3 array, original batch of point clouds
        Return:
          Nx3 array, jittered batch of point clouds
    """
    N, C = batch_data.shape

    assert(clip > 0)

    jittered_data = np.clip(sigma * np.random.randn(N, C).astype(np.float32),
                            -1*clip, clip)
    jittered_data += batch_data

    return jittered_data


def scale_point_cloud(batch_data, sigma=0.25, clip=0.5):
    """ Randomly jitter points. jittering is per point.

        initial code : https://github.com/charlesq34/pointnet

        Input:
          Nx3 array, original batch of point clouds
        Return:
          Nx3 array, jittered batch of point clouds
    """
    assert(clip > 0)

    scaling = np.clip(sigma * np.random.randn(1), -1*clip, clip) + 1
    scale_data = batch_data * scaling
    return scale_data


def move_point_cloud(batch_data, max_dist=.3):
    """ Randomly move point clouds.

        Input:
          Nx3 array, original batch of point clouds
        Return:
          Nx3 array, jittered batch of point clouds
    """

    offset = np.random.rand(3) * 2 * max_dist - max_dist
    batch_data += offset[None]

    return batch_data


def chop_point_cloud(vertices, skel_graph, skel_nodes, radius_nm_min=25000,
                     radius_nm_max=75000):
    radius_nm = np.random.randint(radius_nm_min, radius_nm_max)

    skel_kd_tree = spatial.cKDTree(skel_nodes)
    _, mesh_to_skel = skel_kd_tree.query(vertices)

    _, paths = nx.single_source_dijkstra(skel_graph,
                                         np.random.randint(len(skel_nodes)),
                                         cutoff=radius_nm, weight='weight')
    close_nodes = np.array(list(paths.keys()), dtype=np.int)

    vertex_ids = np.where(np.in1d(mesh_to_skel, close_nodes))[0]

    return vertex_ids


def pull_n_samples(dataset, n):
    """Pulls n random samples from a dataset object"""
    return [dataset[i] for i in np.random.choice(range(len(dataset)),n,replace=False)]
    #
    # samples = []
    # for i_sample in range(len(dataset)):
    #     print(i_sample)
    #     samples.append(dataset[i_sample])
    # return samples

def save_samples(samples, output_prefix="sample"):
    """Saves a list of samples to ply files (with h5 labels)"""

    for (i,vertices) in enumerate(samples):
        #vertices
        vertex_fname = "{pref}{i}_vertices.ply".format(pref=output_prefix, i=i)
        if os.path.dirname(vertex_fname) == "":
            vertex_fname = "./" + vertex_fname
        ply.write_vertices_ply(vertices, vertex_fname)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Saving N samples")

    parser.add_argument('--num_samples', type=int, default=5,
                        help='# samples to save')
    parser.add_argument('--dataset_dirs', nargs="+", help="Location of mesh files",
                        default=[os.path.expanduser("~") + "/seungmount/research/svenmd/pointnet_full_semantic_labels_masked_180401_refined/"])
    parser.add_argument('--phase', type=int, default=3, choices=[1,2,3],
                         help="1=train,2=val,3=full")
    parser.add_argument('--n_points', type=int, default=1000,
                        help='number of points')
    parser.add_argument('--rotation', action="store_true", help='augment with rotation')
    parser.add_argument('--jitter', action="store_true", help='augment with jitter')
    parser.add_argument('--scaling', action="store_true", help='augment with scaling')
    parser.add_argument('--movement', action="store_true", help='augment with movement')
    parser.add_argument('--chopping', action="store_true", help='augment with chopping')
    parser.add_argument('--random_seed', type=int, default=2, help="random seed for mesh selection")

    opt = parser.parse_args()
    print(opt)

    dataset = CellDataset(gt_dirs=opt.dataset_dirs,
                          phase=opt.phase,
                          n_points=opt.n_points,
                          random_seed=opt.random_seed,
                          apply_rotation=opt.rotation,
                          apply_jitter=opt.jitter,
                          apply_scaling=opt.scaling,
                          apply_chopping=opt.chopping,
                          apply_movement=opt.movement,
                          train_test_split_ratio=.8)

    samples = pull_n_samples(dataset, opt.num_samples)
    save_samples(samples)
