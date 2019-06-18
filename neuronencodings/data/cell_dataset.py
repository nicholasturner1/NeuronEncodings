__doc__ = """
Cell Dataset Class - Primary interface to mesh data
"""
import os
import random
import enum
import time

import numpy as np

import torch
import torch.utils.data as data

from meshparty import trimesh_io


from . import transform
from . import utils


class Phase(enum.Enum):
    """ Short descriptor of training phase """
    TRAIN = 1   # Training set
    VAL = 2     # Validation set
    TEST = 3    # Test set
    FULL = 4    # Full dataset


class CellDataset(data.Dataset):
    def __init__(self, gt_dirs, n_points=2500, sample_n_points=2500,
                 phase=Phase.TRAIN, random_seed=0, train_split=0.8,
                 val_split=0.1, test_split=0.1, n_views_per_batch=5,
                 n_max_meshes_per_dataset=None):
        self._gt_dirs = gt_dirs
        self._n_max_meshes_per_dataset = n_max_meshes_per_dataset
        self._gt_files = None  # lazily populated

        self._n_points = n_points
        self._sample_n_points = sample_n_points

        self._n_views_per_batch = n_views_per_batch

        self._phase = Phase(phase)
        self._train_split = train_split
        self._val_split = val_split
        self._test_split = test_split

        self._valid_files = []
        self._train_files = []
        self._val_files = []
        self._test_files = []

        self.meshmeta = trimesh_io.MeshMeta()

        self._random_seed = random_seed
        np.random.seed(random_seed)
        self._pick_files()

    @property
    def gt_dirs(self):
        return self._gt_dirs

    @property
    def n_max_meshes_per_dataset(self):
        return self._n_max_meshes_per_dataset

    @property
    def n_points(self):
        return self._n_points

    @property
    def sample_n_points(self):
        return self._sample_n_points

    @property
    def n_views_per_batch(self):
        return self._n_views_per_batch

    @property
    def random_seed(self):
        return self._random_seed

    @property
    def train_split(self):
        return self._train_split

    @property
    def val_split(self):
        return self._val_split

    @property
    def test_split(self):
        return self._test_split

    @property
    def train_files(self):
        return self._train_files

    @property
    def val_files(self):
        return self._val_files

    @property
    def test_files(self):
        return self._test_files

    @property
    def gt_files(self):
        if self._gt_files is None:
            self._gt_files = []
            for gt_dir in self.gt_dirs:
                dir_paths = utils.files_from_dir(gt_dir)

                if self.n_max_meshes_per_dataset is not None:
                    dir_paths = dir_paths[: self.n_max_meshes_per_dataset]

                self._gt_files += dir_paths

        return self._gt_files

    def __len__(self):
        if self._phase == Phase.TRAIN:
            return len(self._train_files)
        elif self._phase == Phase.VAL:
            return len(self._val_files)
        elif self._phase == Phase.TEST:
            return len(self._test_files)
        else:   # default phase == Phase.FULL
            return len(self._valid_files)

    def __getitem__(self, index):
        if self._phase == Phase.TRAIN:
            mesh_fname = self._train_files[index]
        elif self._phase == Phase.VAL:
            mesh_fname = self._val_files[index]
        elif self._phase == Phase.TEST:
            mesh_fname = self._test_files[index]
        else:   # default phase == Phase.FULL
            mesh_fname = self._valid_files[index]

        mesh = self.meshmeta.mesh(mesh_fname, merge_large_components=False)

        if mesh is None:
            print(f"WARNING: {mesh_fname} has no vertices")

        vertex_list = []
        for i_view in range(self.n_views_per_batch):
            vertices, v_id = mesh.get_local_view(self.n_points, pc_align=True,
                                                 pc_norm=False,
                                                 sample_n_points=self.sample_n_points)
            vertices = vertices.squeeze()

            if len(vertices) < self.n_points:
                # can get here if the whole mesh is too small
                vertices = transform.random_sample(vertices, self.n_points)
            vertex_list.append(vertices)

        vertex_list = np.array(vertex_list)

        # Normalize to unit sphere and mean zero
        vertex_list = transform.norm_to_unit_sphere_many(vertex_list)

        return vertex_list.astype(np.float32)

    def _pick_files(self):
        """
        Picks which files can be used for neuronencodings, and splits them
        into neuronencodings & test
        """
        self._valid_files = self.gt_files

        if self._phase == Phase.FULL:
            self._train_files = self.gt_files
            self._val_files = self.gt_files
            self._test_files = self.gt_files

        else:
            train, val, test = utils.split_files(
                             self.gt_files, self.train_split, self.val_split,
                             self.test_split, self.random_seed)

            self._train_files = train
            self._val_files = val
            self._test_files = test