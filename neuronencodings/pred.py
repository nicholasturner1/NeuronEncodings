import numpy as np
from torch.autograd import Variable
import torch
import os
from scipy import spatial


import models
from datasets_pychg import CellDataset

from meshparty import mesh_io

home = os.path.expanduser("~")
expt_dir = "%s/seungmount/research/nick_and_sven/models_sven/" % home
meshmeta = mesh_io.MeshMeta()


def load_model(expt_name, model_name, chkpt_num, expt_dir=expt_dir,
               bottle_fs=128, n_points=2500, in_dim=3, bn=True, eval_=True):
    model_class = getattr(models, model_name)
    model = model_class(n_points, pt_dim=in_dim, bn=bn, bottle_fs=bottle_fs)

    if eval_:
        model.cuda().eval()
    else:
        model.cuda()

    model_dir = os.path.join(expt_dir, expt_name, "models")

    model_chkpt = "{dir}/model_{iter}.chkpt".format(dir=model_dir,
                                                    iter=chkpt_num)
    model.load_state_dict(torch.load(model_chkpt))

    return model

def predict_points(model, points):
    points_batch = torch.from_numpy(points)
    points_batch = Variable(points_batch)
    points_batch = points_batch.cuda()
    # points_batch = points_batch.transpose(2, 1)

    points_batch = points_batch.cuda()
    _, fs = model.forward(points_batch)
    fs = fs.cpu().data.numpy()

    return fs


def load_orphans(n_points=2500, batch_size=2, dataset_name="pinky"):
    if dataset_name == "pinky":
        dataset_paths = [home + "/seungmount/research/svenmd/pointnet_orphan_axons_gt_180308_refined/"]
    elif dataset_name == "full_cells_pinky":
        dataset_paths = [home + "/seungmount/research/svenmd/pointnet_full_semantic_labels_masked_180401_refined/"]
    elif dataset_name == "fish":
        dataset_paths = [home + "/seungmount/research/svenmd/180831_meshes_ashwin_refined/"]
    else:
        raise()

    dataset = CellDataset(gt_dirs=dataset_paths,
                          phase=3,
                          n_points=n_points,
                          random_seed=0,
                          batch_size=batch_size,
                          apply_rotation=False,
                          apply_jitter=False,
                          apply_scaling=False,
                          apply_chopping=False,
                          apply_movement=False,
                          train_test_split_ratio=.666)

    return dataset


def load_orphan_vertex_block(dataset, fnames, n_points=2500):
    vertices_list = []
    vertex_ids_list = []
    for fname in fnames:
        vertices = dataset.read_vertices(fname).astype(np.float32)

        kdtree = spatial.cKDTree(vertices)
        center_vertex_id = np.random.randint(0, len(vertices))
        _, valid_vertex_ids = kdtree.query(vertices[center_vertex_id],
                                           k=n_points, n_jobs=-1)

        if len(valid_vertex_ids) < n_points:
            valid_vertex_ids = np.arange(len(vertices), dtype=np.int)

        if len(valid_vertex_ids) < n_points:
            vertex_ids = np.random.choice(valid_vertex_ids, n_points,
                                          replace=True)
        elif len(valid_vertex_ids) == n_points:
            vertex_ids = valid_vertex_ids
        else:
            vertex_ids = np.random.choice(valid_vertex_ids, n_points,
                                          replace=False)

        vertices = vertices[vertex_ids]

        # Normalize to unit sphere and mean zero
        vertices -= np.min(vertices, axis=0)[None]
        vertices /= np.max(np.linalg.norm(vertices, axis=1))
        vertices_list.append(vertices)
        vertex_ids_list.append(center_vertex_id)

    return np.array(vertices_list, dtype=np.float32),\
           np.array(vertex_ids_list, dtype=np.int)

def load_vertex_block_pychg(fnames, center_coords=None, n_points=500,
                            local_env=True):
    vertices_list = []
    vertex_ids_list = []

    if center_coords is None:
        center_coords = [None] * len(fnames)

    for i_fname, fname in enumerate(fnames):

        mesh = meshmeta.mesh(fname)

        if local_env:
            vertices, center_vertex_id = mesh.get_local_view(n_points,
                                                             center_coord=center_coords[i_fname],
                                                             pc_align=True,
                                                             method="kdtree")
        else:
            vertices = mesh.vertices[np.random.choice(
                np.arange(len(mesh.vertices), dtype=np.int), n_points,
                replace=False)]

        if len(vertices) < n_points:
            vertices = vertices[np.random.choice(len(vertices), n_points, replace=True)]

        # Normalize to unit sphere and mean zero
        vertices -= np.min(vertices, axis=0)[None]
        vertices /= np.max(np.linalg.norm(vertices, axis=1))

        vertices_list.append(vertices)
        vertex_ids_list.append(center_vertex_id)

    return np.array(vertices_list, dtype=np.float32),\
           np.array(vertex_ids_list)

