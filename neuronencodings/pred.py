import numpy as np
from torch.autograd import Variable
import torch
import os
from scipy import spatial

home = os.path.expanduser("~")
expt_dir = "%s/seungmount/research/nick_and_sven/models_sven/" % home

import models
from datasets import CellDataset


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


def load_orphans(n_points=2500, batch_size=2):
    dataset_paths = [home + "/seungmount/research/svenmd/pointnet_orphan_axons_gt_180308/",
                     home + "/seungmount/research/svenmd/pointnet_orphan_dendrites_gt_180308/"]

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

