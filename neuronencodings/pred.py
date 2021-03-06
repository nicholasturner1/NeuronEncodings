import collections

import numpy as np

import torch

from meshparty import iterator

from .data import transform


def encode_mesh_by_views(model, mesh, n_points, batch_size=10,
                         order="random", pc_align=False, method="kdtree",
                         verbose=False, pt_dim=3, pc_norm=True):
    """
    Runs inference over the local views of a single mesh. Assumes that you have
    at least one full batch worth of local views (otherwise the inference will output
    strange values
    """

    # buffer to hold a batch of local views/centers
    view_batch = np.empty((batch_size, n_points, pt_dim), dtype=np.float32)
    center_batch = np.empty((batch_size, ), dtype=np.uint32)

    # record of all inferred vectors and centers so far
    vectors, centers = list(), list()

    it = iterator.LocalViewIterator(mesh, n_points, order=order,
                                    pc_align=pc_align, method=method,
                                    verbose=False, pc_norm=pc_norm)

    batch_i = 0
    while True:
        view_batch, center_batch, new_sz = fill_batch(view_batch, center_batch,
                                                      it, batch_size)

        if new_sz == 0:
            break

        if verbose:
            print(f"batch {batch_i}")

        new_vectors = unpack_batch(predict_batch(model, view_batch))
        new_centers = unpack_batch(center_batch.copy())

        vectors.extend(new_vectors)
        centers.extend(new_centers)
        batch_i += 1

    return vectors, centers


def unpack_batch(batch):
    """ Splits the contents of a batch into a list """
    return list(batch[i,...] for i in range(batch.shape[0]))


def predict_batch(model, points):
    """ Runs inference on a batch of points """

    points_tensor = torch.from_numpy(points).cuda()

    with torch.no_grad():
        _, fs = model.forward(points_tensor)
    return fs.data.cpu().numpy()


def fill_batch(view_batch, center_batch, it, n):
    """
    Fills as many items in a batch as possible with new views. Leaves
    the rest as-is
    """

    num_added = 0
    for i in range(n):
        try:
            view, center = next(it)
            view = transform.norm_to_unit_sphere(view)
            view_batch[i,...] = view
            center_batch[i] = center
            num_added += 1
        except StopIteration:
            break

    return view_batch, center_batch, num_added


def encode_meshs_by_views(model, meshes, n_points, batch_size=10,
                          order="random", pc_align=False, method="kdtree",
                          verbose=False, pt_dim=3, pc_norm=True, unit_norm=True):
    """
    Runs inference over the local views of multiple meshes. Assumes that you have
    at least one full batch worth of local views (otherwise the inference will output
    strange values
    """

    # buffers to hold a batch of local views, centers, and mesh indices
    view_batch = np.empty((batch_size, n_points, pt_dim), dtype=np.float32)
    center_batch = np.empty((batch_size,), dtype=np.uint32)
    ind_batch = np.empty((batch_size,), dtype=np.uint32)

    its = [iterator.LocalViewIterator(mesh, n_points, order=order,
                                      pc_align=pc_align, method=method,
                                      verbose=False, pc_norm=pc_norm) 
           for mesh in meshes]
    multi_it = collections.deque(list(enumerate(its)))

    # record of the inferred vectors and centers
    vectors = list(list() for mesh in meshes)
    centers = list(list() for mesh in meshes)

    batch_i = 0
    while True:
        (view_batch, center_batch, ind_batch, new_sz) = \
            fill_multi_it_batch(view_batch, center_batch, ind_batch,
                                multi_it, batch_size, unit_norm)

        if new_sz == 0:
            break

        if verbose:
            print(f"batch {batch_i}")

        new_vectors = unpack_batch(predict_batch(model, view_batch))
        new_centers = unpack_batch(center_batch.copy())

        for (i,v) in enumerate(ind_batch[:new_sz]):
            vectors[v].append(new_vectors[i])
            centers[v].append(new_centers[i])
        batch_i += 1

    return vectors, centers


def fill_multi_it_batch(view_batch, center_batch, ind_batch,
                        multi_it, batch_size, unit_norm=True):
    """
    Same as fill_batch above, except uses a deque of its instead to represent
    multiple cells
    """

    num_added = 0
    for i in range(batch_size):
        j, view, center = next_sample(multi_it, unit_norm)
        if j == -1:
            break

        #if there are fewer points than needed for a batch
        if view.shape[0] < view_batch.shape[1]:
            view = transform.random_sample(view, view_batch.shape[1])

        ind_batch[i] = j
        view_batch[i,...] = view
        center_batch[i] = center
        num_added += 1

    return view_batch, center_batch, ind_batch, num_added


def next_sample(multi_it, unit_norm=True):
    """
    Pulls the next sample from a deque of LocalViewIterators
    Returns (-1,-1,-1) if no samples are left
    """

    while len(multi_it) != 0:
        try:
            i, it = multi_it[0]
            view, center = next(it)
            if unit_norm:
                view = transform.norm_to_unit_sphere(view)
            multi_it.rotate(-1)
            return i, view, center

        except StopIteration:
            multi_it.popleft()

    return -1, -1, -1
