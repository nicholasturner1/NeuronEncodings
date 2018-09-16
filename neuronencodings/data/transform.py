__doc__ = """
Mesh Transformations and Augmentations
"""
import numpy as np
import transforms3d
import networkx as nx
from scipy import spatial


def random_sample(vertices, n_points, seed=None):
    """
    Samples the rows of vertices with replacement to keep samples consistent
    """
    if seed is not None:
        np.random.seed(seed)

    inds = np.arange(len(vertices), dtype=np.int)
    return vertices[np.random.choice(inds, n_points, replace=True)]


def norm_to_unit_sphere(vertices):
    """ Normalizes a point cloud to the unit sphere with 0 mean in-place """
    vertices -= np.min(vertices, axis=0)[None]
    vertices /= np.max(np.linalg.norm(vertices, axis=1))

    return vertices


def rotate(vertices, seed=None):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction

        initial code : https://github.com/charlesq34/pointnet

        Input:
          Nx3 array, original batch of point clouds
        Return:
          Nx3 array, rotated batch of point clouds
    """
    if seed is not None:
        np.random.seed(seed)

    euler_angles = np.random.rand(3) * 2 * np.pi
    R = transforms3d.euler.euler2mat(euler_angles[0],
                                     euler_angles[1],
                                     euler_angles[2],
                                     "sxyz").astype(np.float32)
    rotated_data = np.dot(vertices, R.T)

    return rotated_data


def jitter(vertices, sigma=0.0001, clip=0.01, seed=None):
    """ Randomly jitter points. jittering is per point.

        initial code : https://github.com/charlesq34/pointnet

        Input:
          Nx3 array, original batch of point clouds
        Return:
          Nx3 array, jittered batch of point clouds
    """
    if seed is not None:
        np.random.seed(seed)

    assert(clip > 0)

    N, C = vertices.shape
    jittered_data = np.clip(sigma * np.random.randn(N, C).astype(np.float32),
                            -1*clip, clip)
    jittered_data += vertices

    return jittered_data


def scale(vertices, sigma=0.25, clip=0.5, seed=None):
    """ Randomly jitter points. jittering is per point.

        initial code : https://github.com/charlesq34/pointnet

        Input:
          Nx3 array, original batch of point clouds
        Return:
          Nx3 array, jittered batch of point clouds
    """
    if seed is not None:
        np.random.seed(seed)

    assert(clip > 0)

    scaling = np.clip(sigma * np.random.randn(1), -1*clip, clip) + 1
    scale_data = vertices * scaling
    return scale_data


def translate(vertices, max_dist=.3, seed=None):
    """ Randomly move point clouds.

        Input:
          Nx3 array, original batch of point clouds
        Return:
          Nx3 array, jittered batch of point clouds
    """
    if seed is not None:
        np.random.seed(seed)

    offset = np.random.rand(3) * 2 * max_dist - max_dist
    vertices += offset[None]

    return vertices


def chop_point_cloud(vertices, skel_graph, skel_nodes, radius_nm_min=25000,
                     radius_nm_max=75000, seed=None):
    if seed is not None:
        np.random.seed(seed)

    radius_nm = np.random.randint(radius_nm_min, radius_nm_max)

    skel_kd_tree = spatial.cKDTree(skel_nodes)
    _, mesh_to_skel = skel_kd_tree.query(vertices)

    _, paths = nx.single_source_dijkstra(skel_graph,
                                         np.random.randint(len(skel_nodes)),
                                         cutoff=radius_nm, weight='weight')
    close_nodes = np.array(list(paths.keys()), dtype=np.int)

    vertex_ids = np.where(np.in1d(mesh_to_skel, close_nodes))[0]

    return vertex_ids
