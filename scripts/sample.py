__doc__ = """
Pulling some samples from a dataset, and saving them to disk for easy viewing
"""
import os

from neuronencodings import data


def main(dset_descs, num_samples, phase, n_points, local_env,
         rotation, jitter, scaling, movement, chopping,
         random_seed, train_split, val_split, test_split):

    gt_dirs = parse_dset_descriptors(dset_descs)

    dataset = data.CellDataset(gt_dirs=gt_dirs, phase=phase, n_points=n_points,
                               random_seed=random_seed,
                               local_env=local_env, apply_rotation=rotation,
                               apply_jitter=jitter, apply_scaling=scaling,
                               apply_chopping=chopping,
                               apply_movement=movement,
                               train_split=train_split, val_split=val_split,
                               test_split=test_split)

    samples = data.pull_n_samples(dataset, opt.num_samples)
    data.save_samples(samples)


def parse_dset_descriptors(descriptors):
    """
    Determines whether a descriptor is a dataset name (see data.utils) or
    a directory of mesh files, and consolidates all directories from each
    descriptor
    """

    gt_dirs = list()
    for desc in descriptors:
        try:
            dirs = data.fetch_dset_dirs(desc)
            gt_dirs.extend(dirs)

        except AssertionError:
            assert os.path.isdir(desc), f"desc {desc} not a dir or dset name"
            gt_dirs.append(desc)

    return gt_dirs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Saving N samples")

    parser.add_argument('--num_samples', type=int, default=5,
                        help='number of samples to save')
    parser.add_argument('--dset_descs', nargs="+",
                        help="dataset descriptors (dataset names or dirs)",
                        default=data.fetch_dset_dirs("default"))
    parser.add_argument('--phase', type=int, default=4, choices=[1, 2, 3, 4],
                        help="1=train, 2=val, 3=test, 4=full")
    parser.add_argument('--n_points', type=int, default=1000,
                        help='number of points')
    parser.add_argument("--local_env", action="store_true",
                        help='Sample local neighborhoods')
    parser.add_argument('--rotation', action="store_true",
                        help='rotation augmentation')
    parser.add_argument('--jitter', action="store_true",
                        help='jitter augmentation')
    parser.add_argument('--scaling', action="store_true",
                        help='scaling augmentation')
    parser.add_argument('--movement', action="store_true",
                        help='translation augmentation')
    parser.add_argument('--chopping', action="store_true",
                        help='"chopping" augmentation')
    parser.add_argument('--random_seed', type=int, default=2,
                        help="random seed for mesh selection")
    parser.add_argument('--train_split', type=float, default=0.8,
                        help="amount of meshes to use as the training set")
    parser.add_argument('--val_split', type=float, default=0.1,
                        help="amount of meshes to use as the validation set")
    parser.add_argument('--test_split', type=float, default=0.1,
                        help="amount of meshes to use as the test set")

    opt = parser.parse_args()
    print(opt)

    main(**vars(opt))
