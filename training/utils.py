#!/usr/bin/env python
__doc__ = """
Miscellaneous Utils
"""

import os, shutil
import datetime
import h5py
import importlib
import types

import torch


def make_required_dirs(expt_dir, expt_name):

    dirs = get_required_dirs(expt_dir, expt_name)

    for d in dirs:
        if not os.path.isdir(d):
            os.makedirs(d)

    return dirs


def get_required_dirs(expt_dir, expt_name):

    model_dir = os.path.join(expt_dir, expt_name, "models")
    log_dir = os.path.join(expt_dir, expt_name, "logs")
    fwd_dir = os.path.join(expt_dir, expt_name, "inference")
    tb_dir = os.path.join(expt_dir, expt_name, "tb_logs")

    tb_train = os.path.join(tb_dir,"train")
    tb_val = os.path.join(tb_dir,"val")

    return model_dir, log_dir, fwd_dir, tb_train, tb_val


def timestamp():
    return datetime.datetime.now().strftime("%d%m%y_%H%M%S")


def log_tagged_modules(module_fnames, log_dir, phase, tstamp=None):

    if tstamp is None:
        tstamp = timestamp()

    for fname in module_fnames:
        basename = os.path.basename(fname)
        output_basename = "{}_{}_{}".format(tstamp, phase, basename)

        shutil.copyfile(fname, os.path.join(log_dir, output_basename))


def save_args(args_obj, log_dir, tstamp=None):
    """Saves the args within a parse_args objs as a csv"""

    if tstamp is None:
        tstamp = timestamp()

    output_fname = os.path.join(log_dir, "{}_params.csv".format(tstamp))

    with open(output_fname, "w+") as f:
        for (k,v) in vars(args_obj).items():
            f.write("{k}:{v}\n".format(k=k,v=v))


def set_gpus(gpu_list):
    """ Sets the gpus visible to this process """
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_list)


def read_h5(fname):

    with h5py.File(fname) as f:
        d = f["/main"].value

    return d


def write_h5(data, fname):

    if os.path.exists(fname):
      os.remove(fname)

    with h5py.File(fname) as f:
        f.create_dataset("/main",data=data)


def load_source(fname, module_name="something"):
    """ Imports a module from source """
    loader = importlib.machinery.SourceFileLoader(module_name,fname)
    mod = types.ModuleType(loader.name)
    loader.exec_module(mod)
    return mod
