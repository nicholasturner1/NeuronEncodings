#!/usr/bin/env python
__doc__ = """
Miscellaneous Utils
"""

import os
import shutil
import datetime
import importlib
import types

import h5py

import torch

from . import models


def load_model(model_name, model_args, model_kwargs, chkpt_fname=None,
               model_dir=None, chkpt_num=None, eval_=False, state_dict=None):
    """ Generalized model loading function """

    model_class = getattr(models, model_name)
    model = model_class(*model_args, **model_kwargs)

    if eval_:
        model.cuda().eval()
    else:
        model.cuda()

    if (model_dir is not None) and (chkpt_num is not None):
        chkpt_fname = f"{model_dir}/model_{iter}.chkpt"

    if chkpt_fname is not None:
        model.load_state_dict(torch.load(chkpt_fname))
    elif state_dict is not None:
        model.load_state_dict(state_dict)
    return model


def load_autoencoder(model_name, n_pts=2500, pt_dim=3, bottle_fs=128,
                     bn=True, eval_=False, chkpt_fname=None, model_dir=None,
                     chkpt_num=None, state_dict=None):
    """
    A specific loading function for autoencoders with a standard set
    of arguments (the usual case)
    """

    model_kwargs = dict(n_pts=n_pts, pt_dim=pt_dim, bottle_fs=bottle_fs, bn=bn)

    return load_model(model_name, list(), model_kwargs,
                      model_dir=model_dir, chkpt_num=chkpt_num,
                      chkpt_fname=chkpt_fname, eval_=eval_,
                      state_dict=state_dict)


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

    tb_train = os.path.join(tb_dir, "train")
    tb_val = os.path.join(tb_dir, "val")

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
        for (k, v) in vars(args_obj).items():
            f.write("{k}:{v}\n".format(k=k, v=v))


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
        f.create_dataset("/main", data=data)


def load_source(fname, module_name="something"):
    """ Imports a module from source """
    loader = importlib.machinery.SourceFileLoader(module_name, fname)
    mod = types.ModuleType(loader.name)
    loader.exec_module(mod)
    return mod


def blue(text):
    """ Makes text blue when printed """
    return f"\033[94m{text}\033[0m"


def print_loss(tag, loss_val, i_epoch, i_batch, num_batch):
    print(f"[{i_epoch}: {i_batch}/{num_batch}] {tag} loss: {loss_val}")
