#!/usr/bin/env python
__doc__ = """
Miscellaneous Utils
"""

import os
import shutil
import datetime
import importlib
import types
import json
import glob
import re
import numpy as np

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
        chkpt_fname = f"{model_dir}/model_{chkpt_num}.chkpt"
        model.load_state_dict(torch.load(chkpt_fname))
    elif state_dict is not None:
        model.load_state_dict(state_dict)
    elif chkpt_fname is not None:
        model.load_state_dict(torch.load(chkpt_fname))

    return model


def load_autoencoder(model_name, n_points=2500, pt_dim=3, bottle_fs=128,
                     conv_layers=[128, 64], mlp_layers=[24, 48, 96, 192, 384],
                     act="relu", bn=True, eval_=False, chkpt_fname=None,
                     model_dir=None, chkpt_num=None, state_dict=None, gpu=0,
                     **kwargs):
    """
    A specific loading function for autoencoders with a standard set
    of arguments (the usual case)
    """

    model_kwargs = dict(n_pts=n_points, pt_dim=pt_dim, bottle_fs=bottle_fs,
                        mlp1_fs=conv_layers, mlp2_fs=mlp_layers,
                        act=act, bn=bn)

    set_gpus(str(gpu))
    return load_model(model_name, list(), model_kwargs,
                      model_dir=model_dir, chkpt_num=chkpt_num,
                      chkpt_fname=chkpt_fname, eval_=eval_,
                      state_dict=state_dict)


def load_autoencoder_from_file(model_path, **kwargs):
    param_paths = glob.glob(
        f"{os.path.dirname(os.path.dirname(model_path))}/logs/*params.json")

    time_stamps = []
    for p in param_paths:
        time_stamp = re.findall("[\d_]+", p)[-1][:-1]
        time_stamp = datetime.datetime.strptime(time_stamp, "%d%m%y_%H%M%S")
        time_stamps.append(time_stamp)

    param_path = param_paths[np.argmax(time_stamps)]

    with open(param_path, 'r') as f:
        j = f.read()

    params = json.loads(j)
    params["bn"] = not params["nobn"]

    load_params = params.copy()
    load_params.update(kwargs)

    model = load_autoencoder(chkpt_fname=model_path, eval_=True, **load_params)
    return model, params


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

    j = json.dumps(vars(args_obj), sort_keys=True, indent=4,
                   ensure_ascii=False)
    with open(f"{log_dir}/{tstamp}_params.json", 'w') as f:
        f.write(j)


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
