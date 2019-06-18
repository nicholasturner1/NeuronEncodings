__doc__ = """
Training Script

initial code: https://github.com/fxia22/pointnet.pytorch
"""

import os
import argparse

import torch.optim as optim
import torch.utils.data
import tensorboardX

import neuronencodings as ne
from neuronencodings import data
from neuronencodings.data import CellDataset
from neuronencodings import loss
from neuronencodings import utils

HOME = os.path.expanduser("~")
NE_DIR = os.path.dirname(ne.__file__)
DEFAULT_EXPT_DIR = f"{HOME}/seungmount/research/nick_and_sven/models_sven/"
MODULES_TO_RECORD = [__file__,
                     os.path.join(NE_DIR, "utils.py"),
                     os.path.join(NE_DIR, "data", "utils.py"),
                     os.path.join(NE_DIR, "data", "cell_dataset.py"),
                     os.path.join(NE_DIR, "loss", "emd_basic.py"),
                     os.path.join(NE_DIR, "models", "pointnet_ae.py")]

# Init / Parser -------------------------

parser = argparse.ArgumentParser()
parser.add_argument('expt_name')
parser.add_argument('--model_name', default='PointNetAE',
                    help='model to train')
parser.add_argument('--loss_name', default='ApproxEMD',
                    help='loss fn to use')
parser.add_argument('--batch_size', type=int, default=10,
                    help='input batch size')
parser.add_argument('--workers', type=int, default=2,
                    help='number of data loading workers')
parser.add_argument('--nepoch', type=int, default=20000,
                    help='number of epochs to train')
parser.add_argument('--expt_dir', type=str, default=DEFAULT_EXPT_DIR,
                    help='experiment folder')
parser.add_argument('--chkpt_num', type=int, default=None,
                    help='chkpt at which to continue training')
parser.add_argument('--gpus', nargs="+", default=["0"],
                    help='gpu ids')
parser.add_argument('--n_points', type=int, default=250,
                    help='number of points for each sample')
parser.add_argument('--sample_n_points', type=int, default=None,
                    help='sample radius in points')
parser.add_argument('--point_dim', type=int, default=3,
                    help='dimensionality of each input point')
parser.add_argument('--bottle_fs', type=int, default=24,
                    help='number of latent vars (size of max pool layers)')
parser.add_argument('--mlp_layers', type=list, default=[24, 48, 96, 192, 384],
                    help='decoder network')
parser.add_argument('--conv_layers', type=list, default=[128, 64],
                    help='encoder network')
parser.add_argument('--act', type=str, default="relu",
                    help='activation function')
parser.add_argument('--nobn', action="store_true",
                    help="whether to remove batch norm from model")
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate')
parser.add_argument('--lr_decay', type=float, default=0.95,
                    help='learning rate decay every 10 epochs')
parser.add_argument('--n_views_per_batch', type=int, default=5,
                    help="number of views from the same cell per batch")
parser.add_argument('--dataset_name', type=str, default="full_cells",
                    choices=list(data.DATASET_DIRS.keys()),
                    help='ground truth dataset')
parser.add_argument('--eval_val', action="store_true",
                    help='switch to use eval mode during validation')
parser.add_argument('--noval', action="store_true",
                    help='use no validation during training')
parser.add_argument('--manualSeed', type=int, default=2,
                    help='random seed for train/val/test split')
parser.add_argument('--val_intv', type=int, default=20,
                    help='model evaluation interval, set to -1 for no eval')
parser.add_argument('--chkpt_intv', type=int, default=500,
                    help='model checkpoint interval')
parser.add_argument('--loss_intv', type=int, default=10,
                    help='loss display interval')
parser.add_argument('--train_split', type=float, default=0.8,
                    help='amount of data to use for training')
parser.add_argument('--val_split', type=float, default=0.1,
                    help='amount of data to use for validation')
parser.add_argument('--test_split', type=float, default=0.1,
                    help='amount of data to reserve for testing')
parser.add_argument('--use_full_data', action="store_true",
                    help='whether to use the full dataset for training')
parser.add_argument('--local_env', action="store_true",
                    help='whether to use the local neighborhood patches')
parser.add_argument('--cache_meshes', action="store_true",
                    help='whether to cache meshes upon loading -- '
                         'memory intensive')

opt = parser.parse_args()
print(opt)


# Datasets ------------------------------
print("Loading data")
print(opt.noval)

gt_dirs = data.fetch_dset_dirs(opt.dataset_name)

if opt.cache_meshes:
    print("Multiply")
    gt_dirs = gt_dirs * 1000

phase = data.Phase.FULL if opt.use_full_data else data.Phase.TRAIN

# Training Set
train_dset = CellDataset(gt_dirs=gt_dirs,
                         phase=phase,
                         n_points=opt.n_points,
                         n_views_per_batch=opt.n_views_per_batch,
                         sample_n_points=opt.sample_n_points,
                         random_seed=opt.manualSeed,
                         train_split=opt.train_split,
                         val_split=opt.val_split,
                         test_split=opt.test_split)

train_loader = torch.utils.data.DataLoader(train_dset,
                                           batch_size=opt.batch_size,
                                           shuffle=True,
                                           num_workers=int(opt.workers),
                                           # pin_memory=True,
                                           drop_last=True)

# Validation Set
if not(opt.noval):
    val_dset = CellDataset(gt_dirs=gt_dirs,
                           phase=data.Phase.VAL,
                           n_points=opt.n_points,
                           n_views_per_batch=opt.n_views_per_batch,
                           sample_n_points=opt.sample_n_points,
                           random_seed=opt.manualSeed,
                           train_split=opt.train_split,
                           val_split=opt.val_split,
                           test_split=opt.test_split)

    val_loader = torch.utils.data.DataLoader(val_dset,
                                             batch_size=opt.batch_size,
                                             shuffle=True,
                                             num_workers=int(opt.workers),
                                             # pin_memory=True,
                                             drop_last=True)


# Setting Up Experiment Logs -------------
print("Setting up experiment")

(model_dir, save_dir, fwd_dir, tb_train, tb_val) = \
    utils.make_required_dirs(opt.expt_dir, opt.expt_name)

tstamp = utils.timestamp()
utils.save_args(opt, save_dir, tstamp=tstamp)
utils.log_tagged_modules(MODULES_TO_RECORD, save_dir, "train", tstamp=tstamp)

train_writer = tensorboardX.SummaryWriter(tb_train)
val_writer = tensorboardX.SummaryWriter(tb_val)

utils.set_gpus(opt.gpus)
print(os.getenv("CUDA_VISIBLE_DEVICES"))


# Train ----------------------------------
print("Setting up training")

model = utils.load_autoencoder(opt.model_name, pt_dim=opt.point_dim,
                               n_points=opt.n_points, bottle_fs=opt.bottle_fs,
                               conv_layers=opt.conv_layers,
                               mlp_layers=opt.mlp_layers, act=opt.act,
                               bn=not(opt.nobn), model_dir=model_dir,
                               chkpt_num=opt.chkpt_num)

loss_class = getattr(loss, opt.loss_name)
loss_fn = loss_class()

optimizer = optim.Adam(model.parameters(), lr=opt.lr)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30000,
                                         gamma=opt.lr_decay)

num_batch = len(train_dset) // opt.batch_size

print(f"Number of meshes in dataset: {len(train_dset.gt_files)}")

print("Starting training loop")
train_iter = opt.chkpt_num if opt.chkpt_num is not None else 0
for i_epoch in range(opt.nepoch):
    print(f"EPOCH {i_epoch}")
    lr_scheduler.step()

    if not(opt.noval):
        val_data_iter = iter(val_loader)

    for i_batch, points in enumerate(train_loader, 0):
        points = points.reshape(-1, opt.n_points, opt.point_dim)
        points = points.cuda()

        optimizer.zero_grad()

        pred, _ = model(points)

        loss = loss_fn(pred, points)
        loss_val = loss.item()

        loss.backward()
        optimizer.step()

        train_iter += 1

        if (train_iter % opt.loss_intv == 0):
            train_writer.add_scalar("Loss", loss_val, train_iter)

        utils.print_loss("train", loss_val, i_epoch+1, i_batch+1, num_batch)

        # Evaluation on validation set
        if not(opt.noval) and (train_iter % opt.val_intv == 0) and (opt.val_intv > 0):
            points = val_data_iter.next().cuda()

            if opt.eval_val:
                model.eval()   # eval mode during evaluation

            with torch.no_grad():
                pred, _ = model(points)

                loss = loss_fn(pred, points)
                loss_val = loss.item()

            if opt.eval_val:
                model.train()  # eval mode during evaluation

            val_writer.add_scalar("Loss", loss_val, train_iter)
            tag = utils.blue("val")
            utils.print_loss(tag, loss_val, i_epoch+1, i_batch+1, num_batch)

        if (train_iter % opt.chkpt_intv == 0):
            torch.save(model.state_dict(),
                       f'{model_dir}/model_{train_iter}.chkpt')
