"""
initial code: https://github.com/fxia22/pointnet.pytorch
"""

import os
import random
import argparse

import numpy as np

import torch.optim as optim
import torch.utils.data
import tensorboardX

from datasets_pychg import CellDataset
import models
import loss
import utils

home = os.path.expanduser("~")
folder_path = os.path.dirname(__file__)
MODULES_TO_RECORD = [__file__, folder_path + "/utils.py",
                     folder_path + "/datasets.py"]

# Init / Parser -------------------------

parser = argparse.ArgumentParser()
parser.add_argument('expt_name')
parser.add_argument('--model_name', default='PointNetAE',
                    help='model to use for neuronencodings')
parser.add_argument('--loss_name', default='ApproxEMD',
                    help='model to use for neuronencodings')
parser.add_argument('--batch_size', type=int, default=10,
                    help='input batch size')
parser.add_argument('--workers', type=int,
                    help='number of data loading workers', default=4)
parser.add_argument('--nepoch', type=int, default=20000,
                    help='number of epochs to train')
parser.add_argument('--expt_dir', type=str, help='experiment folder',
                    default="%s/seungmount/research/nick_and_sven/models_sven/" % home)
parser.add_argument('--chkpt_num', type=int, default=0,
                    help='chkpt at which to continue neuronencodings')
parser.add_argument('--gpus', nargs="+", default=["0"], help='gpu ids')
parser.add_argument('--n_points', type=int, default=250,
                    help='number of points')
parser.add_argument('--bottle_fs', type=int, default=64,
                    help='number of latent variables (size of max pool layers)')
parser.add_argument('--nobn', action="store_true")
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--lr_decay', type=float, default=0.95, help='learning rate decay every 10 epochs')
parser.add_argument('--validation', action="store_true", help='enables testing')
parser.add_argument('--rotation', action="store_true", help='augment with rotation')
parser.add_argument('--jitter', action="store_true", help='augment with jitter')
parser.add_argument('--scaling', action="store_true", help='augment with scaling')
parser.add_argument('--movement', action="store_true", help='augment with movement')
parser.add_argument('--chopping', action="store_true", help='augment with chopping')
parser.add_argument('--dataset_name', type=str, default="full_cells", help='ground truth dataset',
                    choices=["full_cells", "orphans", "all", "soma_vs_rest","orphans2",
                             "orphan_axons", "orphan_axons_refined", "fish_refined",
                             "full_cells_refined"])
parser.add_argument('--eval_val', action="store_true",
                    help="use eval mode during validation"),

opt = parser.parse_args()
print(opt)

utils.set_gpus(opt.gpus)

opt.manualSeed = 2  # random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)


# Datasets ------------------------------

if opt.dataset_name == "full_cells":
    # dataset_paths = [home + "/seungmount/research/svenmd/pointnet_axoness_gt_rfc_based_masked_180322/"]
    dataset_paths = [home + "/seungmount/research/svenmd/pointnet_axoness_gt_180223/"]
elif opt.dataset_name == "soma_vs_rest":
    dataset_paths = [home + "/seungmount/research/svenmd/pointnet_soma_masked_180401"]
elif opt.dataset_name == "orphans":
    dataset_paths = [home + "/seungmount/research/svenmd/pointnet_orphan_axons_gt_180308/",
                     home + "/seungmount/research/svenmd/pointnet_orphan_dendrites_gt_180308/"]
elif opt.dataset_name == "orphans2":
    dataset_paths = [home + "/research/pointnet/orphan_dataset/train_val_axons",
                     home + "/research/pointnet/orphan_dataset/train_val_dends/"]
elif opt.dataset_name == "orphan_axons":
    dataset_paths = [home + "/seungmount/research/svenmd/pointnet_orphan_axons_gt_180308/"]
elif opt.dataset_name == "orphan_axons_refined":
    dataset_paths = [home + "/seungmount/research/svenmd/pointnet_orphan_axons_gt_180308_refined/"]
elif opt.dataset_name == "fish_refined":
    dataset_paths = [home + "/seungmount/research/svenmd/180831_meshes_ashwin_refined/"]
elif opt.dataset_name == "full_cells_refined":
    dataset_paths = [home + "/seungmount/research/svenmd/pointnet_full_semantic_labels_masked_180401_refined/"]
else:
    dataset_paths = [home + "/seungmount/research/svenmd/pointnet_axoness_gt_rfc_based_masked_180322/",
                     home + "/pointnet_orphan_axons_gt_180308/",
                     home + "/pointnet_orphan_dendrites_gt_180308/"]

print("Loading data...")
#Training Set
dataset = CellDataset(gt_dirs=dataset_paths,
                      phase=3,
                      n_points=opt.n_points,
                      random_seed=opt.manualSeed,
                      batch_size=opt.batch_size,
                      apply_rotation=opt.rotation,
                      apply_jitter=opt.jitter,
                      apply_scaling=opt.scaling,
                      apply_chopping=opt.chopping,
                      apply_movement=opt.movement,
                      train_test_split_ratio=1.)

dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size=opt.batch_size,
                                         shuffle=True,
                                         num_workers=int(opt.workers),
                                         #pin_memory=True,
                                         drop_last=True)

#Validation Set
test_dataset = CellDataset(gt_dirs=dataset_paths, phase=2,
                           n_points=opt.n_points,
                           random_seed=opt.manualSeed,
                           batch_size=opt.batch_size,
                           apply_rotation=False,
                           apply_jitter=False,
                           apply_scaling=False,
                           apply_chopping=False,
                           apply_movement=False,
                           train_test_split_ratio=1.)

testdataloader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=opt.batch_size,
                                             shuffle=True,
                                             num_workers=int(opt.workers),
                                             #pin_memory=True,
                                             drop_last=True)
print("Finished, setting up expt")

(model_dir, save_dir, fwd_dir, tb_train, tb_val) = \
    utils.make_required_dirs(opt.expt_dir, opt.expt_name)

tstamp = utils.timestamp()
utils.save_args(opt, save_dir, tstamp=tstamp)
utils.log_tagged_modules(MODULES_TO_RECORD, save_dir, "train", tstamp=tstamp)

train_writer = tensorboardX.SummaryWriter(tb_train)
val_writer = tensorboardX.SummaryWriter(tb_val)
# Train ----------------------------------

blue = lambda x: '\033[94m' + x + '\033[0m'

in_dim = 3
model_class = getattr(models, opt.model_name)
model = model_class(opt.n_points, bottle_fs=opt.bottle_fs, pt_dim=in_dim,
                    bn=not(opt.nobn))
model.cuda()

loss_class = getattr(loss, opt.loss_name)
loss_fn = loss_class()

if opt.chkpt_num != 0:
    model_chkpt = "{dir}/model_{iter}.chkpt".format(dir=model_dir,
                                                    iter=opt.chkpt_num)
    model.load_state_dict(torch.load(model_chkpt))


optimizer = optim.Adam(model.parameters(), lr=opt.lr)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30000,
                                         gamma=opt.lr_decay)

num_batch = len(dataset) / opt.batch_size


print("Starting Training Loop")
train_iter = opt.chkpt_num
for i_epoch in range(opt.nepoch):
    lr_scheduler.step()

    test_data_iter = iter(testdataloader)

    for i_batch, points in enumerate(dataloader, 0):
        points = points.cuda()

        optimizer.zero_grad()

        pred, _ = model(points)

        loss = loss_fn(pred, points)

        loss.backward()
        optimizer.step()

        train_iter += 1

        if (train_iter % 10 == 0):
            train_writer.add_scalar("Loss", loss.item(), train_iter)
        print('[%d: %d/%d] %s loss: %f ' % (
            i_epoch, i_batch, num_batch, "train", loss.item()))

        #validation
        if (train_iter != 0 and train_iter % 100 == 0) and opt.validation:
            points = test_data_iter.next().cuda()

            if opt.eval_val:
                model.eval()#eval mode during evaluation

            pred, _ = model(points)

            if opt.eval_val:
                model.train()#eval mode during evaluation

            loss = loss_fn(pred, points)

            val_writer.add_scalar("Loss", loss.item(), train_iter)
            print('[%d: %d/%d] %s loss: %f' % (
                i_epoch, i_batch, num_batch, blue('test'), loss.item()))

        if (train_iter != 0 and
            train_iter % 500 == 0):
            torch.save(model.state_dict(),
                       '{dir}/model_{iter}.chkpt'.format(dir=model_dir,
                                                         iter=train_iter))
