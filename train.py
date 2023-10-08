import argparse
import collections
import torch
import pickle
from utils.accuracy import *
from utils.AverageMeter import AverageMeter
import timm.scheduler
import os
import datetime
from torchvision.transforms import transforms
from torchsketch.utils.general_utils.logger import Logger
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from model.mobilenet_v3 import mobilenetv3_large
import logging
import torch.nn as nn
from utils.ensemble_loss import EnsembleLoss
from model.cmps import CMPS
from Data.quickdraw414k_4_rnn_cnn import Quickdraw414k4RNN_CNN
import sys
sys.path.append('../')
logging.getLogger('matplotlib.font_manager').disabled = True
logging.getLogger('matplotlib.pyplot').disabled = True
logging.getLogger('PIL').setLevel(logging.WARNING)

parser = argparse.ArgumentParser(description='CNN_RNN_two_branches_classification')
parser.add_argument('--exp_base', type=str, default="./experiment", help='results_save_base_dir')
parser.add_argument('--exp', type=str, default="CMPS", help='result_save_dic')

# TODO

parser.add_argument('--train_sketch_dir', type=str, default="./coordinate_files/train", help='coordinate_train_dir')
parser.add_argument('--val_sketch_dir', type=str, default="./coordinate_files/val", help='coordinate_val_dir')
parser.add_argument('--train_sketch_list', type=str, default="./coordinate_files/tiny_train_set.txt",
                    help='train_sketch_list')
parser.add_argument('--val_sketch_list', type=str, default="./coordinate_files/tiny_val_set.txt",
                    help='val_sketch_list')
parser.add_argument("--epoch", type=int, default=40, help="epoch")
parser.add_argument("--batch_size", type=int, default=32, help="batch_size")
parser.add_argument("--num_workers", type=int, default=4, help="num_workers")
parser.add_argument('--gpu', type=str, default="0", help='choose GPU')

args = parser.parse_args()

#training config
basic_configs = collections.OrderedDict()
basic_configs['learning_rate'] = 4e-4
learning_rate = basic_configs['learning_rate']

basic_configs["display_step"] = 100

#data config
dataloader_configs = collections.OrderedDict()
dataloader_configs['train_sketch_path_root'] = args.train_sketch_dir
dataloader_configs['val_sketch_path_root'] = args.val_sketch_dir
dataloader_configs['train_sketch_list'] = args.train_sketch_list
dataloader_configs['val_sketch_list'] = args.val_sketch_list

dataloader_configs['num_epochs'] = args.epoch
dataloader_configs['batch_size'] = args.batch_size
dataloader_configs['num_workers'] = args.num_workers
epoch = dataloader_configs['num_epochs']

transform_train = transforms.Compose([
            transforms.ToTensor(),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(30),
            ])

transform_val = transforms.Compose([
            transforms.ToTensor()
            ])


train_dataset = Quickdraw414k4RNN_CNN(dataloader_configs['train_sketch_path_root'], dataloader_configs['train_sketch_list'], trans=transform_train)
val_dataset = Quickdraw414k4RNN_CNN(dataloader_configs['val_sketch_path_root'], dataloader_configs['val_sketch_list'], trans=transform_val)

train_loader = DataLoader(train_dataset, batch_size=dataloader_configs['batch_size'], shuffle=True, num_workers=dataloader_configs['num_workers'])
val_loader = DataLoader(val_dataset, batch_size=dataloader_configs['batch_size'], shuffle=False, num_workers=dataloader_configs['num_workers'])


exp_dir = os.path.join(args.exp_base, args.exp)
exp_log_dir = os.path.join(exp_dir, "log")
if not os.path.exists(exp_log_dir):
    os.makedirs(exp_log_dir)

exp_visual_dir = os.path.join(exp_dir, "visual")
if not os.path.exists(exp_visual_dir):
    os.makedirs(exp_visual_dir)

exp_ckpt_dir = os.path.join(exp_dir, "checkpoints")
if not os.path.exists(exp_ckpt_dir):
    os.makedirs(exp_ckpt_dir)


now_str = datetime.datetime.now().__str__().replace(' ', '_').replace(':', '.')

writer_path = os.path.join(exp_visual_dir, now_str)
writer = SummaryWriter(writer_path)

logger_path = os.path.join(exp_log_dir, args.exp + ".log")
logger = Logger(logger_path).get_logger()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

max_val_acc = 0.0
max_val_acc_epoch = -1

cnn = mobilenetv3_large(num_classes=345)
pretrained_dict = torch.load("./model/mobilenetv3-large-1cd25616.pth")
pretrained_dict = {k: v for k, v in pretrained_dict.items() if "classifier" not in k}
missing_keys, unexpected_keys = cnn.load_state_dict(pretrained_dict, strict=False)

loss_function = EnsembleLoss()

loss_function = loss_function.cuda()
net = CMPS(cnn)

net = net.cuda()

optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=3e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=0.1)


def train_function(epoch_th):
    training_loss = AverageMeter()
    training_acc = AverageMeter()
    train_iter = 0
    net.train()
    count, loss_val, correct, total = train_iter, 0, 0, 0

    lr = optimizer.param_groups[0]['lr']
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    logger.info("set learning rate to: {}".format(lr))

    for idx, (coord, img, label) in enumerate(tqdm(train_loader)):
        coord = coord.cuda()

        img = img.cuda()
        label = label.cuda()
        output_i, output_e, output_c, kl_img, kl_stroke = net(coord, img)
        batch_loss = loss_function(output_i, output_e, output_c, label, kl_img, kl_stroke)
        c_loss = batch_loss.data.item()

        loss_val += c_loss

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        training_loss.update(batch_loss.item(), coord.size(0))
        training_acc.update(accuracy(output_c, label, topk=(1,))[0].item(), coord.size(0))
        if (idx + 1) % basic_configs["display_step"] == 0:
            logger.info(
                "==> Iteration [{}][{}/{}]:".format(epoch_th + 1, idx + 1, len(train_loader)))
            logger.info("current batch loss: {}".format(
                batch_loss.item()))
            logger.info("average loss: {}".format(
                training_loss.avg))
            logger.info("average acc: {}".format(training_acc.avg))

            # TODO
    logger.info("Begin Evaluating")

    validation_loss, validation_acc = validate_function(val_loader)

    return validation_acc


def validate_function(data_loader):
    validation_loss = AverageMeter()

    validation_acc_1 = AverageMeter()
    validation_acc_5 = AverageMeter()
    validation_acc_10 = AverageMeter()
    net.eval()
    # TODO
    with torch.no_grad():

        for idx, (coord, img, label) in enumerate(tqdm(data_loader)):
            coord = coord.cuda()
            label = label.cuda()
            img = img.cuda()

            output_i, output_e, output_c, kl_img, kl_stroke = net(coord, img)
            batch_loss = loss_function(output_i, output_e, output_c, label, kl_img, kl_stroke)

            validation_loss.update(batch_loss.item(), coord.size(0))
            acc_1, acc_5, acc_10 = accuracy(output_c, label, topk=(1, 5, 10))
            validation_acc_1.update(acc_1, coord.size(0))
            validation_acc_5.update(acc_5, coord.size(0))
            validation_acc_10.update(acc_10, coord.size(0))

        logger.info("==> Testing Result: ")
        logger.info("loss: {}  acc@1: {} acc@5: {} acc@10: {}".format(validation_loss.avg, validation_acc_1.avg,
                                                                      validation_acc_5.avg, validation_acc_10.avg))

    return validation_loss, validation_acc_1


if __name__ == '__main__':
    logger.info("training status: ")
    logger.info("Begin Evaluating before training")
    logger.info("optimizer: {} epoch: {}".format(optimizer, epoch))
    # validate_function(val_loader)

    for e in range(epoch):

        validation_acc = train_function(e)
        scheduler.step()
        if validation_acc.avg > max_val_acc:
            max_val_acc = validation_acc.avg
            max_val_acc_epoch = e + 1
        logger.info("max_val_acc: {}  max_val_acc_epoch: {}".format(max_val_acc, max_val_acc_epoch))

        net_checkpoint_name = args.exp + "_net_epoch" + str(e + 1)
        net_checkpoint_path = os.path.join(exp_ckpt_dir, net_checkpoint_name)
        net_state = {"epoch": e + 1,
                     "network": net.state_dict()}
        #torch.save(net_state, net_checkpoint_path)

    logger.info("max_val_acc: {}  max_val_acc_epoch: {}".format(max_val_acc, max_val_acc_epoch))
