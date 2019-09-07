import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'    # change GPU here
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import shutil
import sys
import datetime as dt
import time
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="student")
parser.add_argument("--mode", type=str, default="", help="please add '_' before the mode")
parser.add_argument("--date", type=str, default="_908", help="please add '_' before the date")
parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
parser.add_argument("--batch_size", type=int, default=64, help="batch size")
parser.add_argument("--epochs", type=int, default=150, help="number of training iterations")
parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay")
parser.add_argument('--dataset', default='CUB', type=str)
parser.add_argument('--dataset_dir', default='/home/data2/rzf/KD/dataset/CUB/', type=str)
parser.add_argument('--workers', default=8, type=int)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
parser.add_argument('--gpu', default=0, type=int, help='Set it to 0. Change it above before importing torch.')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N')
parser.add_argument('--print-freq', '-p', default=30, type=int, metavar='N')
parser.add_argument('--classes', default=200, type=int)
parser.add_argument('--logspace', default=True, type=bool)
parser.add_argument('--epoch_step', default=40, type=int)
parser.add_argument('--milestones', type=list, default=[50, 100], help='')
parser.add_argument('--gama', type=float, default=0.1, help='')
parser.add_argument('--save_per_epoch', default=True, type=bool)
parser.add_argument('--resume', default='', type=str, metavar='PATH')
parser.add_argument('--pick_sample', default=False, type=bool)
args = parser.parse_args()

PATH = '/home/data2/rzf/KD/trained_model/' + str(args.model) + str(args.mode) + str(args.date) + '_' + str(args.dataset) + '/'
if not os.path.exists(PATH):
    os.mkdir(PATH)
print(vars(args))

print("Time: {}".format(dt.datetime.now()))
print("Python: {}".format(sys.version))
print("Numpy: {}".format(np.__version__))
print("Pytorch: {}".format(torch.__version__))

if args.seed is not None:
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)   # if you are using multi-GPU.
    cudnn.benchmark = False
    cudnn.deterministic = True


from torch.utils.data import DataLoader
from model.models import *
from function.dataset import SigmaDataSet
from function.logger import Logger


def train_network(PATH, args):

    if args.model == 'vgg16':
        if args.mode == '_without_pretrain':
            model = vgg16_without_pretrained(args.classes, seed=args.seed)
        elif args.mode == '_pretrain' or args.mode == '':
            model = vgg16_pretrained(args.classes, seed=args.seed)
        else:
            sys.exit('Model Name Error')
    elif args.model == 'student':
        model = student_network_bn(args.classes, seed=args.seed)
    elif args.model == 'vgg11':
        model = vgg11_pretrained(args.classes, seed=args.seed)
    elif args.model == 'vgg19':
        model = vgg19_pretrained(args.classes, seed=args.seed)
    elif args.model == 'resnet18':
        model = ResNet18(args.classes, seed=args.seed)
    elif args.model == 'resnet50':
        model = ResNet50(args.classes, seed=args.seed)
    elif args.model == 'resnet101':
        model = ResNet101(args.classes, seed=args.seed)
    elif args.model == 'resnet152':
        model = ResNet152(args.classes, seed=args.seed)
    else:
        sys.exit('Model Name Error')

    if args.gpu is not None:
        model = model.cuda(args.gpu)
    elif args.distributed:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)

    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=args.milestones,gamma=args.gama,last_epoch=-1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=10,verbose=False,
                                                           threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
    if args.logspace:
        logspace_lr = np.logspace(np.log10(args.lr), np.log10(args.lr) - 3, args.epochs)
        print("lr logspace:", logspace_lr, '\n')

    # if 'vgg' in args.model:
    #     large_lr_layers = list(map(id, model.net.classifier.parameters()))
    #     small_lr_layers = list(filter(lambda p: id(p) not in large_lr_layers, model.net.parameters()))
    #     optimizer = torch.optim.SGD([
    #         {"params": model.net.classifier.parameters(), "lr": 1e-3},
    #         {"params": small_lr_layers, "lr": 1e-3}
    #     ], lr=1e-3, momentum=0.9)
    #
    # elif 'resnet' in args.model:
    #     large_lr_layers = list(map(id, model.net.fc.parameters()))
    #     small_lr_layers = list(filter(lambda p: id(p) not in large_lr_layers, model.net.parameters()))
    #     optimizer = torch.optim.SGD([
    #         {"params": model.net.fc.parameters(), "lr": 1e-2},
    #         {"params": small_lr_layers, "lr": 1e-3}
    #     ], lr=1e-3, momentum=0.9)


    train_logger_path = '/home/data2/rzf/KD/logs/{}_{}{}{}/train'.format(args.dataset, args.model, args.mode, args.date)
    val_logger_path = '/home/data2/rzf/KD/logs/{}_{}{}{}/val'.format(args.dataset, args.model, args.mode, args.date)

    if os.path.exists(train_logger_path):
        shutil.rmtree(train_logger_path)
    if os.path.exists(val_logger_path):
        shutil.rmtree(val_logger_path)
    logger_train = Logger(train_logger_path)
    logger_val = Logger(val_logger_path)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.dataset == 'CUB':
        traindir = args.dataset_dir + '/train'
        valdir = args.dataset_dir + '/val'
    elif args.dataset == 'VOC':
        traindir = args.dataset_dir + '/train'
        valdir = args.dataset_dir + '/val'
    elif args.dataset == 'coco':
        traindir = args.dataset_dir + '/train'
        valdir = args.dataset_dir + '/val'

    train_loader = torch.utils.data.DataLoader(SigmaDataSet(traindir), batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True,  worker_init_fn=_init_fn)
    val_loader = torch.utils.data.DataLoader(SigmaDataSet(valdir), batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True,  worker_init_fn=_init_fn)

    best_acc1, acc1 = 0, 0

    if args.model == 'vgg16':
        w_origin = model.net.features[28].weight.clone()
    elif args.model == 'vgg11':
        w_origin = model.net.features[18].weight.clone()
    elif args.model == 'vgg19':
        w_origin = model.net.features[34].weight.clone()
    elif args.model == 'student':
        w_origin = model.features[12].weight.clone()
    weight_diff = torch.zeros(args.epochs, requires_grad=False)
    for epoch in range(args.start_epoch, args.epochs):
        # remember best acc@1 and save checkpoint
        if args.save_per_epoch:
            save_dir = PATH + '{}.pth.tar'.format(epoch)
        else:
            save_dir = PATH + 'checkpoint_{}_{}{}{}.pth.tar'.format(args.dataset, args.model, args.mode, args.date)
        # is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        save_checkpoint({'epoch': epoch, 'state_dict': model.state_dict(), 'best_acc1': best_acc1, 'optimizer': optimizer.state_dict()}, save_dir)


        if args.model == 'vgg16':
            weight_curr = model.net.features[28].weight
        elif args.model == 'vgg11':
            weight_curr = model.net.features[18].weight
        elif args.model == 'vgg19':
            weight_curr = model.net.features[34].weight
        elif args.model == 'student':
            weight_curr = model.features[12].weight
        a = torch.sqrt(torch.sum((weight_curr - w_origin) ** 2))
        print(a)
        weight_diff[epoch] = torch.sqrt(torch.mean((weight_curr - w_origin) ** 2))
        print(weight_diff[epoch])


        if args.logspace:
            for param_group in optimizer.param_groups:
                param_group['lr'] = logspace_lr[epoch]

        lr = get_learning_rate(optimizer)
        print("lr:", lr)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, logger_train)

        # evaluate on validation set
        acc1, val_loss = validate(val_loader, model, criterion, epoch, logger_val)

        if args.model[:3] == 'vgg':
            scheduler.step(val_loss)

        weight_diff_save_path = PATH + 'weight_diff.npy'
        np.save(weight_diff_save_path, weight_diff.detach().numpy())
    print(weight_diff)


def train(train_loader, model, criterion, optimizer, epoch, logger):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    Loss = 0
    # switch to train mode
    model.train()

    end = time.time()
    for i, (id, img_name, input, target) in enumerate(train_loader):
        # print(id)
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)
        # prob = F.softmax(output, dim=1).data.cpu().numpy()
        # predict = np.argmax(prob, axis=1)

        Loss += loss.cpu().item()
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))


        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t''Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t''Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t''Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time, data_time=data_time, loss=losses, top1=top1,
                top5=top5))
    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))
    log_dict = {'Loss': losses.avg, 'top1_prec': top1.avg.item(), 'top5_prec': top5.avg.item()}
    set_tensorboard(log_dict, epoch, logger)



def validate(val_loader, model, criterion, epoch, logger):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        Loss = 0
        Error = 0
        end = time.time()
        for i, (id, img_name, input, target) in enumerate(val_loader):
            input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)
            # prob = F.softmax(output, dim=1).data.cpu().numpy()
            # predict = np.argmax(prob, axis=1)

            Loss += loss.cpu().item()

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
        log_dict = {'Loss': losses.avg, 'top1_prec': top1.avg.item(), 'top5_prec': top5.avg.item()}
        set_tensorboard(log_dict, epoch, logger)
    return top1.avg, losses.avg


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


# For tensorboard
def set_tensorboard(log_dict, epoch, logger):
    # set for tensorboard
    info = log_dict
    for tag, value in info.items():
        logger.scalar_summary(tag, value, epoch + 1)
    return


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // args.epoch_step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def get_error(pre, lbl):
    acc = torch.sum(pre == lbl).item() / len(pre)
    return 1 - acc


def checkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def Pick_Sample(false_times, np_dir):
    length = len(false_times)
    false_seq = np.argsort(np.array(false_times))
    easy_id = false_seq[: int(length * 0.1)]
    hard_id = false_seq[length - int(length * 0.1):]
    np.save(np_dir + 'easy_id.npy', easy_id)
    np.save(np_dir + 'hard_id.npy', hard_id)


def get_learning_rate(optimizer):
    lr=[]
    for param_group in optimizer.param_groups:
          lr +=[ param_group['lr'] ]
    return lr


def _init_fn(worker_id):
    np.random.seed(args.seed + worker_id)


train_network(PATH, args)
