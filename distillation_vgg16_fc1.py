import shutil
import argparse
from torch.autograd import Variable
import random
import os
from torch.utils.data import DataLoader
import sys
sys.path.extend(['/home/data1/chenyilan/KD'])
from model.models import *
import torch.backends.cudnn as cudnn
from function.logger import Logger
from function.dataset import MyDataSet
from function.sub_functions import *


parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="fc1") ## fc1 fc2 fc3 softmax ##
parser.add_argument("--model", type=str, default="vgg16") ## vgg11 vgg16 vgg19 resnet50 resnet101 resnet152 ##
parser.add_argument("--classifier", type=bool, default=True)
parser.add_argument("--date", type=str, default="905")
parser.add_argument("--pretrained", type=bool, default=True)
parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
parser.add_argument("--batch_size", type=int, default=64, help="batch size")
parser.add_argument("--epochs", type=int, default=150, help="number of distillation training iterations")
parser.add_argument("--classify_epochs", type=int, default=100, help="number of classifier training iterations")
parser.add_argument("--start_epoch", type=int, default=0)
parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay")
parser.add_argument('--dataset', default='CUB', type=str)
parser.add_argument('--workers', default=8, type=int)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--world-size', default=1, type=int)
parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
parser.add_argument('--gpu', default=3, type=int, help='GPU id to use.')
parser.add_argument('--resume', default='', type=str, metavar='PATH')
parser.add_argument('--teacher_checkpoint', default='/home/data2/rzf/KD/trained_model/vgg16_904_CUB/149.pth.tar', type=str, metavar='PATH')
parser.add_argument('--student_checkpoint', default='/home/data2/rzf/KD/trained_model/stu_CUB/0.pth.tar', type=str, metavar='PATH')
parser.add_argument('--sample_num', default=-1, type=type)
parser.add_argument('--start-epoch', default=0, type=int, metavar='N')
parser.add_argument('--print-freq', '-p', default=30, type=int,metavar='N')
parser.add_argument('--temperature', default=1, type=int)
parser.add_argument('--alpha', default=1, type=float) ## adjust mse loss ##
parser.add_argument('--classes', default=200, type=int)
parser.add_argument('--logspace', default=True, type=bool)
parser.add_argument('--batch_step', default=30, type=int)
parser.add_argument('--save_per_epoch', default=True, type=bool)
parser.add_argument('--save_best_acc', default=False, type=bool)
args = parser.parse_args()

## get the path for project ##
path = '/home/data2/rzf/KD/'
print(vars(args))
torch.cuda.set_device(args.gpu)
os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)

def distillation(path,args):

    ## choose randonm seed to intialize ##
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)  # 为当前GPU设置随机种子
        cudnn.benchmark = False
        cudnn.deterministic = True

    ## get dataset path ##
    dataset_root = path +'dataset/{}'.format(args.dataset)
    traindir, valdir = dataset_root+'/train', dataset_root+'/val'
    # traindir, valdir = '/home/data1/chenyilan/KD/dataset/coco_tiny/train', '/home/data1/chenyilan/KD/dataset/coco_tiny/val'

    ## prepare dataloader ##
    train_loader = torch.utils.data.DataLoader(MyDataSet(traindir), batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(MyDataSet(valdir), batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)

    ## load teacher checkpoint ##
    if args.model == 'vgg16':
        if args.pretrained:
            teacher_model = vgg16_pretrained(out_planes=args.classes).cuda(args.gpu)
        else:
            teacher_model = vgg16_without_pretrained(out_planes=args.classes).cuda(args.gpu)
    elif args.model == 'vgg11':
        teacher_model = vgg11_pretrained(out_planes=args.classes).cuda(args.gpu)
    elif args.model == 'vgg19':
        teacher_model = vgg19_pretrained(out_planes=args.classes).cuda(args.gpu)
    elif args.model == 'resnet50':
        teacher_model = ResNet50(out_planes=args.classes).cuda(args.gpu)
    elif args.model == 'resnet101':
        teacher_model = ResNet101(out_planes=args.classes).cuda(args.gpu)
    elif args.model == 'resnet152':
        teacher_model = ResNet152(out_planes=args.classes).cuda(args.gpu)

    load_checkpoint(args.teacher_checkpoint, teacher_model)
    ## fetch teacher outputs using teacher_model under eval() mode ##
    teacher_model.eval()
    teacher_output_path = path + 'npy_files/teacher_output_MSE/'
    check_dir(teacher_output_path)
    teacher_train_path = teacher_output_path + '/teacher_{}_{}_{}_train.npy'.format(args.model, args.mode, args.dataset)
    teacher_val_path = teacher_output_path + '/teacher_{}_{}_{}_val.npy'.format(args.model, args.mode, args.dataset)
    if os.path.exists(teacher_train_path):
        teacher_train = np.load(teacher_train_path)
    else:
        print("get teacher train")
        teacher_train = fetch_teacher_outputs(teacher_model, train_loader, args.model, args.mode, args.classes)
        np.save(teacher_train_path, teacher_train)
    if os.path.exists(teacher_val_path):
        teacher_val = np.load(teacher_val_path)
    else:
        print("\nget teacher val")
        teacher_val = fetch_teacher_outputs(teacher_model, val_loader, args.model, args.mode, args.classes)
        np.save(teacher_val_path, teacher_val)
    del teacher_model
    torch.cuda.empty_cache()


    ## prepare dataloader ##
    train_loader = torch.utils.data.DataLoader(MyDataSet(traindir), batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(MyDataSet(valdir), batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)

    ## choose model for student and teacher ##
    model = student_network_bn(out_planes=args.classes)
    if args.gpu is not None:
        model = model.cuda(args.gpu)

    load_checkpoint(args.student_checkpoint, model)
    ## choose optimizer ##
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=10, verbose=False)

    ## resume to load for further training##
    if args.resume:
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_acc1 = checkpoint['best_acc']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

    train_mse_logger_path = path + 'logs/distil_{}_{}_{}_{}/train_mse'.format(args.model, args.mode, args.dataset, args.date)
    val_mse_logger_path = path + 'logs/distil_{}_{}_{}_{}/val_mse'.format(args.model, args.mode, args.dataset, args.date)
    if os.path.exists(train_mse_logger_path):
        shutil.rmtree(train_mse_logger_path)
    if os.path.exists(val_mse_logger_path):
        shutil.rmtree(val_mse_logger_path)
    logger_train_mse = Logger(train_mse_logger_path)
    logger_val_mse = Logger(val_mse_logger_path)

    train_ce_logger_path = path + 'logs/distil_{}_{}_{}_{}/train_ce'.format(args.model, args.mode, args.dataset, args.date)
    val_ce_logger_path = path + 'logs/distil_{}_{}_{}_{}/val_ce'.format(args.model, args.mode, args.dataset, args.date)
    if os.path.exists(train_ce_logger_path):
        shutil.rmtree(train_ce_logger_path)
    if os.path.exists(val_ce_logger_path):
        shutil.rmtree(val_ce_logger_path)
    logger_train_ce = Logger(train_ce_logger_path)
    logger_val_ce = Logger(val_ce_logger_path)

    val_acc = 0
    print('----------- distillation is ready -----------')
    w_origin = model.features[12].weight.clone()
    weight_diff = torch.zeros(args.epochs, requires_grad=False)
    for epoch in range(args.start_epoch, args.epochs):

        save_path = path + 'trained_model/' + 'distil_{}_{}_{}_{}/'.format(args.model, args.mode, args.dataset, args.date)
        check_dir(save_path)
        if args.save_per_epoch:
            save_dir = save_path + '{}.pth.tar'.format(epoch)
        else:
            save_dir = save_path + '{}_{}_{}.pth.tar'.format(args.dataset, args.mode, args.epochs)
        save_checkpoint({'epoch': epoch, 'state_dict': model.state_dict(), 'best_acc': val_acc, 'optimizer': optimizer.state_dict()}, save_dir)

        print(optimizer.param_groups[0]['lr'])

        weight_curr = model.features[12].weight
        weight_diff[epoch] = torch.sqrt(torch.mean((weight_curr - w_origin) ** 2))
        print('weight_diff', weight_diff[epoch])

        train_kd(model, optimizer, loss_fn_kd, train_loader, epoch, logger_train_mse, teacher_train, args.mode)

        val_loss, val_acc = evaluate_kd(model, val_loader, epoch, logger_val_mse, teacher_val, args.mode)
        scheduler.step(val_loss)

        torch.cuda.empty_cache()

        weight_diff_save_path = save_path + 'weight_diff.npy'.format(args.dataset, args.mode, args.epochs)
        np.save(weight_diff_save_path, weight_diff.detach().numpy())
    print(weight_diff)


    # load_checkpoint('/home/data1/chenyilan/KD/trained_model/distil_vgg16_fc1_coco_901/CE_150.pth.tar', model)
    ## fine-tune model to classify ##
    print('------------fine-tune model to classify is ready!------------\n')
    if args.classifier and args.mode != 'fc3':
        if args.model[:3] == 'vgg':
            if args.mode == 'conv':
                for param in model.features.parameters():
                    param.requires_grad = False
                optimizer = torch.optim.SGD(model.classifier.parameters(), 1e-3, momentum=args.momentum, weight_decay=args.weight_decay)
            elif args.mode == 'fc1':
                for param in model.features.parameters():
                    param.requires_grad = False
                for param in model.classifier[:1].parameters():
                    param.requires_grad = False
                optimizer = torch.optim.SGD(model.classifier[1:].parameters(), 1e-3, momentum=args.momentum, weight_decay=args.weight_decay)
            elif args.mode == 'fc2':
                for param in model.features.parameters():
                    param.requires_grad = False
                for param in model.classifier[:-1].parameters():
                    param.requires_grad = False
                optimizer = torch.optim.SGD(model.classifier[-1].parameters(), 1e-3, momentum=args.momentum, weight_decay=args.weight_decay)
        elif args.model[:6] == 'resnet':
            for param in model.features.parameters():
                param.requires_grad = False
            for param in model.classifier[:-1].parameters():
                param.requires_grad = False
            optimizer = torch.optim.SGD(model.classifier[-1].parameters(), 1e-3, momentum=args.momentum, weight_decay=args.weight_decay)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=10, verbose=False)

        for epoch in range(args.epochs, args.epochs + args.classify_epochs):

            # save_path = path + 'trained_model/' + 'distil_{}_{}_{}_{}/'.format(args.model, args.mode, args.dataset, args.date)
            # check_dir(save_path)
            # if args.save_per_epoch:
            #     save_dir = save_path + 'CE_{}.pth.tar'.format(epoch)
            # else:
            #     save_dir = save_path + 'CE_{}_{}_{}.pth.tar'.format(args.dataset, args.mode, args.epochs)
            # save_checkpoint({'epoch': epoch, 'state_dict': model.state_dict(), 'best_acc': val_acc, 'optimizer': optimizer.state_dict()}, save_dir)

            print(optimizer.param_groups[0]['lr'])
            train(model, optimizer, train_loader, epoch, logger_train_ce)

            val_loss, val_acc = evaluate(model, val_loader, epoch, logger_val_ce)
            scheduler.step(val_loss)

            torch.cuda.empty_cache()


def train_kd(model, optimizer, loss_fn_kd, dataloader, epoch, logger, teacher_train, mode):
    # set model to training mode
    model.train()
    MSE = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for i, (img_idx, train_batch, labels_batch) in enumerate(dataloader):
        # move to GPU if available
        train_batch, labels_batch = train_batch.cuda(args.gpu),labels_batch.cuda(args.gpu)
        # convert to torch Variables
        train_batch, labels_batch = Variable(train_batch), Variable(labels_batch)
        output_batch = model(train_batch)

        # compute model output, fetch teacher output, and compute KD loss
        if mode == 'conv':
            output_layer = model.features[:14](train_batch)
        elif mode == 'fc1':
            pre = model.features(train_batch)
            output_layer = model.classifier[:1](pre.view(pre.shape[0], -1))
        elif mode == 'fc2':
            pre = model.features(train_batch)
            output_layer = model.classifier[:4](pre.view(pre.shape[0], -1))
        elif mode == 'fc3':
            pre = model.features(train_batch)
            output_layer = model.classifier(pre.view(pre.shape[0], -1))
        # get one batch output from teacher_outputs list
        output_teacher = Variable(torch.from_numpy(teacher_train[img_idx]).cuda(args.gpu), requires_grad=False)
        mse = loss_fn_kd(output_layer, output_teacher)

        # clear previous gradients, compute gradients of all variables wrt loss
        optimizer.zero_grad()
        mse.backward()

        # performs updates using calculated gradients
        optimizer.step()
        acc1, acc5 = accuracy(output_batch, labels_batch, topk=(1, 5))

        # update the average loss
        top1.update(acc1[0], labels_batch.size(0))
        top5.update(acc5[0], labels_batch.size(0))
        MSE.update(mse.item(), labels_batch.size(0))

        if i % args.print_freq == 0:
            print(
                'Epoch: [{0}][{1}/{2}]\t''MSE {MSE.val:.4f} ({MSE.avg:.4f})\t''Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t''Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, i, len(dataloader), MSE=MSE,top1=top1, top5=top5))
    print('Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    log_dict = {'MSE': MSE.avg, 'top1_prec': top1.avg.item(), 'top5_prec': top5.avg.item()}
    set_tensorboard(log_dict, epoch, logger)


def evaluate_kd(model, dataloader, epoch, logger, teacher_val, mode):

    model.eval()
    # summary for current eval loop
    MSE = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # compute metrics over the dataset
    for i, (img_idx,data_batch, labels_batch) in enumerate(dataloader):

        # move to GPU if available
        data_batch, labels_batch = data_batch.cuda(args.gpu), labels_batch.cuda(args.gpu)
        # fetch the next evaluation batch
        data_batch, labels_batch = Variable(data_batch), Variable(labels_batch)
        output_batch = model(data_batch)

        # compute model output, fetch teacher output, and compute KD loss
        if mode == 'conv':
            output_layer = model.features[:14](data_batch)
        elif mode == 'fc1':
            pre = model.features(data_batch)
            output_layer = model.classifier[:1](pre.view(pre.shape[0], -1))
        elif mode == 'fc2':
            pre = model.features(data_batch)
            output_layer = model.classifier[:4](pre.view(pre.shape[0], -1))
        elif mode == 'fc3':
            pre = model.features(data_batch)
            output_layer = model.classifier(pre.view(pre.shape[0], -1))

        # get one batch output from teacher_outputs list
        output_teacher = Variable(torch.from_numpy(teacher_val[img_idx]).cuda(args.gpu), requires_grad=False)
        mse = loss_fn_kd(output_layer, output_teacher)

        acc1, acc5 = accuracy(output_batch, labels_batch, topk=(1, 5))
        MSE.update(mse.item(), labels_batch.size(0))
        top1.update(acc1[0], labels_batch.size(0))
        top5.update(acc5[0], labels_batch.size(0))

        if i % args.print_freq == 0:
            print(
                'Test: [{0}/{1}]\t''MSE {MSE.val:.4f} ({MSE.avg:.4f})\t''Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t''Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(dataloader), MSE=MSE, top1=top1, top5=top5))
    print('Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    log_dict = {'MSE': MSE.avg, 'top1_prec': top1.avg.item(), 'top5_prec': top5.avg.item()}
    set_tensorboard(log_dict, epoch, logger)

    return MSE.avg, top1.avg.item()


def train(model, optimizer, dataloader, epoch, logger):
    # set model to training mode
    model.train()
    CE = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for i, (img_idx, train_batch, labels_batch) in enumerate(dataloader):
        # move to GPU if available
        train_batch, labels_batch = train_batch.cuda(args.gpu), labels_batch.cuda(args.gpu)
        # convert to torch Variables
        train_batch, labels_batch = Variable(train_batch), Variable(labels_batch)
        output_batch = model(train_batch)

        ## compute loss ##
        ce = nn.CrossEntropyLoss()(output_batch, labels_batch)

        # clear previous gradients, compute gradients of all variables wrt loss
        optimizer.zero_grad()
        ce.backward()

        # performs updates using calculated gradients
        optimizer.step()
        acc1, acc5 = accuracy(output_batch, labels_batch, topk=(1, 5))

        # update the average loss
        top1.update(acc1[0], labels_batch.size(0))
        top5.update(acc5[0], labels_batch.size(0))
        CE.update(ce.item(), labels_batch.size(0))

        if i % args.print_freq == 0:
            print(
                'Epoch: [{0}][{1}/{2}]\t''CE {CE.val:.4f} ({CE.avg:.4f})\t''Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t''Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, i, len(dataloader), CE=CE, top1=top1, top5=top5))
    print('Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    log_dict = {'CE': CE.avg, 'top1_prec': top1.avg.item(), 'top5_prec': top5.avg.item()}
    set_tensorboard(log_dict, epoch, logger)


def evaluate(model, dataloader, epoch, logger):

    model.eval()
    # summary for current eval loop
    CE = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # compute metrics over the dataset
    for i, (img_idx,data_batch, labels_batch) in enumerate(dataloader):

        # move to GPU if available
        data_batch, labels_batch = data_batch.cuda(args.gpu), labels_batch.cuda(args.gpu)
        # fetch the next evaluation batch
        data_batch, labels_batch = Variable(data_batch), Variable(labels_batch)
        output_batch = model(data_batch)

        ce = nn.CrossEntropyLoss()(output_batch, labels_batch)

        acc1, acc5 = accuracy(output_batch, labels_batch, topk=(1, 5))
        CE.update(ce.item(), labels_batch.size(0))
        top1.update(acc1[0], labels_batch.size(0))
        top5.update(acc5[0], labels_batch.size(0))

        if i % args.print_freq == 0:
            print(
                'Test: [{0}/{1}]\t''CE {CE.val:.4f} ({CE.avg:.4f})\t''Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t''Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(dataloader), CE=CE, top1=top1, top5=top5))
    print('Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    log_dict = {'CE': CE.avg, 'top1_prec': top1.avg.item(), 'top5_prec': top5.avg.item()}
    set_tensorboard(log_dict, epoch, logger)

    return CE.avg, top1.avg.item()


def fetch_teacher_outputs(teacher_model, dataloader, model, mode, classes):
    # set teacher_model to evaluation mode
    teacher_model.eval()

    if model[:3] == 'vgg':
        if mode == 'fc1':
            teacher_fc = np.zeros((dataloader.dataset.size, 4096))
            for i, (idx, data_batch, labels_batch) in enumerate(dataloader):
                print(i, idx)
                data_batch, labels_batch = data_batch.cuda(args.gpu), labels_batch.cuda(args.gpu)
                pre = teacher_model.net.features(data_batch)
                pre = teacher_model.net.avgpool(pre)
                pre = torch.flatten(pre, 1)
                output_teacher_fc = teacher_model.net.classifier[:1](pre)
                teacher_fc[idx] = output_teacher_fc.data.cpu().numpy()

        elif mode == 'fc2':
            teacher_fc = np.zeros((dataloader.dataset.size, 4096))
            for i, (idx, data_batch, labels_batch) in enumerate(dataloader):
                print(i, idx)
                data_batch, labels_batch = data_batch.cuda(args.gpu), labels_batch.cuda(args.gpu)
                pre = teacher_model.net.features(data_batch)
                pre = teacher_model.net.avgpool(pre)
                pre = torch.flatten(pre, 1)
                output_teacher_fc = teacher_model.net.classifier[:4](pre)
                teacher_fc[idx] = output_teacher_fc.data.cpu().numpy()

        elif mode == 'fc3':
            teacher_fc = np.zeros((dataloader.dataset.size, classes))
            for i, (idx, data_batch, labels_batch) in enumerate(dataloader):
                print(i, idx)
                data_batch, labels_batch = data_batch.cuda(args.gpu), labels_batch.cuda(args.gpu)
                pre = teacher_model.net.features(data_batch)
                pre = teacher_model.net.avgpool(pre)
                pre = torch.flatten(pre, 1)
                output_teacher_fc = teacher_model.net.classifier(pre)
                teacher_fc[idx] = output_teacher_fc.data.cpu().numpy()

        return teacher_fc

    # elif model[:6] == 'resnet':
    #     teacher_fc = np.zeros((dataloader.dataset.size, 4096))
    #     for i, (idx, data_batch, labels_batch) in enumerate(dataloader):
    #         print(i, idx)
    #         data_batch, labels_batch = data_batch.cuda(args.gpu), labels_batch.cuda(args.gpu)
    #         pre = teacher_model.net.avgpool(teacher_model.net.layer4(teacher_model.net.layer3(teacher_model.net.layer2(teacher_model.net.layer1(
    #             teacher_model.net.maxpool(teacher_model.net.relu(teacher_model.net.bn1(teacher_model.net.conv1(data_batch)))))))))
    #         pre = pre.view(pre.shape[0], -1)
    #         output_teacher_batch = teacher_model.net.fc[:4](pre).data.cpu().numpy()  # 2rd fc before relu
    #         teacher_fc[idx] = output_teacher_batch
    #         torch.cuda.empty_cache()
    #     return teacher_fc


def loss_fn_kd(output, output_teacher):
    alpha = args.alpha
    # T = args.temperature
    mse_loss = nn.MSELoss()(output, alpha * output_teacher.float())
    # KD_loss = mse_loss * (alpha ) + crossentropy_loss* (1. - alpha)
    return mse_loss


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer=None):

    if not os.path.exists(checkpoint):
        raise("File doesn't exist {}".format(checkpoint))
    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint)
    else:
        # this helps avoid errors when loading single-GPU-trained weights onto CPU-model
        checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)

    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint


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


def set_tensorboard(log_dict, epoch, logger):
    # set for tensorboard
    info = log_dict
    for tag, value in info.items():
        logger.scalar_summary(tag, value, epoch + 1)
    return

def check_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)

distillation(path, args)