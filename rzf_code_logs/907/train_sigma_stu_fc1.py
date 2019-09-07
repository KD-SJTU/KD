import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'    # change GPU here
import sys
sys.path.extend(['/home/data2/rzf/KD'])
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import math
import torch.nn.functional as F
import argparse
from PIL import Image
import matplotlib.pyplot as plt
import datetime as dt


parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="student_908") ## fc or conv ##
parser.add_argument("--capacity_layer", type=str, default='fc1')
parser.add_argument("--student", type=bool, default=True) ## student network or teacher##
parser.add_argument("--pretrain", type=bool, default=False)
parser.add_argument('--dataset', default='CUB', type=str)
parser.add_argument('--Top10', default=False, type=bool)
parser.add_argument('--threshold', type=int, default=-3.0, help='threshold of feature loss for selecting sigma')
parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
parser.add_argument("--batch_size", type=int, default=128, help="batch size")
parser.add_argument('--workers', default=8, type=int)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--gpu', default=0, type=int, help='Set it to 0. Change it above before importing torch.')
parser.add_argument('--print_freq', '-p', default=1, type=int, metavar='N')
parser.add_argument('--classes', default=200, type=int)
parser.add_argument('--checkpoint_step', default=3, type=int) ## the step for checkpoint ##
parser.add_argument('--lambda_init', type=float, default=6, help='lambda can be changed.')
parser.add_argument('--sigma_init_decay', type=float, default=1e-3, help='the initialization of sigma is setting same value of all sigma')
parser.add_argument("--image_size", type=int, default=224)
parser.add_argument('--sigma_size', type=int, default=16, help='if the size of image is 224 and the size of sigma is 16, then 1 sigma have 14*14 pixels.')
parser.add_argument('--sigma_upsample_size', type=int, default=18, help='')
parser.add_argument('--lambda_change_ratio', type=float, default=0.3, help='')
parser.add_argument('--epoch', type=int, default=60, help='number of training iterations for sigma')
parser.add_argument('--pic_num', type=int, default=500, help='number of imgs to select threshold')
parser.add_argument('--sigma_num', type=int, default=1000, help='number of imgs to calculate sigma_f')
parser.add_argument('--trick', default=True, type=bool)
args = parser.parse_args()

## get the path for project ##
path = '/home/data2/rzf/KD/'
print(vars(args))
args.seed = 0               # seed set to same number
print('seed:', args.seed)

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


from model.models import *
from function.get_model_noise_layer import get_model_noise_layer
from function.dataset import *


def train_sigma(path,args):

    ## get dataset path ##
    dataset_root = path + 'dataset/{}'.format(args.dataset)
    traindir, segdir = dataset_root + '/train', dataset_root + '/CUB200_seg/train'
    # traindir, segdir = '/home/data1/chenyilan/KD/dataset/coco_tiny/train', '/home/data1/chenyilan/KD/dataset/coco_tiny_seg/train'

    ## prepare dataset ##
    dataloader = torch.utils.data.DataLoader(SelectDataSet(traindir, sample_num=args.sigma_num), batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True, worker_init_fn=_init_fn)

    ## define loss function ##
    criterion = torch.nn.MSELoss().cuda(args.gpu)
    relu_method = nn.ReLU()

    ## define model ##
    if args.student:
        model = student_network_bn(args.classes, seed=args.seed)
    else:
        if args.pretrain:
            model = vgg16_pretrained(out_planes=args.classes, seed=args.seed)
        else:
            model = vgg16_without_pretrained(out_planes=args.classes, seed=args.seed)
    if args.gpu is not None:
        model = model.cuda(args.gpu)


    checkpoint_root = '/home/data2/rzf/KD/trained_model/student_908_CUB/'
    # checkpoint_num = len(os.listdir(checkpoint_root))
    checkpoint_num = 150

    ## get sigma_f for each checkpoint ##
    sigma_f_list_root = path + 'npy_files/sigma_f/' + str(args.dataset) + '_' + str(args.mode) + '_' + str(args.capacity_layer) + '/'
    checkdir(sigma_f_list_root)
    sigma_f_list_path = sigma_f_list_root + 'sigma_f.npy'
    sigma_f_num = int(checkpoint_num / args.checkpoint_step)
    print("sigma_f_num", sigma_f_num)
    if os.path.exists(sigma_f_list_path):
        sigma_f_list = np.load(sigma_f_list_path)
    else:

        sigma_f_list = np.zeros((sigma_f_num, 1))
        # calculate sigma_f for every checkpoint 0 3 6 9...
        for idx in range(sigma_f_num):
            checkpoint_idx = args.checkpoint_step * idx
            checkpoint = checkpoint_root + str(checkpoint_idx) + '.pth.tar'
            print("checkpoint", checkpoint)
            load_checkpoint(checkpoint, model)
            model.eval()
            sigma_f = 0
            for batch_id, (img_idx, image_name, image, label) in enumerate(dataloader):
                if batch_id == 0 and image_name != ['106.Horned_Puffin/Horned_Puffin_0016_100993.jpg']:
                    sys.exit("random seed error")
                # print(image_name)

                noise_layer = get_model_noise_layer(args.gpu, torch.zeros(1, 1, args.sigma_size, args.sigma_size).size(), args.sigma_init_decay * 10, args.image_size)
                unit_vector = torch.ones(args.batch_size, 1, args.sigma_size, args.sigma_size).cuda(args.gpu)
                unit_noise = torch.randn(args.batch_size, 3, args.image_size, args.image_size).cuda(args.gpu)
                image = image.cuda(args.gpu)
                noise_image, penalty = noise_layer(image, unit_vector, unit_noise)
                noise_feature = get_feature(noise_image, model, args.student, layer=args.capacity_layer)
                origin_feature = get_feature(image, model, args.student, layer=args.capacity_layer)
                origin_feature = origin_feature.detach()

                noise_feature = relu_method(noise_feature)
                origin_feature = relu_method(origin_feature)
                origin_feature = origin_feature.expand(noise_feature.size())

                feature_loss = criterion(noise_feature, origin_feature)
                print("batch id", batch_id, "image id", img_idx, "sigma_f", feature_loss)

                sigma_f += feature_loss.data.cpu()

            sigma_f_list[idx] = sigma_f/args.sigma_num
            np.save(sigma_f_list_path, sigma_f_list)
            print("sigma_f_list", sigma_f_list)



    print('---------------- sigma_f is ready ------------------')
    #train a sigma matrix for every checkpoint of every image of every layer
    # all parameters of net is fixed and dropout will lose efficiency
    lambda_param = args.lambda_init
    result_root = os.path.join(path, 'sigma_result')
    result_path = result_root + '/' + str(args.dataset) + '_' + str(args.mode) + '_' + str(args.capacity_layer) + '/'
    checkdir(result_root)
    checkdir(result_path)

    Select_DataSet = SelectDataSet(traindir, sample_num=args.pic_num)
    name_path = result_path + '/full_name_list.txt'
    fw = open(name_path, 'w')  # 将要输出保存的文件地址
    for name in Select_DataSet.name_list:
        fw.write(name)  # 将字符串写入文件中
        fw.write("\n")  # 换行
    fw.close()
    dataloader = torch.utils.data.DataLoader(Select_DataSet, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True, worker_init_fn=_init_fn)

    effect_count, non_effect_count = 0, 0
    ## pic --> checkpoint ##
    ## each img has one folder ##
    for batch_id, (id, img_name, image, label) in enumerate(dataloader):
        if batch_id == 0 and img_name != ['106.Horned_Puffin/Horned_Puffin_0016_100993.jpg']:
            sys.exit("random seed error")
        if effect_count >= 50:
            break
        print("effect count", effect_count, "\tnon_effect count", non_effect_count, '\n')

        image = image.cuda(args.gpu)
        ## define path ##
        result_folder = os.path.join(result_path, 'img_' + str(id.data.cpu().numpy()[0]) + '_non_effect')
        plot_img_path = os.path.join(result_folder,'plot')
        npy_save_path = os.path.join(result_folder,'npy_files') ## to save feature_loss,penalty_loss,ect.##
        checkdir(result_folder), checkdir(plot_img_path), checkdir(npy_save_path)

        ## save origin img and prepare npy_files ##
        get_origin_img(traindir, img_name, args.image_size, result_folder)
        feature_loss_matrix, penalty_loss_matrix, entropy_matrix = np.zeros((sigma_f_num, args.epoch)), np.zeros((sigma_f_num, args.epoch)), np.zeros((sigma_f_num, args.epoch))
        # save the sigma nearest to the threshold
        sigma_matrix = np.zeros((sigma_f_num, args.sigma_size, args.sigma_size))

        # get seg, down-sample it to a bigger size than sigma size and get 16*16 size
        seg_img = torch.from_numpy(np.array(Image.open(segdir + '/' + img_name[0].replace('jpg', 'png')).convert('L'))).cuda(args.gpu)
        # seg_img = torch.from_numpy(np.array(Image.open(segdir + '/' + img_name[0]))).cuda()
        seg_img = F.interpolate(seg_img[np.newaxis, np.newaxis, :, :].float(), size=args.sigma_upsample_size, mode='nearest').squeeze()
        a = seg_img.cpu().numpy()
        print(a.shape, a)
        seg_pos1 = int((args.sigma_upsample_size - args.sigma_size) * 0.5)  # 1
        seg_pos2 = int((args.sigma_upsample_size + args.sigma_size) * 0.5)  # 17
        seg_img = seg_img[seg_pos1:seg_pos2, seg_pos1:seg_pos2]
        a = seg_img.cpu().numpy()
        print(a.shape, a)
        foreground_pos = (seg_img > 0).nonzero().cuda(args.gpu)
        background_pos = (seg_img == 0).nonzero().cuda(args.gpu)

        # train for every checkpoint
        for idx in reversed(range(sigma_f_num)):
        # for idx in [49, 30, 10, 0]:
            checkpoint_idx = args.checkpoint_step * idx
            ## prepare path ##
            sigma_f = torch.from_numpy(sigma_f_list[idx].astype(np.float32)).cuda(args.gpu)
            print('\n', id, img_name)
            print("effect count", effect_count, "\tnon_effect count", non_effect_count, '\n')
            print("sigma_f", sigma_f)
            plot_checkpoint_path = os.path.join(plot_img_path, 'checkpoint_' + str(checkpoint_idx))
            checkdir(plot_checkpoint_path)
            sigma_data_list = []
            train_feature_loss,  train_penalty_loss = AverageMeter(), AverageMeter()

            ## load models ##
            checkpoint = checkpoint_root + str(checkpoint_idx) + '.pth.tar'
            load_checkpoint(checkpoint, model)
            print(checkpoint)
            model.eval()

            noise_layer = get_model_noise_layer(args.gpu, torch.zeros(1, 1, args.sigma_size, args.sigma_size).size(), args.sigma_init_decay, args.image_size)
            optimizer = torch.optim.SGD([{"params": noise_layer.parameters(), 'lr': args.lr, 'initial_lr': args.lr}])

            # train the sigma matrix
            for epoch in range(args.epoch):

                # the parameter lambda is rising over epochs
                # lambda_param = labmda_init * math.e ** (args.lambda_change_ratio * epoch / args.epoch)
                print("lambda", lambda_param)

                train_feature_loss.reset()
                train_penalty_loss.reset()

                # train
                noise_layer.train()
                params_data = optimizer.param_groups[0]['params'][0].data
                sigma_data_list.append(np.array(params_data.data.cpu()))

                params_data = torch.log(params_data) + 0.5 * torch.log(torch.tensor(2*math.pi*math.e))
                visual_data = F.interpolate(params_data, size=args.image_size, mode='nearest')[0][0].data.cpu()
                print('min:',np.min(np.array(visual_data))), print('max:',np.max(np.array(visual_data)))
                visual_path = os.path.join(plot_checkpoint_path,'sigma_'+str(epoch)+'.jpg')
                plot_feature_new(visual_data, visual_path)

                unit_vector = torch.ones(args.batch_size,1,args.sigma_size,args.sigma_size).cuda(args.gpu)
                unit_noise = torch.randn(args.batch_size,3,args.image_size,args.image_size).cuda(args.gpu)
                noise_image, penalty = noise_layer(image, unit_vector, unit_noise)
                noise_feature = get_feature(noise_image, model, args.student, layer=args.capacity_layer)
                origin_feature = get_feature(image, model, args.student, layer=args.capacity_layer).detach()

                ## cal loss ##
                noise_feature = relu_method(noise_feature)
                origin_feature = relu_method(origin_feature)
                origin_feature = origin_feature.expand(noise_feature.size())

                feature_loss = criterion(noise_feature, origin_feature)/sigma_f
                penalty_loss = -penalty*lambda_param
                loss = feature_loss + penalty_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (epoch) % (args.print_freq) == 0:
                    print("Train: [" + str(epoch) + "/" + str(args.epoch) + "]" + "\n"+'feature_loss: '+str(float(feature_loss))+"\n"
                          +'entropy: '+str(float(penalty))+"\n")

                train_feature_loss.update(feature_loss.data.cpu())
                train_penalty_loss.update(penalty_loss.data.cpu())
                feature_loss_matrix[idx,epoch], penalty_loss_matrix[idx,epoch], entropy_matrix[idx,epoch] = train_feature_loss.avg, train_penalty_loss.avg, penalty

            ## the last epoch need to be saved ##
            params_data = optimizer.param_groups[0]['params'][0].data
            sigma_data_list.append(np.array(params_data.data.cpu()))
            params_data = torch.log(params_data) + 0.5 * torch.log(torch.tensor(2 * math.pi * math.e))
            visual_data = F.interpolate(params_data, size=args.image_size, mode='nearest')[0][0].data.cpu()
            visual_path = os.path.join(plot_checkpoint_path, 'sigma_' + str(epoch +1) + '.jpg')
            plot_feature_new(visual_data, visual_path)

            ## save the epoch nearest to threshold ##
            best_idx = np.argmin(np.abs(entropy_matrix[idx] - args.threshold))
            best_sigma = sigma_data_list[best_idx][0][0]
            print("best idx", best_idx), print("best sigma", best_sigma)
            params_data = torch.log(torch.from_numpy(best_sigma[np.newaxis,np.newaxis,:,:])) + 0.5 * torch.log(torch.tensor(2 * math.pi * math.e))
            visual_data = F.interpolate(params_data, size=args.image_size, mode='nearest')[0][0].data.cpu()
            visual_path = os.path.join(plot_checkpoint_path, 'sigma_best_' + str(best_idx) + '.jpg')
            plot_feature_new(visual_data, visual_path)

            ## select effect samples when checkpoint_idx >= 40 ##
            if checkpoint_idx >= 40:
                effect_flag = select_effect_smaples(best_sigma, foreground_pos, background_pos)
                if not effect_flag:
                    non_effect_count += 1
                    name_path = result_path + '/non_effect_name_list.txt'
                    fw = open(name_path, 'a')  # 将要输出保存的文件地址
                    fw.write(img_name[0])  # 将字符串写入文件中
                    fw.write("\n")  # 换行
                    fw.close()
                    break
            ## save the best sigma ##
            sigma_matrix[idx] = best_sigma

            # if effective sample, save data
            if checkpoint_idx == 0:
                np.save(npy_save_path+'/'+'feature_loss.npy', feature_loss_matrix)
                np.save(npy_save_path +'/'+ 'penalty_loss.npy', penalty_loss_matrix)
                np.save(npy_save_path + '/' + 'entropy.npy', entropy_matrix)
                np.save(npy_save_path + '/' + 'best_sigma.npy', sigma_matrix)
                name_path = result_path + '/effect_name_list.txt'
                fw = open(name_path, 'a')  # 将要输出保存的文件地址
                fw.write(img_name[0])  # 将字符串写入文件中
                fw.write("\n")  # 换行
                fw.close()
                os.rename(result_folder, result_folder.replace('non_effect', 'effect'))
                effect_count += 1



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


def load_checkpoint(checkpoint, model, optimizer=None):

    if not os.path.exists(checkpoint):
        raise("File doesn't exist {}".format(checkpoint))
    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint, map_location='cuda:0')
    else:
        # this helps avoid errors when loading single-GPU-trained weights onto CPU-model
        checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)

    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])


def checkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def plot_feature(feature,visual_path):

    feature_size = feature.size()
    new_feature = np.zeros((feature_size[0],feature_size[1],3))
    feature = np.array(feature)
    new_feature[:,:,0] = feature
    new_feature[:,:,1] = feature
    new_feature[:,:,2] = feature
    max = -2.3
    min = -5.5
    new_feature = 255*(new_feature-min)/(max-min)
    new_feature = np.uint8(new_feature)
    feature_visual = Image.fromarray(new_feature, 'RGB')
    feature_visual.save(visual_path)


def plot_feature_new(feature,visual_path):
    feature = np.array(feature)

    fig, ax = plt.subplots()
    # my_dpi = 96
    # plt.figure(figsize=(224 / my_dpi, 224 / my_dpi), dpi=my_dpi)
    im = ax.imshow(feature, cmap=plt.get_cmap('hot'), interpolation='nearest',
                   vmin=-5.4, vmax=-2.3)
    fig.colorbar(im)
    plt.savefig(visual_path, dpi=70)
    plt.close()


def get_feature(input, model, student, layer):

    if student:
        if layer == 'conv':
            decoder_feature = model.features[:14](input)
        elif layer == 'fc1':
            pre = model.features(input)
            pre = pre.view(pre.shape[0], -1)
            decoder_feature = model.classifier[:1](pre)
        elif layer == 'fc2':
            pre = model.features(input)
            pre = pre.view(pre.shape[0], -1)
            decoder_feature = model.classifier[:4](pre)
        elif layer == 'fc3':
            pre = model.features(input)
            pre = pre.view(pre.shape[0], -1)
            decoder_feature = model.classifier(pre)
        else:
            sys.exit('Layer Error')
    else:
        if layer == 'conv':
            decoder_feature = model.net.features[:29](input)
        elif layer == 'fc1':
            pre = model.net.features(input)
            pre = model.net.avgpool(pre)
            pre = torch.flatten(pre, 1)
            decoder_feature = model.net.classifier[:1](pre)
        elif layer == 'fc2':
            pre = model.net.features(input)
            pre = model.net.avgpool(pre)
            pre = torch.flatten(pre, 1)
            decoder_feature = model.net.classifier[:4](pre)
        elif layer == 'fc3':
            pre = model.net.features(input)
            pre = model.net.avgpool(pre)
            pre = torch.flatten(pre, 1)
            decoder_feature = model.net.classifier(pre)
        else:
            sys.exit('Layer Error')

    return decoder_feature


def get_origin_img(dir, image_name, image_size, save_root):
    img = Image.open(dir+'/'+image_name[0])
    img = img.resize((image_size,image_size))
    img.save(save_root+'/'+image_name[0].split('/')[1])


def get_fore_back(best_sigma, img_name, segdir):

    best_sigma = F.interpolate(torch.from_numpy(best_sigma[np.newaxis,np.newaxis,:,:]), size=args.image_size,mode='nearest')
    seg_img = np.array(Image.open(segdir + '/' + img_name[0].replace('jpg', 'png') ).convert('L'))
    seg_img = torch.from_numpy(seg_img)

    foreground_sigma = best_sigma[:, :, seg_img > 0].flatten()
    print("foreground sigma", foreground_sigma.size(), foreground_sigma)
    background_sigma = best_sigma[:, :, seg_img == 0].flatten()
    print("background sigma", background_sigma.size(), background_sigma)

    return np.array(foreground_sigma.data.cpu()), np.array(background_sigma.data.cpu())


def select_effect_smaples(best_sigma, foreground_pos, background_pos):

    best_sigma = torch.from_numpy(best_sigma).cuda(args.gpu)
    best_entropy = torch.log(best_sigma) + 0.5 * torch.log(torch.tensor(2 * math.pi * math.e))
    foreground_entropy = best_entropy[foreground_pos[:, 0], foreground_pos[:, 1]].flatten()
    background_entropy = best_entropy[background_pos[:, 0], background_pos[:, 1]].flatten()
    print("foreground entropy", foreground_entropy.size(), foreground_entropy)
    print("background entropy", background_entropy.size(), background_entropy)

    if args.Top10:
        foreground_entropy = foreground_entropy[np.argsort(foreground_entropy)[: int(len(foreground_entropy) * 0.1)]]
        print("top 10% foreground entropy", foreground_entropy.size(), foreground_entropy)
        background_entropy = background_entropy[np.argsort(background_entropy)[: int(len(background_entropy) * 0.1)]]
        print("top 10% background entropy", background_entropy.size(), background_entropy)

    foreground_mean = torch.mean(foreground_entropy)
    background_mean = torch.mean(background_entropy)
    entropy_diff = background_mean - foreground_mean
    print("foreground mean", foreground_mean, "\tbackground mean", background_mean)
    print('entropy_diff', entropy_diff, '\n')
    if entropy_diff < 0 or torch.isnan(entropy_diff):
        effect_flag = False
    else:
        effect_flag = True

    return effect_flag


def _init_fn(worker_id):
    np.random.seed(args.seed + worker_id)


train_sigma(path, args)