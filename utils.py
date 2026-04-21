import os
import time
import datetime
import torch
import torch.nn as nn
import torchvision.utils as vutils
import numpy as np
import random
import piq
import network


def set_device(opt):
    device = torch.device("cpu")
    if opt.use_gpu:
        for gpu_id in opt.gpu_ids:
            assert 0 <= gpu_id < torch.cuda.device_count()
        device = torch.device(f"cuda:{opt.gpu_ids[0]}" if torch.cuda.is_available() else "cpu")
        torch.backends.cudnn.benchmark = opt.cudnn_benchmark
    if torch.cuda.is_available() and not opt.use_gpu:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    return device


def set_random_seed(seed=None):
    if seed is None:
        seed = (
                os.getpid()
                + int(datetime.datetime.now().strftime("%S%f"))
                + int.from_bytes(os.urandom(2), "big")
        )
        print('Using a generated random seed {}'.format(seed))
    else:
        print('Using a pre-specified random seed {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def load_model(opt, hiding, device):
    if hiding:
        if opt.dual_net:
            model = network.HideNet2(in_ch=3, out_ch=3, d=64)
        else:
            model = network.HideNet1(in_ch=6, out_ch=3, d=64)
    else:
        model = network.RevealNet(in_ch=3, out_ch=3, d=64)

    if opt.pretrained:
        model_path = opt.Hnet_path if hiding else opt.Rnet_path
        model.load_state_dict(torch.load(model_path))
        print("Model loaded")
    else:
        model.apply(weights_init)
        print("Model created")

    if len(opt.gpu_ids) > 1:
        model = torch.nn.DataParallel(model).to(device)

    model = model.to(device)

    return model


def save_model(opt, net_H, net_R, epoch, val_psnr):
    if epoch > opt.save_after_epoch:
        model_name_H = 'modelH_ep%03d_psnr%.4f.pth' % (epoch, val_psnr[0])
        model_name_R = 'modelR_ep%03d_psnr%.4f.pth' % (epoch, val_psnr[1])
        model_path_H = os.path.join(opt.ckpt_dir, model_name_H)
        model_path_R = os.path.join(opt.ckpt_dir, model_name_R)
        torch.save(net_H.state_dict(), model_path_H)
        torch.save(net_R.state_dict(), model_path_R)


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(str(net))
    print('Total number of network parameters: %d' % num_params)


def write_metrics(log_info, log_path):
    if not os.path.exists(log_path):
        fp = open(log_path, "w")
        fp.writelines(log_info + "\n")
    else:
        with open(log_path, 'a+') as f:
            f.writelines(log_info + '\n')


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def make_output_dirs(opt, test=False):
    try:
        if not test:
            opt.ckpt_dir = os.path.join(opt.out_dir, 'ckpt')
            opt.train_img_dir = os.path.join(opt.out_dir, 'train_imgs')
            opt.val_img_dir = os.path.join(opt.out_dir, 'val_imgs')
            check_path(opt.ckpt_dir)
            check_path(opt.train_img_dir)
            check_path(opt.val_img_dir)
            opt.train_metric_path = os.path.join(opt.out_dir, 'train_bs%d_metric.txt' % (opt.batch_size))
            opt.val_metric_path = os.path.join(opt.out_dir, 'val_bs%d_metric.txt' % (opt.batch_size))
        else:
            opt.test_img_dir = os.path.join(opt.out_dir, 'test_imgs')
            check_path(opt.test_img_dir)
            opt.test_metric_path = os.path.join(opt.out_dir, 'test_%d_metric.txt' % (opt.batch_size))
    except OSError:
        print("mkdir failed!")


def get_perm_index(num):
    idx_org = np.arange(num)
    idx_perm = np.arange(num)
    while num > 1 and (idx_org == idx_perm).any():
        random.shuffle(idx_perm)

    return idx_perm


def save_images(img_list, epoch, i, batch_size, save_path):
    img = torch.cat(img_list, 0)
    img_name = 'img_ep%03d_batch%04d.png' % (epoch, i)
    img_path = os.path.join(save_path, img_name)
    vutils.save_image(img, img_path, nrow=batch_size, padding=1, normalize=True)


def avg_pixel_disc(x, y):
    d = (x - y) * 255
    d = d.abs().int()
    d = d.sum() / d.numel()

    return d


def compute_four_metrics(img_list):
    cover_psnr = piq.psnr(img_list[0], img_list[1], data_range=1.0).clone().detach().cpu().numpy()
    secret_psnr = piq.psnr(img_list[2], img_list[3], data_range=1.0).clone().detach().cpu().numpy()

    cover_ssim = piq.ssim(img_list[0], img_list[1], data_range=1.0).clone().detach().cpu().numpy()
    secret_ssim = piq.ssim(img_list[2], img_list[3], data_range=1.0).clone().detach().cpu().numpy()

    cover_apd = avg_pixel_disc(img_list[0], img_list[1]).clone().detach().cpu().numpy()
    secret_apd = avg_pixel_disc(img_list[2], img_list[3]).clone().detach().cpu().numpy()

    cover_LPIPS = piq.LPIPS()(img_list[0], img_list[1]).clone().detach().cpu().numpy()
    secret_LPIPS = piq.LPIPS()(img_list[2], img_list[3]).clone().detach().cpu().numpy()

    metric = [cover_psnr, cover_ssim, cover_apd, cover_LPIPS, secret_psnr, secret_ssim, secret_apd, secret_LPIPS]

    return metric


def compute_two_metrics(img_list):
    cover_psnr = piq.psnr(img_list[0], img_list[1], data_range=1.0).clone().detach().cpu().numpy()
    secret_psnr = piq.psnr(img_list[2], img_list[3], data_range=1.0).clone().detach().cpu().numpy()

    cover_ms_ssim = piq.multi_scale_ssim(img_list[0], img_list[1], data_range=1.0).clone().detach().cpu().numpy()
    secret_ms_ssim = piq.multi_scale_ssim(img_list[2], img_list[3], data_range=1.0).clone().detach().cpu().numpy()

    metric = [cover_psnr, cover_ms_ssim, secret_psnr, secret_ms_ssim]

    return metric


class AverageMeter(object):
    """
    Computes and stores the average and current value.
    """

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


class LossLog(object):
    def __init__(self):
        self.loss_H = AverageMeter()
        self.loss_R = AverageMeter()
        self.loss_sum = AverageMeter()
    
    def update_loss(self, loss_H, loss_R, loss_sum, batch_size):
        self.loss_H.update(loss_H, batch_size)
        self.loss_R.update(loss_R, batch_size)
        self.loss_sum.update(loss_sum, batch_size)


class TimeLog(object):
    def __init__(self, epochs, epoch, iterations):
        self.iters_left = (epochs - epoch) * iterations
        self.prev_time = time.time()
        self.time = AverageMeter()
        self.time_left = 0
    
    def get_time_stats(self):
        self.iters_left = self.iters_left - 1
        time_cost = time.time() - self.prev_time
        self.time.update(time_cost)
        self.time_left = datetime.timedelta(seconds=self.iters_left * self.time.avg)
        self.prev_time = time.time()
    

class MetricLog(object):
    def __init__(self, path):
        self.metric_path = path
        self.psnr_c = AverageMeter()
        self.ssim_c = AverageMeter()
        self.psnr_s = AverageMeter()
        self.ssim_s = AverageMeter()
    
    def compute_update_metric(self, img_list):
        metric = compute_two_metrics(img_list)
        self.psnr_c.update(metric[0])
        self.ssim_c.update(metric[1])
        self.psnr_s.update(metric[2])
        self.ssim_s.update(metric[3])
    
    def write_metric_head(self):
        log = '         Cover Image          Secret Image\n'
        log = log + '       PSNR     MS-SSIM     PSNR     MS-SSIM'
        write_metrics(log, self.metric_path)
    
    def write_metric_line(self, epoch):
        log = '%4d: %.4f, %.6f,   %.4f, %.6f' % (
            epoch, self.psnr_c.avg, self.ssim_c.avg, self.psnr_s.avg, self.ssim_s.avg)
        write_metrics(log, self.metric_path)


def print_screen(save_log_freq, epochs, epoch, iterations, idx, loss_log, time_log, metric_log, batch=False, stage='Training'):
    if batch and (idx + 1) % save_log_freq == 0:
        print("[Epoch {:{}d}/{:d}] [Batch {:{}d}/{:d}] [Loss_H: {:.4f}] [Loss_R: {:.4f}] [Loss_sum: {:.4f}] Time cost: {:.6f} Time left: {}".format(
            (epoch + 1), len(str(epochs)), epochs, idx + 1, len(str(iterations)), iterations, 
            loss_log.loss_H.val, loss_log.loss_R.val, loss_log.loss_sum.val, time_log.time.val, time_log.time_left))
    elif not batch:
        print("[Epoch {:{}d}/{:d}] [{:8s} stats][Loss_H: {:.4f}] [Loss_R: {:.4f}] [Loss_sum: {:.4f}] Time cost: {:.6f}".format(
            (epoch + 1), len(str(epochs)), epochs, stage, loss_log.loss_H.avg, loss_log.loss_R.avg, loss_log.loss_sum.avg, time_log.time.sum))
        print("[Epoch {:{}d}/{:d}] [{:8s} stats][PSNR_C: {:.4f}] [MSSSIM_C: {:.6f}] [PSNR_S: {:.4f}] [MSSSIM_S: {:.6f}]".format(
            (epoch + 1), len(str(epochs)), epochs, stage, metric_log.psnr_c.avg, metric_log.ssim_c.avg, metric_log.psnr_s.avg, metric_log.ssim_s.avg))    

