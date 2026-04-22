import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import utils
from utils import LossLog, TimeLog, MetricLog
from dataset import steg_data_loader
from ast import literal_eval as arg_bool
import wandb


def parse_opts():
    parser = argparse.ArgumentParser()
    # GPU parameters
    parser.add_argument('--use_gpu', type=arg_bool, default=True, choices=[True, False], help='use GPU or not')
    parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0], help='GPU IDs')
    # Dataset parameters
    parser.add_argument('--data_dir', type=str, default='/home/liming/pro/DHVC/data/', help='dataset root directory')
    parser.add_argument('--use_aug', type=arg_bool, default=True, choices=[True, False], help='use data augmentation or not')
    parser.add_argument('--image_height', type=int, default=256, help='hight for resized images')
    parser.add_argument('--image_width', type=int, default=256, help='width for resized images')
    # Model parameters
    parser.add_argument('--pretrained', type=arg_bool, default=False, choices=[True, False], help='use pretrained model or not')
    parser.add_argument('--dual_net', type=arg_bool, default=True, choices=[True, False], help='use dual encoders for hiding net or not')
    parser.add_argument('--Hnet_path', default='', help="path to Hidingnet (to continue training)")
    parser.add_argument('--Rnet_path', default='', help="path to Revealnet (to continue training)")
    # Training parameters
    parser.add_argument('--epochs', type=int, default=1, help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for adam. default=0.999')
    parser.add_argument('--workers', type=int, default=4, help='number of cpu threads')
    parser.add_argument('--lambda_H', type=float, default=1.0, help='loss weight of hiding net')
    parser.add_argument('--lambda_R', type=float, default=0.75, help='loss weight of revealing net')
    parser.add_argument('--use_wandb', type=arg_bool, default=False, choices=[True, False], help='wandb type')
    # Reproducibility
    parser.add_argument('--use_repro', type=arg_bool, default=True, choices=[True, False], help='use reproducibility or not')
    parser.add_argument('--seed', type=int, default=1000, help='random seed for reproducibility')
    parser.add_argument('--cudnn_benchmark', type=arg_bool, default=True, choices=[True, False], help='True for unchanged input data')
    # Output parameters
    parser.add_argument('--out_dir', default='/home/liming/pro/steg/DDH/DDH_2/output/', help='output directory')
    parser.add_argument('--save_log_freq', type=int, default=10, help='the frequency of printing the log on the console')
    parser.add_argument('--save_img_freq', type=int, default=100, help='the frequency of saving the resultPic')
    parser.add_argument('--save_after_epoch', type=int, default=0, help='save checkpoints after certain epochs')

    opts = parser.parse_args()

    print('Parameters:')
    print('\n'.join(f'{k:<20} {v}' for k, v in vars(opts).items()), '\n')

    return opts


def main():
    # Preparation
    opt = parse_opts()

    if opt.use_wandb:
        wandb.login()
        wandb.init(project="DeepHiding")
        wandb.watch_called = False

    device = utils.set_device(opt)
    if opt.use_repro:
        utils.set_random_seed(None if opt.seed < 0 else opt.seed)

    utils.make_output_dirs(opt)

    # Load the dataset
    train_loader = steg_data_loader('train', opt)
    val_loader = steg_data_loader('val', opt)
    
    # Build the model
    net_H = utils.load_model(opt, True, device)
    net_R = utils.load_model(opt, False, device)
    utils.print_network(net_H)
    utils.print_network(net_R)

    # Loss functions
    criterion = nn.MSELoss().to(device)

    # Optimizers
    optim_H = optim.Adam(net_H.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
    scheduler_H = optim.lr_scheduler.MultiStepLR(optim_H, milestones=[40, 60, 80, 90], gamma=0.2)

    optim_R = optim.Adam(net_R.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
    scheduler_R = optim.lr_scheduler.MultiStepLR(optim_R, milestones=[40, 60, 80, 90], gamma=0.2)

    # Training and validation loops
    max_psnr = 0
    print('=' * 130)
    for epoch in range(opt.epochs):
        train(opt, train_loader, epoch, net_H, net_R, optim_H, optim_R, criterion, device)

        with torch.no_grad():
            val_psnr = validation(opt, val_loader, epoch, net_H, net_R, criterion, device)

        scheduler_H.step()
        scheduler_R.step()

        if val_psnr[1] > max_psnr:
            max_psnr = val_psnr[1]
            utils.save_model(opt, net_H, net_R, epoch + 1, val_psnr)
        print('=' * 130)    


def train(opt, train_loader, epoch, net_H, net_R, optim_H, optim_R, criterion, device):
    net_H.train()#设置为训练模式，启用 Batch Normalization 和 Dropout
    net_R.train()
    
    iterations = len(train_loader)
    loss_log = LossLog()
    time_log = TimeLog(opt.epochs, epoch, iterations)
    metric_log = MetricLog(opt.train_metric_path)
    if epoch == 0:
        metric_log.write_metric_head()
    
    for i, data in enumerate(train_loader, 0):
        net_H.zero_grad()
        net_R.zero_grad()

        cover_img = data
        batch_size = cover_img.shape[0]
        idx = utils.get_perm_index(batch_size)  # 生成一个随机排列的索引列表
        secret_img = cover_img[idx, :, :, :].clone().detach()

        cover_img = cover_img.to(device, non_blocking=True)
        secret_img = secret_img.to(device, non_blocking=True)
        
        stego_img = net_H(cover_img, secret_img)
        loss_H = criterion(stego_img, cover_img)

        rev_secret_img = net_R(stego_img)
        loss_R = criterion(rev_secret_img, secret_img)

        loss_sum = opt.lambda_H * loss_H + opt.lambda_R * loss_R
        loss_sum.backward()

        optim_H.step()
        optim_R.step()
        
        loss_log.update_loss(loss_H.data, loss_R.data, loss_sum.data, batch_size)
        time_log.get_time_stats()
        utils.print_screen(opt.save_log_freq, opt.epochs, epoch, iterations, i, loss_log, time_log, None, True)

        img_list = [cover_img, stego_img, secret_img, rev_secret_img]
        if (i + 1) % opt.save_img_freq == 0:
            utils.save_images(img_list, epoch + 1, i + 1, batch_size, opt.train_img_dir)
        if epoch == 3 and i == iterations - 1:
            opt.save_img_freq = iterations
            
        metric_log.compute_update_metric(img_list)
        
    metric_log.write_metric_line(epoch + 1)
    utils.print_screen(opt.save_log_freq, opt.epochs, epoch, iterations, None, loss_log, time_log, metric_log, False)

    if opt.use_wandb:
        wandb.log({'epoch': epoch + 1,
                   'lr_H': optim_H.param_groups[0]['lr'],
                   'lr_R': optim_R.param_groups[0]['lr'],
                   'train_loss_H': loss_log.loss_H.avg,
                   'train_loss_R': loss_log.loss_R.avg,
                   'train_loss_sum': loss_log.loss_sum.avg,
                   'train_PSNR_C': metric_log.psnr_c.avg,
                   'train_PSNR_S': metric_log.psnr_s.avg})


def validation(opt, val_loader, epoch, net_H, net_R, criterion, device):
    net_H.eval()
    net_R.eval()
    
    iterations = len(val_loader)
    loss_log = LossLog()
    time_log = TimeLog(1, 0, iterations)
    metric_log = MetricLog(opt.val_metric_path)
    if epoch == 0:
        metric_log.write_metric_head()
    
    for i, data in enumerate(val_loader, 0):
        cover_img = data
        batch_size = cover_img.shape[0]
        idx = utils.get_perm_index(batch_size)
        secret_img = cover_img[idx, :, :, :].clone()

        cover_img = cover_img.to(device)
        secret_img = secret_img.to(device)
        
        stego_img = net_H(cover_img, secret_img)
        loss_H = criterion(stego_img, cover_img)

        rev_secret_img = net_R(stego_img)
        loss_R = criterion(rev_secret_img, secret_img)
        
        loss_sum = opt.lambda_H * loss_H + opt.lambda_R * loss_R
        loss_log.update_loss(loss_H.data, loss_R.data, loss_sum.data, batch_size)

        img_list = [cover_img, stego_img, secret_img, rev_secret_img]
        if (i + 1) % 50 == 0:
            utils.save_images(img_list, epoch + 1, i + 1, batch_size, opt.val_img_dir)
            
        metric_log.compute_update_metric(img_list)
        
    metric_log.write_metric_line(epoch + 1)
    time_log.get_time_stats()
    utils.print_screen(opt.save_log_freq, opt.epochs, epoch, iterations, None, loss_log, time_log, metric_log, False, 'Testing')

    if opt.use_wandb:
        wandb.log({'val_loss_H': loss_log.loss_H.avg,
                   'val_loss_R': loss_log.loss_R.avg,
                   'val_loss_sum': loss_log.loss_sum.avg,
                   'val_PSNR_C': metric_log.psnr_c.avg,
                   'val_PSNR_S': metric_log.psnr_s.avg})
    
    return [metric_log.psnr_c.avg, metric_log.psnr_s.avg]


if __name__ == '__main__':
    main()
