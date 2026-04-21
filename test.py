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
    parser.add_argument('--batch_input', type=arg_bool, default=True, choices=[True, False], help='a batch of images for input or not')
    parser.add_argument('--data_dir', type=str, default='/home/liming/pro/DHVC/data/', help='dataset root directory')
    parser.add_argument('--image_path', type=str, default='/home/liming/pro/DHVC/data/', help='image path of single input')
    parser.add_argument('--image_height', type=int, default=256, help='hight for resized images')
    parser.add_argument('--image_width', type=int, default=256, help='width for resized images')
    # Model parameters
    parser.add_argument('--pretrained', type=arg_bool, default=False, choices=[True, False], help='use pretrained model or not')
    parser.add_argument('--dual_net', type=arg_bool, default=True, choices=[True, False], help='use dual input for hiding net or not')
    parser.add_argument('--Hnet_path', default='/home/liming/pro/steg/DDH/DDH_1/output/ckpt/modelH_ep001_psnr22.7371.pth', help="path to Hiding net")
    parser.add_argument('--Rnet_path', default='/home/liming/pro/steg/DDH/DDH_1/output/ckpt/modelR_ep001_psnr16.0190.pth', help="path to Revealing net")
    # Testing parameters
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    # Output parameters
    parser.add_argument('--out_dir', default='/home/liming/pro/steg/DDH/DDH_1/output/', help='testing output directory')

    opts = parser.parse_args()

    print('Parameters:')
    print('\n'.join(f'{k:<20} {v}' for k, v in vars(opts).items()), '\n')

    return opts


def main():
    # Preparation
    opt = parse_opts()
    device = utils.set_device(opt)
    utils.make_output_dirs(opt)

    # Load the dataset
    val_loader = steg_data_loader('test', opt)
    
    # Build the model
    net_H = utils.load_model(opt, True, device)
    net_R = utils.load_model(opt, False, device)

    # Testing loops
    print('=' * 130)
    with torch.no_grad():
        if opt.batch_input:
            batch_testing(opt, val_loader, net_H, net_R, device)
    print('=' * 130)


def batch_testing(opt, val_loader, net_H, net_R, device):
    net_H.eval()
    net_R.eval()
    
    iterations = len(val_loader)
    time_log = TimeLog(1, 0, iterations)
    metric_log = MetricLog(opt.val_metric_path)
    
    for i, data in enumerate(val_loader, 0):
        cover_img = data
        batch_size = cover_img.shape[0]
        idx = utils.get_perm_index(batch_size)
        secret_img = cover_img[idx, :, :, :].clone()

        cover_img = cover_img.to(device)
        secret_img = secret_img.to(device)
        
        stego_img = net_H(cover_img, secret_img)

        rev_secret_img = net_R(stego_img)

        img_list = [cover_img, stego_img, secret_img, rev_secret_img]
        if (i + 1) % 50 == 0:
            utils.save_images(img_list, epoch + 1, i + 1, batch_size, opt.val_img_dir)
            
        metric_log.compute_update_metric(img_list)
        
    metric_log.write_metric_line(epoch + 1)
    time_log.get_time_stats()
    utils.print_screen(opt.save_log_freq, opt.epochs, epoch, iterations, None, loss_log, time_log, metric_log, False, 'Testing')


if __name__ == '__main__':
    main()
