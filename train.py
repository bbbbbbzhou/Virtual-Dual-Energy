import os
import argparse
import json
import numpy as np
import torch.utils.data
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.backends.cudnn as cudnn

from torchvision.utils import save_image
from utils import prepare_sub_folder
from datasets import get_datasets
from models import create_model
import scipy.io as sio
import csv
import pdb

parser = argparse.ArgumentParser(description='Dual Energy Chest X-ray')

# model name
parser.add_argument('--experiment_name', type=str, default='train_bone_msunet', help='give a model name before training')
parser.add_argument('--model_type', type=str, default='model_bone', help='model_bone / model_softtissue')
parser.add_argument('--resume', type=str, default=None, help='Filename of the checkpoint to resume')

# dataset
parser.add_argument('--dataset', type=str, default='DE', help='dataset name: DE')
parser.add_argument('--data_root', type=str, default='./example_data/', help='data root folder')

# networks
parser.add_argument('--net_G', type=str, default='msunet', help='generator')
parser.add_argument('--net_D', type=str, default='patchGAN', help='discriminator')
parser.add_argument('--depth', type=int, default=6, help='depth of the network')
parser.add_argument('--wr_recon', type=float, default=100, help='weight for recon loss')

# training options
parser.add_argument('--n_epochs', type=int, default=1000, help='number of epoch')
parser.add_argument('--batch_size', type=int, default=3, help='training batch size')
parser.add_argument('--AUG', default=False, action='store_true', help='use augmentation')
parser.add_argument('--angle', type=int, default=20, help='random rotation angle')
parser.add_argument('--osize', type=int, default=1150, help='random scale')
parser.add_argument('--fineSize', nargs='+', type=float, default=[1024, 1024], help='random crop')

# evaluation options
parser.add_argument('--eval_epochs', type=int, default=4, help='evaluation epochs')
parser.add_argument('--save_epochs', type=int, default=4, help='save evaluation for every number of epochs')

# optimizer
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')

parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for ADAM')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for ADAM')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')

# learning rate policy
parser.add_argument('--lr_policy', type=str, default='step', help='learning rate decay policy')
parser.add_argument('--step_size', type=int, default=1000, help='step size for step scheduler')
parser.add_argument('--gamma', type=float, default=0.5, help='decay ratio for step scheduler')

# logger options
parser.add_argument('--snapshot_epochs', type=int, default=4, help='save model for every number of epochs')
parser.add_argument('--log_freq', type=int, default=100, help='save model for every number of epochs')
parser.add_argument('--output_path', default='./', type=str, help='Output path.')

# other
parser.add_argument('--num_workers', type=int, default=8, help='number of threads to load data')
parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0], help='list of gpu ids')
opts = parser.parse_args()

options_str = json.dumps(opts.__dict__, indent=4, sort_keys=False)
print("------------------- Options -------------------")
print(options_str[2:-2])
print("-----------------------------------------------")

cudnn.benchmark = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = create_model(opts)
model.setgpu(opts.gpu_ids)

num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)

print('Number of parameters: {} \n'.format(num_param))

if opts.resume is None:
    model.initialize()
    ep0 = -1
    total_iter = 0
else:
    ep0, total_iter = model.resume(opts.resume)

model.set_scheduler(opts, ep0)
ep0 += 1
print('Start training at epoch {} \n'.format(ep0))

# select dataset
train_set, val_set = get_datasets(opts)
train_loader = DataLoader(dataset=train_set, num_workers=opts.num_workers, batch_size=opts.batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_set, num_workers=opts.num_workers, batch_size=1, shuffle=False)

# Setup directories
output_directory = os.path.join(opts.output_path, 'outputs', opts.experiment_name)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)

with open(os.path.join(output_directory, 'options.json'), 'w') as f:
    f.write(options_str)

with open(os.path.join(output_directory, 'train_loss.csv'), 'w') as f:
    writer = csv.writer(f)
    writer.writerow(model.loss_names)

# training loop
for epoch in range(ep0, opts.n_epochs + 1):

    train_bar = tqdm(train_loader)
    model.train()
    model.set_epoch(epoch)
    for it, data in enumerate(train_bar):
        total_iter += 1
        model.set_input(data)
        model.optimize()
        train_bar.set_description(desc='[Epoch {}]'.format(epoch) + model.loss_summary)

        if it % opts.log_freq == 0:
            with open(os.path.join(output_directory, 'train_loss.csv'), 'a') as f:
                writer = csv.writer(f)
                writer.writerow(model.get_current_losses().values())

    model.update_learning_rate()

    # save checkpoint
    if (epoch+1) % opts.snapshot_epochs == 0:
        checkpoint_name = os.path.join(checkpoint_directory, 'model_{}.pt'.format(epoch))
        model.save(checkpoint_name, epoch, total_iter)

    # evaluation
    if (epoch+1) % opts.eval_epochs == 0:
        print('Evaluation ......')
        pred = os.path.join(image_directory, 'pred_{:03d}.png'.format(epoch))
        gt = os.path.join(image_directory, 'gt_{:03d}.png'.format(epoch))

        print(model.IT.detach().shape)
        vis_pred = model.IT_fake.detach().type(torch.DoubleTensor)
        save_image(vis_pred, pred, normalize=False, scale_each=False, padding=5)
        vis_gt = model.IT.detach().type(torch.DoubleTensor)
        save_image(vis_gt, gt, normalize=False, scale_each=False, padding=5)

        model.eval()
        with torch.no_grad():
            model.evaluate(val_loader)

        with open(os.path.join(output_directory, 'metrics_table.csv'), 'a') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, model.psnr, model.ssim, model.mse])

    if (epoch+1) % opts.save_epochs == 0:
        sio.savemat(os.path.join(image_directory, 'eval.mat'), model.results)
