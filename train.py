import argparse
import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.models import create_model

from datasets import CUFED

parser = argparse.ArgumentParser(description='PETA: Photo Album Event Recognition')
parser.add_argument('--resume', default=None, help='checkpoint to resume training')
parser.add_argument('--model_name', type=str, default='mtresnetaggregate')
parser.add_argument('--num_classes', type=int, default=23)
parser.add_argument('--dataset', default='cufed', choices=['cufed', 'pec', 'holidays'])
parser.add_argument('--dataset_path', type=str, default='/content/drive/MyDrive/CUFED-Event-Image/CUFED')
parser.add_argument('--dataset_type', type=str, default='ML_CUFED')
parser.add_argument('--batch_size', type=int, default=32, help='batch size') # change
parser.add_argument('--transform_type', type=str, default='squish')
parser.add_argument('--album_sample', type=str, default='rand_permute')
parser.add_argument('--num_workers', type=int, default=2, help='number of workers for data loader')
parser.add_argument('-v', '--verbose', action='store_true', help='show details')
parser.add_argument('--img_size', type=int, default=224)
parser.add_argument('--album_clip_length', type=int, default=32)
parser.add_argument('--remove_model_jit', type=int, default=None)
parser.add_argument('--use_transformer', type=int, default=1)
parser.add_argument('--transformers_pos', type=int, default=1)
parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
parser.add_argument('--milestones', nargs="+", type=int, default=[110, 160], help='milestones of learning decay')
parser.add_argument('--num_epochs', type=int, default=200, help='number of epochs to train')
parser.add_argument('--save_interval', type=int, default=10, help='interval for saving models (epochs)')
parser.add_argument('--save_folder', default='weights', help='directory to save checkpoints')
args = parser.parse_args()

def train(model, train_loader, crit, opt, sched, device):
  epoch_loss = 0
  for batch in train_loader:
    feats, label = batch.to(device)
    opt.zero_grad()
    out_data = model(feats)
    loss = crit(out_data, label)
    loss.backward()
    opt.step()
    epoch_loss += loss.item()

  sched.step()
  return epoch_loss / len(train_loader)


def main():
  if args.dataset == 'cufed':
    dataset = CUFED(root_dir=args.dataset_path, is_train=True, img_size=args.img_size, album_clip_length=args.album_clip_length)
    crit = nn.BCEWithLogitsLoss() # change
  else:
    exit("Unknown dataset!")
  device = torch.device('cuda:0')
  train_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True)

  if args.verbose:
    print("running on {}".format(device))
    print("num samples={}".format(len(dataset)))

  start_epoch = 0
  model = create_model(args).to(device)
  opt = optim.Adam(model.parameters(), lr=args.lr)
  sched = optim.lr_scheduler.MultiStepLR(opt, milestones=args.milestones)
  if args.resume:
      data = torch.load(args.resume)
      start_epoch = data['epoch']
      model.load_state_dict(data['model'], strict=True)
      opt.load_state_dict(data['opt_state_dict'])
      sched.load_state_dict(data['sched_state_dict'])
      if args.verbose:
          print("resuming from epoch {}".format(start_epoch))

  model.train()
  for epoch in range(start_epoch, args.num_epochs):
    t0 = time.perf_counter()
    loss = train(model, train_loader, crit, opt, sched, device)
    t1 = time.perf_counter()

    epoch_cnt = epoch + 1
    if epoch_cnt >= 110 and (epoch_cnt >= 190 or epoch_cnt % args.save_interval == 0): # change
      sfnametmpl = 'model-cufed-{:03d}.pt' # change
      sfname = sfnametmpl.format(epoch_cnt)
      spth = os.path.join(args.save_folder, sfname)
      torch.save({
          'epoch': epoch_cnt,
          'model': model.state_dict(),
          'loss': loss,
          'opt_state_dict': opt.state_dict(),
          'sched_state_dict': sched.state_dict()
      }, spth)
    if args.verbose:
      print("[epoch {}] loss={} dt={:.2f}sec".format(epoch_cnt, loss, t1 - t0))

if __name__ == '__main__':
  main()