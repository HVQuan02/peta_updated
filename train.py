import argparse
import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.models import create_model
from src.loss_functions.asymmetric_loss import AsymmetricLossOptimized
from flash.core.optimizers import LinearWarmupCosineAnnealingLR
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn, update_bn
from datasets import CUFED

parser = argparse.ArgumentParser(description='PETA: Photo Album Event Recognition')
parser.add_argument('--resume', default=None, help='checkpoint to resume training')
parser.add_argument('--model_name', type=str, default='mtresnetaggregate')
parser.add_argument('--num_classes', type=int, default=23)
parser.add_argument('--dataset', default='cufed', choices=['cufed', 'pec', 'holidays'])
parser.add_argument('--dataset_path', type=str, default='/kaggle/input/thesis-cufed/CUFED')
parser.add_argument('--split_path', type=str, default='/kaggle/working/split_dir')
parser.add_argument('--dataset_type', type=str, default='ML_CUFED')
parser.add_argument('--train_batch_size', type=int, default=6, help='train batch size') # change
parser.add_argument('--num_workers', type=int, default=2, help='number of workers for data loader')
parser.add_argument('-v', '--verbose', action='store_true', help='show details')
parser.add_argument('--img_size', type=int, default=224)
parser.add_argument('--album_clip_length', type=int, default=32)
parser.add_argument('--remove_model_jit', type=int, default=None)
parser.add_argument('--use_transformer', type=int, default=1)
parser.add_argument('--transformers_pos', type=int, default=1)
parser.add_argument('--lr', type=float, default=1e-5, help='base learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-3, help='weight decay rate')
parser.add_argument('--warmup_epochs', type=int, default=10, help='number of warmup epochs')
parser.add_argument('--max_epochs', type=int, default=100, help='max number of epochs to train')
parser.add_argument('--save_folder', default='weights', help='directory to save checkpoints')
parser.add_argument('--loss', type=str, default='asymmetric', help='loss function')
parser.add_argument('--patience', type=int, default=10, help='patience of early stopping')
parser.add_argument('--min_delta', type=float, default=1e-3, help='min delta of early stopping') # change
parser.add_argument('--threshold', type=float, default=5e-3, help='val loss threshold of early stopping') # change
args = parser.parse_args()

def train_one_epoch(ema_model, model, train_loader, crit, opt, sched, device):
  model.train()
  epoch_loss = 0
  for batch in train_loader:
    feats, label = batch
    feats = feats.to(device)
    label = label.to(device)
    opt.zero_grad()
    out_data = model(feats)
    loss = crit(out_data, label)
    loss.backward()
    opt.step()
    ema_model.update_parameters(model)
    epoch_loss += loss.item()
    sched.step()
  return epoch_loss / len(train_loader)

class EarlyStopper:
    def __init__(self, patience, min_delta, threshold):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_train_loss = float('inf')
        self.threshold = threshold

    def early_stop(self, train_loss):
        if train_loss <= self.threshold:
            return True, True
        if train_loss < self.min_train_loss:
            self.min_train_loss = train_loss
            self.counter = 0
            return False, True
        elif train_loss > (self.min_train_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True, False
            else:
               return False, False

def main():
  if args.dataset == 'cufed':
    train_dataset = CUFED(root_dir=args.dataset_path, split_dir=args.split_path, is_train=True, train_with_val=False, img_size=args.img_size, album_clip_length=args.album_clip_length) # combine train and val
  else:
    exit("Unknown dataset!")

  if args.loss == 'asymmetric':
    crit = AsymmetricLossOptimized()
  elif args.loss == 'bce':
    crit = nn.BCEWithLogitsLoss()
  else:
    exit("Unknown loss function!")
     
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True)

  if args.verbose:
    print("running on {}".format(device))
    print("num samples of train = {}".format(len(train_dataset)))

  start_epoch = 0
  model = create_model(args).to(device)
  ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.999))
  opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
  sched = LinearWarmupCosineAnnealingLR(opt, args.warmup_epochs, args.max_epochs)
  early_stopper = EarlyStopper(patience=args.patience, min_delta=args.min_delta, threshold=args.threshold)

  if args.resume:
      data = torch.load(args.resume)
      start_epoch = data['epoch']
      model.load_state_dict(data['model'], strict=True)
      opt.load_state_dict(data['opt_state_dict'])
      sched.load_state_dict(data['sched_state_dict'])
      if args.verbose:
          print("resuming from epoch {}".format(start_epoch))

  for epoch in range(start_epoch, args.max_epochs):
    t0 = time.perf_counter()
    train_loss = train_one_epoch(ema_model, model, train_loader, crit, opt, sched, device)
    t1 = time.perf_counter()

    epoch_cnt = epoch + 1
    is_early_stopping, is_save_ckpt = early_stopper.early_stop(train_loss)

    if is_save_ckpt:
      torch.save({
        'epoch': epoch_cnt,
        'model': model.state_dict(),
        'loss': train_loss,
        'opt_state_dict': opt.state_dict(),
        'sched_state_dict': sched.state_dict()
      }, os.path.join(args.save_folder, 'PETA-cufed.pt')) 
         
    if is_early_stopping:
      # Update bn statistics for the ema_model at the end
      update_bn(train_loader, ema_model)
      torch.save({
        'epoch': epoch_cnt,
        'model': ema_model.state_dict()
      }, os.path.join(args.save_folder, 'EMA-PETA-cufed.pt'))
      print('Stop at epoch {}'.format(epoch_cnt)) 
      break

    if args.verbose:
      print("[epoch {}] train_loss={} dt_train={:.2f}sec".format(epoch_cnt, train_loss, t1 - t0))  


if __name__ == '__main__':
  main()