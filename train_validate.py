import time
import os
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from src.utils.evaluation import AP_partial
from src.loss_functions.asymmetric_loss import AsymmetricLossOptimized
from flash.core.optimizers import LinearWarmupCosineAnnealingLR
from datasets import CUFED
from models.models import MTResnetAggregate
from options.train_options import TrainOptions

def validate_one_epoch(model, val_loader, val_dataset, device):
  model.eval()
  scores = torch.zeros((len(val_dataset), len(val_dataset.event_labels)), dtype=torch.float32)
  gidx = 0
  with torch.no_grad():
    for batch in val_loader:
      feats, _, _ = batch
      feats = feats.to(device)
      logits, _, _ = model(feats)
      shape = logits.shape[0]
      scores[gidx:gidx+shape, :] = logits.cpu()
      gidx += shape
  return AP_partial(val_dataset.labels, scores.numpy())[1]

def train_one_epoch(model, train_loader, crit, opt, sched, device):
  model.train()
  epoch_loss = 0
  for batch in train_loader:
    feats, labels, _ = batch
    feats = feats.to(device)
    labels = labels.to(device)
    opt.zero_grad()
    logits, _, _ = model(feats)
    loss = crit(logits, labels)
    loss.backward()
    opt.step()
    epoch_loss += loss.item()
    sched.step() # change
  return epoch_loss / len(train_loader)

class EarlyStopper:
    def __init__(self, patience, min_delta, threshold):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.max_validation_mAP = -float('inf')
        self.threshold = threshold

    def early_stop(self, validation_mAP):
        if validation_mAP >= self.threshold:
            return True, True
        if validation_mAP > self.max_validation_mAP:
            self.max_validation_mAP = validation_mAP
            self.counter = 0
            return False, True
        if validation_mAP < (self.max_validation_mAP - self.min_delta):
            self.counter += 1
            if self.counter > self.patience:
                return True, False
        return False, False

def main():
  args = TrainOptions().parse()
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  torch.cuda.manual_seed(args.seed)

  if args.dataset == 'cufed':
    train_dataset = CUFED(root_dir=args.dataset_path, split_dir=args.split_path, is_train=True, img_size=args.img_size, album_clip_length=args.album_clip_length)
    val_dataset = CUFED(root_dir=args.dataset_path, split_dir=args.split_path, is_train=False, img_size=args.img_size, album_clip_length=args.album_clip_length)
  else:
    exit("Unknown dataset!")
     
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True)
  val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size, num_workers=args.num_workers, shuffle=False)

  if args.verbose:
    print("running on {}".format(device))
    print("num samples of train = {}".format(len(train_dataset)))
    print("num samples of val = {}".format(len(val_dataset)))

  start_epoch = 0
  model = MTResnetAggregate(args).to(device)

  if args.loss == 'asymmetric':
    crit = AsymmetricLossOptimized()
  elif args.loss == 'bce':
    crit = nn.BCEWithLogitsLoss()
  else:
    exit("Unknown loss function!")

  if args.optimizer == 'adam':
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
  elif args.optimizer == 'adamw':
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
  elif args.optimizer == 'sgd':
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
  else:
     exit('Unknown optimizer')
     
  if args.lr_policy == 'cosine':
    sched = LinearWarmupCosineAnnealingLR(opt, args.warmup_epochs, args.max_epochs)
  elif args.lr_policy == 'step':
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=args.lr_step, gamma=args.lr_gamma)
  elif args.lr_policy == 'multi_step':
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=args.lr_milestones, gamma=args.lr_gamma)
  elif args.lr_policy == 'onecycle':
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, steps_per_epoch=len(train_loader), epochs=args.max_epochs, pct_start=0.2)
  else:
     exit('Unknown optimization lr')

  early_stopper = EarlyStopper(patience=args.patience, min_delta=args.min_delta, threshold=args.stop_threshold)

  if args.resume:
      data = torch.load(args.resume)
      start_epoch = data['epoch']
      model.load_state_dict(data['model_state_dict'], strict=True)
      opt.load_state_dict(data['opt_state_dict'])
      sched.load_state_dict(data['sched_state_dict'])
      if args.verbose:
          print("resuming from epoch {}".format(start_epoch))

  for epoch in range(start_epoch, args.max_epochs):
    t0 = time.perf_counter()
    train_loss = train_one_epoch(model, train_loader, crit, opt, sched, device)
    t1 = time.perf_counter()

    t2 = time.perf_counter()
    val_mAP = validate_one_epoch(model, val_loader, val_dataset, device)
    t3 = time.perf_counter()

    epoch_cnt = epoch + 1
    model_config = {
      'epoch': epoch_cnt,
      'model_state_dict': model.state_dict(),
      'loss': train_loss,
      'opt_state_dict': opt.state_dict(),
      'sched_state_dict': sched.state_dict()
    }

    torch.save(model_config, os.path.join(args.save_folder, 'last-PETA-{}.pt'.format(args.dataset)))

    is_early_stopping, is_save_ckpt = early_stopper.early_stop(val_mAP)

    if is_save_ckpt:
      torch.save(model_config, os.path.join(args.save_folder, 'best-PETA-{}.pt'.format(args.dataset)))
         
    if is_early_stopping:
      print('Early stop at epoch {}'.format(epoch_cnt)) 
      break

    if args.verbose:
      print("[epoch {}] train_loss={} val_mAP={} dt_train={:.2f}sec dt_val={:.2f}sec dt={:.2f}sec".format(epoch_cnt, train_loss, val_mAP, t1 - t0, t3 - t2, t1 - t0 + t3 - t2))  

if __name__ == '__main__':
  main()