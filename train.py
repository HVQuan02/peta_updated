import time
import os
import torch
import numpy as np
import torch.nn as nn
from dataset import CUFED, CUFED_VIT, CUFED_VIT_CLIP, PEC_VIT_CLIP
from torch.utils.data import DataLoader
from models.models import MTResnetAggregate
from src.utils.evaluation import AP_partial
from options.train_options import TrainOptions
from flash.core.optimizers import LinearWarmupCosineAnnealingLR
from src.loss_functions.asymmetric_loss import AsymmetricLossOptimized
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn, update_bn

args = TrainOptions().parse()


def validate_one_epoch(model, val_dataset, val_loader, device):
  model.eval()
  gidx = 0
  scores = torch.zeros((len(val_dataset), len(val_dataset.event_labels)), dtype=torch.float32)
  
  with torch.no_grad():
    for batch in val_loader:
      if isinstance(val_dataset, PEC_VIT_CLIP):
        feats, _ = batch
      else:
        feats, _, _ = batch
      feats = feats.to(device)
      logits, _ = model(feats)
      shape = logits.shape[0]
      scores[gidx:gidx+shape, :] = logits.cpu()
      gidx += shape

  return AP_partial(val_dataset.labels, scores.numpy())[2]


def train_one_epoch(ema_model, model, train_dataset, train_loader, crit, opt, sched, device):
  model.train()
  epoch_loss = 0
    
  for batch in train_loader:
    if isinstance(train_dataset, PEC_VIT_CLIP):
      feats, labels = batch
    else:
      feats, labels, _ = batch
    feats = feats.to(device)
    labels = labels.to(device)
    opt.zero_grad()
    logits, _ = model(feats)
    loss = crit(logits, labels)
    loss.backward()
    opt.step()
    ema_model.update_parameters(model)
    epoch_loss += loss.item()
    sched.step() # change
  return epoch_loss / len(train_loader)


class EarlyStopper:
    def __init__(self, patience, min_delta, threshold):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.max_validation_map = -float('inf')
        self.threshold = threshold

    def early_stop(self, validation_mAP):
        if validation_mAP >= self.threshold:
            return True, True
        if validation_mAP > self.max_validation_map:
            self.max_validation_map = validation_mAP
            self.counter = 0
            return False, True
        if validation_mAP < (self.max_validation_map - self.min_delta):
            self.counter += 1
            if self.counter > self.patience:
                return True, False
        return False, False


def main():
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  if args.seed:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

  if not os.path.exists(args.save_dir):
     os.mkdir(args.save_dir)

  if args.dataset == 'cufed':
    num_classes = CUFED.NUM_CLASS
  elif args.dataset == 'pec':
    num_classes = PEC_VIT_CLIP.NUM_CLASS
  else:
    pass
  model = MTResnetAggregate(args, num_classes)
  ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.999))

  if args.dataset == 'cufed':
    if args.backbone is not None:
      train_dataset = CUFED(root_dir=args.dataset_path, split_dir=args.split_path, img_size=args.img_size, album_clip_length=args.album_clip_length, ext_model=model.feature_extraction)
      val_dataset = CUFED(root_dir=args.dataset_path, split_dir=args.split_path, is_train=False, img_size=args.img_size, album_clip_length=args.album_clip_length, ext_model=model.feature_extraction)
    elif args.use_clip:  
      train_dataset = CUFED_VIT_CLIP(root_dir=args.dataset_path, feats_dir=args.feats_dir, split_dir=args.split_path, album_clip_length=args.album_clip_length)
      val_dataset = CUFED_VIT_CLIP(root_dir=args.dataset_path, feats_dir=args.feats_dir, split_dir=args.split_path, album_clip_length=args.album_clip_length, is_train=False)
    else:
      train_dataset = CUFED_VIT(root_dir=args.dataset_path, feats_dir=args.feats_dir, split_dir=args.split_path, album_clip_length=args.album_clip_length)
      val_dataset = CUFED_VIT(root_dir=args.dataset_path, feats_dir=args.feats_dir, split_dir=args.split_path, album_clip_length=args.album_clip_length, is_train=False)
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size, num_workers=args.num_workers)
  elif args.dataset == 'pec':
    if args.backbone is not None:
        pass
#       train_dataset = CUFED(root_dir=args.dataset_path, split_dir=args.split_path, img_size=args.img_size, album_clip_length=args.album_clip_length, ext_model=model.feature_extraction)
#       val_dataset = CUFED(root_dir=args.dataset_path, split_dir=args.split_path, is_train=False, img_size=args.img_size, album_clip_length=args.album_clip_length, ext_model=model.feature_extraction)
    elif args.use_clip:
      train_dataset = PEC_VIT_CLIP(root_dir=args.dataset_path, feats_dir=args.feats_dir, split_dir=args.split_path, album_clip_length=args.album_clip_length)
      val_dataset = PEC_VIT_CLIP(root_dir=args.dataset_path, feats_dir=args.feats_dir, split_dir=args.split_path, album_clip_length=args.album_clip_length, is_train=False)
    else:
        pass
#       train_dataset = CUFED_VIT(root_dir=args.dataset_path, feats_dir=args.feats_dir, split_dir=args.split_path, album_clip_length=args.album_clip_length)
#       val_dataset = CUFED_VIT(root_dir=args.dataset_path, feats_dir=args.feats_dir, split_dir=args.split_path, album_clip_length=args.album_clip_length, is_train=False)
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, num_workers=args.num_workers, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size, num_workers=args.num_workers)
  else:
    exit("Unknown dataset!")
     
  if args.verbose:
    print("running on {}".format(device))
    print("train_set={}".format(len(train_dataset)))
    print("test_set={}".format(len(val_dataset)))

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
    sched = LinearWarmupCosineAnnealingLR(opt, args.warmup_epochs, args.max_epoch)
  elif args.lr_policy == 'step':
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=args.lr_step, gamma=args.lr_gamma)
  elif args.lr_policy == 'multi_step':
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=args.lr_milestones, gamma=args.lr_gamma)
  elif args.lr_policy == 'onecycle':
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, steps_per_epoch=len(train_loader), epochs=args.max_epoch, pct_start=0.2)
  else:
     exit('Unknown optimization lr')

  early_stopper = EarlyStopper(patience=args.patience, min_delta=args.min_delta, threshold=args.stopping_threshold)

  start_epoch = 0
  if args.resume:
      data = torch.load(args.resume, map_location=device)
      start_epoch = data['epoch']
      model.load_state_dict(data['model_state_dict'], strict=True)
      opt.load_state_dict(data['opt_state_dict'])
      sched.load_state_dict(data['sched_state_dict'])
      if args.verbose:
          print("resuming from epoch {}".format(start_epoch))

  for epoch in range(start_epoch, args.max_epoch):
    epoch_cnt = epoch + 1
    model = model.to(device)

    t0 = time.perf_counter()
    train_loss = train_one_epoch(ema_model, model, train_dataset, train_loader, crit, opt, sched, device)
    t1 = time.perf_counter()

    t2 = time.perf_counter()
    val_map = validate_one_epoch(model, val_dataset, val_loader, device)
    t3 = time.perf_counter()

    model_config = {
      'epoch': epoch_cnt,
      'model_state_dict': model.state_dict(),
      'loss': train_loss,
      'opt_state_dict': opt.state_dict(),
      'sched_state_dict': sched.state_dict()
    }

    torch.save(model_config, os.path.join(args.save_dir, 'last_updated_peta_{}.pt'.format(args.dataset)))

    is_early_stopping, is_save_ckpt = early_stopper.early_stop(val_map)

    if is_save_ckpt:
      torch.save(model_config, os.path.join(args.save_dir, 'best_updated_peta_{}.pt'.format(args.dataset)))
    
    if is_early_stopping or epoch_cnt == args.max_epoch:
      # Update bn statistics for the ema_model at the end
      update_bn(train_loader, ema_model)

      # save ema model
      torch.save({
        'epoch': epoch_cnt,
        'model_state_dict': ema_model.state_dict()
      }, os.path.join(args.save_dir, 'ema_updated_peta_{}.pt'.format(args.dataset)))

      print('Stop at epoch {}'.format(epoch_cnt)) 
      break

    if args.verbose:
      print("[epoch {}] train_loss={} val_map={} dt_train={:.2f}sec dt_val={:.2f}sec dt={:.2f}sec".format(epoch_cnt, train_loss, val_map, t1 - t0, t3 - t2, t1 - t0 + t3 - t2))  


if __name__ == '__main__':
  main()