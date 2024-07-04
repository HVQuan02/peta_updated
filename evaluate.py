import time
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from models.models import MTResnetAggregate
from options.test_options import TestOptions
from dataset import CUFED, CUFED_VIT, CUFED_VIT_CLIP, PEC_VIT_CLIP
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from src.utils.evaluation import AP_partial, spearman_correlation, showCM
from sklearn.metrics import accuracy_score, multilabel_confusion_matrix, classification_report

args = TestOptions().parse()


def evaluate(model, test_dataset, test_loader, device):
  model.eval()
  gidx = 0
  attentions = []
  is_pec = isinstance(test_dataset, PEC_VIT_CLIP)
  importance_labels = []
  scores = torch.zeros((len(test_dataset), len(test_dataset.event_labels)), dtype=torch.float32)
  
  with torch.no_grad():
    for batch in test_loader:
      if is_pec:
        feats, _ = batch
      else:
        feats, _, importances = batch
      feats = feats.to(device)
      logits, attention = model(feats)
      shape = logits.shape[0]
      scores[gidx:gidx+shape, :] = logits.cpu()
      gidx += shape
      if not is_pec:
          attentions.append(attention)
          importance_labels.append(importances)

    if is_pec:
        scores = scores.numpy()
        preds = np.zeros(scores.shape, dtype=np.float32)

        # Find the index of the maximum value along each row
        max_indices = np.argmax(scores, axis=1)

        # Set the corresponding elements in 'preds' to 1
        preds[np.arange(preds.shape[0]), max_indices] = 1
    else:
      m = nn.Sigmoid()
      preds = m(scores)
      preds[preds >= args.threshold] = 1
      preds[preds < args.threshold] = 0

      scores = scores.numpy()
      preds = preds.numpy()
      
      # Ensure no row has all zeros
      for i in range(preds.shape[0]):
        if np.sum(preds[i]) == 0:
            preds[i][np.argmax(scores[i])] = 1

    # inaccurate albums
    # fidx = ~np.all(preds == test_dataset.labels, axis=1)
    # f_albums = np.array(test_dataset.videos)[fidx]
    # print('inaccurate albums', f_albums)
    
    acc = accuracy_score(test_dataset.labels, preds)
    cms = multilabel_confusion_matrix(test_dataset.labels, preds)
    cr = classification_report(test_dataset.labels, preds)

    map_micro, map_macro = AP_partial(test_dataset.labels, scores)[1:3]
    
    if not is_pec:
        attention_tensor = torch.cat(attentions).to(device)
        importance_labels = torch.cat(importance_labels).to(device)
        spearman = spearman_correlation(attention_tensor[:, 0, 1:], importance_labels)

        return map_micro, map_macro, acc, spearman, cms, cr
    return map_micro, map_macro, acc, cms, cr

def main():
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
  if args.dataset == 'cufed':
    if args.backbone is not None:
      test_dataset = CUFED(root_dir=args.dataset_path, split_dir=args.split_path, is_train=False, img_size=args.img_size, album_clip_length=args.album_clip_length, ext_model=model.feature_extraction)
    elif args.use_clip:
      test_dataset = CUFED_VIT_CLIP(root_dir=args.dataset_path, feats_dir=args.feats_dir, split_dir=args.split_path, album_clip_length=args.album_clip_length, is_train=False)
    else:
      test_dataset = CUFED_VIT(root_dir=args.dataset_path, feats_dir=args.feats_dir, split_dir=args.split_path, album_clip_length=args.album_clip_length, is_train=False)
  elif args.dataset == 'pec':
    if args.backbone is not None:
        pass
    elif args.use_clip:
      test_dataset = PEC_VIT_CLIP(root_dir=args.dataset_path, feats_dir=args.feats_dir, split_dir=args.split_path, album_clip_length=args.album_clip_length, is_train=False)
    else:
        pass
  else:
    exit("Unknown dataset!")
     
  test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, num_workers=args.num_workers)
    
  if args.verbose:
    print("running on {}".format(device))
    print("test_set = {}".format(len(test_dataset)))

  model = MTResnetAggregate(args, test_dataset.NUM_CLASS)
  if args.ema:
    model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.999))
  state = torch.load(args.model_path, map_location=device)
  print('load model from epoch {}'.format(state['epoch']))
  model.load_state_dict(state['model_state_dict'], strict=True)
  model = model.to(device)
  
  t0 = time.perf_counter()
  if args.dataset == 'cufed':
      map_micro, map_macro, acc, spearman, cms, cr = evaluate(model, test_dataset, test_loader, device)
  else:
      map_micro, map_macro, acc, cms, cr = evaluate(model, test_dataset, test_loader, device)
  t1 = time.perf_counter()

  if args.dataset == 'cufed':
    print("map_micro={:.2f} map_macro={:.2f} accuracy={:.2f} spearman={:.3f} dt={:.2f}sec".format(map_micro, map_macro, acc * 100, spearman, t1 - t0))
  else: 
    print("map_micro={:.2f} map_macro={:.2f} accuracy={:.2f} dt={:.2f}sec".format(map_micro, map_macro, acc * 100, t1 - t0))
  print(cr)
  showCM(cms)


if __name__ == '__main__':
  main()