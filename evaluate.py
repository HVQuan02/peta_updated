import time
import torch
import numpy as np
import torch.nn as nn
from dataset import CUFED_VIT_CLIP
from torch.utils.data import DataLoader
from models.models import MTResnetAggregate
from options.test_options import TestOptions
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from src.utils.evaluation import AP_partial, spearman_correlation, showCM
from sklearn.metrics import accuracy_score, multilabel_confusion_matrix, classification_report

args = TestOptions().parse()


def evaluate(model, test_dataset, test_loader, device):
  model.eval()
  gidx = 0
  attentions = []
  importance_labels = []
  scores = torch.zeros((len(test_dataset), len(test_dataset.event_labels)), dtype=torch.float32)
  
  with torch.no_grad():
    for batch in test_loader:
      feats, _, importances = batch
      feats = feats.to(device)
      logits, attention, imp = model(feats)
      shape = logits.shape[0]
      scores[gidx:gidx+shape, :] = logits.cpu()
      gidx += shape
      attentions.append(attention)
      importance_labels.append(importances)

    m = nn.Sigmoid()
    preds = m(scores)
    preds[preds >= args.threshold] = 1
    preds[preds < args.threshold] = 0

    scores = scores.numpy()
    preds = preds.numpy()
    if preds.sum() == 0:
        preds[np.argmax(scores)] = 1

    attention_tensor = torch.cat(attentions).to(device)
    importance_labels = torch.cat(importance_labels).to(device)
    
    acc = accuracy_score(test_dataset.labels, preds)
    cms = multilabel_confusion_matrix(test_dataset.labels, preds)
    cr = classification_report(test_dataset.labels, preds)

    map_micro, map_macro = AP_partial(test_dataset.labels, scores)[1:3]
    spearman = spearman_correlation(attention_tensor[:, 0, 1:], importance_labels) # debug lay tu 0 hay 1

    return map_micro, map_macro, acc, spearman, cms, cr

def main():
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  model = MTResnetAggregate(args)
    
  if args.dataset == 'cufed':
    test_dataset = CUFED_VIT_CLIP(root_dir=args.dataset_path, feats_dir=args.feats_dir, split_dir=args.split_path, is_train=False)
  else:
    exit("Unknown dataset!")
     
  test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, num_workers=args.num_workers)

  if args.verbose:
    print("running on {}".format(device))
    print("test_set = {}".format(len(test_dataset)))

  if args.ema:
    model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.999))
  state = torch.load(args.model_path, map_location=device)
  print('load model from epoch {}'.format(state['epoch']))
  model.load_state_dict(state['model_state_dict'], strict=True)
  model = model.to(device)
  
  t0 = time.perf_counter()
  map_micro, map_macro, acc, spearman, cms, cr = evaluate(model, test_dataset, test_loader, device)
  t1 = time.perf_counter()

  print("map_micro={} map_macro={} accuracy={} spearman={} dt={:.2f}sec".format(map_micro, map_macro, acc * 100, spearman, t1 - t0))
  print(cr)
  showCM(cms)


if __name__ == '__main__':
  main()