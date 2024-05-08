import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from src.utils.evaluation import AP_partial, spearman_correlation
from datasets import CUFED
from models.models import MTResnetAggregate
from options.test_options import TestOptions

def evaluate(model, test_loader, test_dataset, device):
  model.eval()
  scores = torch.zeros((len(test_dataset), len(test_dataset.event_labels)), dtype=torch.float32)
  importance_labels = []
  gidx = 0
  with torch.no_grad():
    for batch in test_loader:
      feats, labels, importance_scores = batch
      feats = feats.to(device)
      logits, importance, attention = model(feats)
      shape = logits.shape[0]
      scores[gidx:gidx+shape, :] = logits.cpu()
      gidx += shape
      importance_labels.append(importance_scores)
  map = AP_partial(test_dataset.labels, scores.numpy())[1]
  importance = importance.view(importance.shape[0] // 32, 32, -1)
  importance_labels = torch.cat(importance_labels)
  spearman = spearman_correlation(importance, importance_labels)
  return map, spearman

def main():
  args = TestOptions().parse()
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  torch.cuda.manual_seed(args.seed)

  if args.dataset == 'cufed':
    test_dataset = CUFED(root_dir=args.dataset_path, split_dir=args.split_path, is_train=False, img_size=args.img_size, album_clip_length=args.album_clip_length)
  else:
    exit("Unknown dataset!")
     
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  test_loader = DataLoader(test_dataset, batch_size=args.val_batch_size, num_workers=args.num_workers, shuffle=False)

  if args.verbose:
    print("running on {}".format(device))
    print("num samples of test = {}".format(len(test_dataset)))

  print("create model ...")
  model = MTResnetAggregate(args).to(device)
  state = torch.load(args.model_path)
  model.load_state_dict(state['model_state_dict'])
  print("done")

  t0 = time.perf_counter()
  map, spearman = evaluate(model, test_loader, test_dataset, device)
  t1 = time.perf_counter()

  if args.verbose:
    print("mAP={} spearman={} dt={:.2f}sec".format(map, spearman, t1 - t0)) 

if __name__ == '__main__':
  main()