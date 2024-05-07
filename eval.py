import time
import torch
import numpy as np
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
from src.utils.evaluation import AP_partial
from datasets import CUFED
from models.models import MTResnetAggregate
from options.test_options import TestOptions

def cov(m):
    # m = m.type(torch.double)  # uncomment this line if desired
    fact = 1.0 / (m.shape[-1] - 1)  # 1 / N
    m -= torch.mean(m, dim=(1, 2), keepdim=True)
    mt = torch.transpose(m, 1, 2)  # if complex: mt = m.t().conj()
    return fact * m.matmul(mt).squeeze()

def rankmin(x: torch.Tensor) -> torch.Tensor:
    tmp = x.argsort()
    ranks = torch.zeros_like(tmp)
    ranks[tmp] = torch.arange(len(x))
    return ranks

def compute_rank_correlation(x, y):
    x, y = rankmin(x), rankmin(y)
    return corrcoef(x, y)

def corrcoef(x, y):
    batch_size = x.shape[0]
    x = torch.stack((x, y), 1)
    # calculate covariance matrix of rows
    c = cov(x)
    # normalize covariance matrix
    d = torch.diagonal(c, dim1=1, dim2=2)
    stddev = torch.pow(d, 0.5)
    stddev = stddev.repeat(1, 2).view(batch_size, 2, 2)
    c = c.div(stddev)
    c = c.div(torch.transpose(stddev, 1, 2))
    return c[:, 1, 0]

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
  N = importance.shape[0]
  importance = importance.view(N // 32, 32, -1)
  importance_labels = torch.cat(importance_labels)
  spearman = compute_rank_correlation(importance, importance_labels)
  print('yoyo: ', spearman)
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