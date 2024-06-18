import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, multilabel_confusion_matrix, classification_report
from src.utils.evaluation import AP_partial, spearman_correlation, showCM
from datasets import CUFED
from models.models import MTResnetAggregate
from options.test_options import TestOptions

args = TestOptions().parse()

def evaluate(model, test_dataset, test_loader, device):
  model.eval()
  scores = torch.zeros((len(test_dataset), len(test_dataset.event_labels)), dtype=torch.float32)
  attentions = []
  importance_labels = []
  gidx = 0
    
  with torch.no_grad():
    for batch in test_loader:
      feats, _, importance_scores = batch
      feats = feats.to(device)
      logits, attention = model(feats)
      shape = logits.shape[0]
      scores[gidx:gidx+shape, :] = logits.cpu()
      gidx += shape
      attentions.append(attention)
      importance_labels.append(importance_scores)

  m = nn.Softmax(dim=1)
  preds = m(scores)
  preds[preds >= args.threshold] = 1
  preds[preds < args.threshold] = 0

  scores = scores.numpy()
  preds = preds.numpy()

  attention_tensor = torch.cat(attentions).to(device)
  importance_labels = torch.cat(importance_labels).to(device)
  
  acc = accuracy_score(test_dataset.labels, preds)

  cms = multilabel_confusion_matrix(test_dataset.labels, preds)
  cr = classification_report(test_dataset.labels, preds)

  map_micro, map_macro = AP_partial(test_dataset.labels, scores)[1:3]
  spearman = spearman_correlation(attention_tensor[:, 0, 1:], importance_labels)

  return map_micro, map_macro, acc, spearman, cms, cr

def main():

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  model = MTResnetAggregate(args).to(device)
    
  if args.dataset == 'cufed':
    test_dataset = CUFED(root_dir=args.dataset_path, split_dir=args.split_path, is_train=False, img_size=args.img_size, album_clip_length=args.album_clip_length, ext_model=model.feature_extraction)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, num_workers=args.num_workers)
  else:
    exit("Unknown dataset!")
     
  if args.verbose:
    print("running on {}".format(device))
    print("test_set={}".format(len(test_dataset)))

  state = torch.load(args.model_path)
  print('load model from epoch', state['epoch'])
  model.load_state_dict(state['model_state_dict'])
  t0 = time.perf_counter()
  map_micro, map_macro, acc, spearman, cms, cr = evaluate(model, test_dataset, test_loader, device)
  t1 = time.perf_counter()

  if args.verbose:
    print("map_micro={} map_macro={} accuracy={} spearman={} dt={:.2f}sec".format(map_micro, map_macro, acc, spearman, t1 - t0))
    print(cr)
    showCM(cms)

if __name__ == '__main__':
  main()