import argparse
import time
import torch
from torch.utils.data import DataLoader
from src.models import create_model
from sklearn.metrics import average_precision_score, accuracy_score

from datasets import CUFED

parser = argparse.ArgumentParser(description='Photo Album Event Recognition')
parser.add_argument('--model_path', type=str, default='./models_local/peta_32.pth')
parser.add_argument('--model_name', type=str, default='mtresnetaggregate')
parser.add_argument('--num_classes', type=int, default=23)
parser.add_argument('--dataset', default='cufed', choices=['cufed', 'pec', 'holiday'])
parser.add_argument('--dataset_path', type=str, default='/content/drive/MyDrive/CUFED-Event-Image/CUFED')
parser.add_argument('--dataset_type', type=str, default='ML_CUFED')
parser.add_argument('--batch_size', type=int, default=32, help='batch size') # change
parser.add_argument('--transform_type', type=str, default='squish')
parser.add_argument('--album_sample', type=str, default='rand_permute')
parser.add_argument('--num_workers', type=int, default=2, help='number of workers for data loader')
parser.add_argument('--save_scores', action='store_true', help='save the output scores')
parser.add_argument('--save_path', default='scores.txt', help='output path')
parser.add_argument('-v', '--verbose', action='store_true', help='show details')
parser.add_argument('--img_size', type=int, default=224)
parser.add_argument('--album_clip_length', type=int, default=32)
parser.add_argument('--remove_model_jit', type=int, default=None)
parser.add_argument('--use_transformer', type=int, default=1)
parser.add_argument('--transformers_pos', type=int, default=1)
parser.add_argument('--path_output', type=str, default='./outputs')
args = parser.parse_args()

def evaluate(model, dataset, loader, scores, out_file, device):
    gidx = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            feats = batch.to(device)
            out_data = model(feats)
            shape = out_data.shape[0]

            if out_file:
                for j in range(shape):
                    video_name = dataset.videos[gidx + j]
                    out_file.write("{} ".format(video_name))
                    out_file.write(' '.join([str(x.item()) for x in out_data[j, :]]))
                    out_file.write('\n')

            scores[gidx:gidx+shape, :] = out_data.cpu()
            gidx += shape

def main():
  if args.dataset == 'cufed':
    dataset = CUFED(root_dir=args.dataset_path, is_train=False, img_size=args.img_size, album_clip_length=args.album_clip_length)
  device = torch.device('cuda:0')
  val_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)

  if args.verbose:
    print("running on {}".format(device))
    print("num samples={}".format(len(dataset)))

  out_file = None
  if args.save_scores:
    out_file = open(args.save_path, 'w')

  # Setup model
  print('creating and loading the model...')
  state = torch.load(args.model_path, map_location='cpu')
  model = create_model(args).cuda()
  model.load_state_dict(state['model'], strict=True)

  num_test = len(dataset)
  scores = torch.zeros((num_test, len(dataset.event_labels)), dtype=torch.float32)

  t0 = time.perf_counter()
  evaluate(model, dataset, val_loader, scores, out_file, device)
  t1 = time.perf_counter()
  
  # # Change tensors to 1d-arrays
  scores = scores.numpy()

  if args.save_scores:
    out_file.close()

  if args.dataset == 'cufed':
    ap = average_precision_score(dataset.labels, scores) 
    # ap = accuracy_score(dataset.labels, scores)
    print('top1={:.2f}% dt={:.2f}sec'.format(100 * ap, t1 - t0))

if __name__ == '__main__':
  main()