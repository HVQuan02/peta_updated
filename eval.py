import argparse
import time
import torch
from torch.utils.data import DataLoader
from src.utils.evaluation import AP_partial
from src.report_manager.utils import accuracy
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from datasets import CUFED
from models.models import MTResnetAggregate

parser = argparse.ArgumentParser(description='PETA: Photo Album Event Recognition')
parser.add_argument('--model_path', type=str, default='./weights/PETA-cufed.pt')
parser.add_argument('--model_name', type=str, default='mtresnetaggregate')
parser.add_argument('--num_classes', type=int, default=23)
parser.add_argument('--dataset', default='cufed', choices=['cufed', 'pec', 'holidays'])
parser.add_argument('--metric', default='map', choices=['map', 'accuracy'])
parser.add_argument('--dataset_path', type=str, default='/kaggle/input/thesis-cufed/CUFED')
parser.add_argument('--split_path', type=str, default='/kaggle/working/split_dir')
parser.add_argument('--dataset_type', type=str, default='ML_CUFED')
parser.add_argument('--batch_size', type=int, default=32, help='batch size') # change
parser.add_argument('--num_workers', type=int, default=4, help='number of workers for data loader')
parser.add_argument('--save_scores', action='store_true', help='save the output scores')
parser.add_argument('--ema', action='store_true', help='use ema model or not')
parser.add_argument('--save_path', default='scores.txt', help='output path of predicted scores')
parser.add_argument('-v', '--verbose', action='store_true', help='show details')
parser.add_argument('--img_size', type=int, default=224)
parser.add_argument('--album_clip_length', type=int, default=32)
parser.add_argument('--remove_model_jit', type=int, default=None)
parser.add_argument('--use_transformer', type=int, default=1)
parser.add_argument('--transformers_pos', type=int, default=1)
parser.add_argument('--path_output', type=str, default='./outputs')
parser.add_argument('--backbone', type=str, default='resnet101')
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--transform_type', type=str, default='squish')
parser.add_argument('--album_sample', type=str, default='rand_permute')
parser.add_argument('--path_output', type=str, default='./outputs')
parser.add_argument('--top_k', type=int, default=3)
parser.add_argument('--threshold', type=float, default=0.85)
parser.add_argument('--attention', type=str, default='multihead')
args = parser.parse_args()

def load_model(net, path):
    if path is not None and path.endswith(".ckpt"):
        state_dict = torch.load(path, map_location='cpu')

        if "model" in state_dict:
            state_dict = state_dict["model"]
        compatible_state_dict = {}
        for k, v in state_dict.items():
            k = k[4:]
            compatible_state_dict[k] = v

        net.load_state_dict(compatible_state_dict)

    return net

def evaluate(model, dataset, loader, scores, out_file, device):
    gidx = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            feats, _ = batch
            feats = feats.to(device)
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
    dataset = CUFED(root_dir=args.dataset_path, split_dir=args.split_path, is_train=False, img_size=args.img_size, album_clip_length=args.album_clip_length)
  else:
    exit("Unknown dataset!")
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  eval_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

  if args.verbose:
    print("running on {}".format(device))
    print("num samples={}".format(len(dataset)))

  out_file = None
  if args.save_scores:
    out_file = open(args.save_path, 'w')

  # Setup model
  print('creating and loading the model...')
  net = MTResnetAggregate(args).to(device)
  if args.ema:
    net = AveragedModel(net, multi_avg_fn=get_ema_multi_avg_fn(0.999))
  model = load_model(net, args.model_path)

  num_test = len(dataset)
  scores = torch.zeros((num_test, len(dataset.event_labels)), dtype=torch.float32)

  t0 = time.perf_counter()
  evaluate(model, dataset, eval_loader, scores, out_file, device)
  t1 = time.perf_counter()
  
  # # Change tensors to 1d-arrays
  scores = scores.numpy()

  if args.save_scores:
    out_file.close()

  if args.metric == 'map':
    mark = AP_partial(dataset.labels, scores)[1]
  else:
    mark = accuracy(dataset.labels, scores)
  print('top1_{}={:.2f}% dt={:.2f}sec'.format(args.metric, mark, t1 - t0))

if __name__ == '__main__':
  main()