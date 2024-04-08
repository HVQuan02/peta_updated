import argparse
# import time
# import torch
from torch.utils.data import DataLoader
# from sklearn.metrics import average_precision_score, accuracy_score
# import numpy as np

from datasets import CUFED

parser = argparse.ArgumentParser(description='Photo Album Event Recognition')
# parser.add_argument('model', nargs=1, help='trained model')
# parser.add_argument('--gcn_layers', type=int, default=2, help='number of gcn layers')
parser.add_argument('--dataset', default='cufed', choices=['cufed', 'pec', 'holiday'])
# parser.add_argument('--dataset_root', default='/home/dimidask/Projects/FCVID', help='dataset root directory')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
# parser.add_argument('--num_objects', type=int, default=50, help='number of objects with best DoC')
parser.add_argument('--num_workers', type=int, default=2, help='number of workers for data loader')
# parser.add_argument('--ext_method', default='VIT', choices=['VIT', 'RESNET'], help='Extraction method for features')
# parser.add_argument('--save_scores', action='store_true', help='save the output scores')
# parser.add_argument('--save_path', default='scores.txt', help='output path')
# parser.add_argument('-v', '--verbose', action='store_true', help='show details')
parser.add_argument('--img_size', type=int, default=224)
parser.add_argument('--album_clip_length', type=int, default=32)
args = parser.parse_args()

# def evaluate(model, dataset, loader, scores, out_file, device):
#     gidx = 0
#     model.eval()
#     with torch.no_grad():
#         for i, batch in enumerate(loader):
#             feats, feat_global, _, _ = batch

#             # Run model with all frames
#             feats = feats.to(device)
#             feat_global = feat_global.to(device)
#             out_data = model(feats, feat_global, device)

#             shape = out_data.shape[0]

#             if out_file:
#                 for j in range(shape):
#                     video_name = dataset.videos[gidx + j]
#                     out_file.write("{} ".format(video_name))
#                     out_file.write(' '.join([str(x.item()) for x in out_data[j, :]]))
#                     out_file.write('\n')

#             scores[gidx:gidx+shape, :] = out_data.cpu()
#             gidx += shape

def main():
  if args.dataset == 'cufed':
    dataset = CUFED(args.dataset_root, is_train=False, args.img_size, args.album_clip_length)
  # device = torch.device('cuda:0')
  print(len(dataset))
  val_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)

  for i, batch in enumerate(val_loader):
    print(batch.shape)
  # out_file = None
  #   if args.save_scores:
  #     out_file = open(args.save_path, 'w')

# Setup model
  # print('creating and loading the model...')
  # state = torch.load(args.model_path, map_location='cpu')
  # # args.num_classes = state['num_classes']
  # model = create_model(args).cuda()
  # model.load_state_dict(state['model'], strict=True)

  # num_test = len(dataset)
  # scores = torch.zeros((num_test, dataset.NUM_CLASS), dtype=torch.float32)

  # t0 = time.perf_counter()
  # evaluate(model, dataset, loader, scores, out_file, device)
  # t1 = time.perf_counter()
  
  # # Change tensors to 1d-arrays
  # scores = scores.numpy()

  # if args.save_scores:
  #   out_file.close()

  # if args.dataset == 'actnet':
  #   ap = average_precision_score(dataset.labels, scores)
  #   print('top1={:.2f}% dt={:.2f}sec'.format(100 * ap, t1 - t0))

if __name__ == '__main__':
  main()