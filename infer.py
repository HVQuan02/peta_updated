import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from models.models import MTResnetAggregate
from options.infer_options import InferOptions
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

args = InferOptions().parse()


def get_album(args, device):
    files = os.listdir(args.album_path)
    idx_fetch = np.linspace(0, len(files) - 1, args.album_clip_length, dtype=int)
    tensor_batch = torch.zeros(len(idx_fetch), args.input_size, args.input_size, 3)
    for i, id in enumerate(idx_fetch):
        im = Image.open(os.path.join(args.album_path, files[id]))
        im_resize = im.resize((args.input_size, args.input_size))
        np_img = np.array(im_resize, dtype=np.uint8)
        tensor_batch[i] = torch.from_numpy(np_img).float() / 255.0
    tensor_batch = tensor_batch.permute(0, 3, 1, 2)   # HWC to CHW
    montage = make_grid(tensor_batch).permute(1, 2, 0).cpu()
    tensor_batch = torch.unsqueeze(tensor_batch, 0).to(device)
    return tensor_batch, montage


def inference(tensor_batch, model, classes_list, args):
    logits, attention = model(tensor_batch)
    output = torch.squeeze(torch.sigmoid(logits))
    np_output = output.cpu().detach().numpy()
    idx_sort = np.argsort(-np_output)

    importance = torch.squeeze(attention[:, 0, 1:])
    np_importance = importance.cpu().detach().numpy()
    top_idx = np.argsort(-np_importance)
    worst_idx = np.argsort(np_importance)
    album_np = tensor_batch.squeeze(0).cpu().detach().numpy()
    top_frames = album_np[top_idx][:args.n_frames]
    worst_frames = album_np[worst_idx][:args.n_frames]
    top_montage = make_grid(torch.from_numpy(top_frames)).permute(1, 2, 0).cpu()
    worst_montage = make_grid(torch.from_numpy(worst_frames)).permute(1, 2, 0).cpu()

    # Top-k
    detected_classes = np.array(classes_list)[idx_sort][:args.top_k]
    scores = np_output[idx_sort][: args.top_k]
    # Threshold
    idx_th = scores > args.threshold

    return detected_classes[idx_th], scores[idx_th], top_montage, worst_montage


def display_image(montage, tags, filename, path_dest):
    if not os.path.exists(path_dest):
        os.makedirs(path_dest)
    plt.figure()
    plt.imshow(montage)
    plt.axis('off')
    plt.rcParams["axes.titlesize"] = 16
    plt.title("Predicted classes: {}".format(tags))
    plt.savefig(os.path.join(path_dest, filename))


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    classes_list = np.array(['Architecture', 'BeachTrip', 'Birthday', 'BusinessActivity',
        'CasualFamilyGather', 'Christmas', 'Cruise', 'Graduation', 'GroupActivity',
        'Halloween', 'Museum', 'NatureTrip', 'PersonalArtActivity',
        'PersonalMusicActivity', 'PersonalSports', 'Protest', 'ReligiousActivity',
        'Show', 'Sports', 'ThemePark', 'UrbanTrip', 'Wedding', 'Zoo'])

    model = MTResnetAggregate(args)
    if args.ema:
        model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.999))

    state = torch.load(args.model_path, map_location=device)
    print('load model from epoch {}'.format(state['epoch']))
    model.load_state_dict(state['model_state_dict'], strict=True)
    model = model.to(device)
    model.eval()

    # Get album
    tensor_batch, montage = get_album(args, device)

    # Inference
    tags, confs, top_montage, worst_montage = inference(tensor_batch, model, classes_list, args)

    # Visualization
    display_image(montage, tags, 'montage_event.jpg', os.path.join(args.path_output, args.album_path).replace("./albums", ""))
    print('confs:', confs)
    display_image(top_montage, 'Best frames', 'top_montage.jpg', os.path.join(args.path_output, args.album_path).replace("./albums", ""))
    display_image(worst_montage, 'Worst frames', 'worst_montage.jpg', os.path.join(args.path_output, args.album_path).replace("./albums", ""))


if __name__ == '__main__':
    main()