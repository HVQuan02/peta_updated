import torch
import os
import matplotlib.pyplot as plt
import torchvision.utils
from PIL import Image
import numpy as np
from src.models import create_model
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from options.infer_options import InferOptions

args = InferOptions().parse()

def get_album(args, device):
    files = os.listdir(args.album_path)
    idx_fetch = np.linspace(0, len(files)-1, args.album_clip_length, dtype=int)
    tensor_batch = torch.zeros(len(idx_fetch), args.input_size, args.input_size, 3)
    for i, id in enumerate(idx_fetch):
        im = Image.open(os.path.join(args.album_path, files[id]))
        im_resize = im.resize((args.input_size, args.input_size))
        np_img = np.array(im_resize, dtype=np.uint8)
        tensor_batch[i] = torch.from_numpy(np_img).float() / 255.0
    tensor_batch = tensor_batch.permute(0, 3, 1, 2)   # HWC to CHW
    montage = torchvision.utils.make_grid(tensor_batch).permute(1, 2, 0).cpu()
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
    top_montage = torchvision.utils.make_grid(top_frames).permute(1, 2, 0).cpu()
    worst_montage = torchvision.utils.make_grid(worst_frames).permute(1, 2, 0).cpu()

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
    plt.axis('tight')
    plt.rcParams["axes.titlesize"] = 16
    plt.title("Predicted classes: {}".format(tags))
    plt.savefig(os.path.join(path_dest, filename))

def show_top_frames(model, tensor_batch):
    _, attention = model(tensor_batch)
    importance = torch.squeeze(attention[:, 0, 1:])
    np_importance = importance.cpu().detach().numpy()
    idx_sort = np.argsort(-np_importance)
    top_frames = tensor_batch.squeeze(0).cpu().detach().numpy()[idx_sort][:args.n_frames]
    montage = torchvision.utils.make_grid(top_frames).permute(1, 2, 0).cpu()

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Setup model
    state = torch.load(args.model_path, map_location='cpu')
    model = create_model(args).to(device)
    if args.ema:
        model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.999))
    print('load model from epoch {}'.format(state['epoch']))
    model.load_state_dict(state['model_state_dict'], strict=True)
    model.eval()
    classes_list = np.array(['Architecture', 'BeachTrip', 'Birthday', 'BusinessActivity',
        'CasualFamilyGather', 'Christmas', 'Cruise', 'Graduation', 'GroupActivity',
        'Halloween', 'Museum', 'NatureTrip', 'PersonalArtActivity',
        'PersonalMusicActivity', 'PersonalSports', 'Protest', 'ReligiousActivity',
        'Show', 'Sports', 'ThemePark', 'UrbanTrip', 'Wedding', 'Zoo'])

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
