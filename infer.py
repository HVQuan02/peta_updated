import os
import cv2
import json
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from torchvision.utils import make_grid
from models.models import MTResnetAggregate
from options.infer_options import InferOptions
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

args = InferOptions().parse()


def get_album_importance(album_imgs, album_importance):
        img_to_score = {}
        for _, image, score in album_importance:
            img_to_score[image.split('/')[1]] = score
        importance = np.zeros(len(album_imgs))
        for i, image in enumerate(album_imgs):
            importance[i] = img_to_score[image[:-4]]
        return importance


def get_album(args, device):
    album_name = args.album_path.split('/')[-1]

    if args.backbone:
        files = os.listdir(args.album_path)
        idx_fetch = np.linspace(0, len(files) - 1, args.album_clip_length, dtype=int)
        imgs = [files[id] for id in idx_fetch]
    elif args.use_clip:
        albimg_path = 'album_imgs_mask.json'
        with open(os.path.join(args.split_path, albimg_path), 'r') as f:
            album_imgs = json.load(f)
        imgs = album_imgs[album_name]
    else:
        albimg_path = 'album_imgs.json'
        with open(os.path.join(args.split_path, albimg_path), 'r') as f:
            album_imgs = json.load(f)
        imgs = [img.split('/')[-1] + '.jpg' for img in imgs]

    with open(os.path.join(args.dataset_path, "image_importance.json"), 'r') as f:
        album_importance = json.load(f)
    importance_album = album_importance[album_name]
    t_importance = get_album_importance(imgs, importance_album)

    tensor_batch = torch.zeros(len(imgs), args.input_size, args.input_size, 3)
    for i, img in enumerate(imgs):
        im = Image.open(os.path.join(args.album_path, img))
        im_resize = im.resize((args.input_size, args.input_size))
        np_img = np.array(im_resize, dtype=np.uint8)
        tensor_batch[i] = torch.from_numpy(np_img).float() / 255.0
    tensor_batch = tensor_batch.permute(0, 3, 1, 2)   # HWC to CHW
    montage = make_grid(tensor_batch).permute(1, 2, 0).cpu()
    tensor_batch = torch.unsqueeze(tensor_batch, 0).to(device)
    
    if args.backbone:
        global_feat = None
    elif args.use_clip:
        global_path = os.path.join(args.feats_dir, 'clip_global', album_name + '.npy')
        global_feat = torch.from_numpy(np.load(global_path))
        global_feat = global_feat.unsqueeze(0).to(device)
    else:
        global_path = os.path.join(args.feats_dir, 'vit_global', album_name + '.npy')
        global_feat = torch.from_numpy(np.load(global_path))
        global_feat = global_feat.unsqueeze(0).to(device)
    return global_feat, t_importance, tensor_batch, montage


def inference(global_feat, t_importance, tensor_batch, model, classes_list, output_path, args, nrow=5):
    if global_feat is not None:
        logits, attention = model(global_feat)
    else:
        logits, attention = model(tensor_batch)
    output = torch.squeeze(torch.sigmoid(logits))
    np_output = output.cpu().detach().numpy()
    idx_sort = np.argsort(-np_output)

    importance = torch.squeeze(attention[:, 0, 1:])
    np_importance = importance.cpu().detach().numpy()

    spearman = spearmanr(t_importance, np_importance).statistic
    print('spearman = {:.3f}'.format(spearman))

    top_idx = np.argsort(-np_importance)
    worst_idx = np.argsort(np_importance)
    album_np = tensor_batch.squeeze(0).cpu().detach().numpy()
    top_frames = album_np[top_idx][:args.n_frames]
    for i, top_frame in enumerate(top_frames):
        save_image(top_frame, 'salient_{}'.format(i + 1), output_path)
    worst_frames = album_np[worst_idx][:args.n_frames]
    top_montage = make_grid(torch.from_numpy(top_frames), nrow=nrow).permute(1, 2, 0).cpu()
    worst_montage = make_grid(torch.from_numpy(worst_frames), nrow=nrow).permute(1, 2, 0).cpu()

    # Top-k
    detected_classes = np.array(classes_list)[idx_sort][:args.top_k]
    scores = np_output[idx_sort][: args.top_k]
#     print('detected_classes', detected_classes)
#     print('scores', scores)
    # Threshold
    idx_th = scores > args.threshold

    return detected_classes[idx_th], scores[idx_th], top_montage, worst_montage


def display_montage(montage, tags, filename, path_dest, dpi=300, file_format='jpg'):
    ext = '.' + file_format
    if not os.path.exists(path_dest):
        os.makedirs(path_dest)
    plt.figure(figsize=(montage.shape[1] / dpi, montage.shape[0] / dpi), dpi=dpi)
    plt.axis('off')
    plt.rcParams["axes.titlesize"] = 16
    plt.title(tags)
    plt.imshow(montage)
    plt.savefig(os.path.join(path_dest, filename + ext), dpi=dpi, format=file_format, bbox_inches='tight', pad_inches=0)


def save_image(image, filename, path_dest, file_format='jpg'):
    image = np.transpose(image, (1, 2, 0)) * 255
    image = image[:, :, ::-1].copy()
    ext = '.' + file_format
    if not os.path.exists(path_dest):
        os.makedirs(path_dest)
    cv2.imwrite(os.path.join(path_dest, filename + ext), image)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    classes_list = np.array(['Architecture', 'BeachTrip', 'Birthday', 'BusinessActivity',
                    'CasualFamilyGather', 'Christmas', 'Cruise', 'Graduation',
                    'GroupActivity', 'Halloween', 'Museum', 'NatureTrip',
                    'PersonalArtActivity', 'PersonalMusicActivity', 'PersonalSports',
                    'Protest', 'ReligiousActivity', 'Show', 'Sports', 'ThemePark',
                    'UrbanTrip', 'Wedding', 'Zoo'])

    model = MTResnetAggregate(args)
    if args.ema:
        model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.999))

    state = torch.load(args.model_path, map_location=device)
    print('load model from epoch {}'.format(state['epoch']))
    model.load_state_dict(state['model_state_dict'], strict=True)
    model.eval()
    model = model.to(device)

    output_path = os.path.join(args.path_output, args.album_path.split('/')[-1])

    # Get album
    global_feat, t_importance, tensor_batch, montage = get_album(args, device)

    # Inference
    tags, confs, top_montage, worst_montage = inference(global_feat, t_importance, tensor_batch, model, classes_list, output_path, args)

    # Visualization
#     display_montage(montage, 'Predicted classes: {}'.format(tags), 'montage', output_path)
#     display_montage(top_montage, None, 'best_montage', output_path)
#     display_montage(worst_montage, None, 'worst_montage', output_path)


if __name__ == '__main__':
    main()