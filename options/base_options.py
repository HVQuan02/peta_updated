import argparse
import os
import torch

class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--scaled_width', type=int, default=1248, help='input width of the model')
        parser.add_argument('--scaled_height', type=int, default=384, help='input height of the model')
        parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment')
        parser.add_argument('--num_classes', type=int, default=23, help='number of labels')
        parser.add_argument('--pretrained', type=str, help='pretrained model path (.ckpt or .pth)')
        parser.add_argument('--num_threads', default=8, type=int, help='# threads for loading data')
        parser.add_argument('--seed', type=int, default=2024, help='seed for random generators')
        parser.add_argument('--use_ocr', action='store_true', help='apply OCR')
        parser.add_argument('--normalization', default='imagenet', help='normalization type: imagenet, default (1/255)')
        parser.add_argument('--val_root', required=False, help='root folder containing images for validation')
        parser.add_argument('--val_list', required=False, help='.txt file containing validation image list')
        parser.add_argument('--save_dir', type=str, default='weights', help='where checkpoints and log are save')
        parser.add_argument('--batch_size', type=int, default=2, help='input batch size')

        # Distributed
        parser.add_argument('--gpus', type=str, default='0', help='gpu ids for training, testing. e.g. 0 or 0,1,2')
        parser.add_argument('--accelerator', type=str, default='ddp', help='DataParallel (dp), DistributedDataParallel (ddp)')

        # For dataset paths configurations
        parser.add_argument(
            '--train_root', help='root folder containing images for training')
        parser.add_argument(
            '--train_list', help='.txt file containing training image list')
        parser.add_argument('--album_clip_length', type=int,
                            help='length of album', default=32)
        parser.add_argument('--input_size', type=int,
                            help='expected size of input image', default=224)
        parser.add_argument('--event_type_pth', type=str,
                            default='../CUFED_split/event_type.json')
        parser.add_argument('--image_importance_pth', type=str,
                            default='../CUFED_split/image_importance.json')
        parser.add_argument('--threshold', type=float, default=0.75)
        parser.add_argument('--ckpt_path', type=str, default=None)

        # accelerator
        parser.add_argument('--devices', type=int, default=1)
        parser.add_argument('--acc', type=str, default='gpu', choices=['cpu', 'gpu', 'tpu'])
        parser.add_argument('--strat', type=str, default='ddp')

        # Augmentations
        parser.add_argument('--crop_width', type=int,
                            default=1024, help='cropping width during training')
        parser.add_argument('--crop_height', type=int,
                            default=512, help='cropping height during training')
        parser.add_argument('--rotate', type=int, default=20,
                            help='angle for rotation')
        parser.add_argument('--color_aug_ratio', type=float, default=0.3,
                            help='ratio of images applied color augmentation')
        parser.add_argument('--only_flip', action='store_true',
                            help='apply only left-right flipping in augmentation')
        parser.add_argument('--use_color_aug', action='store_true',
                            help='apply color jitter in augmentation')
        parser.add_argument('--resume', type=str,
                          help='resume path for continue training (.ckpt)')
        parser.add_argument('--max_epoch', type=int,
                            default=100, help='maximum epochs')
        
        # adam
        parser.add_argument('--beta1', type=float,
                            default=0.5, help='momentum term of adam')
        # sgd
        parser.add_argument('--momentum', type=float,
                            default=0.9, help='momentum factor for sgd')

        # learning rate scheduler
        parser.add_argument('--lr_step', type=int, default=10,
                            help='step_size for step lr scheduler')
        parser.add_argument('--lr_milestones', metavar='N', nargs="+",
                            type=int, help='milestones for multi step lr scheduler')
        parser.add_argument('--lr_gamma', type=float,
                            default=0.9, help='gamma factor for lr_scheduler')

        # For early stopping
        parser.add_argument('--stopping_threshold', type=float, default=99.99, 
                            help='stops training when reach this threshold measured in mAP metric')

        # Choosing loss functions
        parser.add_argument('--loss', type=str, nargs='+', default='asymmetric', choices=[
                            'focal', 'asymmetric', 'bce'], help='loss function')
        parser.add_argument('--use_aux', action='store_true',
                            help='apply auxilary loss while training. Currently use with OCR net')
        parser.add_argument('--ocr_alpha_loss', type=float,
                            default=0.4, help='weight for OCR aux loss')

        parser.add_argument('--album_sample', type=str,
                            default='rand_permute')
        
        # use transformer
        parser.add_argument('--transformers_pos', type=int, default=1)
        parser.add_argument('--use_transformer', type=int, default=1)
        parser.add_argument('--gamma_neg', type=int, default=4)
        parser.add_argument('--gamma_pos', type=int, default=0.05)
        parser.add_argument('--clip', type=int, default=0.05)
        parser.add_argument('--num_classes_to_remove_negative_backprop', type=int, default=23)
        parser.add_argument('--partial_loss_mode', type=str, default='negative_backprop')
        parser.add_argument('--infer', type=int, default=0)
        parser.add_argument('--attention', type=str, default='multihead')
        
        # peta
        parser.add_argument('--model_name', type=str, default='mtresnetaggregate')
        parser.add_argument('--use_clip', action='store_true', help='use vit clip or vit')
        parser.add_argument('--num_workers', type=int, default=4, help='number of workers for data loader')
        parser.add_argument('--dataset', default='cufed', choices=['cufed', 'pec'])
        parser.add_argument('--dataset_path', type=str, default='/kaggle/input/thesis-cufed/CUFED')
        parser.add_argument('--feats_dir', type=str, default='/kaggle/input/mask-cufed-feats-bb')
        parser.add_argument('--split_path', type=str, default='/kaggle/input/cufed-full-split')
        parser.add_argument('--dataset_type', type=str, default='ML_CUFED')
        parser.add_argument('--img_size', type=int, default=224)
        parser.add_argument('--remove_model_jit', type=int, default=None)
        parser.add_argument('--backbone', type=str, default=None, help='feature extraction backbone network')
        parser.add_argument('-v', '--verbose', action='store_true', help='show details')

        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        self.parser = parser

        return parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        # expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        # util.mkdirs(expr_dir)
        # file_name = os.path.join(expr_dir, 'opt.txt')
        # with open(file_name, 'wt') as opt_file:
        #     opt_file.write(message)
        #     opt_file.write('\n')

    def parse(self):
        opt = self.gather_options()

        self.print_options(opt)

        # set gpu ids
        # str_ids = opt.gpu_ids.split(',')
        # opt.gpu_ids = []
        # for str_id in str_ids:
        #     id = int(str_id)
        #     if id >= 0:
        #         opt.gpu_ids.append(id)
        # if len(opt.gpu_ids) > 0:
        #     torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt