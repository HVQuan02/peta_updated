from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)

        # For dataset paths configurations
        parser.add_argument(
            '--train_root', help='root folder containing images for training')
        parser.add_argument(
            '--train_list', help='.txt file containing training image list')

        parser.add_argument('--album_clip_length', type=int,
                            help='length of album', default=32)
        parser.add_argument('--event_type_pth', type=str,
                            default='../CUFED_split/event_type.json')
        parser.add_argument('--image_importance_pth', type=str,
                            default='../CUFED_split/image_importance.json')
        parser.add_argument('--threshold', type=float, default=0.85)
        parser.add_argument('--ckpt_path', type=str, default='/kaggle/working/peta_k18_thesis/model_ckpt')

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
                            default=150, help='maximum epochs')
        
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
        parser.add_argument('--stopping_threshold', type=float, default=90, 
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
        parser.add_argument(
            '--num_classes_to_remove_negative_backprop', type=int, default=23)
        parser.add_argument('--partial_loss_mode', type=str,
                            default='negative_backprop')
        parser.add_argument('--infer', type=int, default=0)
        parser.add_argument('--attention', type=str,
                            default='multihead')
        # PETA
        parser.add_argument('--model_name', type=str, default='mtresnetaggregate')
        parser.add_argument('--dataset', default='cufed', choices=['cufed', 'pec'])
        parser.add_argument('--dataset_path', type=str, default='/kaggle/input/thesis-cufed/CUFED')
        parser.add_argument('--split_path', type=str, default='/kaggle/input/cufed-full-split')
        parser.add_argument('--dataset_type', type=str, default='ML_CUFED')
        parser.add_argument('--train_batch_size', type=int, default=14, help='train batch size')
        parser.add_argument('--val_batch_size', type=int, default=32, help='validate batch size')
        parser.add_argument('--num_workers', type=int, default=4, help='number of workers for data loader')
        parser.add_argument('-v', '--verbose', action='store_true', help='show details')
        parser.add_argument('--img_size', type=int, default=224)
        parser.add_argument('--remove_model_jit', type=int, default=None)
        parser.add_argument('--optimizer', type=str, default='adamw', choices=['sgd', 'adam', 'adamw'])
        parser.add_argument('--lr_policy', type=str, default='cosine', choices=['cosine', 'step', 'multi_step', 'onecycle'])
        parser.add_argument('--lr', type=float, default=1e-5, help='base learning rate')
        parser.add_argument('--weight_decay', type=float, default=1e-3, help='weight decay rate')
        parser.add_argument('--warmup_epochs', type=int, default=10, help='number of warmup epochs')
        parser.add_argument('--save_folder', default='/kaggle/working/weights', help='directory to save checkpoints')
        parser.add_argument('--patience', type=int, default=20, help='patience of early stopping')
        parser.add_argument('--min_delta', type=float, default=0.1, help='min delta of early stopping')

        return parser