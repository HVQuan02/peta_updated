from .base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--dataset', default='cufed', choices=['cufed', 'pec'])
        parser.add_argument('--dataset_path', type=str, default='/kaggle/input/thesis-cufed/CUFED')
        parser.add_argument('--split_path', type=str, default='/kaggle/input/cufed-full-split')
        parser.add_argument('--dataset_type', type=str, default='ML_CUFED')
        parser.add_argument('--test_batch_size', type=int, default=16, help='test batch size')
        parser.add_argument('--backbone', type=str, default='vit_large_patch14_clip_224.openai', help='feature extraction backbone network')
        return parser