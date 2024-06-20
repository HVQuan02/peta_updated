from .base_options import BaseOptions

class InferOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--model_path', type=str, default=None)
        parser.add_argument('--album_path', type=str, default='/kaggle/working/peta_updated/albums/Graduation/0_92024390@N00')
        parser.add_argument('--top_k', type=int, default=3)
        parser.add_argument('--n_frames', type=int, default=5)
        parser.add_argument('--path_output', type=str, default='/kaggle/working/outputs')
        parser.add_argument('--ema', action='store_true', help='use ema model or not')
        return parser