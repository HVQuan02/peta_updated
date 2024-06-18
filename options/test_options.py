from .base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--model_path', type=str, default=None)
        parser.add_argument('--test_batch_size', type=int, default=16, help='test batch size')
        parser.add_argument('--ema', action='store_true', help='use ema model or not')
        return parser