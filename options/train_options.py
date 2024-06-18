from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--train_batch_size', type=int, default=16, help='train batch size')
        parser.add_argument('--val_batch_size', type=int, default=16, help='validate batch size')
        parser.add_argument('--optimizer', type=str, default='adamw', choices=['sgd', 'adam', 'adamw'])
        parser.add_argument('--lr_policy', type=str, default='cosine', choices=['cosine', 'step', 'multi_step', 'onecycle'])
        parser.add_argument('--lr', type=float, default=1e-5, help='base learning rate')
        parser.add_argument('--weight_decay', type=float, default=1e-3, help='weight decay rate')
        parser.add_argument('--warmup_epochs', type=int, default=10, help='number of warmup epochs')
        parser.add_argument('--patience', type=int, default=20, help='patience of early stopping')
        parser.add_argument('--min_delta', type=float, default=0.1, help='min delta of early stopping')
        return parser