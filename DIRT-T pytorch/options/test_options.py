from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes training options.
    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)

        parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--gpu_ids', '--gpu', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--is_train', default=False, type=bool, help='training')
        parser.add_argument('--test_db', default='', type=str, help='lmdb path for train')
        parser.add_argument('--test_trans', default='', type=str, help='transforms for train', nargs='+')     
        parser.add_argument('--train_db', default='/home/liguangrui/data/train_all.lmdb', type=str, help='lmdb path for train')
        parser.add_argument('--val_db', default='/home/liguangrui/data/val.lmdb', type=str, help='lmdb path for validation')

        self.isTrain = False
        return parser
