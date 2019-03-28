from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.
    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)

        parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--gpu_ids', '--gpu', type=str, default='1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--is_train', default=True, type=bool, help='training')
        parser.add_argument('--src_train_db', default='/home/liguangrui/data/digits/mnist/trainset/train.lmdb', type=str, help='lmdb path for train')
        parser.add_argument('--src_val_db', default='/home/liguangrui/data/digits/mnist/testset/test.lmdb', type=str, help='lmdb path for validation')

        parser.add_argument('--tgt_train_db', default='/home/liguangrui/data/digits/svhn/trainset/train.lmdb', type=str, help='lmdb path for train')
        parser.add_argument('--tgt_val_db', default='/home/liguangrui/data/digits/svhn/testset/test.lmdb', type=str, help='lmdb path for validation')
        parser.add_argument('--dataset', default='digit', type=str, help='which dataset for training')
        parser.add_argument('--source',type=str, default='svhn')
        parser.add_argument('--target', type=str, default='mnist')
        parser.add_argument('--train_trans', default='', type=str, help='transforms for train', nargs='+')
        parser.add_argument('--val_trans', default='', type=str, help='transforms for validation',nargs="+")       
        parser.add_argument('--num_cls', default='1000', type=int, help='number of classes in classification task')
        parser.add_argument('--ins_norm', default=True, type=bool, help='training')
        parser.add_argument('--small', default=True, type=bool)
        parser.add_argument('--noise', default=True, type=bool)
        parser.add_argument('--radius', default=3.5, type=float)
        parser.add_argument('--mode', default='vada', type=str)
        self.isTrain = True
        return parser
