import importlib
from .networks import *
import torch.nn as nn
from .dirt_t import *
def create_model(opt, optimizer):
    scheduler = networks.get_scheduler(optimizer,opt)
    criterion = nn.CrossEntropyLoss().cuda()
    #return optimizer, criterion
    return criterion#, scheduler

def find_model_using_name(model_name):
    """Import the module "models/[model_name]_model.py".

    In the file, the class called DatasetNameModel() will
    be instantiated. It has to be a subclass of BaseModel,
    and it is case-insensitive.
    """
    model_filename = "models." + model_name
    modellib = importlib.import_module(model_filename)
    model = None
    target_model_name = model_name.replace('_', '')
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower(): 
            model = cls

    if model is None:
        print("In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase." % (model_filename, target_model_name))
        exit(0)

    return model

def create_model(opt):
    """Create a model given the option.

    This function warps the class CustomDatasetDataLoader.
    This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from models import create_model
        >>> model = create_model(opt)
    """
    model = find_model_using_name(opt.model)
    instance = model(opt)
    init_weights(instance, init_type='kaiming')
    discriminator = Discriminator(opt)
    discriminator = discriminator.cuda()
    init_weights(discriminator, init_type='kaiming')
    print("model [%s] was created" % type(instance).__name__)
#    instance = nn.DataParallel(instance).cuda()
    instance=instance.cuda() 
    return instance, discriminator

