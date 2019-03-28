import time
from options.train_options import TrainOptions
import models
from util.util import *
import os.path as osp

import os, sys
from PIL import Image
import six
import string

import lmdb
import pickle
import msgpack
import tqdm
import pyarrow as pa

import torch.utils.data as data
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

def dumps_pyarrow(obj):
    """
    Serialize an object.
    Returns:
        Implementation-dependent bytes-like object
    """
    return pa.serialize(obj).to_buffer()

def raw_reader(path):
    with open(path, 'rb') as f:
        bin_data = f.read()
    return bin_data




dpath = sys.argv[1]
name = sys.argv[2]
write_frequency=5000
image_dataset = datasets.ImageFolder(dpath,loader=raw_reader)

data_loader = DataLoader(image_dataset, num_workers=16, collate_fn=lambda x: x)

lmdb_path = osp.join(dpath, "%s.lmdb" % name)
isdir = os.path.isdir(lmdb_path)

print("Generate LMDB to %s" % lmdb_path)
db = lmdb.open(lmdb_path, subdir=isdir,
               map_size=1099511627776 * 2, readonly=False,
               meminit=False, map_async=True)

txn = db.begin(write=True)
for idx, data in enumerate(data_loader):
    # print(type(data), data)
    image, label = data[0]
    txn.put(u'{}'.format(idx).encode('ascii'), dumps_pyarrow((image, label)))
    if idx % write_frequency == 0:
        print("[%d/%d]" % (idx, len(data_loader)))
        txn.commit()
        txn = db.begin(write=True)

# finish iterating through dataset
txn.commit()
keys = [u'{}'.format(k).encode('ascii') for k in range(idx + 1)]
with db.begin(write=True) as txn:
    txn.put(b'__keys__', dumps_pyarrow(keys))
    txn.put(b'__len__', dumps_pyarrow(len(keys)))

print("Flushing database ...")
db.sync()
db.close()
