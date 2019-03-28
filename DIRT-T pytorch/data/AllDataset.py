import torch.utils.data as data
import random
from PIL import Image
import os
import os.path
import numpy as np
from numpy.random import randint
from .datasets_video import VideoRecord, VideoRecordCha

class AllDataSet(data.Dataset):
    def __init__(self, root_path, list_file,num_frame=1,
                 new_length=1, modality='RGB',
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 force_grayscale=False, random_shift=True, test_mode=False):

        self.root_path = root_path
        self.list_file = list_file
        self.num_frame = num_frame
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        if 'charades' in self.list_file:
            self.image_tmpl = '{}-{:06d}.jpg'
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode

        if self.modality == 'RGBDiff':
            self.new_length += 1# Diff needs one more image to calculate diff

        self._parse_list()

    def _load_image(self, directory, idx):
        if not 'charades' in self.list_file:
            if self.modality == 'RGB' or self.modality == 'RGBDiff':
                try:
                    return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(idx))).convert('RGB')]
                except Exception:
                    print('error loading image:', os.path.join(self.root_path, directory, self.image_tmpl.format(idx)))
                    return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(1))).convert('RGB')]
            elif self.modality == 'Flow':
                try:
                    idx_skip = 1 + (idx-1)*5
                    flow = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(idx_skip))).convert('RGB')
                except Exception:
                    print('error loading flow file:', os.path.join(self.root_path, directory, self.image_tmpl.format(idx_skip)))
                    flow = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(1))).convert('RGB')
                # the input flow file is RGB image with (flow_x, flow_y, blank) for each channel
                flow_x, flow_y, _ = flow.split()
                x_img = flow_x.convert('L')
                y_img = flow_y.convert('L')

                return [x_img, y_img]
        else:
            if self.modality == 'RGB' or self.modality == 'RGBDiff':
                try:
                    return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(directory,idx))).convert('RGB')]
                except Exception:
                    print('error loading image:', os.path.join(self.root_path, directory, self.image_tmpl.format(idx)))
                    return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(1))).convert('RGB')]



    def _parse_list(self):
        # check the frame number is large >3:
        # usualy it is [video_id, num_frames, class_idx]
        tmp = [x.strip().split(' ') for x in open(self.list_file)]
        tmp = [item for item in tmp if int(item[1])>=3]
        if 'charades' in self.list_file:
            self.video_list = [VideoRecordCha(item) for item in tmp]
        else:
            self.video_list = [VideoRecord(item) for item in tmp]
        print('video number:%d'%(len(self.video_list)))


    def __getitem__(self, index):
        record = self.video_list[index]
        #frame_inde = np.linspace(0, record.num_frames, 30)
        #start_f = random.randint(1, record.num_frames-65)
        indices = np.linspace(1, record.num_frames, num=self.num_frame)
        if 'charades' in self.list_file:
            indices += record.start_frame
        return self.get(record, indices)

    def get(self, record, indices):
        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                seg_imgs = self._load_image(record.path, p)
                images.extend(seg_imgs)
                if p < record.num_frames:
                    p += 1

        process_data = self.transform(images)
        return process_data, record.label

    def __len__(self):
        return len(self.video_list)

