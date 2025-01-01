import torch
from torch.utils.data import Dataset 
import json
import os 
from PIL import Image
from utils_kaist_multi import transform


class KaistDataset(Dataset): # Dataset : 미리 준비해둔 데이터셋
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, split, keep_difficult=False):
        """
        :param data_folder: folder where data files are stored
        :param split: split, one of 'TRAIN' or 'TEST'
        :param keep_difficult: keep or discard objects that are considered difficult to detect?
        """
        self.split = split.upper()

        assert self.split in {'TRAIN', 'TEST'}

        self.data_folder = '/home/urp1/workspace/kaist_dataset'
        self.keep_difficult = keep_difficult
        
        # Read data files
        with open(os.path.join(data_folder, self.split + '_visible_images.json'), 'r') as j:
            # /home/urp1/workspace/kaist_dataset/TRAIN_visible_images.json
            self.images_R = json.load(j)
        with open(os.path.join(data_folder, self.split + '_visible_objects.json'), 'r') as j:
            self.objects_R = json.load(j)
            
        with open(os.path.join(data_folder, self.split + '_lwir_images.json'), 'r') as j:
            self.images_T = json.load(j)
        with open(os.path.join(data_folder, self.split + '_lwir_objects.json'), 'r') as j:
            self.objects_T = json.load(j)
                
        assert len(self.images_R) == len(self.objects_R)
        assert len(self.images_T) == len(self.objects_T)
                
    def __getitem__(self, i):
        # Read image
        image_R = Image.open(self.images_R[i], mode='r') 
        # <PIL.PngImagePlugin.PngImageFile image mode=RGB size=640x512 at 0x7FF6AEC53A10>
        image_R = image_R.convert('RGB') # 3채널로 바뀐다 <-> gray
        # <PIL.Image.Image image mode=RGB size=640x512 at 0x7FF6C4833310>
        
        image_T = Image.open(self.images_T[i], mode='r')
        image_T = image_T.convert('RGB') 
        
        # Read objects in this image (bounding boxes, labels, difficulties)
        objects = self.objects_R[i]
        boxes = torch.FloatTensor(objects['bbox'])  # (n_objects, 4)
        labels = torch.LongTensor(objects['category_id'])  # (n_objects)
        difficulties = torch.ByteTensor(objects['is_crowd'])  # (n_objects)

        # Discard difficult objects, if desired
        if not self.keep_difficult:
            boxes = boxes[1 - difficulties]
            labels = labels[1 - difficulties]
            difficulties = difficulties[1 - difficulties]

        # Apply transformations
        image_R, image_T, boxes, labels, difficulties = transform(image_R, image_T, boxes, labels, difficulties, split=self.split)
        # image_T, boxes, labels, difficulties = transform(image_T, boxes, labels, difficulties, split=self.split)
        # image_R, boxes, labels, difficulties = transform(image_R, boxes, labels, difficulties, split=self.split)
        # Transform 두번 적용하면 같은 이미지에 대해서 다르게 Transform됌
        # 근데 수정은 했는데 잘 됐는지 확인 법은?        
        
        # import pdb;pdb.set_trace()

        return image_R, image_T, boxes, labels, difficulties

    def __len__(self):
        return len(self.images_T)

    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).
        이미지마다 객체 수가 다를 수 있다.

        This describes how to combine these tensors of different sizes. We use lists.
        크기 다른 텐서 결합

        Note: this need not be defined in this Class, can be standalone.

        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """
        
        images_R = list()
        images_T = list()
        boxes = list()
        labels = list()
        difficulties = list()

        for b in batch:
            images_R.append(b[0])
            images_T.append(b[1])
            boxes.append(b[2])
            labels.append(b[3])
            difficulties.append(b[4])

        images_R = torch.stack(images_R, dim=0)
        images_T = torch.stack(images_T, dim=0)

        return images_R, images_T, boxes, labels, difficulties  # tensor (N, 3, 300, 300), 3 lists of N tensors each


