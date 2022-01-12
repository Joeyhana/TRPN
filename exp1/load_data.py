# from __future__ import print_function
import random
import os
import pickle
import numpy as np
from PIL import Image as pil_image
from itertools import islice
from torchvision import transforms
import torch.utils.data as data
from torchtools import *


class MiniImagenetLoader(data.Dataset):
    def __init__(self, root, partition='train'):
        super(MiniImagenetLoader, self).__init__()
        # set dataset information
        self.root = root
        self.partition = partition
        self.data_size = [3, 84, 84]
        
        # set normalizer
        mean_pix = [x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]
        std_pix = [x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]
        normalize = transforms.Normalize(mean=mean_pix, std=std_pix)

        # set transformer
        if self.partition == 'train':
            self.transform = transforms.Compose([transforms.RandomCrop(84, padding=4),
                                                 transforms.RandomHorizontalFlip(),
                                                 lambda x: np.asarray(x),
                                                 transforms.ToTensor(),
                                                 normalize])
        else:  # 'val' or 'test' ,
            self.transform = transforms.Compose([lambda x: np.asarray(x),
                                                 transforms.ToTensor(),
                                                 normalize])
        # load data
        self.data = self.load_dataset()

    def load_dataset(self):
        # 获取数据路径
        dataset_path = os.path.join(self.root, 'mini_imagenet_%s.txt' % self.partition)
        with open(dataset_path, "rb") as f:
            line = f.read()
            data = pickle.loads(line)
        return data
     
    def get_task_batch(self,num_tasks=5,num_ways=20,num_shots=1,num_queries=1,seed=None):
        if seed is not None:
            random.seed(seed)
        # init task batch data
        support_data, support_label, query_data, query_label = [], [], [], []
        for _ in range(num_ways * num_shots):
            data = np.zeros(shape=[num_tasks] + self.data_size,dtype='float32')
            label = np.zeros(shape=[num_tasks],dtype='float32')
            support_data.append(data)
            support_label.append(label)
        for _ in range(num_ways * num_queries):
            data = np.zeros(shape=[num_tasks] + self.data_size,dtype='float32')
            label = np.zeros(shape=[num_tasks],dtype='float32')
            query_data.append(data)
            query_label.append(label)

        # 获取全部类别
        full_class_list = list(self.data.keys())
        label_list = list(range(0,num_ways))
        #随机定义类别
        random.shuffle(label_list)
        # 遍历每个batch
        for batch_idx in range(num_tasks):
            # 每个batch 随机抽取5类
            task_class_list = random.sample(full_class_list, num_ways)

            # 每个batch 随机抽取5类 的 每类 抽取5个support , 1 个query
            for class_idx in range(num_ways):
                # sample data for support and query (num_shots + num_queries)
                class_data_list = random.sample(self.data[task_class_list[class_idx]], num_shots + num_queries)
                # 存放support_set
                for shot_idx in range(num_shots):
                    support_data[shot_idx + class_idx * num_shots][batch_idx] = self.transform(class_data_list[shot_idx])
                    support_label[shot_idx + class_idx * num_shots][batch_idx] = label_list[class_idx]
                # 存放query_set
                for query_idx in range(num_queries):
                    query_data[query_idx + class_idx * num_queries][batch_idx] = self.transform(class_data_list[num_shots + query_idx])
                    query_label[query_idx + class_idx * num_queries][batch_idx] = label_list[class_idx]

        # reshape to tensor (num_tasks x (num_ways * (num_supports + num_queries)) x ...)(20*5*6,3,84,84)
        support_data = torch.stack([torch.from_numpy(data).float().to(tt.arg.device) for data in support_data], 1)
        support_label = torch.stack([torch.from_numpy(label).float().to(tt.arg.device) for label in support_label], 1)
        query_data = torch.stack([torch.from_numpy(query_data[i]).float().to(tt.arg.device) for i in label_list], 1)
        query_label = torch.stack([torch.from_numpy(query_label[i]).float().to(tt.arg.device) for i in label_list], 1)

        return [support_data, support_label, query_data, query_label]