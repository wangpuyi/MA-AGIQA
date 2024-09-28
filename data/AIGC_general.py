import os
import torch
import numpy as np
from PIL import Image
import cv2
import torch.nn.functional as F


class AIGCgeneral(torch.utils.data.Dataset):
    # def __init__(self, dis_path, txt_file_name, list_name, transform, keep_ratio):
    def __init__(self, dis_path, labels, pic_names, transform, keep_ratio, **kwargs):
        super(AIGCgeneral, self).__init__()
        self.dis_path = dis_path # 图片路径
        self.transform = transform
        for k, v in kwargs.items():
            setattr(self, k, v)

        # reshape score_list (1xn -> nx1)
        score_data = np.array(labels)
        # score_data = self.normalization(score_data)
        # [1, 2, 3, 4, 5] -> [[1], [2], [3], [4], [5]]
        score_data = list(score_data.astype('float').reshape(-1, 1))

        self.data_dict = {'d_img_list': pic_names, 'score_list': score_data}

    def normalization(self, data):
        range = np.max(data) - np.min(data)
        return (data - np.min(data)) / range

    def __len__(self):
        return len(self.data_dict['d_img_list'])
    
    def __getitem__(self, idx):
        d_img_name = self.data_dict['d_img_list'][idx]
        d_img = cv2.imread(os.path.join(self.dis_path, d_img_name), cv2.IMREAD_COLOR)
        d_img = cv2.resize(d_img, (224, 224), interpolation=cv2.INTER_CUBIC)
        d_img = cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)
        # d_img = Image.open(os.path.join(self.dis_path, d_img_name)).convert("RGB")
        # d_img = d_img.resize((224, 224), Image.BICUBIC)
        d_img = np.array(d_img).astype('float32') / 255
        d_img = np.transpose(d_img, (2, 0, 1))
        score = self.data_dict['score_list'][idx]
        name = self.data_dict['d_img_list'][idx]

        sample = {
            'd_img_org': d_img,
            'score': score,
        }
        if self.transform:
            sample = self.transform(sample)
        # add name to sample
        sample['name_'] = name

        # include tensor
        tensor_path = os.path.join(self.tensor_root, d_img_name.split('.')[0] + '.pt')
        if d_img_name.endswith('.jpg'):
            img_name = d_img_name.replace('.jpg', '')
        elif d_img_name.endswith('.png'):
            img_name = d_img_name.replace('.png', '')
        tensor1_path = os.path.join(self.tensor_root, img_name + '_semantic_content.pt')
        tensor2_path = os.path.join(self.tensor_root, img_name + '_coherence.pt')
        # sample['tensor'] = torch.load(tensor_path)
        sample['tensor_1'] = torch.load(tensor1_path, map_location=torch.device('cpu'))
        sample['tensor_2'] = torch.load(tensor2_path, map_location=torch.device('cpu'))
        # print('tensor_1:', sample['tensor_1'])
        return sample
