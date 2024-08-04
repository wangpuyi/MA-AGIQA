'''
relu 单fc 784, 后gating
在AGIQA-3k上  SRCC 0.8939,    median PLCC 0.9273,     median KRCC 0.7211,     median RMSE 0.3756
'''

import torch
import torch.nn as nn
import timm

from timm.models.vision_transformer import Block
from torch import nn
from einops import rearrange
from models.swin import SwinTransformer


class TABlock(nn.Module):
    def __init__(self, dim, drop=0.1):
        super().__init__()
        self.c_q = nn.Linear(dim, dim)
        self.c_k = nn.Linear(dim, dim)
        self.c_v = nn.Linear(dim, dim)
        self.norm_fact = dim ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.proj_drop = nn.Dropout(drop)

    def forward(self, x):
        _x = x
        B, C, N = x.shape
        q = self.c_q(x)
        k = self.c_k(x)
        v = self.c_v(x)

        attn = q @ k.transpose(-2, -1) * self.norm_fact
        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, C, N)
        x = self.proj_drop(x)
        x = x + _x
        return x


class SaveOutput:
    def __init__(self):
        self.outputs = []
    
    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)
    
    def clear(self):
        self.outputs = []


class MA_AGIQA(nn.Module):
    def __init__(self, embed_dim=72, num_outputs=1, patch_size=8, drop=0.1, 
                    depths=[2, 2], window_size=4, dim_mlp=768, num_heads=[4, 4],
                    img_size=224, num_tab=2, scale=0.8, **kwargs):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.input_size = img_size // patch_size #224//8 = 28
        self.patches_resolution = (img_size // patch_size, img_size // patch_size)
        
        self.vit = timm.create_model('vit_base_patch8_224', pretrained=True)
        self.save_output = SaveOutput()
        hook_handles = []
        for layer in self.vit.modules():
            if isinstance(layer, Block):
                handle = layer.register_forward_hook(self.save_output)
                hook_handles.append(handle)

        self.tablock1 = nn.ModuleList()
        for i in range(num_tab):
            tab = TABlock(self.input_size ** 2) #52
            self.tablock1.append(tab)

        self.conv1 = nn.Conv2d(embed_dim * 4, embed_dim, 1, 1, 0) #3072 768
        self.swintransformer1 = SwinTransformer(
            patches_resolution=self.patches_resolution,
            depths=depths,
            num_heads=num_heads,
            embed_dim=embed_dim,
            window_size=window_size,
            dim_mlp=dim_mlp,
            scale=scale
        )

        self.tablock2 = nn.ModuleList()
        for i in range(num_tab):
            tab = TABlock(self.input_size ** 2)
            self.tablock2.append(tab)

        self.conv2 = nn.Conv2d(embed_dim, embed_dim // 2, 1, 1, 0)
        self.swintransformer2 = SwinTransformer(
            patches_resolution=self.patches_resolution,
            depths=depths,
            num_heads=num_heads,
            embed_dim=embed_dim // 2,
            window_size=window_size,
            dim_mlp=dim_mlp,
            scale=scale
        )
        
        self.fc_score = nn.Sequential(
            nn.Linear(embed_dim // 2, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(embed_dim // 2, num_outputs),
            nn.ReLU()
        )
        self.fc_weight = nn.Sequential(
            nn.Linear(embed_dim // 2, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(embed_dim // 2, num_outputs),
            nn.Sigmoid()
        )
        self.fc39_784 = nn.Sequential(
            nn.Linear(39, 392),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(392, 784),
            nn.Sigmoid()
        )
        self.fc_output = nn.Sequential(
            nn.Linear(784, 392),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(392, 1)
        )

        # type6
        self.fc4096_784_a = nn.Linear(4096, 784)
        self.fc4096_784_b = nn.Linear(4096, 784)
        self.quality = self.quality_regression(784*3, 128, 1)

        # MA-AGIQA
        self.fc4096_784 = nn.Sequential(
            nn.Linear(4096, 784),
            nn.ReLU(),
            nn.Dropout(drop),
        )
        self.fc4096_784 = nn.Sequential( 
            nn.Linear(4096, 784),
            nn.ReLU(),
            nn.Dropout(drop),
        )
        self.fc784x3_1 = self.quality_regression(784*3, 128, 1)

        self.fc784_784 = nn.Linear(784, 784)
        self.gating = nn.Linear(784*3, 3)
        self.output_fc = nn.Linear(784, 1)


    
    def extract_feature(self, save_output):
        x6 = save_output.outputs[6][:, 1:]
        x7 = save_output.outputs[7][:, 1:]
        x8 = save_output.outputs[8][:, 1:]
        x9 = save_output.outputs[9][:, 1:]
        x = torch.cat((x6, x7, x8, x9), dim=2)
        return x
    
    def quality_regression(self,in_channels, middle_channels, out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.Linear(middle_channels, out_channels),          
        )
        return regression_block

    def forward(self, x, **kwargs):
        _x = self.vit(x) # x  torch.Size([1, 3, 224, 224])
        x = self.extract_feature(self.save_output)
        # print('x: ', x.shape) #torch.Size([1, 784, 3072])
        self.save_output.outputs.clear()

        # stage 1
        x = rearrange(x, 'b (h w) c -> b c (h w)', h=self.input_size, w=self.input_size)
        # print("x_before tablock1: ", x.shape) #torch.Size([1, 3072, 784])
        # print('h: ', self.input_size, 'w: ', self.input_size) #h:  28 w:  28
        
        #x in tablock1_0:  torch.Size([1, 3072, 784])
        #x in tablock1_1:  torch.Size([1, 3072, 784])
        for index,tab in enumerate(self.tablock1):
            x = tab(x)
            # print("x in tablock1_{}: ".format(index), x.shape)


        x = rearrange(x, 'b c (h w) -> b c h w', h=self.input_size, w=self.input_size)
        # print("x_before conv1: ", x.shape) #torch.Size([1, 3072, 28, 28])
        x = self.conv1(x)
        # print("x_after conv1: ", x.shape) #torch.Size([1, 768, 28, 28])
        x = self.swintransformer1(x)
        # print("x_after swintransformer1: ", x.shape) #torch.Size([1, 768, 28, 28])

        # stage2
        x = rearrange(x, 'b c h w -> b c (h w)', h=self.input_size, w=self.input_size)
        for tab in self.tablock2:
            x = tab(x)
        x = rearrange(x, 'b c (h w) -> b c h w', h=self.input_size, w=self.input_size)
        #torch.Size([1, 768, 24, 24])
        x = self.conv2(x)
        x = self.swintransformer2(x) #torch.Size([1, 384, 24, 24])

        x = rearrange(x, 'b c h w -> b (h w) c', h=self.input_size, w=self.input_size) #torch.Size([1, 784, 384])
        # print("x_before score: ", x.shape) #torch.Size([1, 784, 384])

        score = torch.tensor([]).cuda()
        # f: torch.Size([batch, 784, 1]) w: torch.Size([batch, 784, 1]) key: torch.Size([batch, 784, 1])
        # query1: torch.Size([batch, 1, 4096]) query2: torch.Size([batch, 1, 4096])
        batch_size = x.shape[0]
        f = self.fc_score(x)
        w = self.fc_weight(x)
        key = (f*w).squeeze(2)
        query1 = kwargs['tensor1'].type(torch.float32)
        query2 = kwargs['tensor2'].type(torch.float32)
        
        key = self.fc784_784(key) #torch.Size([batch, feature])
        query1 = self.fc4096_784(query1.squeeze(1)) #torch.Size([batch, feature])
        query2 = self.fc4096_784(query2.squeeze(1))
        gating_weights = self.gating(torch.cat([query1, query2, key], dim=1)) #torch.Size([batch, 1, 3])
        expert_outputs = torch.stack([query1, query2, key], dim=1) #torch.Size([batch, 3, feature])
        mixed_experts = torch.bmm(gating_weights.unsqueeze(1), expert_outputs).squeeze(1) #torch.Size([batch, feature])
        score = self.output_fc(mixed_experts) #torch.Size([batch, 1])
        return score
