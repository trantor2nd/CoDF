import torch
import torch.nn as nn
import torch.nn.functional as F


class Pooling(nn.Module):
    """
    Implementation of pooling for PoolFormer
    --pool_size: pooling size
    """
    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool2d(
            pool_size, stride=1, padding=pool_size//2, count_include_pad=False)

    def forward(self, x):
        return x#self.pool(x) - x


class FreqSplitBlock(nn.Module):
    def __init__(self, keep_ratio=0.2):
        super().__init__()
        self.keep_ratio = keep_ratio
        self.low_processor = Pooling()
        self.high_processor = Pooling()

    def forward(self, x):
        B, C, H, W = x.shape
        freq = torch.fft.fft2(x, dim=(-2, -1))  # [B,C,H,W] -> complex freq

        # Create low/high frequency mask
        k = int(H * W * self.keep_ratio)
        mask = torch.zeros(H * W, device=x.device)
        mask[:k] = 1
        idx = torch.randperm(H * W)
        low_mask = mask[idx].reshape(1, 1, H, W)
        high_mask = 1.0 - low_mask

        # Broadcast to all channels
        low_mask = low_mask.expand(B, C, -1, -1)
        high_mask = high_mask.expand(B, C, -1, -1)

        # Separate frequency
        low_freq = freq * low_mask
        high_freq = freq * high_mask

        # Inverse transform
        low_spatial = torch.fft.ifft2(low_freq, dim=(-2, -1)).real
        high_spatial = torch.fft.ifft2(high_freq, dim=(-2, -1)).real

        # Process separately
        low_feat = self.low_processor(low_spatial)
        high_feat = self.high_processor(high_spatial)

        return low_feat + high_feat


class PatchEmbed(nn.Module):
    """
    Patch Embedding that is implemented by a layer of conv. 
    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H/stride, W/stride]
    """
    def __init__(self, patch_size=1, stride=1, padding=0, 
                 in_chans=3, embed_dim=64, norm_layer=None):
        super().__init__()

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, 
                              stride=stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


#feature encoder
class F2SF(nn.Module):
    def __init__(self,embed_dim,hidden_dim,drop=0.1):
        super().__init__()
    
        self.mixer =nn.Sequential(
            nn.BatchNorm2d(embed_dim),
            FreqSplitBlock(keep_ratio=0.4)#Pooling()
        )

        self.mlp= nn.Sequential(
            nn.BatchNorm2d(embed_dim),
            nn.Conv2d(in_channels=embed_dim,out_channels=hidden_dim,kernel_size=1,stride=1,padding=0,bias = True),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Conv2d(in_channels=hidden_dim,out_channels=embed_dim,kernel_size=1,stride=1,padding=0,bias = True),
            nn.Dropout(drop)
        )

    def forward(self,tensor):
        tensor = self.mixer(tensor)+tensor
        tensor = self.mlp(tensor)+tensor
        return tensor


class MCP(nn.Module):
    def __init__(self,embed_dim,hidden_dim,drop=0.1):
        super().__init__()

        #in1------------------------------------------------------------
        self.norm1 =nn.BatchNorm2d(embed_dim)
        self.pool1 = Pooling()
        self.mlp1= nn.Sequential(
            nn.BatchNorm2d(embed_dim),
            nn.Conv2d(in_channels=embed_dim,out_channels=hidden_dim,kernel_size=1,stride=1,padding=0,bias = True),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Conv2d(in_channels=hidden_dim,out_channels=embed_dim,kernel_size=1,stride=1,padding=0,bias = True),
            nn.Dropout(drop)
        )
        #in2------------------------------------------------------------
        self.norm2 =nn.BatchNorm2d(embed_dim)
        self.pool2 = Pooling()
        self.mlp2= nn.Sequential(
            nn.BatchNorm2d(embed_dim),
            nn.Conv2d(in_channels=embed_dim,out_channels=hidden_dim,kernel_size=1,stride=1,padding=0,bias = True),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Conv2d(in_channels=hidden_dim,out_channels=embed_dim,kernel_size=1,stride=1,padding=0,bias = True),
            nn.Dropout(drop)
        )

    def forward(self,tensor1,tensor2):
        # tensor1_norm = self.norm1(tensor1)
        # tensor2_norm = self.norm2(tensor2)
        
        # tensor_add=self.pool1(tensor1_norm+tensor2_norm)
        # tensor_sub12=self.pool2(tensor1_norm-tensor2_norm)
        # tensor_sub21=self.pool2(tensor2_norm-tensor1_norm)

        # tensor1_mix = tensor1 * F.sigmoid(tensor_add) +  tensor_sub21
        # tensor2_mix = tensor2 * F.sigmoid(tensor_add) +  tensor_sub12

        # tensor1 = self.mlp1(tensor1_mix)+tensor1_mix
        # tensor2 = self.mlp2(tensor2_mix)+tensor2_mix
        
        return tensor1 , tensor2

class Classifier(nn.Module):
    def __init__(self,in_dim,out_dim,hidden_dim,drop=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.drop = drop
        self.refine = self._make_layer(3,2*in_dim,in_dim)
        self.predict = self._make_layer(3,in_dim,out_dim)
    def _make_layer(self,layer_num,in_dim,embed_dim):
        layers=[PatchEmbed(1,1,0,in_dim,embed_dim,nn.BatchNorm2d)]
        for i in range(layer_num):
            layers.append(F2SF(embed_dim,self.hidden_dim,self.drop))
        return nn.Sequential(*layers)
    def forward(self,tensor):
        tensor = self.refine(tensor)
        tensor = self.predict(tensor)
        return tensor

class CoDF(nn.Module):
    def __init__(self,
        in_dim1,
        in_dim2,
        out_dim, 
        layers=[2,6,2],
        embed_dim=[64,128,256],
        hidden_dim=128,
        drop=0.1
    ):
        super().__init__()

        assert layers != None and embed_dim != None , "CoDF : layers or embed_dim should not be None"
        self.layers = layers
        self.hidden_dim = hidden_dim
        self.drop = drop

        #in1------------------------------------------------------------
        list1 = []
        dim1 = [in_dim1]
        dim1.extend(embed_dim)
        for i in range(len(layers)) :
            list1.append(self._make_layer(layers[i],dim1[i],dim1[i+1]))
        self.block_in1 = nn.ModuleList(list1)

        #in2------------------------------------------------------------   
        list2 = []
        dim2 = [in_dim2]
        dim2.extend(embed_dim)
        for i in range(len(layers)) :
            list2.append(self._make_layer(layers[i],dim2[i],dim2[i+1]))
        self.block_in2 = nn.ModuleList(list2)

        #fuse-----------------------------------------------------------
        fu = []
        for i in range(len(layers)-1) :
            fu.append(MCP(embed_dim[i],self.hidden_dim,drop))
        self.fuse_in12 = nn.ModuleList(fu)

        #classifier--------------------------------------------------------
        self.classifier = Classifier(embed_dim[-1],out_dim,self.hidden_dim,drop)
    
    def _make_layer(self,layer_num,in_dim,embed_dim):
        layers=[PatchEmbed(1,1,0,in_dim,embed_dim,nn.BatchNorm2d)]
        for i in range(layer_num):
            layers.append(F2SF(embed_dim,self.hidden_dim,self.drop))
        return nn.Sequential(*layers)
    
    def criterion_kl_mse(self,tensor,target):
        batch = tensor.shape[0]
        tensor = F.sigmoid(tensor.mean(dim=1))
        target = F.sigmoid(target.mean(dim=1))

        tensor_p = F.softmax(tensor.view(batch,-1),dim=1)
        target_p = F.softmax(target.view(batch,-1),dim=1)

        kl_loss = F.kl_div(tensor_p.log(),target_p,reduction='batchmean')
        mse_loss = F.mse_loss(tensor,target)
        loss = kl_loss+0.5*mse_loss
        return loss
    
    def forward(self,tensor1,tensor2):
        self.loss = 0
        self.p1 = []
        self.p2 = []
        self.p3 = 0

        for i in range(len(self.layers)-1) :
            tensor1 = self.block_in1[i](tensor1)
            tensor2 = self.block_in2[i](tensor2)
            tensor1,tensor2 = self.fuse_in12[i](tensor1,tensor2)

            self.loss += self.criterion_kl_mse(tensor1,tensor2)
            #----------------------
            self.p1.append(tensor1)
            self.p2.append(tensor2)
            #----------------------

        tensor1 = self.block_in1[-1](tensor1)
        tensor2 = self.block_in2[-1](tensor2)
        #----------------------
        self.p1.append(tensor1)
        self.p2.append(tensor2)
        #----------------------
        tensor = tensor = torch.cat([tensor1,tensor2],dim=1)
        #----------------------
        self.p3 = tensor
        #----------------------
        tensor = self.classifier(tensor)
        
        return tensor





