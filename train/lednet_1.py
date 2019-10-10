import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.functional import interpolate as interpolate 

def split(x):
    c = int(x.size()[1])
    c1 = round(c * 0.5)
    x1 = x[:, :c1, :, :].contiguous()
    x2 = x[:, c1:, :, :].contiguous()

    return x1, x2

def channel_shuffle(x,groups):
    batchsize, num_channels, height, width = x.data.size()
    
    channels_per_group = num_channels // groups
    
    #reshape
    x = x.view(batchsize,groups,
        channels_per_group,height,width)
    
    x = torch.transpose(x,1,2).contiguous()
    
    #flatten
    x = x.view(batchsize,-1,height,width)
    
    return x
    

class Conv2dBnRelu(nn.Module):
    def __init__(self,in_ch,out_ch,kernel_size=3,stride=1,padding=0,dilation=1,bias=True):
        super(Conv2dBnRelu,self).__init__()
		
        self.conv = nn.Sequential(
		nn.Conv2d(in_ch,out_ch,kernel_size,stride,padding,dilation=dilation,bias=bias),
		nn.BatchNorm2d(out_ch, eps=1e-3),
		nn.ReLU(inplace=True)
	)

    def forward(self, x):
        return self.conv(x)


##after Concat -> BN, you also can use Dropout like SS_nbt_module may be make a good result!
class DownsamplerBlock (nn.Module):
    """
    下采样模块
                  input
                 /     \    
            conv/s2    pool/s2
         (out_c-in_c)  (in_c)
              |___________|
                    |
                  concat
                  (out_c)
                    |
                    bn
                    |
                   relu
    """
    def __init__(self, in_channel, out_channel):
        super(DownsamplerBlock,self).__init__()

        self.conv = nn.Conv2d(in_channel, out_channel-in_channel, (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(out_channel, eps=1e-3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        output = self.bn(output)
        output = self.relu(output)
        return output
      

class SS_nbt_module(nn.Module):
    """
    最主要的基础构建模块SS_nbt_module(Split-Shuffle-non-bottleneck Module)
    构造函数参数：
            chann: 输入通道数 
            dropprob: dropout比率
            dilated: 

    ---------------input
    |             (in_c)
    |                |
    |              split
    |             /      \   
    |        input1      input2
    |       (in_c//2)  (in_c//2)
    |           |          |
    |        conv3x1     conv3x1   
    |           |          |
    |          relu       relu
    |           |          |
    |        conv1x3    conv1x3
    |           |          | 
    |           bn        bn
    |           |          |
    |          relu       relu
    |           |          |
    |    conv3x1(dilated) conv3x1(dilated)
    |           |          |
    |          relu       relu
    |           |          |
    |    conv1x3(dilated) conv1x3(dilated)
    |           |          | 
    |           bn         bn
    |           |          |
    |        dropout     dropout
    |           |__________|
    |                |
    |              concat
    |              (in_c)
    |________________|
                    add
                     |
                    relu
                     |
              channel_shuffle
                   (in_c)
    """
    def __init__(self, chann, dropprob, dilated):        
        super(SS_nbt_module,self).__init__()

        oup_inc = chann//2
        
        #dw
        self.conv3x1_1_l = nn.Conv2d(oup_inc, oup_inc, (3,1), stride=1, padding=(1,0), bias=True)

        self.conv1x3_1_l = nn.Conv2d(oup_inc, oup_inc, (1,3), stride=1, padding=(0,1), bias=True)

        self.bn1_l = nn.BatchNorm2d(oup_inc, eps=1e-03)

        self.conv3x1_2_l = nn.Conv2d(oup_inc, oup_inc, (3,1), stride=1, padding=(1*dilated,0), bias=True, dilation = (dilated,1))

        self.conv1x3_2_l = nn.Conv2d(oup_inc, oup_inc, (1,3), stride=1, padding=(0,1*dilated), bias=True, dilation = (1,dilated))

        self.bn2_l = nn.BatchNorm2d(oup_inc, eps=1e-03)
        
        #dw
        self.conv3x1_1_r = nn.Conv2d(oup_inc, oup_inc, (3,1), stride=1, padding=(1,0), bias=True)

        self.conv1x3_1_r = nn.Conv2d(oup_inc, oup_inc, (1,3), stride=1, padding=(0,1), bias=True)

        self.bn1_r = nn.BatchNorm2d(oup_inc, eps=1e-03)

        self.conv3x1_2_r = nn.Conv2d(oup_inc, oup_inc, (3,1), stride=1, padding=(1*dilated,0), bias=True, dilation = (dilated,1))

        self.conv1x3_2_r = nn.Conv2d(oup_inc, oup_inc, (1,3), stride=1, padding=(0,1*dilated), bias=True, dilation = (1,dilated))

        self.bn2_r = nn.BatchNorm2d(oup_inc, eps=1e-03)       
        
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropprob)       
        
    @staticmethod
    def _concat(x,out):
        return torch.cat((x,out),1)    
    
    def forward(self, input):

        # x1 = input[:,:(input.shape[1]//2),:,:]
        # x2 = input[:,(input.shape[1]//2):,:,:]

        residual = input
        x1, x2 = split(input)

        output1 = self.conv3x1_1_l(x1)
        output1 = self.relu(output1)
        output1 = self.conv1x3_1_l(output1)
        output1 = self.bn1_l(output1)
        output1 = self.relu(output1)

        output1 = self.conv3x1_2_l(output1)
        output1 = self.relu(output1)
        output1 = self.conv1x3_2_l(output1)
        output1 = self.bn2_l(output1)

        output2 = self.conv1x3_1_r(x2)
        output2 = self.relu(output2)
        output2 = self.conv3x1_1_r(output2)
        output2 = self.bn1_r(output2)
        output2 = self.relu(output2)

        output2 = self.conv1x3_2_r(output2)
        output2 = self.relu(output2)
        output2 = self.conv3x1_2_r(output2)
        output2 = self.bn2_r(output2)

        if (self.dropout.p != 0):
            output1 = self.dropout(output1)
            output2 = self.dropout(output2)

        out = self._concat(output1,output2)
        out = F.relu(residual + out, inplace=True)

        return channel_shuffle(out, 2)


class Encoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # input: 1024x512x3  output: 512x256x32
        self.initial_block = DownsamplerBlock(3,32)
        

        self.layers = nn.ModuleList()

        # input: 512x256x32  output: 512x256x32
        for x in range(0, 3):
            self.layers.append(SS_nbt_module(32, 0.03, 1))
        
        # input: 512x256x32  output: 256x128x64
        self.layers.append(DownsamplerBlock(32,64))
        
        # input: 256x128x64  output: 256x128x64
        for x in range(0, 2):
            self.layers.append(SS_nbt_module(64, 0.03, 1))
  
        # input: 256x128x64  output: 128x64x128
        self.layers.append(DownsamplerBlock(64,128))
        
        # input: 128x64x128  output: 128x64x128
        for x in range(0, 1):    
            self.layers.append(SS_nbt_module(128, 0.3, 1))
            self.layers.append(SS_nbt_module(128, 0.3, 2))
            self.layers.append(SS_nbt_module(128, 0.3, 5))
            self.layers.append(SS_nbt_module(128, 0.3, 9))
            
        # input: 128x64x128  output: 128x64x128
        for x in range(0, 1):    
            self.layers.append(SS_nbt_module(128, 0.3, 2))
            self.layers.append(SS_nbt_module(128, 0.3, 5))
            self.layers.append(SS_nbt_module(128, 0.3, 9))
            self.layers.append(SS_nbt_module(128, 0.3, 17))
                    

        #Only in encoder mode:
        self.output_conv = nn.Conv2d(128, num_classes, 1, stride=1, padding=0, bias=True)

    def forward(self, input, predict=False):
        
        output = self.initial_block(input)

        for layer in self.layers:
            output = layer(output)

        # Only in encoder mode
        if predict:
            output = self.output_conv(output)

        return output

class Interpolate(nn.Module):
    def __init__(self,size,mode):
        super(Interpolate,self).__init__()
        
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
    def forward(self,x):
        x = self.interp(x,size=self.size,mode=self.mode,align_corners=True)
        return x
        

class APN_Module(nn.Module):
    """
    解码器APN_Module(Attention Pyramid Mudule)
    
    """
    def __init__(self, in_ch, out_ch):
        super(APN_Module, self).__init__()
        # global pooling branch
        self.branch1 = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                Conv2dBnRelu(in_ch, out_ch, kernel_size=1, stride=1, padding=0)
	)
		
        # midddle branch
        self.mid = nn.Sequential(
		Conv2dBnRelu(in_ch, out_ch, kernel_size=1, stride=1, padding=0)
	)
		
        self.down1 = Conv2dBnRelu(in_ch, 128, kernel_size=7, stride=2, padding=3)
		
        self.down2 = Conv2dBnRelu(128, 128, kernel_size=5, stride=2, padding=2)
		
        self.down3 = nn.Sequential(
		Conv2dBnRelu(128, 128, kernel_size=3, stride=2, padding=1),
		Conv2dBnRelu(128, 20, kernel_size=1, stride=1, padding=0)
	)
		
        self.conv2 = Conv2dBnRelu(128, 20, kernel_size=1, stride=1, padding=0)
        self.conv1 = Conv2dBnRelu(128, 20, kernel_size=1, stride=1, padding=0)
	
    def forward(self, x):
        
        h = x.size()[2]
        w = x.size()[3]
        
        ### global pooling branch
        # input: 128x64x128   output:1x1xC
        b1 = self.branch1(x)
        # input: 1x1xC   output:128x64xC
        #b1 = Interpolate(size=(h, w), mode="bilinear")(b1)
        b1= interpolate(b1, size=(h, w), mode="bilinear", align_corners=True)

        ### midddle branch
        # input: 128x64x128   output:128x64xC
        mid = self.mid(x)
		
        ### third branch
        # input: 128x64x128   output:64x32x128
        x1 = self.down1(x)
        # input: 64x32x128   output:32x16x128
        x2 = self.down2(x1)
        # input: 32x16x128   output:16x8xC
        x3 = self.down3(x2)
        # input: 16x8xC   output:32x16xC
        #x3 = Interpolate(size=(h // 4, w // 4), mode="bilinear")(x3)
        x3= interpolate(x3, size=(h // 4, w // 4), mode="bilinear", align_corners=True)	
        # input: 32x16x128  output:32x16xC
        x2 = self.conv2(x2)
        # input: 32x16xC/32x16xC  output:32x16xC
        x = x2 + x3
        # input: 32x16xC/32x16xC  output:64x32xC
        #x = Interpolate(size=(h // 2, w // 2), mode="bilinear")(x)
        x= interpolate(x, size=(h // 2, w // 2), mode="bilinear", align_corners=True)
        # input: 64x32x128   output:64x32xC
        x1 = self.conv1(x1)
        # input: 64x32xC/64x32xC   output:64x32xC
        x = x + x1
        # input: 64x32xC   output:128x64xC
        #x = Interpolate(size=(h, w), mode="bilinear")(x)
        x= interpolate(x, size=(h, w), mode="bilinear", align_corners=True)
        
        ### midddle branch与third branch融合
        # input: 128x64xC/128x64xC   output:128x64xC
        x = torch.mul(x, mid)

        ### 与global pooling branch融合
        # input: 128x64xC/128x64xC   output:128x64xC
        x = x + b1

        return x
         
class Decoder (nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.apn = APN_Module(in_ch=128,out_ch=20)
        #self.upsample = Interpolate(size=(512, 1024), mode="bilinear")
        #self.output_conv = nn.ConvTranspose2d(16, num_classes, kernel_size=4, stride=2, padding=1, output_padding=0, bias=True)
        #self.output_conv = nn.ConvTranspose2d(16, num_classes, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)
        #self.output_conv = nn.ConvTranspose2d(16, num_classes, kernel_size=2, stride=2, padding=0, output_padding=0, bias=True)

    def forward(self, input):
        ### 调用APN模块
        # input: 128x64x128   output:128x64xC
        output = self.apn(input)
        
        ### 上采样8倍回到输入尺寸
        # input: 128x64xC   output:1024x512xC
        out = interpolate(output, size=(512, 1024), mode="bilinear", align_corners=True)
        #out = self.upsample(output)
        return out

# LEDNet
class Net(nn.Module):
    def __init__(self, num_classes, encoder=None):  
        super().__init__()

        # 若没有加载预训练的encoder
        if (encoder == None):
            self.encoder = Encoder(num_classes)
        # 若给到预训练的encoder
        else:
            self.encoder = encoder
        self.decoder = Decoder(num_classes)

    def forward(self, input, only_encode=False):
        if only_encode:
            return self.encoder.forward(input, predict=True)
        else:
            # input: 1024x512x3  output: 128x64x128
            output = self.encoder(input)   
            # input: 128x64x128  output: 1024x512xC
            return self.decoder.forward(output)
