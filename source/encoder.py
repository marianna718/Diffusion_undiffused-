import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock

# the residiual block is very simmilar to the one used in resnet

class VAE_Encoder(nn.Sequential):
    # reduce the dimantion of data but at the same time increase the features 
    def __init__(self):
        super().__init__(
            # (Batch_size, Channel, Height, Width) --> (Batch_size, 128, Height, Width)
            nn.Conv2d(3,128, kernel_size=3, padding = 1),
            # this dont change the imiage w and h (size) becouse we have added the padding

            # the block bellow is a combination of normalizations and convalution
            # (Batch_zixe, 128, Height, Width) -> (Batch_size, 128, Height, Width)
            VAE_ResidualBlock(128, 128),

            # (Batch_zixe, 128, Height, Width) -> (Batch_size, 128, Height, Width)
            VAE_ResidualBlock(128, 128),

            # (Batch_zixe, 128, Height, Width) -> (Batch_size, 128, Height/2, Width/2)
            nn.Conv2d(128,128, kernel_size = 3, stride=2, padding =0),
            # the upper code will change the sizes 
            
            # (Batch_zixe, 128, Height/2, Width/2) -> (Batch_size, 256, Height/2, Width/2)
            VAE_ResidualBlock(128,256),
           
            # (Batch_zixe, 256, Height/2, Width/2) -> (Batch_size, 256, Height/2, Width/2)
            VAE_ResidualBlock(256,256),

            # (Batch_zixe, 256, Height/2, Width/2) -> (Batch_size, 256, Height/4, Width/4)
            nn.Conv2d(256,256, kernel_size=3, stride=2, padding = 0),

            # (Batch_zixe, 256, Height/4, Width/4) -> (Batch_size, 512, Height/4, Width/4)
            VAE_ResidualBlock(256,512),

            # (Batch_zixe, 512, Height/4, Width/4) -> (Batch_size, 512, Height/4, Width/4)
            VAE_ResidualBlock(512,512),

            # (Batch_zixe, 512, Height/4, Width/4) -> (Batch_size, 512, Height/8, Width/8)
            nn.Conv2d(512,512, kernel_size=3, stride=2, padding = 0),


            VAE_ResidualBlock(512,512),

            VAE_ResidualBlock(512,512),
            
            # (Batch_zixe, 512, Height/8, Width/8) -> (Batch_size, 512, Height/8, Width/8)
            VAE_ResidualBlock(512,512),

            # (Batch_zixe, 512, Height/8, Width/8) -> (Batch_size, 512, Height/8, Width/8)
            VAE_AttentionBlock(512)
            # attention itself is a way to ralate tokens in the sentance
            # so in the image the attention is a way to ralate pixels to each other

            # (Batch_zixe, 512, Height/8, Width/8) -> (Batch_size, 512, Height/8, Width/8)
            VAE_ResidualBlock(512,512),

            # and then we have a group normalization
            # number of groups 32, number of chanels 512
            
            # (Batch_zixe, 512, Height/8, Width/8) -> (Batch_size, 512, Height/8, Width/8)
            nn.GroupNorm(32,512)


            # and the activation function
            # (Batch_zixe, 512, Height/8, Width/8) -> (Batch_size, 512, Height/8, Width/8)
            nn.SiLU(),
            # similar to ReLU, just works better in this case

            # (Batch_zixe, 512, Height/8, Width/8) -> (Batch_size, 8, Height/8, Width/8)
            nn.Conv2d(512, 8 , kernel_size = 3, padding=1),

            # (Batch_zixe, 8, Height/8, Width/8) -> (Batch_size, 8, Height/8, Width/8)
            nn.Conv2d(8,8, kernel_size=1, padding=0)
        # the size does not chnage becouse the kernel size is 1

        
        )

    
    def forward(self, x:torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # x:(Batch_size, channel, Height, Wigth)
        # noise: (Batch_size, out_chanels, height/8, wigth/8)

        for module in self:
            # in convolutions that have stride we need to apply a 
            # special emmbedding 
            # so its like checking if there is a stride
            # and adding padding to the right and bottom
            if getattr(module, 'stride', None) == (2,2):
                # (padding_left,padding_right,padding_top, padding_bottom)
                x = F.pad(x, (0,1,0,1))
            x = module(x)


        # variational autoencoder part of our code takes the output
        # of the last layer of our net aboive 
        # and devides it into 2 tensors with dim
        # (batch_size, 8, height/8, width/8) ->(two tensors) (Batch_size, 4, height/8, width/8)
        mean, log_variance = torch.chunk(x, 2, dim=1)

        # put the variance into some range(not too big oro too smal)
        # (Batch_size, 4, height/8, width/8)
        log_variance = torch.clamp(log_variance, -30, 20)
        # this didnt change the shape

        # then we take the exponent to get reed of log
        # (Batch_size, 4, height/8, width/8)
        varinace = log_variance.exp()

        # then calculating the standard deviation
        # (Batch_size, 4, height/8, width/8)
        stdev = varinace.sqrt()

        # now with given mean and variance we need to sample from this gausian normal distribution
        # we know that if we sample from [0,1] then 
        # we can transform the samples to any distribution
        # Z = N(0,1) -> N(mean, variance) = X
        # X = mean +stdev * Z

        x = mean + stdev * noise 

        # Scale the output by constant that is jus also defined in the paper
        x *= 0.182515

        return x 
    
    