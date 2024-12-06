import torch 
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention


class VAE_AtteentionBlock(nn.Module):

    def __init__(self, channels:int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)

        # the main purpose of useing normalization and in this case group normalization is 
        # when we use normalization by gropup the items or the 
        # pixesls(or outputs of convalutions) in this case are more siminlar
        # in one segment of the image those the distribution is nearly ther same , and 
        # this way we will get different means and variances for 
        # each group 
        self.attention = SelfAttention(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (Batch_size, features, height, width)
        \
        residue = x

        n,c,h,w = x.shape

        # then we will do self attention beetween all the pixels of ther image4

        #  (Batch_size, features, height, width) - > (batch_size, featurers, height* width)
        # pixelvise view
        x = x.view(n,c,h*w)

        # Batch_size, features, height*width) -> (batch_size, height*width, features)
        x = x.transpose(-1,-2)
                        
        # now we go with the SelfAttention)
        # (Batch_size, height*width, features ) ->(batch_size, height*width, features)
        x = self.attention(x)

        # now transform back
        # (batch_size, height* width, features) -> (batch_size,features, height*width)
        x = x.transpose(-1,-2)

        # (batch_size, Features, height*width) -> (batch_size, Features, Height, width)
        x = x.view((n,c,h,w))

        x+= residue

        return x

#  the variational autoencoder is traind to lear the latent space of our vectors 
class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__() 
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding=1)

        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # we have the skip connections
        #  which means we take the inputs and we skip some layers
        # and then we connect it with the output
        #  and if the 2 channels in and out  are different then we need to create an intermwediate leyer
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size =1, padding=0)


    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # x: (Batch_size, in_channels, height, width)

        # make a copy
        residue = x

        x = self.groupnorm_1(x)

        x = F.silu(x)
        x = self.conv_1(x)
        x = self.groupnorm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)
        x = x + self.residual_layer(residue)

        return x
    



class VAE_Decoder(nn.Sequential):

    def __init__(self):
        super().__init__(


            nn.Conv2D(4, 4, kernel_size = 1, padding = 0),
            nn.Conv2D(4, 512, kernel_size = 3, padding = 1),

            VAE_ResidualBlock(512,512),
            VAE_AtteentionBlock(512),
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),

            # (Batch_size, 512, wight/8, Width/8) --> (Batch_size, 512, Heigh/8, Width/8)
            VAE_ResidualBlock(512,512),

            # now we want to extand the dimention
            # so we are going to do upsempling

            # Umpsample extends the image by replicating each pixel by the scale factor

            # (batch_size, 512, Heigh/8, width/8) -> (Batch_size,512,height/4, width/4)
            nn.Upsample(scale_factor=2),

            nn.Conv2d(512,512, kernel_size=3, padding=1),

            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            
            # (batch_size, 512, Heigh/4, width/4) -> (Batch_size,512,height/2, width/2)
            nn.Upsample(scale_factor=2),

            nn.Conv2d(512,512, kernel_size=3, padding=1),

            VAE_ResidualBlock(512, 256),
            VAE_ResidualBlock(512, 256),
            VAE_ResidualBlock(512, 256),


            # (batch_size, 512, Heigh/2, width/2) -> (Batch_size,512,height, width)
            nn.Upsample(scale_factor=2),



            nn.Conv2d(256,256, kernel_size=3, padding=1),

            VAE_ResidualBlock(256, 128),
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),

            nn.GroupNorm(32,128),

            nn.SiLU(),

            nn.Conv2d(128,3, kernel_size=3, padding=1)



        )


    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # x:(Batch_size, 4, height/8, width/8)

        x/= 0.18215

        for module in self:
            x = module(x)

        # (Batch_size,3 , height, width)
        return x
    