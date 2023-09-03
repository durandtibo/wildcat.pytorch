import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


### 
### 2D UNET Implementation
### from: https://gist.githubusercontent.com/johschmidt42/b9b0d55ca575d559267390f8adcf1f7c/raw/71641f14b2284d83a40cc5960443d73a88815c28/unet.py
### from: https://towardsdatascience.com/creating-and-training-a-u-net-model-with-pytorch-for-2d-3d-semantic-segmentation-model-building-6ab09d6a0862
###

@torch.jit.script
def autocrop(encoder_layer: torch.Tensor, decoder_layer: torch.Tensor):
    """
    Center-crops the encoder_layer to the size of the decoder_layer,
    so that merging (concatenation) between levels/blocks is possible.
    This is only necessary for input sizes != 2**n for 'same' padding and always required for 'valid' padding.
    """
    if encoder_layer.shape[2:] != decoder_layer.shape[2:]:
        ds = encoder_layer.shape[2:]
        es = decoder_layer.shape[2:]
        assert ds[0] >= es[0]
        assert ds[1] >= es[1]
        if encoder_layer.dim() == 4:  # 2D
            encoder_layer = encoder_layer[
                            :,
                            :,
                            ((ds[0] - es[0]) // 2):((ds[0] + es[0]) // 2),
                            ((ds[1] - es[1]) // 2):((ds[1] + es[1]) // 2)
                            ]
        elif encoder_layer.dim() == 5:  # 3D
            assert ds[2] >= es[2]
            encoder_layer = encoder_layer[
                            :,
                            :,
                            ((ds[0] - es[0]) // 2):((ds[0] + es[0]) // 2),
                            ((ds[1] - es[1]) // 2):((ds[1] + es[1]) // 2),
                            ((ds[2] - es[2]) // 2):((ds[2] + es[2]) // 2),
                            ]
    return encoder_layer, decoder_layer


def conv_layer(dim: int):
    if dim == 3:
        return nn.Conv3d
    elif dim == 2:
        return nn.Conv2d


def get_conv_layer(in_channels: int,
                   out_channels: int,
                   kernel_size: int = 3,
                   stride: int = 1,
                   padding: int = 1,
                   bias: bool = True,
                   dim: int = 2):
    return conv_layer(dim)(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                           bias=bias)


def conv_transpose_layer(dim: int):
    if dim == 3:
        return nn.ConvTranspose3d
    elif dim == 2:
        return nn.ConvTranspose2d


def get_up_layer(in_channels: int,
                 out_channels: int,
                 kernel_size: int = 2,
                 stride: int = 2,
                 dim: int = 3,
                 up_mode: str = 'transposed',
                 ):
    if up_mode == 'transposed':
        return conv_transpose_layer(dim)(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
    else:
        return nn.Upsample(scale_factor=2.0, mode=up_mode)


def maxpool_layer(dim: int):
    if dim == 3:
        return nn.MaxPool3d
    elif dim == 2:
        return nn.MaxPool2d


def get_maxpool_layer(kernel_size: int = 2,
                      stride: int = 2,
                      padding: int = 0,
                      dim: int = 2):
    return maxpool_layer(dim=dim)(kernel_size=kernel_size, stride=stride, padding=padding)


def get_activation(activation: str):
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'leaky':
        return nn.LeakyReLU(negative_slope=0.1)
    elif activation == 'elu':
        return nn.ELU()


def get_normalization(normalization: str,
                      num_channels: int,
                      dim: int):
    if normalization == 'batch':
        if dim == 3:
            return nn.BatchNorm3d(num_channels)
        elif dim == 2:
            return nn.BatchNorm2d(num_channels)
    elif normalization == 'instance':
        if dim == 3:
            return nn.InstanceNorm3d(num_channels)
        elif dim == 2:
            return nn.InstanceNorm2d(num_channels)
    elif 'group' in normalization:
        num_groups = int(normalization.partition('group')[-1])  # get the group size from string
        return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)


class Concatenate(nn.Module):
    def __init__(self):
        super(Concatenate, self).__init__()

    def forward(self, layer_1, layer_2):
        x = torch.cat((layer_1, layer_2), 1)

        return x


class DownBlock(nn.Module):
    """
    A helper Module that performs 2 Convolutions and 1 MaxPool.
    An activation follows each convolution.
    A normalization layer follows each convolution.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 pooling: bool = True,
                 activation: str = 'relu',
                 normalization: str = None,
                 dim: str = 2,
                 conv_mode: str = 'same'):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling
        self.normalization = normalization
        if conv_mode == 'same':
            self.padding = 1
        elif conv_mode == 'valid':
            self.padding = 0
        self.dim = dim
        self.activation = activation

        # conv layers
        self.conv1 = get_conv_layer(self.in_channels, self.out_channels, kernel_size=3, stride=1, padding=self.padding,
                                    bias=True, dim=self.dim)
        self.conv2 = get_conv_layer(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=self.padding,
                                    bias=True, dim=self.dim)

        # pooling layer
        if self.pooling:
            self.pool = get_maxpool_layer(kernel_size=2, stride=2, padding=0, dim=self.dim)

        # activation layers
        self.act1 = get_activation(self.activation)
        self.act2 = get_activation(self.activation)

        # normalization layers
        if self.normalization:
            self.norm1 = get_normalization(normalization=self.normalization, num_channels=self.out_channels,
                                           dim=self.dim)
            self.norm2 = get_normalization(normalization=self.normalization, num_channels=self.out_channels,
                                           dim=self.dim)

    def forward(self, x):
        y = self.conv1(x)  # convolution 1
        y = self.act1(y)  # activation 1
        if self.normalization:
            y = self.norm1(y)  # normalization 1
        y = self.conv2(y)  # convolution 2
        y = self.act2(y)  # activation 2
        if self.normalization:
            y = self.norm2(y)  # normalization 2

        before_pooling = y  # save the outputs before the pooling operation
        if self.pooling:
            y = self.pool(y)  # pooling
        return y, before_pooling


class UpBlock(nn.Module):
    """
    A helper Module that performs 2 Convolutions and 1 UpConvolution/Upsample.
    An activation follows each convolution.
    A normalization layer follows each convolution.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 activation: str = 'relu',
                 normalization: str = None,
                 dim: int = 3,
                 conv_mode: str = 'same',
                 up_mode: str = 'transposed'
                 ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalization = normalization
        if conv_mode == 'same':
            self.padding = 1
        elif conv_mode == 'valid':
            self.padding = 0
        self.dim = dim
        self.activation = activation
        self.up_mode = up_mode

        # upconvolution/upsample layer
        self.up = get_up_layer(self.in_channels, self.out_channels, kernel_size=2, stride=2, dim=self.dim,
                               up_mode=self.up_mode)

        # conv layers
        self.conv0 = get_conv_layer(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0,
                                    bias=True, dim=self.dim)
        self.conv1 = get_conv_layer(2 * self.out_channels, self.out_channels, kernel_size=3, stride=1,
                                    padding=self.padding,
                                    bias=True, dim=self.dim)
        self.conv2 = get_conv_layer(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=self.padding,
                                    bias=True, dim=self.dim)

        # activation layers
        self.act0 = get_activation(self.activation)
        self.act1 = get_activation(self.activation)
        self.act2 = get_activation(self.activation)

        # normalization layers
        if self.normalization:
            self.norm0 = get_normalization(normalization=self.normalization, num_channels=self.out_channels,
                                           dim=self.dim)
            self.norm1 = get_normalization(normalization=self.normalization, num_channels=self.out_channels,
                                           dim=self.dim)
            self.norm2 = get_normalization(normalization=self.normalization, num_channels=self.out_channels,
                                           dim=self.dim)

        # concatenate layer
        self.concat = Concatenate()

    def forward(self, encoder_layer, decoder_layer):
        """ Forward pass
        Arguments:
            encoder_layer: Tensor from the encoder pathway
            decoder_layer: Tensor from the decoder pathway (to be up'd)
        """
        up_layer = self.up(decoder_layer)  # up-convolution/up-sampling
        cropped_encoder_layer, dec_layer = autocrop(encoder_layer, up_layer)  # cropping

        if self.up_mode != 'transposed':
            # We need to reduce the channel dimension with a conv layer
            up_layer = self.conv0(up_layer)  # convolution 0
        up_layer = self.act0(up_layer)  # activation 0
        if self.normalization:
            up_layer = self.norm0(up_layer)  # normalization 0

        merged_layer = self.concat(up_layer, cropped_encoder_layer)  # concatenation
        y = self.conv1(merged_layer)  # convolution 1
        y = self.act1(y)  # activation 1
        if self.normalization:
            y = self.norm1(y)  # normalization 1
        y = self.conv2(y)  # convolution 2
        y = self.act2(y)  # acivation 2
        if self.normalization:
            y = self.norm2(y)  # normalization 2
        return y


class UNet(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 2,
                 n_blocks: int = 4,
                 start_filters: int = 32,
                 activation: str = 'relu',
                 normalization: str = 'batch',
                 conv_mode: str = 'same',
                 dim: int = 2,
                 up_mode: str = 'transposed'
                 ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_blocks = n_blocks
        self.start_filters = start_filters
        self.activation = activation
        self.normalization = normalization
        self.conv_mode = conv_mode
        self.dim = dim
        self.up_mode = up_mode

        self.down_blocks = []
        self.up_blocks = []

        # create encoder path
        for i in range(self.n_blocks):
            num_filters_in = self.in_channels if i == 0 else num_filters_out
            num_filters_out = self.start_filters * (2 ** i)
            pooling = True if i < self.n_blocks - 1 else False

            down_block = DownBlock(in_channels=num_filters_in,
                                   out_channels=num_filters_out,
                                   pooling=pooling,
                                   activation=self.activation,
                                   normalization=self.normalization,
                                   conv_mode=self.conv_mode,
                                   dim=self.dim)

            self.down_blocks.append(down_block)

        # create decoder path (requires only n_blocks-1 blocks)
        for i in range(n_blocks - 1):
            num_filters_in = num_filters_out
            num_filters_out = num_filters_in // 2

            up_block = UpBlock(in_channels=num_filters_in,
                               out_channels=num_filters_out,
                               activation=self.activation,
                               normalization=self.normalization,
                               conv_mode=self.conv_mode,
                               dim=self.dim,
                               up_mode=self.up_mode)

            self.up_blocks.append(up_block)

        # final convolution
        self.conv_final = get_conv_layer(num_filters_out, self.out_channels, kernel_size=1, stride=1, padding=0,
                                         bias=True, dim=self.dim)

        # add the list of modules to current module
        self.down_blocks = nn.ModuleList(self.down_blocks)
        self.up_blocks = nn.ModuleList(self.up_blocks)

        # initialize the weights
        self.initialize_parameters()

    @staticmethod
    def weight_init(module, method, **kwargs):
        if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose3d, nn.ConvTranspose2d)):
            method(module.weight, **kwargs)  # weights

    @staticmethod
    def bias_init(module, method, **kwargs):
        if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose3d, nn.ConvTranspose2d)):
            method(module.bias, **kwargs)  # bias

    def initialize_parameters(self,
                              method_weights=nn.init.xavier_uniform_,
                              method_bias=nn.init.zeros_,
                              kwargs_weights={},
                              kwargs_bias={}
                              ):
        for module in self.modules():
            self.weight_init(module, method_weights, **kwargs_weights)  # initialize weights
            self.bias_init(module, method_bias, **kwargs_bias)  # initialize bias

    def forward(self, x: torch.tensor):
        encoder_output = []

        # Encoder pathway
        for module in self.down_blocks:
            x, before_pooling = module(x)
            encoder_output.append(before_pooling)

        # Decoder pathway
        for i, module in enumerate(self.up_blocks):
            before_pool = encoder_output[-(i + 2)]
            x = module(before_pool, x)

        x = self.conv_final(x)

        return x

    def __repr__(self):
        attributes = {attr_key: self.__dict__[attr_key] for attr_key in self.__dict__.keys() if '_' not in attr_key[0] and 'training' not in attr_key}
        d = {self.__class__.__name__: attributes}
        return f'{d}'

### WildCat stuff

from wildcat_pytorch.wildcat.pooling import WildcatPool2d, ClassWisePool

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Upsample(nn.Module):
    def __init__(self, in_ch, in_skipch, out_ch):
        super(Upsample, self).__init__()
        # self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = double_conv(in_ch//2+in_skipch, out_ch)

    def forward(self, x1, x2):
        
        # Upsample the input
        x1 = self.up(x1)
        
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        # Pad so that the input and the skip connection are same size
        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))
        
        # Creates the skip connection 
        x = torch.cat([x2, x1], dim=1)
        
        # Doubl convolution
        x = self.conv(x)
        return x


# This model implements a GMM layer
import time


class GMMLayerSlow(nn.Module):
    
    def __init__(self):
        
        super(GMMLayerSlow, self).__init__()       
        
    def forward(self, Pij, I):
        """
        The forward pass takes two inputs: the posteriors of the classes Pij 
        the image I. Pij is of dimensions [N,K,W,H] where N is the minibatch size,
        K is the number of classes, W and H are image dimensions. I is of dimensions
        [N,C,W,H] where C is the number of channels.
        
        The output are new posteriors of the same dimension as the input posteriors
        """
        N = Pij.shape[2] * Pij.shape[3]
        K = Pij.shape[1]
        D = I.shape[1]
        B = I.shape[0]

        SPij = torch.sum(Pij, (2,3))
        alpha = SPij / N
        mu = torch.zeros((I.shape[0], K, D), device=I.device)
        sig = torch.zeros((I.shape[0], K, D, D), device=I.device)
        
        # Epsilon to avoid divisions by zero
        eps = 1.0e-5
        t0 = time.perf_counter()
        # Epsilon matrix to add to the covariance matrix to avoid singularity
        eps_sigma = torch.eye(3, device=I.device).repeat(B,1,1) * 1e-3

        # Estimate the parameters (taking special care to avoid division by zero)
        for k in range(K):
            mu[:,k,:] = torch.sum(I * Pij[:,k:k+1,:,:], (2,3)) / (eps + SPij[:,k:k+1])
            Iz = (I - mu[:,k,:].unsqueeze(-1).unsqueeze(-1)).permute(0,2,3,1).unsqueeze(-1)
            t01 = (eps + SPij[:,k:k+1]).unsqueeze(-1)
            t02 = Pij[:,k:k+1,:,:].permute(0,2,3,1).unsqueeze(-1)
            sig[:,k,:,:] = torch.sum((torch.matmul(Iz, Iz.permute(0,1,2,4,3)) * t02),(1,2)) / t01 + eps_sigma
            
        if torch.any(torch.isnan(mu)):
            print('Mu has nans')

        elif torch.any(torch.isnan(sig)):
            print('Sigma has nans')

        t1 = time.perf_counter()
        # Print the model parameters
        #for b in range(mu.shape[0]):
        #    for k in range(K):
        #        print('MB %d Cls %d  Mu = %f,%f,%f  Sigma = %f,%f' % 
        #              (b,k, mu[b,k,0].item(),mu[b,k,1].item(),mu[b,k,2].item(),torch.det(sig[b,k,:,:]).item(),torch.trace(sig[b,k,:,:]).item()))
        
        # Update the posteriors
        Qij = torch.zeros_like(Pij)
        for k in range(K):
            alpha_k = alpha[:,k].unsqueeze(-1).unsqueeze(-1)
            distro = torch.distributions.multivariate_normal.MultivariateNormal(mu[:,k,:], sig[:,k,:,:])
            Qij[:,k,:,:] = alpha_k * torch.exp(distro.log_prob(I.permute(2,3,0,1))).permute(2,0,1)
            
        # Compute the new posteriors (again, avoid division by zero)
        Pij_new = Qij / (eps + Qij.sum(1,keepdim=True))
        
        if torch.any(torch.isnan(Pij_new)):
            print('Pij has nans')

        t2 = time.perf_counter()
        # print('E %f   M %f', (t1-t0), (t2-t1))

        # Return the new posteriors
        return Pij_new
    

class GMMLayer(nn.Module):
    
    def __init__(self):
        
        super(GMMLayer, self).__init__()      
        
    def forward(self, x, img):
        """
        The forward pass takes two inputs: the class posteriors x and the image img
        Posterior x is of dimensions [N,K,W,H] where N is the minibatch size,
        K is the number of classes, W and H are image dimensions. Image is of 
        dimensions [N,C,W,H] where C is the number of channels.
        
        The output are new posteriors of the same dimension as the input posteriors
        """
        # Get all the dimensions
        b,k,w,h = x.shape
        d = img.shape[1]
        
        # Compute the weighted means and covariance matrices
        t0 = time.perf_counter()
        sx = torch.sum(x, (2,3))
        eps = 1.e-5
        eps_sigma = (torch.eye(d, device=img.device) * 1e-3).view(1,d,d,1)
        
        mu = torch.sum(img.view(b,d,1,w,h) * x.view(b,1,k,w,h), (3,4)) / (sx.view(b,1,k) + eps)
        iz = img.view(b,d,1,w,h) - mu.view(b,d,k,1,1)
        sig = torch.einsum('bkwhy,bxkwh->bxyk', 
                           torch.einsum('bkwh,bykwh->bkwhy', x, iz), iz) / (eps + sx).view(b,1,1,k) + eps_sigma
        alpha = sx / (w*h)

        # Compute the new posteriors (again, avoid division by zero)
        dd = torch.distributions.multivariate_normal.MultivariateNormal(mu.permute(0,2,1), 
                                                                        sig.permute(0,3,1,2))
        
        y = alpha.view(b,k,1,1) * torch.exp(dd.log_prob(img.permute(2,3,0,1).view(w,h,b,1,d)).permute(2,3,0,1))
        
        y = y / (eps + y.sum(1,keepdim=True))
        
        # Return the new posteriors
        return y


    

class SphericalGMMLayer(nn.Module):
    
    def __init__(self):
        
        super(SphericalGMMLayer, self).__init__()      
        
    def forward(self, x, img):
        """
        The forward pass takes two inputs: the class posteriors x and the image img
        Posterior x is of dimensions [N,K,W,H] where N is the minibatch size,
        K is the number of classes, W and H are image dimensions. Image is of 
        dimensions [N,C,W,H] where C is the number of channels.
        
        The output are new posteriors of the same dimension as the input posteriors
        """
        # Get all the dimensions
        b,k,w,h = x.shape
        d = img.shape[1]
        
        # Compute the weighted means and covariance matrices
        sx = torch.sum(x, (2,3))
        eps = 1.e-5
        eps_sigma = 1e-3
        
        # Compute the weighted means (b x d x k tensor)
        mu = torch.sum(img.view(b,d,1,w,h) * x.view(b,1,k,w,h), (3,4)) / (sx.view(b,1,k) + eps)

        # Compute the squared distance from each point to each class mean (result is [b,k,w,h])
        iz2 = torch.sum((img.view(b,d,1,w,h) - mu.view(b,d,k,1,1))**2, axis=1)

        # Compute the sample variance for every class ([b,k])
        sigma_sq = torch.sum(x * iz2, axis=(2,3)) / (eps + sx * d)

        # Compute the new class posteriors
        alpha = sx / (w*h)

        # Compute the new posteriors
        y = alpha.view(b,k,1,1) * torch.exp(-0.5 * iz2 / (sigma_sq.view(b,k,1,1) + eps_sigma))

        # Normalize the probabilities across classes
        y = y / (eps + y.sum(1,keepdim=True))
        
        # Return the new posteriors
        return y


class FuzzyCMeansLayer(nn.Module):
    
    def __init__(self):
        
        super(FuzzyCMeansLayer, self).__init__()      
        
    def forward(self, x, img):
        """
        The forward pass takes two inputs: the class posteriors x and the image img
        Posterior x is of dimensions [N,K,W,H] where N is the minibatch size,
        K is the number of classes, W and H are image dimensions. Image is of 
        dimensions [N,C,W,H] where C is the number of channels.
        
        The output are new posteriors of the same dimension as the input posteriors
        """
        # Get all the dimensions
        b,k,w,h = x.shape
        d = img.shape[1]

        # Compute the cluster centers
        c_means = torch.einsum('bkij,bcij->bkc',x,img) / torch.einsum('bkij->bk',x).unsqueeze(-1)

        # Compute the distance to the clusters for every pixel
        c_sqdev = torch.sum((img.unsqueeze(1) - c_means.unsqueeze(-1).unsqueeze(-1))**2, axis=2)

        # Compute the new posteriors
        x = (torch.sum(c_sqdev**-1, axis=1).unsqueeze(1) * c_sqdev)**-1
        
        # Line below is the c-means objective
        # torch.sum((x**2) * c_sqdev).item()

        # Return the new posteriors
        return x


# This model combines weakly supervised learning with a Gaussian mixture model
class UNet_WSL_GMM(nn.Module):

    def __init__(self, num_classes, mix_per_class=4, kmax=1, kmin=None, alpha=1, gmm_iter=10,
                 fu_blocks=0, fu_dim=3, mode='gmm'):
        super(UNet_WSL_GMM, self).__init__()
        
        # Store parameters
        self.mix_per_class = mix_per_class
        self.num_classes = num_classes
        self.gmm_iter = gmm_iter
        
        # Create the u-net portion of the model
        self.unet = UNet(in_channels=3,
                         out_channels=num_classes * mix_per_class,
                         dim=2)
        
        # Create a feature u-net if requested by user
        if fu_blocks > 0:
            self.feature_unet = UNet(in_channels=3, out_channels=fu_dim, n_blocks=fu_blocks, dim=2)
        else:
            self.feature_unet = None
        
        # Create the softmax layer
        self.softmax = torch.nn.Softmax(dim=1)
        
        # Create the GMM layers
        self.gmm = []
        for i in range(self.gmm_iter):
            if mode == 'gmm':
                self.gmm.append(GMMLayer())
            elif mode == 'sgmm':
                self.gmm.append(SphericalGMMLayer())
            elif mode == 'fcm':
                self.gmm.append(FuzzyCMeansLayer())
            else:
                raise ValueError('Unknown mode {}, options are "gmm", "sgmm", and "fcm"'.format(mode))
            
        # Create the classwise pooling layer
        ### self.classwise_pooling = ClassWisePool(mix_per_class)
        
        # Create the spatial pooling layer
        ### self.spatial_pooling = WildcatPool2d(kmax, kmin, alpha)
        self.spatial_pooling = nn.Sequential()
        self.spatial_pooling.add_module('class_wise', ClassWisePool(mix_per_class))
        self.spatial_pooling.add_module('spatial', WildcatPool2d(kmax, kmin, alpha))
        
        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]
        
        # New linear layer and softmax layer
        self.fc_pooled = torch.nn.Linear(num_classes * mix_per_class, num_classes)
        self.softmax_pooled = torch.nn.Softmax(dim=1)
        
    def forward_to_classifier(self, x):
        
        # Take the image through the u-net (result is a num_classes * mix_per_class stack of images)
        Qij = self.unet(x)
        
        # Take the softmax of the posteriors
        Pij = self.softmax(Qij)

        # Optionally, pass the inputs themselves through a u-net
        if self.feature_unet:
            x = self.feature_unet.forward(x)

        # Pass the posteriors through the GMM
        for j in range(len(self.gmm)):
            Pij = self.gmm[j](Pij, x)
            
        # Return the class posteriors
        return Pij
        
    def forward(self, x):
        # Get the final per-class posteriors
        Pij = self.forward_to_classifier(x)
        
        # Apply spatial pooling to the probabilities (take average of max-values)
        z1 = self.spatial_pooling(Pij)
        
        # Apply the linear transformation to mix probabilities for classification
        z2 = self.fc_pooled(z1)
        z3 = self.softmax_pooled(z2)
        return z3
        
        # Default code
        # z1 = self.classwise_pooling(Pij) * self.mix_per_class
        # z2 = self.spatial_pooling(z1)
        # return z2
