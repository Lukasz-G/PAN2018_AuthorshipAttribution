import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math, random, time, numpy
from unicodedata import bidirectional

##from cuda_functional import SRU, SRUCell


use_cuda = False

def repackage_variable(h, requires_grad= None):
    if type(h) == Variable:
        return Variable(h.data, requires_grad = requires_grad)
    else:
        return tuple(repackage_variable(v) for v in h)

def mse_loss(input, target):
    return torch.sum((input - target)**2) / input.data.nelement()



def tensor_decomp(tensor):
    
    pass





class Encoder(nn.Module):
    
    def __init__(self, embedding=128,  nb_words=0, frag_size=0,
                 channels = 0, kernel_size=0, padding=0, 
                 stride = 0, dilation = 0):
        super(Encoder,self).__init__()

        
        self.channels = channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.embedding = embedding
        self.nb_words = nb_words
        self.frag_size = frag_size
        
        self.channel = 300
        self.len_encoding = 500
        
        
        
        

        self.embedder_words = nn.Embedding(self.nb_words, self.embedding, max_norm=5.0, norm_type=2.0)
        #self.embedder_words.weight.requires_grad=False
        
        self.lstm = nn.LSTM(self.embedding,self.embedding, num_layers=3, dropout=0.5,bidirectional=True)
        
        self.sru1 = SRU(self.embedding, self.embedding,
                num_layers = 4,          # number of stacking RNN layers
                dropout = 0.2,           # dropout applied between RNN layers
                rnn_dropout = 0.2,       # variational dropout applied on linear transformation
                use_tanh = 1,            # use tanh?
                use_relu = 0,            # use ReLU?
                #use_selu = 0,            # use SeLU?
                bidirectional = True  # bidirectional RNN ?
               
                )
        self.sru2 = SRU(self.embedding*2*4, self.embedding,
                num_layers = 4,          # number of stacking RNN layers
                dropout = 0.2,           # dropout applied between RNN layers
                rnn_dropout = 0.2,       # variational dropout applied on linear transformation
                use_tanh = 1,            # use tanh?
                use_relu = 0,            # use ReLU?
                #use_selu = 0,            # use SeLU?
                bidirectional = True  # bidirectional RNN ?
               
                )
        
        
        
        
        self.att1 = nn.Parameter(torch.randn(self.embedding*2*4)/100)
        self.att2 = nn.Parameter(torch.randn(self.embedding*2*4)/100)
        self.att3 = nn.Parameter(torch.randn(self.len_encoding)/100)
        
        
        
        #self.channel = 256
        
        self.fc1 = nn.Linear(1502,300)
        self.fc2 = nn.Linear(300, 300)
        self.fc3 = nn.Linear(300, 300)
        self.fc4 = nn.Linear(1000,500)
        #self.fc2 = nn.Linear(128, 64)
        #self.fc1 = nn.Linear(20,1)
        #self.fc2 = nn.Linear(128, 64)
        
        
        #self.fc3 = nn.Linear(self.embedding*2 ,self.embedding)
        
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.elu = nn.ELU()
        self.leaky = nn.LeakyReLU()
        self.drop1 = nn.Dropout2d(0.5)
        self.softmax = nn.Softmax(dim=0) 
        #for encoding
        
        
        
        self.conv_1= nn.Conv2d(1, self.channel, kernel_size=(3,self.embedding), padding=(1,0), 
                               stride=2, dilation=1, groups=1)
        output_size1 = ((self.frag_size + (2*0) - 1*(3-1)-1)/2) + 1
        output_size2 = ((self.embedding + (2*0) - 1*(self.embedding-1)-1)/2) + 1
        #print(output_size1,output_size2,'after conv1')
        
       
        
        self.conv_2 = nn.Conv2d(self.channel, self.channel, 
                                kernel_size=(500,1), padding=(0,0), 
                                stride=2, dilation=1, groups=1)
        output_size1 = ((output_size1 + (2*0) - 1*(3-1)-1)/2) + 1
        output_size2 = ((output_size2 + (2*0) - 1*(1-1)-1)/2) + 1
        
        #print(output_size1,output_size2,'after conv2')
        
        self.conv_3 = nn.Conv2d(self.channel, self.len_encoding,
                                kernel_size=(1,1), padding=(0,0), 
                                stride=2, dilation=1, groups=1)
        
        
        
       
        
        
        
        
        
        
        
        kernel_value_reversed= self.dilation*(self.kernel_size-1)+2 #look at website of pytorch
        
        self.conv_4 = nn.ConvTranspose2d(100, self.channel, kernel_size=(10,1), 
                                         padding=(0,0), stride=10, 
                                         dilation=1, groups=1)
        output_size1 = ((output_size1-1)*(self.stride)-(2*self.padding)+(kernel_value_reversed))
        output_size2 = ((output_size2-1)*(self.stride)-(2*self.padding)+(kernel_value_reversed))
        #print(output_size1,output_size2,'after deconvpierwszy')
        
        self.conv_5 = nn.ConvTranspose2d(self.channel, self.channel, kernel_size=(10,1), 
                                         padding=(0,0), stride=10, 
                                         dilation=1, groups=1)
        
        
        
        
        #print(output_size1, output_size2,
        #      output_size1*output_size2*(self.channels),self.embedding,'after maxpool2')
                
        self.conv_6= nn.ConvTranspose2d(self.channel, 1, kernel_size=(10,self.embedding), 
                                        padding=(0,0), stride=10, 
                                        dilation=1, groups=1)
        output_size1 = ((output_size1-1)*(self.stride)-(2*self.padding)+(kernel_value_reversed))
        output_size2 = ((output_size2-1)*(self.stride)-(2*self.padding)+(kernel_value_reversed))        
        #print(output_size1,output_size2,'after decconv1drugi')
        
        
     
        
        
        
        output_size1 = ((output_size1-1)*(self.stride)-(2*self.padding)+(kernel_value_reversed))
        output_size2 = ((output_size2-1)*(self.stride)-(2*self.padding)+(kernel_value_reversed))  
        
        #print(output_size1,output_size2,'after decconv1dtrzeci')
        
        
        
        #print(output_size1,output_size2,'last deconv')
        self.splitter = nn.Linear(1,50)
        self.merger = nn.Linear(50,1)
        
        self.final_layer_words = torch.nn.Linear(self.embedding,self.nb_words)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
   


    def Conv_encoder(self, frag = None):
        
        #if max_length % 2 != 0:
        #      max_length = max_length + 1
        
        #frag_length = frag.size()[0]
        #print(frag.size(),'sdfdsgdf')
        #if self.training:
        #    frag = frag + Variable(torch.randn(frag.size())/10).cuda()
        
        #tensor = frag
        #frag = self.dropout1(frag)#.unsqueeze(1)
        #tensor_mean = torch.mean(tensor,0)
        #tensor = tensor - tensor_mean.expand_as(tensor)
        #frag = tensor
        #print(frag)
        #frag = self.softmax(frag)
        #frag = self.fc1(frag)
        #frag = self.elu(frag)
        
        #frag = self.fc2(frag) 
        #frag = self.elu(frag)
        #frag = self.fc3(frag) 
        #frag = self.elu(frag)
        #frag = self.fc4(frag) 
        
        
        
        return frag
    
    def Conv_decoder(self, frag_encoded = None):
        
        #if max_length % 2 != 0:
        #      max_length = max_length + 1
        
        #diff = self.frag_size - frag_length
        
        
        #self.padding = torch.nn.ZeroPad2d((0,0,0,diff))
        #if use_cuda:
            #self.padding = self.padding.cuda()
        
        #print(frag_encoded.size(),'sdfdsfgdsfgfdgfdgf')
        frag = self.relu(frag_encoded)
        frag = self.conv_4(frag)
        frag = self.drop1(frag)
        frag = self.relu(frag)
        #print(frag.size(),'after conv 4')
        
        frag = self.conv_5(frag)
        frag = self.drop1(frag)
        frag = self.relu(frag)
        #print(frag.size(),'after conv 5')
        frag = self.conv_6(frag)
        frag = self.drop1(frag)
        frag = self.relu(frag)
        #print(frag.size(),'after conv 6')
      
        #frag = self.conv_8(frag)
        #frag = self.drop1(frag)
        #frag = self.relu(frag)
        #print(frag.size(),'after conv 8')
        
        
        #frag = self.drop1(frag)
        #frag = self.tanh(frag)
        #quit()
        #print(frag.size())
        
        
        frag = frag.squeeze().unsqueeze(0)

        #print(frag,'gdfgdfgdfgdfgdfgdfgfdgdfgfdg')
        #quit()
        #
        # normalize
        norm_frag = torch.norm(frag, 2, dim=2, keepdim=True)
        rec_frag = frag / (norm_frag+0.001)
        
        #print(norm_frag,rec_frag,'dfgfdghfghgfhtrztrzrt')
        
        
        # compute probability
        norm_w = Variable(self.embedder_words.weight.data).t()
        prob_logits = torch.bmm(rec_frag, 
            norm_w.unsqueeze(0).expand(rec_frag.size(0), *norm_w.size())) / 0.07
        log_prob = F.log_softmax(prob_logits, dim=2)
        
        
        #quit()
        
        return log_prob
        
       
    def forward(self, frag=None):
        
        #length_frag = frag.size(0)
        #part_to_change = length_frag//4
        
        #random.seed(time.time())
        #words_to_change = random.sample(range(length_frag),part_to_change)
        #for change in words_to_change:
        
        
        
        #idx = torch.LongTensor([1, 0, 0, 2])
        #j = torch.arange(frag.size(0)).long().cuda()
    
        #for change in words_to_change:
             
            #print(change)   
            #new_order = random.sample(range(self.nb_words),self.nb_words)
        #    new_order = torch.LongTensor(new_order)
        #    print(new_order)
        #    words_to_change = words_to_change[change]
            #new_order = random.sample(range(self.nb_words),self.nb_words)
            
            #new_order = torch.FloatTensor(new_order)
            
            #insert = torch.zeros(self.nb_words)
            #place_to_insert = random.randint(0,self.nb_words-1)
            #insert[place_to_insert] = place_to_insert
        #    instert = torch.LongTensor(torch.randperm(self.nb_words))
        #    if use_cuda:
        #       instert = instert.cuda()
            
        #    frag[change] = torch.gather(frag[change],0,instert)
        
        
        
        
        
        
        
        #one_hot = torch.LongTensor(frag.size()[0], self.nb_words).zero_()
        #if use_cuda:
            #one_hot = one_hot.cuda()
        
        #frag_encoded = one_hot.scatter_(1, frag, 1)
        
        #suma = torch.sum(frag_encoded,dim=0, keepdim=False)
        
        #print(frag.nonzero(), 'sdfdsfdsfsf')
        
        #quit()
        frag_encoded = Variable(frag)
        if use_cuda:
            frag_encoded = frag_encoded.cuda()
        
        #embedded_frag = self.embedder_words(frag_encoded)
        
        frag_encoded = self.Conv_encoder(frag_encoded)
        #frag_quadr = self.splitter(frag_encoded.squeeze().unsqueeze(1))
        #frag_flat = self.merger(frag_quadr)
        
        #frag_decoded = self.Conv_decoder(frag_encoded)
        
        #print(embedded_frag, frag_encoded, 'fdsfgdgfdfg')
        #quit()
        return None, frag_encoded, None#, frag_decoded


class Parafac(nn.Module):
    def __init__(self, input_shape, rank):
        """
        input_shape : tuple
        Ex. (3,28,28)
        """
        super(Parafac, self).__init__()

        l, m, n = input_shape
        self.input_shape = input_shape
        self.rank = rank

        self.U = torch.nn.Parameter(torch.randn(rank, l)/100., requires_grad=True)
        self.V = torch.nn.Parameter(torch.randn(rank, m)/100., requires_grad=True)
        self.W = torch.nn.Parameter(torch.randn(rank, n)/100., requires_grad=True)

    def forward_one_rank(self, u, v, w):
        """
        input
            u : torch.FloatTensor of size l
            v : torch.FloatTensor of size m
            w : torch.FloatTensor of size n
            z : torch.FloatTensor of size o
        output
            outputs : torch.FloatTensor of size lxmxnxo
        """
        l, m, n = self.input_shape
        UV = torch.ger(u, v)
        UV = UV.unsqueeze(2).repeat(1,1,n)
        W = w.unsqueeze(0).unsqueeze(1).repeat(l,m,1)
        UVW = UV * W
        
        return UVW
    

    def forward(self):
        l, m, n = self.input_shape
        output = self.forward_one_rank(self.U[0], self.V[0], self.W[0])

        for i in range(1, self.rank):
            one_rank = self.forward_one_rank(self.U[i], self.V[i], self.W[i])
            output = output + one_rank
        return output




class GlimpseWindow:
    """
    Generates glimpses from images using Cauchy kernels.

    Args:
        glimpse_h (int): The height of the glimpses to be generated.
        glimpse_w (int): The width of the glimpses to be generated.

    """

    def __init__(self, glimpse_h=1, glimpse_w=1, controller_out=1):
        self.glimpse_h = glimpse_h
        self.glimpse_w = glimpse_w
        self.controller_out = controller_out
        self.compressor_wind = nn.Linear(self.controller_out*2,self.glimpse_w)
        if use_cuda:
            self.compressor_wind.cuda()
        
    @staticmethod
    def _get_filterbanks(delta_caps=None, center_caps=None, image_size=0, glimpse_size=None):
        """
        Generates Cauchy Filter Banks along a dimension.

        Args:
            delta_caps (B,):  A batch of deltas [-1, 1]
            center_caps (B,): A batch of [-1, 1] reals that dictate the location of center of cauchy kernel glimpse.
            image_size (int): size of images along that dimension
            glimpse_size (int): size of glimpses to be generated along that dimension

        Returns:
            (B, image_size, glimpse_size): A batch of filter banks

        """

        # convert dimension sizes to float. lots of math ahead.
        image_size = float(image_size)
        glimpse_size = float(glimpse_size)

        # scale the centers and the deltas to map to the actual size of given image.
        centers = (image_size - 1) * (center_caps + 1) / 2.0  # (B)
        deltas = (float(image_size) / glimpse_size) * (1.0 - torch.abs(delta_caps))

        # calculate gamma for cauchy kernel
        gammas = torch.exp(1.0 - 2 * torch.abs(delta_caps))  # (B)

        # coordinate of pixels on the glimpse
        glimpse_pixels = Variable(torch.arange(0, glimpse_size) - (glimpse_size - 1.0) / 2.0)  # (glimpse_size)
        if use_cuda:
            glimpse_pixels = glimpse_pixels.cuda()

        # space out with delta
        glimpse_pixels = deltas[:, None] * glimpse_pixels[None, :]  # (B, glimpse_size)
        # center around the centers
        glimpse_pixels = centers[:, None] + glimpse_pixels  # (B, glimpse_size)

        # coordinates of pixels on the image
        image_pixels = Variable(torch.arange(0, image_size))  # (image_size)
        if use_cuda:
            image_pixels = image_pixels.cuda()

        fx = image_pixels - glimpse_pixels[:, :, None]  # (B, glimpse_size, image_size)
        fx = fx / gammas[:, None, None]
        fx = fx ** 2.0
        fx = 1.0 + fx
        fx = math.pi * gammas[:, None, None] * fx
        fx = 1.0 / fx
        fx = fx / (torch.sum(fx, dim=2) + 1e-4)[:, :, None]  # we add a small constant in the denominator division by 0.

        return fx.transpose(1, 2)

    def get_attention_mask(self, glimpse_params=None, mask_h=None, mask_w=None):
        """
        For visualization, generate a heat map (or mask) of which pixels got the most "attention".

        Args:
            glimpse_params (B, hx):  A batch of glimpse parameters.
            mask_h (int): The height of the image for which the mask is being generated.
            mask_w (int): The width of the image for which the mask is being generated.

        Returns:
            (B, mask_h, mask_w): A batch of masks with attended pixels weighted more.

        """

        batch_size, _ = glimpse_params.size()

        # (B, image_h, glimpse_h)
        F_h = self._get_filterbanks(delta_caps=glimpse_params[:, 2], center_caps=glimpse_params[:, 0],
                                    image_size=mask_h, glimpse_size=self.glimpse_h)

        # (B, image_w, glimpse_w)
        F_w = self._get_filterbanks(delta_caps=glimpse_params[:, 2], center_caps=glimpse_params[:, 1],
                                    image_size=mask_w, glimpse_size=self.glimpse_w)

        # (B, glimpse_h, glimpse_w)
        glimpse_proxy = Variable(torch.ones(batch_size, self.glimpse_h, self.glimpse_w))

        # find the attention mask that lead to the glimpse.
        mask = glimpse_proxy
        mask = torch.bmm(F_h, mask)
        mask = torch.bmm(mask, F_w.transpose(1, 2))

        # scale to between 0 and 1.0
        mask = mask - mask.min()
        mask = mask / mask.max()
        mask = mask.float()

        return mask

    def get_glimpse(self, images=None, glimpse_params=None):
        """
        Generate glimpses given images and glimpse parameters. This is the main method of this class.

        The glimpse parameters are (h_center, w_center, delta). (h_center, w_center)
        represents the relative position of the center of the glimpse on the image. delta determines
        the zoom factor of the glimpse.

        Args:
            images (B, h, w):  A batch of images
            glimpse_params (B, 3):  A batch of glimpse parameters (h_center, w_center, delta)

        Returns:
            (B, glimpse_h, glimpse_w): A batch of glimpses.

        """
        
        
        #images = images.permute(1,0,2)
        
        #print(images.size(), glimpse_params.size(),'gdfgdfgdfgdfdfgdg')
        batch, image_h, image_w = images.size()
        
        #print(images.size(), glimpse_params.size(),'gdfgdfgdfgdfdfgdg')
        #quit()
        # (B, image_h, glimpse_h)
        F_h = self._get_filterbanks(delta_caps=glimpse_params[:, 2], center_caps=glimpse_params[:, 0],
                                    image_size=image_h, glimpse_size=self.glimpse_h)#self.glimpse_h

        # (B, image_w, glimpse_w)
        F_w = self._get_filterbanks(delta_caps=glimpse_params[:, 2], center_caps=glimpse_params[:, 1],
                                    image_size=image_w, glimpse_size=self.glimpse_w)#self.glimpse_w

        #print(F_w.size(),F_h.size(),images.size(),'fsdfsdfsdfsdfdsf')
        
        #F_w = self.compressor_wind(F_w)
        
        #print(F_w,F_h,'fsdfsdfsdfsdfdsf')
        # F_h.T * images * F_w
        glimpses = images
        glimpses = torch.bmm(F_h.transpose(1, 2), glimpses)
        glimpses = torch.bmm(glimpses, F_w)

        #print(glimpses,'fsdhfsjdfsdjfsdnfsdfs')
        return glimpses  # (B, glimpse_h, glimpse_w)


class ARC(nn.Module):
    """
    This class implements the Attentive Recurrent Comparators. This module has two main parts.

    1.) controller: The RNN module that takes as input glimpses from a pair of images and emits a hidden state.

    2.) glimpser: A Linear layer that takes the hidden state emitted by the controller and generates the glimpse
                    parameters. These glimpse parameters are (h_center, w_center, delta). (h_center, w_center)
                    represents the relative position of the center of the glimpse on the image. delta determines
                    the zoom factor of the glimpse.

    Args:
        num_glimpses (int): How many glimpses must the ARC "see" before emitting the final hidden state.
        glimpse_h (int): The height of the glimpse in pixels.
        glimpse_w (int): The width of the glimpse in pixels.
        controller_out (int): The size of the hidden state emitted by the controller.

    """

    def __init__(self, num_glimpses=8, glimpse_h=8, glimpse_w=8, controller_out=128,  nb_layers=2, nb_words=0):
        super(ARC,self).__init__()

        self.num_glimpses = num_glimpses
        self.glimpse_h = glimpse_h
        self.glimpse_w = glimpse_w
        self.controller_out = controller_out
        self.nb_layers = nb_layers
        #self.encoder_size = encoder_size
        self.nb_words = nb_words
        #dummy frag size
        
        
        
       
        
        
        # main modules of ARC
        
        self.embedder_words = nn.Embedding(self.nb_words, self.controller_out, max_norm=1.0, norm_type=2.0)
        #self.embedder_words.weight.requires_grad=False
      
        
        #self.controller = SRU(input_size=self.glimpse_h*self.glimpse_w, hidden_size=self.controller_out,
        #               num_layers = 4,
        #               dropout = 0.1,           # dropout applied between RNN layers
        #              rnn_dropout = 0.1,       # variational dropout applied on linear transformation         
        #              use_relu = 0,
        #             use_tanh = 1,
        #               bidirectional=True)
        self.controller = nn.LSTMCell(input_size=(glimpse_h * glimpse_w), 
                                  #num_layers=1,
                                  hidden_size=self.controller_out,
                                  #dropout=0.0,
                                  #bidirectional=True
                                  )
        self.compressor = nn.Linear(2,1)
        self.glimpser = nn.Linear(in_features=self.controller_out, out_features=3)
        
        
        self.fc1 = nn.Linear(50,100)
        self.fc2 = nn.Linear(100,250)
        #self.fc3 = nn.Linear(250,100)
        self.fc3 = nn.Linear(250,50)
        
        #self.encoder_text = SRU(input_size=self.encoder_size, hidden_size=self.encoder_size,
        #               num_layers = self.nb_layers,
        #               dropout = 0.0,           # dropout applied between RNN layers
        #               rnn_dropout = 0.0,       # variational dropout applied on linear transformation         
        #               use_relu = 1,
        #               use_tanh = 0,
        #               bidirectional=True)
       
        
       
        
        
        
        
        
        self.dropout1 = nn.Dropout(0.0)
        self.dropout2 = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        # this will actually generate glimpses from images using the glimpse parameters.
        self.glimpse_window = GlimpseWindow(glimpse_h=self.glimpse_h, glimpse_w=self.glimpse_w,controller_out=self.controller_out)
    
    
    
    
    
            
    
    def forward(self, image_pairs=None):
        """
        The method calls the internal _forward() method which returns hidden states for all time steps. This i

        Args:
            image_pairs (B, 2, h, w):  A batch of pairs of images

        Returns:
            (B, controller_out): A batch of final hidden states after each pair of image has been shown for num_glimpses
            glimpses.

        """
        pairs_=[]
        
        for text in image_pairs:
            frag_to_encode = text.view(1,-1,1)#.t()
            pairs_.append(frag_to_encode)
        
            
            '''
            mean_frag = torch.mean(frag_to_encode,0)
            #print(mean_frag)
            frag_norm = frag_to_encode - mean_frag.expand_as(frag_to_encode)
            #print(frag_norm)
            U,S,V = torch.svd(frag_norm.t(), some=False)
        
            _u = Variable(U[:,:32].data, requires_grad=True)
            if use_cuda:
                _u = _u.cuda()
            
            #print(frag_norm.size(),_u.size())
            frag_pca = torch.mm(frag_norm,_u)
            pairs_.append(frag_pca.view(1,32,32))
            '''
            
        
        
        #if self.training:
        #Hx = Variable(torch.zeros(self.nb_layers,1, self.controller_out*2))
        #    if use_cuda:
        #        Hx = Hx.cuda()
        # return only the last hidden state
            #for pair in image_pairs:
        
             #   all_hidden = self._forward(pair)  # (2*num_glimpses, B, controller_out)
             #   last_hidden = all_hidden[-self.nb_layers:, :, :]  # (B, controller_out)
             #   out_, Hx = self.encoder_results(last_hidden, c0=Hx)
        #else:
            #Hx = Variable(torch.zeros(self.nb_layers,1, self.controller_out*2))
            #if use_cuda:
            #    Hx = Hx.cuda()
        last_hidden = self._forward(pairs_)  # (2*num_glimpses, B, controller_out)
        #last_hidden = all_hidden[-2:, :, :]  # (B, controller_out)
        
        return last_hidden

    def _forward(self, pair):
        """
        The main forward method of ARC. But it returns hidden state from all time steps (all glimpses) as opposed to
        just the last one. See the exposed forward() method.

        Args:
            image_pairs: (B, 2, h, w) A batch of pairs of images

        Returns:
            (2*num_glimpses, B, controller_out) Hidden states from ALL time steps.

        """
        
        
        
        
            
        
        #print(pair)
        #quit()
        
        
        
        # convert to images to float.
        #image_pairs = image_pairs.float()
        
        # calculate the batch size
        batch_size = 1#image_pairs.size()[1]

        # an array for collecting hidden states from each time step.
        all_hidden = []

        # initial hidden state of the LSTM.
        #Hx = Variable(torch.zeros(self.nb_layers,batch_size, self.controller_out*2))  # (B, controller_out)
        #Cx = Variable(torch.zeros(batch_size, self.controller_out))  # (B, controller_out)
        
        Hx = Variable(torch.zeros(1, self.controller_out))  # (B, controller_out)
        Cx = Variable(torch.zeros(1, self.controller_out))  # (B, controller_out)
        
        
        if use_cuda:
            Hx = Hx.cuda()#, Cx.cuda()
            Cx = Cx.cuda()
        # take `num_glimpses` glimpses for both images, alternatingly.
        numb_images = len(pair)
        for turn in range(numb_images*self.num_glimpses):
            # select image to show, alternate between the first and second image in the pair
            #print(turn % 2,'dfgdfgfd')
            images_to_observe = pair[turn % numb_images]  # (B, h, w)
            #Hx.data.clamp_(-1.0,1.0)#!!!!!!!!!!!!!!!!!!!!!
            
            #print(images_to_observe)
            #quit()
            # choose a portion from image to glimpse using attention
            #Hx_ = self.compressor(Hx.permute(2,1,0).contiguous()).permute(2,1,0).contiguous()
            glimpse_params = torch.tanh(self.glimpser(Hx.view(-1))).view(-1,3)  # (B, 3)  a batch of glimpse params (x, y, delta)
            glimpses = self.glimpse_window.get_glimpse(images_to_observe, glimpse_params)  # (B, glimpse_h, glimpse_w)
            flattened_glimpses = glimpses.view(1, -1)  # (B, glimpse_h * glimpse_w), one time-step

            # feed the glimpses and the previous hidden state to the LSTM.
            #print(Hx,flattened_glimpses, self.controller.out_size)
            #Hx.data.clamp_(-1.0,1.0)#!!!!!!!!!!!!!!!!!!!!!
            #Hx = torch.tanh(Hx)
            #Cx = torch.tanh(Cx)
            #flattened_glimpses.data.clamp_(-1.0,1.0)
            Hx, Cx = self.controller(flattened_glimpses,(Hx, Cx))
            #Hx = self.dropout2(Hx)
            #Cx = self.dropout2(Cx)
            #Cx, Hx = self.controller(flattened_glimpses, c0=Hx)  # (B, controller_out), (B, controller_out)
            
            #print(Hx.size(),'fndbsfmdsbfs')
            #quit()
            
            # append this hidden state to all states
            #all_hidden.append(Hx)

        #all_hidden = torch.cat(all_hidden)  # (2*num_glimpses, B, controller_out)

        #all_hidden = self.dropout2(all_hidden)
        #print(all_hidden.size(),'sdfsdfsfsdfd')
        #quit()
        
        # return a batch of all hidden states.
        return Hx


class ArcBinaryClassifier(nn.Module):
    """
    A binary classifier that uses ARC.
    Given a pair of images, feeds them the ARC and uses the final hidden state of ARC to
    classify the images as belonging to the same class or not.

    Args:
        num_glimpses (int): How many glimpses must the ARC "see" before emitting the final hidden state.
        glimpse_h (int): The height of the glimpse in pixels.
        glimpse_w (int): The width of the glimpse in pixels.
        controller_out (int): The size of the hidden state emitted by the controller.

    """

    def __init__(self, num_glimpses=8, glimpse_h=8, glimpse_w=8, controller_out= 128, 
                 nb_layers=2,nb_words=0):
        super(ArcBinaryClassifier,self).__init__()
        self.arc = ARC(
            num_glimpses=num_glimpses,
            glimpse_h=glimpse_h,
            glimpse_w=glimpse_w,
            controller_out=controller_out,
            nb_layers=nb_layers,
            nb_words=nb_words
            
            )

        # two dense layers, which take the hidden state from the controller of ARC and
        # classify the images as belonging to the same class or not.
        self.dense1 = nn.Linear(controller_out, 64)
        #self.compr = nn.Linear(2, 1)
        self.dense2 = nn.Linear(64, 1)
        self.dropout2 = nn.Dropout(0.0)
        
    def forward(self, image_pairs=None):
        
        if use_cuda:
            self.arc = self.arc.cuda()
        
        
        arc_out = self.arc(image_pairs)
        
        arc_out = self.dropout2(arc_out)
        #print(arc_out)
        #arc_out = self.compr(arc_out.permute(2,1,0).contiguous()).permute(2,1,0).contiguous()
        
        #arc_out = self.dropout2(arc_out)
        #print(arc_out)
        #quit()
        
        d1 = F.elu(self.dense1(arc_out.view(-1)))
        d1 = self.dropout2(d1)
        decision = torch.sigmoid(self.dense2(d1))

        return decision

    def save_to_file(self, file_path=None):
        torch.save(self.state_dict(), file_path)

class BinaryMeasurer(nn.Module):
    """
    A binary classifier that uses ARC.
    Given a pair of images, feeds them the ARC and uses the final hidden state of ARC to
    classify the images as belonging to the same class or not.

    Args:
        num_glimpses (int): How many glimpses must the ARC "see" before emitting the final hidden state.
        glimpse_h (int): The height of the glimpse in pixels.
        glimpse_w (int): The width of the glimpse in pixels.
        controller_out (int): The size of the hidden state emitted by the controller.

    """

    def __init__(self, nb_words=[], nb_authors=0, freq_feature=[], nb_fcat=0, freq_masking=[]):
        super(BinaryMeasurer,self).__init__()
        
        #nb_bi, nb_tri, nb_tetra = nb_words
        #print(nb_words)
        self.freq_feature = freq_feature
        self.nb_fcat=nb_fcat
        self.freq_masking = freq_masking
        
        #nb_words_limits = [s.squeeze().size(0) for x in self.freq_masking for s in x]
        #print(len(freq_feature[0]))
        #quit()
        #self.n_neurons = 1502#*4*2
        #self.n_neurons_2 = 600
        #self.fc1 = nn.Linear(self.n_neurons,self.n_neurons_2//2)
        #self.fc2 = nn.Linear(self.n_neurons_2//2,self.n_neurons_2//2)
        
        
        #if use_cuda:
        #    module
        
        #self.list_modules = []
        
        #for cat in range(nb_fcat):
            
        nw = sum([x//10 for x in nb_words])    
        #print(nw)
        #quit()
        self.layer1 = nn.ModuleList([nn.Linear(nb_words[r],nb_words[r]) for r in range(nb_fcat)])   
        self.layer2 = nn.ModuleList([nn.Linear(nb_words[r],nb_words[r]) for r in range(nb_fcat)])
        self.layer3 = nn.ModuleList([nn.Linear(nb_words[r],nb_words[r]) for r in range(nb_fcat)])
        
        #self.fl = nn.ModuleList([nn.Linear(nb_words[r],nb_authors) for r in range(nb_fcat)])
        self.par =  nn.Parameter(torch.FloatTensor([1.0]))# for r in range(nb_fcat)])
        #self.attn2 =  nn.Parameter(torch.randn(nw)/nw)# for r in range(nb_fcat)])
        #self.fc1 = nn.Linear(len(nb_words),nb_authors)
        self.fc2 = nn.Linear(nb_authors*2,nb_authors)
        self.fc3 = nn.Linear(nw,nb_authors)
        self.fc4 = nn.Linear(nw,nb_authors)
        #self.conv1 = nn.Conv1d(1,nb_authors,nw, padding=0)
        #kernel_size_2 = (nw+2*2-(5-1)-1)+1 
        #self.conv2 = nn.Conv1d(nb_authors,nb_authors,1, padding=0)
        
        #(nw+2*2-(5-1)-1)+1 
        
        
        
        #self.fc5 = nn.Linear(nb_bi,nb_bi)
        #self.fc6 = nn.Linear(nb_bi,nb_bi)
        #self.fc7 = nn.Linear(nb_bi,nb_bi)
        #self.fc8 = nn.Linear(nb_bi,nb_bi)
        
        #self.fc9 = nn.Linear(nb_tri,nb_tri)
        #self.fc10 = nn.Linear(nb_tri,nb_tri)
        #self.fc11 = nn.Linear(nb_tri,nb_tri)
        #self.fc12 = nn.Linear(nb_tri,nb_tri)
        
        
        #self.fc13 = nn.Linear(nb_tetra,nb_tetra)
        #self.fc14 = nn.Linear(nb_tetra,nb_tetra)
        #self.fc15 = nn.Linear(nb_tetra,nb_tetra)
        #self.fc16 = nn.Linear(nb_tetra,nb_tetra)
        
        
       
        
        
        self.dropout1 = nn.Dropout(0.5)
        self.dropout1_2 = nn.Dropout(0.0)
        self.dropout2 = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.softmax_ = nn.Softmax(dim=0) 
        self.softmax = nn.LogSoftmax(dim=0) 
        
    
    def _siam(self,tensor):
        
        #print(tensor.size())
        #tensor = Variable(text.view(1,-1))
        #if use_cuda:
        #    tensor = tensor.cuda()
        #print(tensor,self.fc1)
        
        #tensor_mean = torch.mean(tensor,0)
        #tensor = tensor - tensor_mean.expand_as(tensor)
        
        #tensor = self.dropout1_2(tensor)
        #tensor = self.elu(tensor)
        #tensor = self.fc1(tensor)
        #tensor = self.dropout1(tensor)
        #tensor = self.elu(tensor)
        #tensor = self.fc2(tensor)
        #tensor = self.dropout1(tensor)
        #tensor = self.elu(tensor)
        #tensor = self.fc3(tensor)
        #tensor = self.dropout1(tensor)
        #tensor = self.elu(tensor)
        #tensor = self.fc4(tensor)
        #tensor = self.dropout1(tensor)
        #tensor = self.elu(tensor)
        #tensor = self.fc5(tensor)
        #tensor = self.dropout1(tensor)
        #tensor = self.elu(tensor)
        #tensor = self.fc6(tensor)
        #tensor = self.elu(tensor)
        #tensor = self.dropout1(tensor)
        #tensor = self.fc7(tensor)
        #tensor = self.elu(tensor)
        #tensor = self.dropout1(tensor)
        #tensor = self.fc8(tensor)
        
        
        
        
        return tensor
    
    def _bi(self,tensor):
        
        #print(self.fc6, tensor)
        #tensor = self.dropout1(tensor)
        tensor = self.fc6(tensor)
        #print(tensor.norm(p=2))
        tensor = self.dropout1(tensor)
        tensor = self.elu(tensor)
        tensor = self.fc7(tensor)
        #print(tensor.norm(p=2))
        #tensor = self.dropout1(tensor)
        #tensor = self.elu(tensor)
        #tensor = self.fc8(tensor)
        #print(tensor.norm(p=2))
        #tensor = tensor.data.cpu().numpy().tolist()
        
        
        
        return tensor
        
    def _tri(self,tensor):
        
        #tensor = self.dropout1(tensor)
        tensor = self.fc10(tensor)
        #print(tensor.norm(p=2))
        tensor = self.dropout1(tensor)
        tensor = self.elu(tensor)
        tensor = self.fc11(tensor)
        #print(tensor.norm(p=2))
        #tensor = self.dropout1(tensor)
        #tensor = self.elu(tensor)
        #tensor = self.fc12(tensor)
        #print(tensor.norm(p=2))
        #tensor = tensor.data.cpu().numpy().tolist()
        
        
        
        return tensor
    
    def _tetra(self,tensor):
        
        
        #tensor = self.dropout1(tensor)
        tensor = self.fc14(tensor)
        #print(tensor.norm(p=2))
        tensor = self.dropout1(tensor)
        tensor = self.elu(tensor)
        tensor = self.fc15(tensor)
        #print(tensor.norm(p=2))
        #tensor = self.dropout1(tensor)
        #tensor = self.elu(tensor)
        #tensor = self.fc16(tensor)
        #print(tensor.norm(p=2))
        #tensor = tensor.data.cpu().numpy().tolist()
        
        
        
        return tensor
    
        
    
    def forward(self, text, unknown):
        
        tensors = []
        
        #print(text)
        #quit()
        for nb_1, tensor in enumerate(text):
            
            #if nb_ == 1:
            #    break
            
            #for freq_ in self.freq_feature:
                #for nb_2, cut_ in enumerate(cut):
            #print(tensor.size(),cut_.size())
                    #tensor_cut = torch.index_select(tensor.squeeze(),0, cut_.squeeze())
            #print(tensor.size(),self.freq_feature[nb_1][0].size(),self.freq_feature[nb_1][1].size())
                    #tensor_mean = torch.mean(tensor,0)
            #print(tensor_cut.size(),self.freq_feature[nb_1][nb_2][0].size(), cut_.squeeze().size())
            #tensor_ = tensor# * self.freq_feature[nb_1][2]
            #tensor_ = torch.log(tensor_)
            #tensor_mean = torch.mean(tensor,0)
            #std = torch.std(tensor)
            #print(std)
            #tensor_ = (tensor - tensor_mean)/ std
          
            #tensor_ = tensor - self.freq_feature[nb_1][0]
            #print(tensor.norm(p=2))
            #std_ = self.freq_feature[nb_1][1]
            
            #std_[std_==0.0] = 0.000001
            #tensor_ = tensor_ / std_
           
            
            #std = torch.std(tensor)
            #print(tensor.norm(p=2), self.freq_feature[nb_][1].nonzero().size())
            #quit()
            #if self.training:
            #    torch.manual_seed(int(time.time()*1000))
            #    noise = torch.randn(tensor.size(0)) * 0.1
            
            #print(tensor.norm(p=2))
            #quit()
            
            #    tensor_ = tensor_ + noise
            
            

            tensor_ = Variable(tensor.squeeze().float())
            if use_cuda:
                tensor_ = tensor_.cuda()
            
            
            tensors.append(tensor_)
            
        
        
       
        results = []
        #results2 = []
        #print(len( self.layer1),len(tensors))
        
        for x in range(self.nb_fcat):#self.nb_fcat
            
            if True:#  x != 2:# and x != 5:
                #print(x)
                t = tensors[x]
                #print(t, self.layer1[x])
                #t = self.dropout1_2(t)
                t = self.layer1[x](t)
                t = self.elu(t)
                #t = self.dropout1(t)
                t = self.layer2[x](t)
                #t = self.elu(t1)
                t = self.dropout1(t)
                #t = self.layer3[x](t)
                #t = self.elu(t)
                #t = self.dropout1(t)
                #t = self.softmax(t)
            
            #result = self.list_modules[3](tensors[3])
            
                #results.append(t.unsqueeze(0))
                results.append(t.squeeze())
                #results2.append(t.squeeze())
        #result1 = self._bi(tensors[0])
        #result2 = self._tri(tenltssors[1])
        #result3 = self._tetra(tensors[2])
        
        
        #at1 = self.softmax_(self.attn1)
        #at2 = self.softmax_(self.attn2)
        
        #result1 = self.softmax(result1)
        #result2 = self.softmax(result2)
        #result3 = self.softmax(result3)
        
        #copying weights!!!!!!
        
        if self.training:
            teach = numpy.random.binomial(1, 0.5)
        else:
            teach = False
        #print(unknown)
        k = 0.5
        result1, result2, result_alter = None, None, None
        #if not(unknown):
            #print(results1)
        result_training_set = torch.cat(results,dim=0)#.unsqueeze(0).unsqueeze(0)
        result_training_set = self.elu(result_training_set)
            #result = self.dropout1(result)
        result_training_set = self.fc3(result_training_set)
        result_training_set = self.softmax(result_training_set)
        #else:
        result_test_set = torch.cat(results,dim=0)#.unsqueeze(0).unsqueeze(0)
        result_test_set = self.elu(result_test_set)
            #result = self.dropout1(result)
        result_test_set = self.fc4(result_test_set)
        result_test_set = self.softmax(result_test_set)
        
        result_mix_set = torch.cat(results,dim=0)
        result_mix_set1 = self.fc3(result_mix_set)
        result_mix_set2 = self.fc4(result_mix_set)
        result_mix_set1 = self.elu(result_mix_set1)
        result_mix_set1 = self.elu(result_mix_set2)
        result_mix_set =  self.fc2(torch.cat([result_mix_set1,result_mix_set2]))
        result_mix_set = self.softmax(result_mix_set)
        
        
        
        guessed_label = (result_training_set+result_test_set).data.cpu().max(dim=0)[1][0]
        
        self.fc4.weight.data = self.fc3.weight.data*(1-k) + self.fc4.weight.data*k
        self.fc4.bias.data = self.fc3.bias.data*(1-k) + self.fc4.bias.data*k
        
            #print('piko')
            #self.fc4.weight.data = (self.fc4.weight.data*(1-k)) + (self.fc4.weight.data*k)#polyakov avariging
            #self.fc4.bias.data = (self.fc4.bias.data*(1-k)) + (self.fc4.bias.data*k)
            #for x in range(self.nb_fcat):
            #    self.layer1[x].weight.data = self.layer1[x].weight.data**2
            #    self.layer1[x].bias.data = self.layer1[x].bias.data**2
        #print(result.norm(p=2))
        
        
        
        
        
        #quit()
        return result_training_set, result_test_set, result_mix_set, guessed_label

class StyloMeasurer(nn.Module):
  

    def __init__(self, nb_words=[], nb_authors=0, freq_feature=[], nb_fcat=0, freq_masking=[]):
        super(StyloMeasurer,self).__init__()
        
        #nb_bi, nb_tri, nb_tetra = nb_words
        #print(nb_words)
        self.freq_feature = freq_feature
        self.nb_fcat=nb_fcat
        self.freq_masking = freq_masking
        self.nb_authors = nb_authors
        nw = sum([x for x in nb_words])    
        #print(nw)
        #quit()
        self.layer1 = nn.ModuleList([nn.Linear(nb_words[r],nb_words[r]) for r in range(nb_fcat)])   
        self.layer2 = nn.ModuleList([nn.Linear(nb_words[r],nb_words[r]) for r in range(nb_fcat)])
        #self.layer3 = nn.ModuleList([nn.Linear(nb_words[r],nb_words[r]) for r in range(nb_fcat)])
        
        #self.fl = nn.ModuleList([nn.Linear(nb_words[r],nb_authors) for r in range(nb_fcat)])
        #self.par1 =  nn.Parameter(torch.FloatTensor([0.5]))# for r in range(nb_fcat)])
        #self.par2 =  nn.Parameter(torch.FloatTensor(torch.rand(1)/100))
        #self.attn2 =  nn.Parameter(torch.randn(nw)/nw)# for r in range(nb_fcat)])
        #self.fc1 = nn.Linear(len(nb_words),nb_authors)
        #self.fc2 = nn.Linear(nb_authors*2,nb_authors)
        #self.fc3 = nn.Linear(nw,nw)
        self.fc4 = nn.Linear(nw,nb_authors)
   
       
        self.register = [[] for _ in range(nb_authors)]
        
        self.dropout1 = nn.Dropout(0.0)
        self.dropout1_2 = nn.Dropout(0.0)
        self.dropout2 = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.softmax_ = nn.Softmax(dim=0) 
        self.softmax = nn.LogSoftmax(dim=0) 

     
    def forward(self, text, unknown):
        
        tensors = []
        
        #print(text)
        #quit()
        for nb_1, tensor in enumerate(text):
            
        
            
            

            tensor_ = Variable(tensor.squeeze().float())
            if use_cuda:
                tensor_ = tensor_.cuda()
            
            
            tensors.append(tensor_)
            
        
        
       
        results1 = []
        results2 = []
        #print(len( self.layer1),len(tensors))
        
        for x in range(self.nb_fcat):#self.nb_fcat
            
            if True:#  x != 2:# and x != 5:
                #print(x)
                t = tensors[x]
                #print(t, self.layer1[x])
                #t = self.dropout1_2(t)
                t = self.layer1[x](t)
                t = self.elu(t)
                #t = self.dropout1(t)
                t1 = self.layer2[x](t)
                #t = self.elu(t1)
                #t = self.dropout1(t)
                #t = self.layer3[x](t)
                #t = self.elu(t)
                #t = self.dropout1(t)
                #t = self.softmax(t)
            
            #result = self.list_modules[3](tensors[3])
            
                #results.append(t.unsqueeze(0))
                results1.append(t1.squeeze())
                #results2.append(t.squeeze())
        #result1 = self._bi(tensors[0])
        #result2 = self._tri(tenltssors[1])
        #result3 = self._tetra(tensors[2])
        
        
        #at1 = self.softmax_(self.attn1)
        #at2 = self.softmax_(self.attn2)
        
        #result1 = self.softmax(result1)
        #result2 = self.softmax(result2)
        #result3 = self.softmax(result3)
        
        #copying weights!!!!!!
        #prob = (self.par1 + self.par2).data.cpu()[0]
        alter_label = None
        if self.training:
            teach = numpy.random.binomial(1, 0.5)
        else:
            teach, alter_label = False, False 
        #print(unknown)
        #k = 0.5
        result1, result2, result_alter = None, None, None
        
        
        
        #if not(unknown):
            #print(results1)
        result_training_set = torch.cat(results1,dim=0).squeeze()#.unsqueeze(0).unsqueeze(0)
        
        #result_training_set = self.elu(result_training_set)
        #if unknown:
        #result_training_set = self.dropout1(result_training_set)
        #result_training_set = self.fc3(result_training_set)
        #if unknown:
        #result_training_set = self.dropout1(result_training_set)
        
        result_training_set = self.elu(result_training_set)
            #result = self.dropout1(result)
        result_training_set = self.fc4(result_training_set)
        
        
        #if unknown and teach:
            #print(self.par1, self.par2)
        #if unknown:
            #best_label = result_training_set.data.cpu().topk(1,dim=0)[1][0]
        #    alter_label = result_training_set.data.cpu().topk(self.nb_authors,dim=0)[1].numpy().tolist()
            
        #print(alter_label)
        #quit()
        result_training_set = self.softmax(result_training_set)
        
        
            
        
        
        
        
        #quit()
        return result_training_set, alter_label, None, results2, None
    
class StyloMentoring(nn.Module):
  

    def __init__(self, nb_words=[], nb_authors=0, freq_feature=[], nb_fcat=0, freq_masking=[]):
        super(StyloMentoring,self).__init__()
        
        self.fc3 = nn.Linear(nb_authors,nb_authors)
        
        


