import torch
import torch.nn as nn
from networkUtils.modelArchitectures import *
from networkUtils.lossFunctions import *
from networkUtils.physicsloss import *

class Model():
    def __init__(self, params):
        '''
        Model wrapper class to setup a model depending on parameters.
        
        params = {
        
        'model': (str),                                    # class of model from modelArchitectures.py
        'optimizer': (str),                                # optimizer for model from torch.optim, currently only using 'Adam'
        'loss_fxn': (str),                                 # one of the loss functions from lossFunctions.py or torch.nn
        'learn_rate': (float),                             # starting learning rate
        'channels': (int),                                 # channels (energy levels) of input data of shape [ch, z, x, y]
        'features': (int),                                 # number of depth points. z in [ch, z, x, y]
        #'loss_w_range': (tuple),                          # range in Mm to weight loss function (not working since we switched to column mass)
        #'loss_scale': (float),                            # scale weight loss function (not working since we switched to column mass)
        #'height_vector': (np array),                      # atmosphere height vector in Mm (not working since we switched to column mass)
        'cuda': {'use_cuda': (bool), 'multi_gpu': (bool)}, # whether to use cuda and multi GPU when training
        'mode': (string)                                   # either 'training' or 'testing'
        
        }        
        '''


        ## plotting only needs 1 forward pass so this gets skipped ##
        if params['mode'] == 'training':
        
        
            # ######################### SUNNYNET ARCHITECTURES ############################ 
            # if params['model'] == 'SunnyNet_1x1':
            #     self.network = SunnyNet_1x1(params['in_channels'], params['features'],1,1)
                
            # elif params['model'] == 'SunnyNet_3x3':
            #     self.network = SunnyNet_3x3(params['in_channels'], params['out_channels'], params['features'],3,3)
                
            # elif params['model'] == 'SunnyNet_5x5':
            #     self.network = SunnyNet_5x5(params['in_channels'], params['out_channels'], params['features'],5,5)
                
            # elif params['model'] == 'SunnyNet_7x7':
            #     self.network = SunnyNet_7x7(params['in_channels'], params['out_channels'], params['features'],7,7)
            # ######################### SUNNYNET ARCHITECTURES ############################
            
            # else:
            #     raise Exception("!!Invalid model architecture!!")

            self.network = SunnyNet(params['in_channels'], params['out_channels'], params['features'], params['window_size'])

            ## set CPU/GPU ##
            if params['cuda']['use_cuda']:
                self.device = torch.device("cuda:0")
                if params['cuda']['multi_gpu']:    
                    if torch.cuda.device_count() > 1:
                        print(f" Using {torch.cuda.device_count()} GPUs")
                        self.network = nn.DataParallel(self.network)
                    else:
                        print(f"Using 1 GPU")
                else:
                    print(f"Using 1 GPU")
            else:
                self.device = torch.device("cpu")
                print(f"Using CPU")

            ## send to CPU/GPU ##
            self.network.to(self.device)

            self.complex_loss = False
            ## set loss function ##
            if params['loss_fxn'] == 'WeightedMSE':
                self.loss_fxn = WeightedMSE()
            elif params['loss_fxn'] == 'RelativeMSE':
                self.loss_fxn = RelativeMSE()
            elif params['loss_fxn'] == 'MSELoss':
                self.loss_fxn = nn.MSELoss()
            elif params['loss_fxn'] == 'PhysicsLoss':
                self.loss_fxn = NLTECompositeLoss(
                    chi=params['chi'],
                    lines=params['lines'],
                    wave=params['wave'],
                    levels=params['levels'],
                    data_loss_func=WeightedMSE(),
                    atom_names=params['atom_names'],
                    debug=False
                )
                self.complex_loss = True
            else:
                raise Exception("!!Invalid loss function: {}!!".format(params['loss_fxn']))

            self.loss_fxn.to(self.device)

            if self.complex_loss is True:
                self.loss_fxn.data_loss.to(self.device)

            ## set optimizer ##
            if params['optimizer'] == 'Adam':
                self.optimizer = torch.optim.Adam(self.network.parameters(), lr=params['learn_rate'])
            else:
                raise Exception("!!Invalid optimizer!!")
        
        ## plotting mode, just a forward pass
        elif params['mode'] == 'testing':
            
            ######################### SUNNYNET ARCHITECTURES ############################ 
            # elif params['model'] == 'SunnyNet_1x1':
            #     self.network = SunnyNet_1x1(params['in_channels'], params['out_channels'], params['features'],1,1)
                
            # elif params['model'] == 'SunnyNet_3x3':
            #     self.network = SunnyNet_3x3(params['in_channels'], params['out_channels'], params['features'],3,3)
                
            # elif params['model'] == 'SunnyNet_5x5':
            #     self.network = SunnyNet_5x5(params['in_channels'], params['out_channels'], params['features'],5,5)
                
            # elif params['model'] == 'SunnyNet_7x7':
            #     self.network = SunnyNet_7x7(params['in_channels'], params['out_channels'], params['features'],7,7)
            ######################### SUNNYNET ARCHITECTURES ############################
                
            self.network = SunnyNet(params['in_channels'], params['out_channels'], params['features'], params['window_size'])
                           
            ## set CPU/GPU ##
            if params['cuda']:
                self.device = "cuda"
            else:
                self.device = "cpu"
            
            self.complex_loss = False
            ## set loss function ##
            if params['loss_fxn'] == 'WeightedMSE':
                self.loss_fxn = WeightedMSE()
            elif params['loss_fxn'] == 'RelativeMSE':
                self.loss_fxn = RelativeMSE()
            elif params['loss_fxn'] == 'MSELoss':
                self.loss_fxn = nn.MSELoss()
            elif params['loss_fxn'] == 'PhysicsLoss':
                self.loss_fxn = NLTECompositeLoss(
                    chi=params['chi'],
                    lines=params['lines'],
                    wave=params['wave'],
                    levels=params['levels'],
                    data_loss_func=WeightedMSE(),
                    atom_names=params['atom_names'],
                    debug=False
                )
                self.complex_loss = True
            else:
                raise Exception("!!Invalid loss function: {}!!".format(params['loss_fxn'])) 

            self.loss_fxn.to(self.device)

            if self.complex_loss is True:
                self.loss_fxn.data_loss.to(self.device)
      
        else:
            raise Exception("!! Invalid model mode")
        
        return