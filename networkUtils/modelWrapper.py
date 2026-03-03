import torch
import torch.nn as nn
from networkUtils.modelArchitectures import *
from networkUtils.context_to_column_3D import ContextToColumn3D
from networkUtils.lossFunctions import *
from networkUtils.physicsloss import *

class Model():
    def __init__(self, params):

        ## plotting only needs 1 forward pass so this gets skipped ##
        if params['mode'] == 'training':

            if params['model'] == 'SunnyNet':
                self.network = SunnyNet(
                    params['in_channels'],
                    params['out_channels'],
                    params['features'],
                    params['window_size']
                )
            elif params['model'] == 'ContextToColumn3D':
                self.network = ContextToColumn3D(
                    params['in_channels'],
                    params['out_channels'],
                    params['features'],
                    params['window_size']
                )
                print ("Using ContextToColumn3D model")
            else:
                raise Exception(
                    "!!Invalid Model: {}!!".format(
                        params['model']
                        )
                    )

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
                
            if params['model'] == 'SunnyNet':
                self.network = SunnyNet(
                    params['in_channels'],
                    params['out_channels'],
                    params['features'],
                    params['window_size']
                )
            elif params['model'] == 'ContextToColumn3D':
                self.network = ContextToColumn3D(
                    params['in_channels'],
                    params['out_channels'],
                    params['features'],
                    params['window_size']
                )
                print ("Using ContextToColumn3D model")
            else:
                raise Exception(
                    "!!Invalid Model: {}!!".format(
                        params['model']
                        )
                    )
                           
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