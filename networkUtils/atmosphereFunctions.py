import os
import numpy
import torch
import h5py
import json
from collections import OrderedDict
from networkUtils.dataSets import PopulationDataset3d
from networkUtils.modelWrapper import Model
from torch.utils.data import DataLoader

def predict_populations(pop_path, train_data_path, config):

    train_data = PopulationDataset3d(train_data_path, train = False)

    ##### LOAD MODEL #####
    model = Model(config)
    print(f'Loading model...')
    state_dict = torch.load(config['model_path'], map_location='cpu')

    # Detect if keys are prefixed with "module."
    has_module_prefix = any(k.startswith("module.") for k in state_dict.keys())

    if has_module_prefix:
        new_state_dict = OrderedDict(
            (k.replace("module.", "", 1), v) for k, v in state_dict.items()
        )
    else:
        new_state_dict = state_dict

    model.network.load_state_dict(new_state_dict)

    if config['cuda']: 
        model.network.to('cuda')
    else:
        model.network.to('cpu')
    model.network.eval()
    
    if hasattr(model.network, "enable_diagnostics"):
        model.network.enable_diagnostics()

    with h5py.File(pop_path, 'r') as f:
        lte = f['lte test windows'][:]

    non_lte = numpy.zeros_like(lte)  # placeholder only
    
    data = [list(a) for a in zip(lte,non_lte)]
    
    print(f'Forward pass of data throught model...')
    loader = DataLoader(data, batch_size=256, pin_memory=True, num_workers=8)
    pred_list = []
    for point in loader:
        with torch.no_grad():
            model.network.eval()
            X = point[0].to(model.device, torch.float, non_blocking=True)
            y_pred = model.network(X)
            # Optional context sensitivity measurement
            if hasattr(model.network, "measure_context_sensitivity"):
                delta = model.network.measure_context_sensitivity(X)
                if hasattr(model.network, "diagnostics"):
                    model.network.diagnostics.add_scalar("context_sensitivity", delta)
            pred_list.append(y_pred)
    
    print(f'Fixing up dimensions...')
    pred_list = torch.cat(pred_list, dim = 0)

    # 🔥 Save prediction diagnostics
    if hasattr(model.network, "get_diagnostics"):
        stats = model.network.get_diagnostics()

        diag_path = os.path.join(os.path.dirname(pop_path), "predict_diagnostics.json")

        with open(diag_path, "w") as f:
            json.dump(stats, f, indent=2)

        print(f"Prediction diagnostics saved to {diag_path}")

    pred_final = pred_list.squeeze(3).squeeze(3).detach().cpu().numpy()
    pred_final = numpy.transpose(pred_final,(0,2,1))

    dim = config['output_XY']
    dimz = config['features']
    dimc = config['out_channels']
    pred_final = pred_final.reshape((dim, dim, dimz, dimc))
    
    return pred_final