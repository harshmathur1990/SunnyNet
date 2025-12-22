# SunnyNet Utilities. Makes use of networkUtils for the heavy lifting.

import numpy as np
import h5py
import os
import sys
import torch
from torch.utils.data import DataLoader
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid as cumtrapz
from networkUtils.atmosphereFunctions import predict_populations
from networkUtils.modelWrapper import Model
from networkUtils.dataSets import PopulationDataset3d
from networkUtils.trainingFunctions import train


def interpolate_everything(rho_super_arr, z_scale, pops_super_array, new_cmass_scale):

    def iterpolate_data(rho_arr, z_scale, pops_array, new_cmass_scale):
        cmass_arr = cumtrapz(rho_arr, -z_scale, initial=0)
        interp_func = interp1d(
            cmass_arr, pops_array,
            kind='linear', axis=0, fill_value='extrapolate'
        )
        return interp_func(new_cmass_scale)

    vec_iterpolate_data = np.vectorize(iterpolate_data, signature='(n),(n),(n,o),(m)->(m,o)')

    return vec_iterpolate_data(rho_super_arr, z_scale, pops_super_array, new_cmass_scale)


def build_solving_set(
    rho, z_scale, temp, vx,
    vy, vz, ne,
    save_path="example.hdf5",
    ndep=400, pad=1
):
    """
    Prepares populations from 3D simulation into a file to be fed into a trained network
    to make population predictions.

    Parameters
    ----------
    lte_pops : 4D array
        Array with LTE populations. Shape should be (nx, ny, nz, nlevels),
        units in m^-3.
    rho_mean : 1D array
        Array with horizontally-averaged mass density. Shape should be (nz,),
        units in kg m^-3.
    z_scale : 1D array
        Height scale in m. First point should be top of atmosphere.
    save_path : str
        Name of file where the output will be written to.
    ndep : int, optional
        Number of output height points, to which the populations will be 
        interpolated on a mean column mass scale. Does not need to be the
        same as the input nz. Default is 400.
    pad : int, optional
        How many pixels to pad the each population column of interest in the
        x and y dimensions. Should be consistent with the window size:
        window size is 1 + 2*pad, so use pad=0 for 1x1, pad=1 for 3x3, 
        and so on. Default is 1.
    """
    # check save path validity
    if os.path.isfile(save_path):
        raise IOError("Output file already exists. Refusing to overwrite.")
    # check sim dims and see whats gonna happen with interpolation
    nx, ny, nz = temp.shape
    print(f'Sim shape: ({nx}, {ny}, {nz})')
    assert nx == ny, "Resizing function needs X / Y to be equal in length"
    grid = nx + 2*pad  # account for padding periodic BC's
    npad = ((pad,pad),(pad,pad),(0,0),(0,0))
    new_cmass_scale = np.logspace(-6, 2, ndep)
    temp_in = np.log10(temp)
    vx_in = vx / 100 # divide by 100 km/s
    vy_in = vy / 100 # divide by 100 km/s
    vz_in = vy / 100 # divide by 100 km/s
    ne_in = np.log10(ne)
    merged_input = np.stack(
        [
            temp_in,
            vx_in,
            vy_in,
            vz_in,
            ne_in
        ],
        axis=-1
    )
    merged_input = interpolate_everything(rho, z_scale, merged_input, new_cmass_scale)
    print('Padding for periodic boundary conditions...')
    lte = np.pad(merged_input, pad_width=npad, mode='wrap')
    print(f'LTE shape after padding: {lte.shape}')
    print('Rearranging and taking Log10...')
    lte = np.transpose(lte, (3,2,0,1))
    print('Splitting into windows and columns...')
    lte_list = []
    for i in range(pad, grid - pad):
        for j in range(pad, grid - pad):
            sample = lte[:, :, i-pad:i+(pad+1), j-pad:j+(pad+1)]
            lte_list.append(sample)
    lte = np.array(lte_list)
    print(f'Output shape {lte.shape}')
    print(f'Saving into {save_path}...')
    with h5py.File(save_path, 'w') as f:
        dset1 = f.create_dataset("lte test windows", data=lte, dtype='f')


def build_training_set(
    temp_list, vx_list, vy_list, vz_list,
    ne_list, lte_pops_list, nlte_pops_list,
    rho_list, z_scale_list,
    save_path="example.hdf5", ndep=400,
    pad=1, tr_percent=85
):
    """
    Prepares populations from 3D simulation into a file to be fed into a network
    for training and validation.

    Parameters
    ----------
    lte_pops : list or array_like
        List of 4D arrays with LTE populations to use for training. Each item
        in list could be populations from a different snapshot and/or simulation.
        List should have at least one element. The shape of each array should be
        (nx, ny, nz, nlevels), units in m^-3.
    nlte_pops : list or array_like
        List of 4D arrays with LTE populations to use for training. Same shape
        and units as lte_pops.
    rho_mean : list or array_like
        List with 1D arrays of spatially averaged mass density. Each item
        in list could be average density from a different snapshot and/or simulation.
        List should have at least one element. The shape of each array should be
        (nz,), units in kg m^-3.
    z_scale : list or array like.
        List with 1D arrays of height. Each item in list could be height from a 
        different snapshot and/or simulation. List should have at least one element.
        The shape of each array should be (nz,), units in m. First point 
        should be top of atmosphere.
    save_path : str
        Name of file where the output will be written to.
    ndep : int, optional
        Number of output height points, to which the populations will be 
        interpolated on a mean column mass scale. Does not need to be the
        same as the input nz. Default is 400.
    pad : int, optional
        How many pixels to pad the each population column of interest in the
        x and y dimensions. Should be consistent with the window size:
        window size is 1 + 2*pad, so use pad=0 for 1x1, pad=1 for 3x3, 
        and so on. Default is 1.
    tr_percent : int, optional
        Percent of data to be used as a training set (the rest will be used
        for validation).
    """

    nx, ny, _, _ = lte_pops_list[0].shape
    k = nx * ny  # number of training/validation instances combined (helps control output file size)
    grid = nx + 2*pad  # accounts for expanding to include periodic BC's
    npad = ((pad,pad),(pad,pad),(0,0),(0,0))
    # check save path validity
    if os.path.isfile(save_path):
        raise IOError("Output file already exists. Refusing to overwrite.")
    lte_list = []
    non_lte_list = []

    for temp_in, vx_in, vy_in, vz_in, ne_in, lte_in, nlte_in, rho_in, z_in in zip(
        temp_list, vx_list, vy_list, vz_list,
        ne_list, lte_pops_list, nlte_pops_list,
        rho_list, z_scale_list
    ):
        print(f'rho shape {rho_in.shape}')
        print(f'z shape {z_in.shape}')
        new_cmass_scale = np.logspace(-6, 2, ndep)
        temp = np.log10(temp_in)
        vx = vx_in / 100 # divide by 100 km/s
        vy = vy_in / 100 # divide by 100 km/s
        vz = vz_in / 100 # divide by 100 km/s
        ne = np.log10(ne_in)
        merged_input = np.stack(
            [
                temp,
                vx,
                vy,
                vz,
                ne
            ],
            axis=-1
        )
        departure_coeff = nlte_in / lte_in
        departure_coeff = np.log10(departure_coeff)
        merged_input = interpolate_everything(rho_in, z_in, merged_input, new_cmass_scale)
        departure_coeff = interpolate_everything(rho_in, z_in, departure_coeff, new_cmass_scale)
        print('Padding data...')
        merged_input = np.pad(merged_input, pad_width=npad, mode='wrap')
        departure_coeff = np.pad(departure_coeff, pad_width=npad, mode='wrap')
        print('Log and transpose...')
        merged_input = np.transpose(merged_input, (3,2,0,1))
        departure_coeff = np.transpose(departure_coeff, (3,2,0,1))
        print('Scaling data...')
        print(f"Splitting simulation into corresponding window / columns...")
        for i in range(pad, grid-pad):
            for j in range(pad, grid-pad):
                # lte window
                sample = merged_input[:, :, i-pad:i+(pad+1), j-pad:j+(pad+1)]
                lte_list.append(sample)
                # non lte in middle of window
                true = departure_coeff[:, :, i, j][:, :, np.newaxis, np.newaxis]
                non_lte_list.append(true)
    print(f"Train / Test Split...")
    # Get train/test indicies
    full_idx = np.arange(len(lte_list))
    tr = int(k * tr_percent/100)
    idx = np.random.choice(full_idx, size=k, replace=False)
    tr_idx = np.random.choice(idx, size=tr, replace=False)
    val_idx = np.setxor1d(tr_idx, idx)
    lte = np.array(lte_list)
    non_lte = np.array(non_lte_list)
    lmean = lte.mean(axis=(0, 2, 3, 4), keepdims=True)
    lstd  = lte.std(axis=(0, 2, 3, 4), keepdims=True)
    l_norm = (lte - lmean) / lstd
    nlmean = non_lte.mean(axis=(0, 2, 3, 4), keepdims=True)
    nlstd  = non_lte.std(axis=(0, 2, 3, 4), keepdims=True)
    nl_norm = (non_lte - nlmean) / nlstd
    lte = l_norm
    non_lte = nl_norm
    # split sets
    lte_train = lte[tr_idx]
    non_lte_train = non_lte[tr_idx]
    train_len = len(lte_train)
    lte_test = lte[val_idx]
    non_lte_test = non_lte[val_idx]
    test_len = len(lte_test)
    print(f'Input shape {lte_train.shape}')
    print(f'Output shape {non_lte_train.shape}')
    print(f'Saving into {save_path}...')
    with h5py.File(save_path, 'w') as f:
        dset1 = f.create_dataset("lte training windows", data=lte_train, dtype='f')
        dset1.attrs["mu"] = lmean
        dset1.attrs["std"] = lstd
        dset1.attrs["len"] = train_len
        dset2 = f.create_dataset("non lte training points", data=non_lte_train, dtype='f')
        dset2.attrs["mu"] = nlmean
        dset2.attrs["std"] = nlstd
        dset3 = f.create_dataset("lte test windows", data=lte_test, dtype='f')
        dset3.attrs["mu"] = lmean
        dset3.attrs["std"] = lstd
        dset3.attrs["len"] = test_len
        dset4 = f.create_dataset("non lte test points", data=non_lte_test, dtype='f')
        dset4.attrs["mu"] = nlmean
        dset4.attrs["std"] = nlstd


def read_train_params(train_file):
    """
    Reads the parameters training size, testing size, channels, pad, 
    and ndep from existing SunnyNet training HDF5 file.
    """
    tmp = h5py.File(train_file, 'r')
    in_buf = tmp['lte training windows'].shape
    out_buf = tmp['non lte training points'].shape
    train_size = in_buf[0]
    test_size = tmp['lte test windows'].shape[0]
    in_channels = in_buf[1]
    out_channels = out_buf[1]
    ndep = in_buf[2]
    pad = (in_buf[-1] - 1) // 2
    tmp.close()
    return train_size, test_size, in_channels, out_channels, ndep, pad


def read_solve_params(solve_file):
    """
    Reads the atmospheric dimensions from existing SunnyNet "solving" HDF5 file.
    """
    tmp = h5py.File(solve_file, 'r')
    in_buf = tmp['lte test windows'].shape
    nx = int(np.sqrt(in_buf[0]))
    in_channels = in_buf[1]
    out_channels = 6
    ndep = in_buf[2]
    pad = (in_buf[-1] - 1) // 2
    return nx, in_channels, ndep, pad
    

def check_model_compat(model_type, pad):
    if model_type == 'SunnyNet_1x1':
        m_pad = 0
    elif model_type == 'SunnyNet_3x3':
        m_pad = 1
    elif model_type == 'SunnyNet_5x5':
        m_pad = 2
    elif model_type == 'SunnyNet_7x7':
        m_pad = 3
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    if m_pad == pad:
        return True
    else:
        return False


def sunnynet_train_model(train_path, save_folder, save_file, model_type='SunnyNet_3x3',
                         loss_function='MSELoss', alpha=1e-3, cuda=True):
    """
    Trains a SunnyNet neural network model to be used to predict non-LTE populations.
    Needs a "train" file prepared with build_training_set(). Common options can
    be entered as keywords. More advanced options can be edited on the dictionary
    'config' below.

    Parameters
    ----------
    train_path : str
        Filename of saved training data, obtained after running 
        build_training_set() from populations from a 3D atmosphere. The 
        format is HDF5.
    save_folder : str
        Folder where to place the output files.
    save_file : str
        Name of output file where to save the file. Usually has .pt extension.
    model_type : str, optional
        Type of network to use. The available types are the names of 
        SunnyNet_* classes in networkUtilities/modelArchitectures.py.
        Currently supported networks are:
        - 'SunnyNet_1x1' : 6 levels, 400 depth points, 1x1 window size
        - 'SunnyNet_3x3' : 6 levels, 400 depth points, 3x3 window size
        - 'SunnyNet_5x5' : 6 levels, 400 depth points, 5x5 window size
        - 'SunnyNet_7x7' : 6 levels, 400 depth points, 7x7 window size
        Should be consistent with choice of channels and ndep.
    loss_function : str, optional
        Type of loss function to use. Could be a class name of pytorch
        loss functions (e.g. 'MSELoss', the default), or a class name 
        from networkUtils/lossFunctions.py. 
    alpha : float or None, optional
        Weight in loss calculation between mass conservation and cell by
        cell error. Default is 0.2. To switch off entirely set to None.
    cuda : bool, optional
        Whether to use GPU acceleration through CUDA (default True).
    """
    if os.path.exists(os.path.join(save_folder, save_file)):
        raise IOError("Output file already exists, refusing to overwrite.")
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)
    train_size, test_size, in_channels, out_channels, ndep, pad = read_train_params(train_path)
    if not check_model_compat(model_type, pad):
        raise ValueError(f"Incompatible sizes between model {model_type} "
                         f"and training set (pad={pad})")
    params = {
        'model': model_type, # pick one from networkUtilities/modelArchitectures.py
        # only works with Adam right now, 
        # can add others from torch.optim to networkUtils/modelWrapper.py:
        'optimizer': 'Adam',  
        'loss_fxn': loss_function, 
        'learn_rate': 1e-3,
        'in_channels': in_channels,
        'out_channels': out_channels,
        'features': ndep,
        'cuda': {'use_cuda': cuda, 'multi_gpu': False},
        'mode': 'training'
    }
    # training configuration 
    config = {
        'data_path': train_path,
        'save_folder': save_folder,
        'model_save': save_file,
        'early_stopping': 5, # iterations without lower loss before breaking training loop
        'num_epochs': 50,    # training epochs
        'train_size': train_size, # manually calculate from your train / test split
        'batch_size_train': 128,
        'val_size': test_size,    # manually calculate from your train / test split
        'batch_size_val': 128,
        'num_workers': 64,   # CPU threads
        'alpha': alpha    # weight in loss calc. between mass conservation and cell by cell error
    }
    print('Python VERSION:', sys.version)
    print('pyTorch VERSION:', torch.__version__)
    print('CUDA VERSION: ', torch.version.cuda)
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print('GPU name:', torch.cuda.get_device_name())
        print(f'Number of GPUS: {torch.cuda.device_count()}')
    print(f"Using {params['model']} architecture...")
    print('Creating dataset...')
    tr_data = PopulationDataset3d(config['data_path'], train=True)
    val_data = PopulationDataset3d(config['data_path'], train=False)
    print('Creating data loaders...')
    loader_dict = {}
    train_loader = DataLoader(
        tr_data,
        batch_size = config['batch_size_train'],
        pin_memory = True,
        num_workers = config['num_workers']
    )
    val_loader = DataLoader(
        val_data,
        batch_size = config['batch_size_val'],
        pin_memory = True,
        num_workers = config['num_workers']
    )
    loader_dict['train'] = train_loader
    loader_dict['val'] = val_loader
    h_model = Model(params)
    epoch_loss = train(config, h_model, loader_dict)
    # save epoch losses for plotting
    with open(f"{config['save_folder']}{config['model_save'][:-3]}_loss.txt", "w") as f:
        for i in range(len(epoch_loss['train'])):
            f.write(str(epoch_loss['train'][i]) + '   ' + str(epoch_loss['val'][i]) + '\n')


def sunnynet_predict_populations(model_path, train_path, test_path, save_path, 
                                 cuda=True, model_type='SunnyNet_3x3', loss_function='MSELoss',
                                 alpha=1e-3):
    """
    Predicts non-LTE populations using SunnyNet, using an existing trained set,
    model data, and input LTE populations pre-prepared with build_solving_set()

    Parameters
    ----------
    model_path : str
        Filename of saved neural network model, obtained after running
        train_model_3d(). Typical extension is .pt.
    train_path : str
        Filename of saved training data, obtained after running 
        build_training_set() from populations from a 3D atmosphere. The 
        format is HDF5.
    test_path : str
        Filename with pre-prepared data from the input LTE populations
        for which to predict the NLTE populations. This file should be
        obtained by running build_solving_set(). The format is HDF5. 
    save_path : str
        Filename of output file to save predicted populations. The format
        is HDF5.
    cuda : bool, optional
        Whether to use GPU acceleration through CUDA (default True).
    model_type : str, optional
        Type of network to use. The available types are the names of 
        SunnyNet_* classes in networkUtilities/modelArchitectures.py.
        Currently supported networks are:
        - 'SunnyNet_1x1' : 6 levels, 400 depth points, 1x1 window size
        - 'SunnyNet_3x3' : 6 levels, 400 depth points, 3x3 window size
        - 'SunnyNet_5x5' : 6 levels, 400 depth points, 5x5 window size
        - 'SunnyNet_7x7' : 6 levels, 400 depth points, 7x7 window size
        Should be consistent with choice of channels and ndep.
    loss_function : str, optional
        Type of loss function to use. Could be a class name of pytorch
        loss functions (e.g. 'MSELoss', the default), or a class name 
        from networkUtils/lossFunctions.py. 
    alpha : float or None, optional
        Weight in loss calculation between mass conservation and cell by
        cell error. Default is 0.2. To switch off entirely set to None.
    """
    _, _, in_channels, out_channels, ndep, pad = read_train_params(train_path)
    nx, in_channels1, ndep1, npad1 = read_solve_params(test_path)
    assert in_channels == in_channels1, "Inconsistent number of input channels between train and test data"
    assert ndep == ndep1, "Inconsistent number of depth points between train and test data"
    assert pad == npad1, "Inconsistent padding number between train and test data"
    if not check_model_compat(model_type, pad):
        raise ValueError(f"Incompatible sizes between model {model_type} "
                         f"and training set (pad={pad}, channels={channels})")    
    if os.path.isfile(save_path):
        raise IOError("Output file already exists. Refusing to overwrite.")
    pred_config = {
        'cuda': cuda,      
        'model': model_type,
        'model_path': model_path,
        'in_channels': in_channels,   # number of atomic levels
        'out_channels': out_channels,
        'features': ndep,       # z dimension
        'mode': 'testing',
        'multi_gpu_train': False,
        'loss_fxn': loss_function,
        'alpha': alpha,   # weight in loss calc. between mass conservation and cell by cell error
        'output_XY': nx,  # number of pixels in horizontal dimensions
    }
    populations, cmass_scale = predict_populations(test_path, train_path, pred_config)
    print('Exponentiate')
    populations = 10**populations
    print(f'Atmos shape: {populations.shape}')
    with h5py.File(save_path, 'w') as f:
        dset = f.create_dataset("populations", data=populations, dtype='f')
        dset.attrs['cmass_scale'] = cmass_scale