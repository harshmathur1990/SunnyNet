import os
import torch
import numpy as np
from tqdm import tqdm

def train(params, model, dataLoaders):
    full_path = os.path.join(params['save_folder'], params['model_save'])
    loss_dict = {'train':[], 'val':[]}
    no_improv = 0
    min_loss = np.inf
    stop = False
    
    for epoch in range(params['num_epochs']):
        ## train forwards ##
        model.network.train()
        tr_loss = run_epoch('train', model, epoch, dataLoaders)
        loss_dict['train'].append(tr_loss)
        ## eval forward ##
        with torch.no_grad():
            model.network.eval()
            val_loss = run_epoch('val', model, epoch, dataLoaders)
            loss_dict['val'].append(val_loss)
        
        ## check los ##
        if val_loss < min_loss:
            no_improv = 0
            min_loss = val_loss
            print(f'New min loss of {min_loss:.4f}, saving checkpoint...')
            torch.save(model.network.state_dict(), full_path)
        else:
            no_improv += 1
            if (epoch + 1 > params['early_stopping']) and no_improv == params['early_stopping']:
                stop = True
        if stop == True:
            print(f'Early stopping condition met, stopping at epoch {epoch}...')
            break
            
    return loss_dict


def extract_temperature(X):
    """
    X: (B, 5, Nz, k, k)
    returns T: (B, Nz)
    """
    k = X.shape[-1]
    center = k // 2

    logT = X[:, 0, :, center, center]   # (B, Nz)
    T = 10.0 ** logT                    # convert log10(T) → T

    return T


def run_epoch(mode, model, cur_epoch, dataLoaders, verbose = True):
    '''
    Runs epoch given the params in train()
    '''
    
    X_size = model.network.height
    if X_size == 1:
        k = 0
    elif X_size == 3:
        k = 1
    elif X_size == 5:
        k = 2
    elif X_size == 7:
        k = 3
    else:
        raise AttributeError('Currently only support models with square X/Y input dimmensions of: 1, 3, 5, 7')
           
    epoch_loss = 0
    loss1_running = 0.0
    loss2_running = 0.0

    if verbose:
        print('-'*10, f'Epoch {cur_epoch}: {mode}', '-'*10)

    pbar = tqdm(
        dataLoaders[mode],
        desc=f"{mode.upper()} | Epoch {cur_epoch}",
        leave=True,
        ncols=100
    )

    for i, instance in enumerate(pbar):
    
        X = instance[0].to(model.device, non_blocking=True)   # LTE
        y_true = instance[1].to(model.device, non_blocking=True)  # non-LTE

        # ------------ FORWARD -------------- #
        y_pred = model.network(X)
        if model.complex_loss is True:
            T = extract_temperature(X)
            
            loss_1 = model.loss_fxn1(y_pred, y_true)
            loss_2 = model.loss_fxn2(y_pred, T)

            batch_loss = loss_1 + loss_2

            loss1_running += loss_1.item()
            loss2_running += loss_2.item()

        else:
            batch_loss = model.loss_fxn(y_pred, y_true)

        # ------------ BACKWARD ------------ #
        if mode == 'train':
            model.optimizer.zero_grad()
            batch_loss.backward(retain_graph=True)
            model.optimizer.step()

        epoch_loss += batch_loss.item()

        # update tqdm postfix (clean, no extra lines)
        if model.complex_loss is True:
            pbar.set_postfix({
                "L": f"{epoch_loss / (i + 1):.3e}",
                "L1": f"{loss1_running / (i + 1):.3e}",
                "L2": f"{loss2_running / (i + 1):.3e}",
            })
        else:
            pbar.set_postfix(loss=f"{epoch_loss / (i + 1):.3e}")

    epoch_loss /= len(dataLoaders[mode])
    print(f"TOTAL {mode.upper()} LOSS = {epoch_loss:.8f}")

    return epoch_loss

