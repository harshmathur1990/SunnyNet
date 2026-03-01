import os
import torch
import numpy as np
import math
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
        tr_loss, loss_data_running, loss_source_running, loss_source_atoms_running = run_epoch('train', model, epoch, dataLoaders)
        loss_dict['train'].append(tr_loss)
        ## eval forward ##
        with torch.no_grad():
            model.network.eval()
            val_loss, _, _, _ = run_epoch('val', model, epoch, dataLoaders)
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


def update_lambda(
    criterion,
    reg_avg,
    data_avg,
    target=(0.1, 1.0),
    eta=0.3,
    lam_min=1e-12,
    lam_max=1e6,
):
    """
    Log-space feedback controller for regularization weight.
    Call once per epoch.
    """
    ratio = reg_avg / (data_avg + 1e-12)

    # Target ratio = geometric mean (center in log-space)
    r_target = math.sqrt(target[0] * target[1])

    # Log error
    err = math.log(r_target) - math.log(ratio + 1e-12)

    # Multiplicative update
    criterion.lam *= math.exp(eta * err)

    # Safety clamps
    criterion.lam = max(lam_min, min(lam_max, criterion.lam))


def run_epoch(mode, model, cur_epoch, dataLoaders, verbose = True):
    '''
    Runs epoch given the params in train()
    '''
           
    epoch_loss = 0.0
    loss_data_running = 0.0
    loss_source_running = 0.0
    loss_source_atoms_running = None   # will become tensor after first batch

    if verbose:
        print('-'*10, f'Epoch {cur_epoch}: {mode}', '-'*10)

    # if model.complex_loss is True:
    #     desc = f"{mode.upper()} | Epoch {cur_epoch} | λ = {model.loss_fxn.lam:.2e}"
    # else:
    desc = f"{mode.upper()} | Epoch {cur_epoch}"

    pbar = tqdm(
        dataLoaders[mode],
        desc=desc,
        leave=True,
        ncols=150
    )

    for i, instance in enumerate(pbar):
    
        X = instance[0].to(model.device, non_blocking=True)   # LTE
        y_true = instance[1].to(model.device, non_blocking=True)  # non-LTE

        # ------------ FORWARD -------------- #
        y_pred = model.network(X)
        
        if model.complex_loss is True:
            T = extract_temperature(X)
            
            y_pred_squeezed = y_pred.squeeze(-1).squeeze(-1)
            y_true_squeezed = y_true.squeeze(-1).squeeze(-1)
            
            batch_loss, components = model.loss_fxn(
                T,
                y_pred_squeezed,
                y_true_squeezed
            )

            # ---- accumulate main components ----
            loss_data_running += components["data"].item()
            loss_source_running += components["source"].item()

            # ---- accumulate per-atom losses ----
            atom_losses = components["source_per_atom"].detach().cpu()

            if loss_source_atoms_running is None:
                loss_source_atoms_running = torch.zeros_like(atom_losses)

            loss_source_atoms_running += atom_losses

        else:
            batch_loss = model.loss_fxn(y_pred, y_true)

        # ------------ BACKWARD ------------ #
        if mode == 'train':
            model.optimizer.zero_grad()
            batch_loss.backward()
            model.optimizer.step()

        epoch_loss += batch_loss.item()

        # update tqdm postfix (clean, no extra lines)
        if model.complex_loss is True:
            postfix = {
                "L":  f"{epoch_loss/(i+1):.3e}",
                "Ld": f"{loss_data_running/(i+1):.3e}",
                "Ls": f"{loss_source_running/(i+1):.3e}",
            }

            # add atom losses dynamically
            if loss_source_atoms_running is not None:
                avg_atoms = loss_source_atoms_running / (i+1)
                atom_names = components.get("atom_names", None)

                if atom_names is None:
                    atom_names = [f"A{a}" for a in range(len(avg_atoms))]

                for a, name in enumerate(atom_names):
                    postfix[name] = f"{avg_atoms[a]:.2e}"

            pbar.set_postfix(postfix)

        else:
            pbar.set_postfix(loss=f"{epoch_loss / (i + 1):.3e}")

    n_batches = len(dataLoaders[mode])

    epoch_loss /= len(dataLoaders[mode])

    if model.complex_loss:
        loss_data_running /= n_batches
        loss_source_running /= n_batches
        loss_source_atoms_running /= n_batches

    print(f"TOTAL {mode.upper()} LOSS = {epoch_loss:.8f}")

    if model.complex_loss:
        return (
            epoch_loss,
            loss_data_running,
            loss_source_running,
            loss_source_atoms_running
        )
    else:
        return epoch_loss, None, None, None

