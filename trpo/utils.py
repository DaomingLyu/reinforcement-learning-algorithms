import torch
import numpy as np

"""
This module will contain the optimization algorithms

that will be used for training the trpo agent

1. Conjugated Gradient Algorighm
2. Line Search Algorithm

"""

# the conjugated gradient algorithms...
# https://github.com/joschu/modular_rl/blob/master/modular_rl/trpo.py
def conjugated_gradient(fvp, b, update_steps, state_batch_tensor, action_mean_old, action_std_old, \
                                                                                        residual_tol=1e-10):
    # assume the inital guess of the x is zero...
    x = torch.zeros(b.size())
    # due to x is 0 in the inital, the resulting r = b - Ax = b(which is surrogate_grad)
    r = b.clone()
    p = b.clone()  # here p = r...
    rdotr = torch.dot(r, r)
    for i in range(update_steps):
        fv_product = fvp(p, state_batch_tensor, action_mean_old, action_std_old)
        alpha = rdotr / torch.dot(p, fv_product)
        x = x + alpha * p
        r = r - alpha * fv_product
        new_rdotr = torch.dot(r, r)
        beta = new_rdotr / rdotr
        p = r + beta * p
        rdotr = new_rdotr

        if rdotr < residual_tol:
            break

    return x

# then the line search 
def line_search(model, loss_function, x, full_step, expected_improve_rate, \
                        state_batch_tensor, advantages, action_batch_tensor, old_action_prob, \
                            max_backtracks=10, accept_ratio=0.1):
    
    fval = loss_function(state_batch_tensor, advantages, action_batch_tensor, old_action_prob).data
    
    for (_n_backtracks, stepfrac) in enumerate(0.5**np.arange(max_backtracks)):
        xnew = x + stepfrac * full_step
        set_flat_params_to(model, xnew)
        new_fval = loss_function(state_batch_tensor, advantages, action_batch_tensor, old_action_prob).data
        actual_improve = fval - new_fval
        expected_improve = expected_improve_rate * stepfrac
        ratio = actual_improve / expected_improve
        if ratio[0] > accept_ratio and actual_improve[0] > 0:
            return True, xnew
    
    return False, x


# set the flat params back to the models...
# https://github.com/ikostrikov/pytorch-trpo/blob/master/utils.py 
def set_flat_params_to(model, flat_params):
    prev_indx = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(flat_params[prev_indx:prev_indx + flat_size].view(param.size()))
        prev_indx += flat_size

