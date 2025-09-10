import math

import numpy as np
import torch
import torch.nn as nn


class WrapperBlock(nn.Module):
    def __init__(self, block, in_features, max_nodes, batch_size):
        """
        A wrapper module to reshape inputs/outputs for a graph block
        so the full graph Jacobian can be computed more easily.

        Args:
            block (nn.Module): The graph block to apply (e.g., a GNN layer).
            in_features (int): Number of input node features.
            max_nodes (int): Maximum number of nodes in a graph.
            batch_size (int): Batch size for input graphs.
        """
        super().__init__()
        self.block = block
        self.in_features = in_features
        self.max_nodes = max_nodes
        self.batch_size = batch_size

    def forward(self, x, edge_index, edge_weight, edge_attr):
        x = x.view(-1, self.in_features)
        out = self.block(x, edge_index, edge_weight, edge_attr)
        return out.view(self.batch_size, self.max_nodes, -1)


def resflow_logdet(self, x, edge_index=None, edge_weight=None, edge_attr=None):
    '''
        From Residual Flow
        Returns g(x) and logdet|d(x+g(x))/dx|
    '''
    x = x.requires_grad_(True)
    with torch.enable_grad():

        lamb = 2.
        def sample_fn(m): return poisson_sample(lamb, m)
        def rcdf_fn(k, offset): return poisson_1mcdf(lamb, k, offset)

        n_samples = sample_fn(1)
        n_exact_terms = 2
        n_power_series = max(n_samples) + n_exact_terms
        def coeff_fn(k): return 1 / rcdf_fn(k, n_exact_terms) * \
            sum(n_samples >= k - n_exact_terms) / len(n_samples)

        vareps = torch.randn_like(x)

        estimator_fn = neumann_logdet_estimator

        if edge_index is None:
            g = self.block(x, edge_index, edge_weight, edge_attr)
            g, logdetgrad = mem_eff_wrapper(
                estimator_fn, g, self.block, x, n_power_series, vareps, coeff_fn, True)
            return g, logdetgrad.view(-1, 1)
        else:
            batch_size = x.shape[0] // self.max_nodes 
            x = x.view(batch_size, self.max_nodes, -1)
            vareps = torch.randn_like(x)

            wrapper_block = WrapperBlock(
                self.block, self.in_features, self.max_nodes, batch_size).to(x.device)

            g = wrapper_block(x, edge_index, edge_weight, edge_attr)

            g, logdetgrad = mem_eff_wrapper(
                estimator_fn, g, wrapper_block, x, n_power_series, vareps, coeff_fn, True)
            g = g.view(-1, self.in_features)

        return g, logdetgrad.view(-1, 1)


def mem_eff_wrapper(estimator_fn, g, gnet, x, n_power_series, vareps, coeff_fn, training):
    if not isinstance(gnet, nn.Module):
        raise ValueError('g is required to be an instance of nn.Module.')

    return MemoryEfficientLogDetEstimator.apply(
        estimator_fn, g, x, n_power_series, vareps, coeff_fn, training,
        * list(gnet.parameters())
    )


class MemoryEfficientLogDetEstimator(torch.autograd.Function):

    @staticmethod
    def forward(ctx, estimator_fn, g, x, n_power_series, vareps, coeff_fn, training, *g_params):
        ctx.training = training
        with torch.enable_grad():
            ctx.g = g
            ctx.x = x
            logdetgrad = estimator_fn(
                g, x, n_power_series, vareps, coeff_fn, training)

            if training:
                grad_x, *grad_params = torch.autograd.grad(
                    logdetgrad.sum(), (x,) + g_params, retain_graph=True, allow_unused=True
                )
                if grad_x is None:
                    grad_x = torch.zeros_like(x)
                ctx.save_for_backward(grad_x, *g_params, *grad_params)

        return safe_detach(g), safe_detach(logdetgrad)

    @staticmethod
    def backward(ctx, grad_g, grad_logdetgrad):
        training = ctx.training
        if not training:
            raise ValueError('Provide training=True if using backward.')

        with torch.enable_grad():
            grad_x, *params_and_grad = ctx.saved_tensors
            g, x = ctx.g, ctx.x

            # Precomputed gradients.
            g_params = params_and_grad[:len(params_and_grad) // 2]
            grad_params = params_and_grad[len(params_and_grad) // 2:]

            dg_x, * \
                dg_params = torch.autograd.grad(
                    g, [x] + g_params, grad_g, allow_unused=True, retain_graph=True)

        # Update based on gradient from logdetgrad.
        dL = grad_logdetgrad[0].detach()
        with torch.no_grad():
            grad_x.mul_(dL)
            grad_params = tuple(
                [g.mul_(dL) if g is not None else None for g in grad_params])

        # Update based on gradient from g.
        with torch.no_grad():
            grad_x.add_(dg_x)
            grad_params = tuple([dg.add_(
                djac) if djac is not None else dg for dg, djac in zip(dg_params, grad_params)])

        return (None, None, grad_x, None, None, None, None) + grad_params


def neumann_logdet_estimator(g, x, n_power_series, vareps, coeff_fn, training):
    vjp = vareps
    neumann_vjp = vareps
    with torch.no_grad():
        for k in range(1, n_power_series + 1):
            vjp = torch.autograd.grad(g, x, vjp, retain_graph=True)[0]
            neumann_vjp = neumann_vjp + (-1)**k * coeff_fn(k) * vjp
    vjp_jac = torch.autograd.grad(g, x, neumann_vjp, create_graph=training)[0]
    logdetgrad = torch.sum(vjp_jac.contiguous().view(
        x.shape[0], -1) * vareps.contiguous().view(x.shape[0], -1), 1)
    return logdetgrad


def poisson_sample(lamb, n_samples):
    return np.random.poisson(lamb, n_samples)


def poisson_1mcdf(lamb, k, offset):
    if k <= offset:
        return 1.
    else:
        k = k - offset
    """P(n >= k)"""
    sum = 1.
    for i in range(1, k):
        sum += lamb**i / math.factorial(i)
    return 1 - np.exp(-lamb) * sum


def safe_detach(tensor):
    return tensor.detach().requires_grad_(tensor.requires_grad)
