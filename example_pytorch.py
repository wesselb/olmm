# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
import numpy as np
import torch
import wbml.out
import wbml.plot
import stheno.torch
import lab.torch as B
from varz.torch import Vars, minimise_l_bfgs_b

from olmm import model, objective, predict, project
from data import load

# Load the data, which are Pandas data frames.
locs, data = load()

# Convert to NumPy.
locs = locs.to_numpy()
x_data = data.index.to_numpy()[:, None]
y_data = data.to_numpy()

# Inputs for two-months ahead predictions.
x_pred = np.arange(1, x_data.max() + 60, dtype=np.float64)[:, None]

# Normalise data.
data_mean = np.mean(y_data, axis=0, keepdims=True)
data_scale = np.std(y_data, axis=0, keepdims=True)
y_data_norm = (y_data - data_scale) / data_mean

# Convert to PyTorch.
locs = torch.tensor(locs)
x_pred = torch.tensor(x_pred)
x_data = torch.tensor(x_data)
y_data_norm = torch.tensor(y_data_norm)

# Model parameters:
n = data.shape[0]  # Number of data points
p = data.shape[1]  # Number of outputs
m = 10  # Number of latent processes

# Learn.
vs = Vars(torch.float64)
minimise_l_bfgs_b(lambda vs_: objective(vs_, m, x_data, y_data_norm, locs),
                  vs=vs,
                  trace=True,
                  iters=200)
wbml.out.kv('Learned spatial scales', vs['scales'])

# Predict.
lat_preds, obs_preds = predict(vs, m, x_data, y_data_norm, locs, x_pred)

# Convert to NumPy and undo normalisation.
obs_preds = [tuple(x * data_mean[0, i] + data_scale[0, i]
                   for x in B.to_numpy(tup))
             for i, tup in enumerate(obs_preds)]

# Plot first four latent processes.
plt.figure(figsize=(15, 5))
y_proj, _, S, _ = B.to_numpy(project(vs, m, y_data_norm, locs))
xs, _, _ = model(vs, m)
for i in range(4):
    plt.subplot(2, 2, i + 1)
    mean, lower, upper = lat_preds[i]
    plt.title(f'Latent Process {i + 1} (${100 * S[i] / np.sum(S):.1f}\\%$) \n'
              f'{xs[i].display(wbml.out.format)}')
    plt.plot(x_data, y_proj[i], c='tab:blue')
    plt.plot(x_pred, mean, c='tab:green')
    plt.plot(x_pred, lower, c='tab:green', ls='--')
    plt.plot(x_pred, upper, c='tab:green', ls='--')
    wbml.plot.tweak(legend=False)

# Plot four random outputs.
plt.figure(figsize=(10, 5))
for i, j in enumerate(sorted(np.random.permutation(p)[:4])):
    plt.subplot(2, 2, i + 1)
    mean, lower, upper = obs_preds[j]
    plt.title(data.columns[j])
    plt.plot(x_data, y_data[:, j], c='tab:blue')
    plt.plot(x_pred, mean, c='tab:green')
    plt.plot(x_pred, lower, c='tab:green', ls='--')
    plt.plot(x_pred, upper, c='tab:green', ls='--')
    wbml.plot.tweak(legend=False)

plt.show()
