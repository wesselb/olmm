# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import wbml.out
import wbml.plot
from lab.tensorflow import B
from stheno.tensorflow import (
    Graph,
    GP,
    EQ,
    Matern52,
    Delta,
    dense,
    Normal,
    Diagonal
)
from varz.tensorflow import Vars, minimise_l_bfgs_b

from data import load

locs, data = load()

# Initialise length scales to two wiggles along the latitude and longitude.
scales_init = (locs.max() - locs.min()).to_numpy() / 2

# Convert to NumPy.
locs = locs.to_numpy()
x_data = data.index.to_numpy()[:, None]
y_data = data.to_numpy()

# Predict two months ahead.
x_pred = np.arange(1, x_data.max() + 60, dtype=np.float64)[:, None]

# Normalise data.
data_mean = np.mean(y_data, axis=0, keepdims=True)
data_scale = np.std(y_data, axis=0, keepdims=True)
y_data = (y_data - data_scale) / data_mean

# Convert to TF.
locs = tf.constant(locs)
x_pred = tf.constant(x_pred)
x_data = tf.constant(x_data)
y_data = tf.constant(y_data)

# Model parameters:
n = data.shape[0]  # Number of data points
p = data.shape[1]  # Number of outputs
m = 20  # Number of latent processes


def project(vs, locs):
    """Create projection of data."""
    _, noise_obs, noises_latent = model(vs)

    # Compute mixing matrix and projection.
    scales = vs.bnd(scales_init, name='scales')
    K = dense(Matern52().stretch(scales)(locs))
    U, S, _ = B.svd(K)
    S = S[:m]
    H = U[:, :m] * S[None, :] ** .5
    T = B.transpose(U[:, :m]) / S[:, None] ** .5

    # Project data and unstack over latent processes.
    y_proj = B.unstack(B.matmul(T, y_data, tr_b=True))

    # Observation noises:
    noises_obs = noise_obs * B.ones(B.dtype(noise_obs), p)

    return y_proj, H, S, noises_obs


def model(vs):
    """Construct model."""
    g = Graph()

    # Observation noise:
    noise_obs = vs.bnd(0.2, name='noise_obs')

    def make_latent_process(i):
        # Long-term trend:
        variance = vs.bnd(1.0, name=f'{i}/long_term/var')
        scale = vs.bnd(6 * 30, lower=30, name=f'{i}/long_term/scale')
        k = variance * EQ().stretch(scale)

        # We could model more trends here.

        return GP(k, graph=g)

    # Latent processes:
    xs = [make_latent_process(i) for i in range(m)]

    # Latent noises:
    noises_latent = vs.bnd(0.2 * B.ones(m), name='noises_latent')

    return xs, noise_obs, noises_latent


def objective(vs):
    """NLML objective."""
    y_proj, _, S, noises_obs = project(vs, locs)
    xs, noise_obs, noises_latent = model(vs)

    # Add contribution of latent processes.
    lml = 0
    for i, (x, y) in enumerate(zip(xs, y_proj)):
        e = GP((noise_obs / S[i] + noises_latent[i]) * Delta(), graph=x.graph)
        lml += (x + e)(x_data).logpdf(y)
        e = GP(noise_obs / S[i] * Delta(), graph=x.graph)
        lml -= e(x_data).logpdf(y)

    # Add regularisation contribution.
    lml += B.sum(Normal(Diagonal(noises_obs)).logpdf(B.transpose(y_data)))

    # Return negative the evidence, normalised by the number of data points.
    return -lml / (n * p)


# Perform learning.
vs = Vars(tf.float64)
minimise_l_bfgs_b(tf.function(objective, autograph=False),
                  vs, trace=True, iters=200)
wbml.out.kv('Spatial scales', vs['scales'])
wbml.out.kv('Length scale of first latent process', vs['0/long_term/scale'])

# Construct model and project data for prediction.
xs, noise_obs, noises_latent = model(vs)
y_proj, H, S, noises_obs = project(vs, locs)
L = noise_obs / S + noises_latent

# Condition latent processes.
xs_posterior = []
for x, noise, y in zip(xs, L, y_proj):
    e = GP(noise * Delta(), graph=x.graph)
    xs_posterior.append(x | ((x + e)(x_data), y))
xs = xs_posterior

# Extract posterior means and variances of the latent processes.
x_means, x_vars = zip(*[(x.mean(x_pred)[:, 0],
                         x.kernel.elwise(x_pred)[:, 0]) for x in xs])

# Construct predictions for latent processes.
lat_preds = [B.to_numpy(mean,
                        mean - 2 * (var + L[i]) ** .5,
                        mean + 2 * (var + L[i]) ** .5)
             for i, (mean, var) in enumerate(zip(x_means, x_vars))]

# Pull means through mixing matrix.
x_means = B.stack(*x_means, axis=0)
y_means = B.matmul(H, x_means)

# Pull variances through mixing matrix.
x_vars = B.stack(*x_vars, axis=0)
y_vars = B.matmul(H ** 2, x_vars + noises_latent[:, None]) + noise_obs

# Construct predictions for outputs.
preds = [(mean, mean - 2 * var ** .5, mean + 2 * var ** .5)
         for mean, var in zip(y_means, y_vars)]

# Convert to NumPy and undo normalisation.
preds = [tuple(x * data_mean[0, i] + data_scale[0, i] for x in B.to_numpy(tup))
         for i, tup in enumerate(preds)]
y_data = B.to_numpy(y_data) * data_mean + data_scale

# Plot first four latent processes.
plt.figure(figsize=(10, 5))
for i in range(4):
    plt.subplot(2, 2, i + 1)
    mean, lower, upper = lat_preds[i]
    plt.title(f'Latent Process {i + 1} (${100 * S[i] / np.sum(S):.1f}\\%$)')
    plt.plot(x_data, y_proj[i], c='tab:blue')
    plt.plot(x_pred, mean, c='tab:green')
    plt.plot(x_pred, lower, c='tab:green', ls='--')
    plt.plot(x_pred, upper, c='tab:green', ls='--')
    wbml.plot.tweak(legend=False)

# Plot four random outputs.
plt.figure(figsize=(10, 5))
for i, j in enumerate(sorted(np.random.permutation(p)[:4])):
    plt.subplot(2, 2, i + 1)
    mean, lower, upper = preds[j]
    plt.title(data.columns[j])
    plt.plot(x_data, y_data[:, j], c='tab:blue')
    plt.plot(x_pred, mean, c='tab:green')
    plt.plot(x_pred, lower, c='tab:green', ls='--')
    plt.plot(x_pred, upper, c='tab:green', ls='--')
    wbml.plot.tweak(legend=False)

plt.show()
