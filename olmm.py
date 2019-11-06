from stheno import (
    B,  # Linear algebra backend
    Graph,  # Graph that keep track of the graphical model
    GP,  # Gaussian process
    EQ,  # Squared-exponential kernel
    Matern12,  # Matern-1/2 kernel
    Matern52,  # Matern-5/2 kernel
    Delta,  # Noise kernel
    Normal,  # Gaussian distribution
    Diagonal,  # Diagonal matrix
    dense,  # Convert matrix objects to regular matrices
)

__all__ = ['model', 'project', 'objective', 'predict']


def model(vs, m):
    """Construct model.

    Args:
        vs (:class:`varz.Vars`): Variable container.
        m (int): Number of latent processes.

    Returns:
        tuple: Tuple containing a list of the latent processes, the
            observation noise, and the noises on the latent processes.
    """
    g = Graph()

    # Observation noise:
    noise_obs = vs.bnd(0.1, name='noise_obs')

    def make_latent_process(i):
        # Long-term trend:
        variance = vs.bnd(0.9, name=f'{i}/long_term/var')
        scale = vs.bnd(2 * 30, name=f'{i}/long_term/scale')
        kernel = variance * EQ().stretch(scale)

        # Short-term trend:
        variance = vs.bnd(0.1, name=f'{i}/short_term/var')
        scale = vs.bnd(20, name=f'{i}/short_term/scale')
        kernel += variance * Matern12().stretch(scale)

        return GP(kernel, graph=g)

    # Latent processes:
    xs = [make_latent_process(i) for i in range(m)]

    # Latent noises:
    noises_latent = vs.bnd(0.1 * B.ones(m), name='noises_latent')

    return xs, noise_obs, noises_latent


def project(vs, m, y_data, locs):
    """Project the data.

    Args:
        vs (:class:`varz.Vars`): Variable container.
        m (int): Number of latent processes.
        y_data (tensor): Observations.
        locs (tensor): Spatial locations of observations.

    Returns:
        tuple: Tuple containing the projected outputs, the mixing matrix,
            S from the mixing matrix, and the observation noises.
    """
    _, noise_obs, noises_latent = model(vs, m)

    # Construct mixing matrix and projection.
    scales = vs.bnd(B.ones(2), name='scales')
    K = dense(Matern52().stretch(scales)(locs))
    U, S, _ = B.svd(K)
    S = S[:m]
    H = U[:, :m] * S[None, :] ** .5
    T = B.transpose(U[:, :m]) / S[:, None] ** .5

    # Project data and unstack over latent processes.
    y_proj = B.unstack(B.matmul(T, y_data, tr_b=True))

    # Observation noises:
    noises_obs = noise_obs * B.ones(B.dtype(noise_obs), B.shape(y_data)[1])

    return y_proj, H, S, noises_obs


def objective(vs, m, x_data, y_data, locs):
    """NLML objective.

    Args:
        vs (:class:`varz.Vars`): Variable container.
        m (int): Number of latent processes.
        x_data (tensor): Time stamps of the observations.
        y_data (tensor): Observations.
        locs (tensor): Spatial locations of observations.

    Returns:
        scalar: Negative log-marginal likelihood.
    """
    y_proj, _, S, noises_obs = project(vs, m, y_data, locs)
    xs, noise_obs, noises_latent = model(vs, m)

    # Add contribution of latent processes.
    lml = 0
    for i, (x, y) in enumerate(zip(xs, y_proj)):
        e_signal = GP((noise_obs / S[i] + noises_latent[i]) * Delta(),
                      graph=x.graph)
        lml += (x + e_signal)(x_data).logpdf(y)

        e_noise = GP(noise_obs / S[i] * Delta(), graph=x.graph)
        lml -= e_noise(x_data).logpdf(y)

    # Add regularisation contribution.
    lml += B.sum(Normal(Diagonal(noises_obs)).logpdf(B.transpose(y_data)))

    # Return negative the evidence, normalised by the number of data points.
    n, p = B.shape(y_data)
    return -lml / (n * p)


def predict(vs, m, x_data, y_data, locs, x_pred):
    """Make predictions.

    Args:
        vs (:class:`varz.Vars`): Variable container.
        m (int): Number of latent processes.
        x_data (tensor): Time stamps of the observations.
        y_data (tensor): Observations.
        locs (tensor): Spatial locations of observations.
        x_pred (tensor): Time stamps to predict at.

    Returns:
        tuple: Tuple containing the predictions for the latent processes and
            predictions for the observations.
    """
    # Construct model and project data for prediction.
    xs, noise_obs, noises_latent = model(vs, m)
    y_proj, H, S, noises_obs = project(vs, m, y_data, locs)
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

    # Pull variances through mixing matrix and add noise.
    x_vars = B.stack(*x_vars, axis=0)
    y_vars = B.matmul(H ** 2, x_vars + noises_latent[:, None]) + noise_obs

    # Construct predictions for observations.
    obs_preds = [(mean, mean - 2 * var ** .5, mean + 2 * var ** .5)
                 for mean, var in zip(y_means, y_vars)]

    return lat_preds, obs_preds
