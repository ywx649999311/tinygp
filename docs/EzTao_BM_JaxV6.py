# ## Benchmark Plot

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from eztao.carma import DRW_term
from eztao.ts import addNoise, gpSimFull
import os

os.environ["JAX_ENABLE_X64"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["XLA_FLAGS"] = (
    os.environ.get("XLA_FLAGS", "")
    + " --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
)

import jax
import jax.numpy as jnp

print(jax.__version__)

import tinygp
from tinygp import kernels, GaussianProcess

# mpl style
import matplotlib as mpl

mpl.rc_file("/home/jovyan/configs/viz/yuBasic.rc")

# import proj

# data_dir = f"{proj.__root__}/data"
# data_dir
# -

# ### Fresh Re-Run

amps = {"g": 0.35, "i": 0.25}
taus = {"g": 100, "i": 150}
snrs = {"g": 5, "i": 3}
sampling_seeds = {"g": 2, "i": 5}
noise_seeds = {"g": 111, "i": 2}

# #### Sim Full LC

ts, ys, yerrs = {}, {}, {}
ys_noisy = {}
seed = 1
for band in "gi":
    DRW_kernel = DRW_term(np.log(amps[band]), np.log(taus[band]))
    t, y, yerr = gpSimFull(
        DRW_kernel,
        snrs[band],
        365 * 10,
        100_000,
        lc_seed=seed,
    )

    # add to dict
    ts[band] = t
    ys[band] = y
    yerrs[band] = yerr
    ys_noisy[band] = addNoise(ys[band], yerrs[band], seed=noise_seeds[band] + seed)

# #### Jax Functions

# +
jax.config.update("jax_enable_x64", True)


def amp_transform(params):
    return jnp.insert(jnp.atleast_1d(params["log_amp_delta"]), 0, 0.0)


def lag_transform(X, lags):
    t, band = X
    new_t = t - lags[band]
    inds = jnp.argsort(new_t)
    return (new_t, band), inds


# GP class
@tinygp.helpers.dataclass
class MB(tinygp.kernels.quasisep.Wrapper):
    amplitudes: jnp.ndarray
    lags: jnp.ndarray

    def coord_to_sortable(self, X):
        return X[0]

    def observation_model(self, X):
        return self.amplitudes[X[1]] * self.kernel.observation_model(X[0])

    # def transition_matrix(self, X1, X2):
    #     t1 = self.coord_to_sortable(X1) - self.lags[X1[1]]
    #     t2 = self.coord_to_sortable(X2) - self.lags[X2[1]]
    #     return jnp.where(
    #         t1 < t2,
    #         self.kernel.transition_matrix(t1, t2),
    #         self.kernel.transition_matrix(t2, t1),
    #     )


def build_mb_drw_lag(params, X, diag):
    t = X[0]
    band = X[1]

    # scale params
    log_amps = amp_transform(params)

    lags = jnp.insert(jnp.atleast_1d(params["lag"]), 0, 0.0)
    new_X, inds = lag_transform(X, lags)
    t, band = new_X

    kernel = MB(
        amplitudes=jnp.exp(log_amps),
        lags=lags,
        kernel=kernels.quasisep.Exp(*jnp.exp(params["log_kernel_param"])),
    )
    return (
        GaussianProcess(
            kernel,
            (t[inds], band[inds]),
            diag=diag[inds],
            mean=0.0,
            assume_sorted=True,
        ),
        inds,
    )


def build_mb_m52_lag(params, X, diag):
    t = X[0]
    band = X[1]

    # scale params
    log_amps = amp_transform(params)

    lags = jnp.insert(jnp.atleast_1d(params["lag"]), 0, 0.0)
    new_X, inds = lag_transform(X, lags)
    t, band = new_X

    kernel = MB(
        amplitudes=jnp.exp(log_amps),
        lags=lags,
        kernel=kernels.quasisep.Matern52(*jnp.exp(params["log_kernel_param"])),
    )
    return (
        GaussianProcess(
            kernel,
            (t[inds], band[inds]),
            diag=diag[inds],
            mean=0.0,
            assume_sorted=True,
        ),
        inds,
    )


def build_mb_m32_lag(params, X, diag):
    t = X[0]
    band = X[1]

    # scale params
    log_amps = amp_transform(params)

    lags = jnp.insert(jnp.atleast_1d(params["lag"]), 0, 0.0)
    new_X, inds = lag_transform(X, lags)
    t, band = new_X

    kernel = MB(
        amplitudes=jnp.exp(log_amps),
        lags=lags,
        kernel=kernels.quasisep.Matern32(*jnp.exp(params["log_kernel_param"])),
    )
    return (
        GaussianProcess(
            kernel,
            (t[inds], band[inds]),
            diag=diag[inds],
            mean=0.0,
            assume_sorted=True,
        ),
        inds,
    )


def build_mb_carma_lag(params, X, diag, p):
    t = X[0]
    band = X[1]

    # scale params
    log_amps = amp_transform(params)

    lags = jnp.insert(jnp.atleast_1d(params["lag"]), 0, 0.0)
    new_X, inds = lag_transform(X, lags)
    t, band = new_X

    kernel = MB(
        amplitudes=jnp.exp(log_amps),
        lags=lags,
        kernel=kernels.quasisep.CARMA.init(
            alpha=jnp.exp(params["log_kernel_param"][:p]),
            beta=jnp.exp(params["log_kernel_param"][p:]),
        ),
    )
    return (
        GaussianProcess(
            kernel,
            (t[inds], band[inds]),
            diag=diag[inds],
            mean=0.0,
            assume_sorted=True,
        ),
        inds,
    )


# -

ns = jnp.floor(jnp.logspace(jnp.log10(30), 5, 20)).astype(int)
params = {
    "log_kernel_param": jnp.log(jnp.array([100, 0.1])),
    "log_amp_delta": jnp.log(0.6),
    "lag": jnp.array(10),
}
ns

# #### Single-Band JAX

cpu_time_single = []
for n in ns[:]:
    inds = jnp.argsort(jnp.concatenate((ts["g"], ts["i"])))
    X = (
        jnp.concatenate((ts["g"], ts["i"]))[inds][:n],
        jnp.concatenate(
            (jnp.zeros_like(ts["g"], dtype=int), jnp.ones_like(ts["i"], dtype=int))
        )[inds][:n],
    )
    ys_noisy["g"] -= jnp.median(ys_noisy["g"])
    ys_noisy["i"] -= jnp.median(ys_noisy["i"])
    y = jnp.concatenate((ys_noisy["g"], ys_noisy["i"]))[inds][:n]
    diag = jnp.concatenate((yerrs["g"], yerrs["i"]))[inds][:n] ** 2

    @jax.jit
    def loss(params):
        kernel = kernels.quasisep.Exp(*jnp.exp(params["log_kernel_param"]))
        inds = jnp.argsort(X[0])
        gp = GaussianProcess(
            kernel,
            X[0],
            diag=diag,
            mean=0.0,
            assume_sorted=True,
        )
        return -gp.log_probability(y)

    loss(params).block_until_ready()
    # results = %timeit -o loss(params).block_until_ready()
    cpu_time_single.append(results.average)

# #### Two Bands DRW + Lag

cpu_time_mb_drw_lag = []
for n in ns[:]:
    inds = jnp.argsort(jnp.concatenate((ts["g"], ts["i"])))
    X = (
        jnp.concatenate((ts["g"], ts["i"]))[inds][:n],
        jnp.concatenate(
            (jnp.zeros_like(ts["g"], dtype=int), jnp.ones_like(ts["i"], dtype=int))
        )[inds][:n],
    )
    ys_noisy["g"] -= jnp.median(ys_noisy["g"])
    ys_noisy["i"] -= jnp.median(ys_noisy["i"])
    y = jnp.concatenate((ys_noisy["g"], ys_noisy["i"]))[inds][:n]
    diag = jnp.concatenate((yerrs["g"], yerrs["i"]))[inds][:n] ** 2

    @jax.jit
    def loss(params):
        gp, inds = build_mb_drw_lag(params, X, diag)
        return -gp.log_probability(y[inds])

    loss(params).block_until_ready()
    # results = %timeit -o loss(params).block_until_ready()
    cpu_time_mb_drw_lag.append(results.average)

# #### Two Bands Matern52 + Lag

cpu_time_mb_m52_lag = []
for n in ns[:]:
    inds = jnp.argsort(jnp.concatenate((ts["g"], ts["i"])))
    X = (
        jnp.concatenate((ts["g"], ts["i"]))[inds][:n],
        jnp.concatenate(
            (jnp.zeros_like(ts["g"], dtype=int), jnp.ones_like(ts["i"], dtype=int))
        )[inds][:n],
    )
    ys_noisy["g"] -= jnp.median(ys_noisy["g"])
    ys_noisy["i"] -= jnp.median(ys_noisy["i"])
    y = jnp.concatenate((ys_noisy["g"], ys_noisy["i"]))[inds][:n]
    diag = jnp.concatenate((yerrs["g"], yerrs["i"]))[inds][:n] ** 2

    @jax.jit
    def loss(params):
        gp, inds = build_mb_m52_lag(params, X, diag)
        return -gp.log_probability(y[inds])

    loss(params).block_until_ready()
    # results = %timeit -o loss(params).block_until_ready()
    cpu_time_mb_m52_lag.append(results.average)

# #### Two Band CARMA(2,0) + Lag

# +
p = 2
carma_params = params = {
    "log_kernel_param": jnp.log(jnp.array([0.1, 0.1, 0.1])),
    "log_amp_delta": jnp.log(0.6),
    "lag": jnp.array(10),
}
cpu_time_mb_carma20_lag = []

for n in ns[:]:
    inds = jnp.argsort(jnp.concatenate((ts["g"], ts["i"])))
    X = (
        jnp.concatenate((ts["g"], ts["i"]))[inds][:n],
        jnp.concatenate(
            (jnp.zeros_like(ts["g"], dtype=int), jnp.ones_like(ts["i"], dtype=int))
        )[inds][:n],
    )
    ys_noisy["g"] -= jnp.median(ys_noisy["g"])
    ys_noisy["i"] -= jnp.median(ys_noisy["i"])
    y = jnp.concatenate((ys_noisy["g"], ys_noisy["i"]))[inds][:n]
    diag = jnp.concatenate((yerrs["g"], yerrs["i"]))[inds][:n] ** 2

    @jax.jit
    def loss(params):
        gp, inds = build_mb_carma_lag(params, X, diag, p)
        return -gp.log_probability(y[inds])

    loss(params).block_until_ready()
    # results = %timeit -o loss(params).block_until_ready()
    cpu_time_mb_carma20_lag.append(results.average)
# -

# #### Two Band CARMA(3,0) + Lag

# +
p = 3
carma_params = params = {
    "log_kernel_param": jnp.log(jnp.array([0.1, 0.1, 0.1, 0.1])),
    "log_amp_delta": jnp.log(0.6),
    "lag": jnp.array(10),
}
cpu_time_mb_carma30_lag = []

for n in ns[:]:
    inds = jnp.argsort(jnp.concatenate((ts["g"], ts["i"])))
    X = (
        jnp.concatenate((ts["g"], ts["i"]))[inds][:n],
        jnp.concatenate(
            (jnp.zeros_like(ts["g"], dtype=int), jnp.ones_like(ts["i"], dtype=int))
        )[inds][:n],
    )
    ys_noisy["g"] -= jnp.median(ys_noisy["g"])
    ys_noisy["i"] -= jnp.median(ys_noisy["i"])
    y = jnp.concatenate((ys_noisy["g"], ys_noisy["i"]))[inds][:n]
    diag = jnp.concatenate((yerrs["g"], yerrs["i"]))[inds][:n] ** 2

    @jax.jit
    def loss(params):
        gp, inds = build_mb_carma_lag(params, X, diag, p)
        return -gp.log_probability(y[inds])

    loss(params).block_until_ready()
    # results = %timeit -o loss(params).block_until_ready()
    cpu_time_mb_carma30_lag.append(results.average)
# -

# ### Plot

# +
fig = plt.figure(figsize=(6, 5.5))
plt.loglog(
    ns,
    cpu_time_single[:],
    label="DRW (single band)",
    # color=color_cycle[i],
    linewidth=2,
)
plt.loglog(
    ns,
    np.array(cpu_time_mb_drw_lag[:]) * 1,
    label="DRW (two band + lag)",
    # color=color_cycle[i],
    linewidth=2,
)

# plt.loglog(
#     ns,
#     np.array(cpu_time_mb_m52_lag[:]) * 1,
#     label="M52 (two band + lag)",
#     # color=color_cycle[i],
#     linewidth=2,
# )

# plt.loglog(
#     ns,
#     np.array(cpu_time_mb_carma20_lag[:]) * 1,
#     label="CARMA(2,0) (two band + lag)",
#     # color=color_cycle[i],
#     linewidth=2,
# )

# plt.loglog(
#     ns,
#     np.array(cpu_time_mb_carma30_lag[:]) * 1,
#     label="CARMA(3,0) (two band + lag)",
#     # color=color_cycle[i],
#     linewidth=2,
# )

plt.legend()
# plt.title("DRW Benchmark")
plt.xlabel("Number of Data Points", labelpad=15)
plt.ylabel("Runtime [second]", labelpad=15)
plt.xlim(50, 1e5 * 1.1)
plt.fill_betweenx(
    np.logspace(-5.5, -2.6, 10), x1=50, x2=3000, alpha=0.1, color="tab:orange"
)

plt.tight_layout()
# -

# #### Save Benchmark

bm_df = pd.DataFrame(
    {
        "data points": ns,
        "drw_lag": cpu_time_mb_drw_lag,
        "m52_lag": cpu_time_mb_m52_lag,
        "c20_lag": cpu_time_mb_carma20_lag,
        "c30_lag": cpu_time_mb_carma30_lag,
        "drw": cpu_time_single,
    }
)
bm_df

bm_dir = f"{data_dir}/Benchmark/"
bm_df.to_csv(bm_dir + "eztao_jax2.csv", index=False)

# #### -------------------------------------------------------------------------------------------

# #### Save Simulated LC

# +
lc1 = np.array([ts["g"], ys["g"], yerrs["g"]])
lc2 = np.array([ts["i"], ys["i"], yerrs["i"]])

np.savetxt(f"{data_dir}/sim_lc/bm_lc1.dat", lc1)
np.savetxt(f"{data_dir}/sim_lc/bm_lc2.dat", lc2)
