"""Run Premier League prediction model using numpyro + blackjax
"""

import pandas as pd
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer.util import initialize_model
import jax
import jax.numpy as jnp
from jax import random
from functools import partial
import blackjax.nuts as nuts
import blackjax.stan_warmup as stan_warmup
import arviz as az

jax.devices()

__author__ = "Theo Rashid"
__email__ = "tar15@ic.ac.uk"

pl_data = pd.read_csv("../../data/premierleague.csv")

ng = len(pl_data)  # number of games
npr = 50  # predict the last 5 rounds of games
ngob = ng - npr  # number of games to train

teams = pl_data["Home"].unique()
teams = pd.DataFrame(teams, columns=["Team"])
teams["i"] = teams.index
nt = len(teams)  # number of teams

df = pd.merge(pl_data, teams, left_on="Home", right_on="Team", how="left")
df = df.rename(columns={"i": "Home_id"}).drop("Team", axis=1)
df = pd.merge(df, teams, left_on="Away", right_on="Team", how="left")
df = df.rename(columns={"i": "Away_id"}).drop("Team", axis=1)

df["split"] = np.where(df.index + 1 <= ngob, "train", "predict")

train = df[df["split"] == "train"]


def model(home_id, away_id, score1_obs=None, score2_obs=None):
    # priors
    alpha = numpyro.sample("alpha", dist.Normal(0.0, 1.0))
    sd_att = numpyro.sample(
        "sd_att",
        dist.FoldedDistribution(dist.StudentT(3.0, 0.0, 2.5)),
    )
    sd_def = numpyro.sample(
        "sd_def",
        dist.FoldedDistribution(dist.StudentT(3.0, 0.0, 2.5)),
    )

    home = numpyro.sample("home", dist.Normal(0.0, 1.0))  # home advantage

    nt = len(np.unique(home_id))

    # team-specific model parameters
    with numpyro.plate("plate_teams", nt):
        attack = numpyro.sample("attack", dist.Normal(0, sd_att))
        defend = numpyro.sample("defend", dist.Normal(0, sd_def))

    # likelihood
    theta1 = jnp.exp(alpha + home + attack[home_id] - defend[away_id])
    theta2 = jnp.exp(alpha + attack[away_id] - defend[home_id])

    with numpyro.plate("data", len(home_id)):
        numpyro.sample("s1", dist.Poisson(theta1), obs=score1_obs)
        numpyro.sample("s2", dist.Poisson(theta2), obs=score2_obs)


rng_key = random.PRNGKey(0)

# translate the model into a log-probability function
init_params, potential_fn_gen, *_ = initialize_model(
    rng_key,
    model,
    model_args=(
        train["Home_id"].values,
        train["Away_id"].values,
        train["score1"].values,
        train["score2"].values,
    ),
    dynamic_args=True,
)

logprob = lambda position: -potential_fn_gen(
    train["Home_id"].values,
    train["Away_id"].values,
    train["score1"].values,
    train["score2"].values,
)(position)

initial_position = init_params.z
initial_state = nuts.new_state(initial_position, logprob)

# run the window adaptation (warmup)
kernel_factory = lambda step_size, inverse_mass_matrix: nuts.kernel(
    logprob, step_size, inverse_mass_matrix
)

last_state, (step_size, inverse_mass_matrix), _ = stan_warmup.run(
    rng_key, kernel_factory, initial_state, 1000
)


@partial(jax.jit, static_argnums=(1, 3))
def inference_loop(rng_key, kernel, initial_state, num_samples):
    def one_step(state, rng_key):
        state, info = kernel(rng_key, state)
        return state, (state, info)

    keys = jax.random.split(rng_key, num_samples)
    _, (states, infos) = jax.lax.scan(one_step, initial_state, keys)

    return states, infos


# Build the kernel using the step size and inverse mass matrix returned from the window adaptation
kernel = kernel_factory(step_size, inverse_mass_matrix)

# Sample from the posterior distribution
states, infos = inference_loop(rng_key, kernel, last_state, 100_000)
states.position["home"].block_until_ready()


acceptance_rate = np.mean(infos.acceptance_probability)
num_divergent = np.mean(infos.is_divergent)
print(f"Acceptance rate: {acceptance_rate:.2f}")
print(f"% divergent transitions: {100*num_divergent:.2f}")


states.position["attack"] = states.position["attack"][jnp.newaxis, ...]
states.position["defend"] = states.position["defend"][jnp.newaxis, ...]

coords = {"teams": np.arange(20)}
dims = {"attack": ["teams"], "defend": ["teams"]}
fit = az.convert_to_inference_data(states.position, coords=coords, dims=dims)


# Plot posterior
az.plot_forest(
    fit,
    var_names=("alpha", "home", "sd_att", "sd_def"),
    backend="bokeh",
)

az.plot_trace(fit, var_names=("alpha", "home", "sd_att", "sd_def"), backend="bokeh")
