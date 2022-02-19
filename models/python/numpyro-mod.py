"""Run Premier League prediction model using numpyro
"""

import arviz as az
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from jax import random
from numpyro.infer import MCMC, NUTS, Predictive
from utils import plot_quality, score_table

numpyro.set_platform("cpu")
numpyro.set_host_device_count(2)

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


nuts_kernel = NUTS(model)

mcmc = MCMC(nuts_kernel, num_samples=10000, num_warmup=5000, num_chains=2)
rng_key = random.PRNGKey(0)
mcmc.run(
    rng_key,
    home_id=train["Home_id"].values,
    away_id=train["Away_id"].values,
    score1_obs=train["score1"].values,
    score2_obs=train["score2"].values,
)

fit = az.from_numpyro(mcmc)

# Plot posterior
az.plot_forest(
    fit,
    var_names=("alpha", "home", "sd_att", "sd_def"),
    backend="bokeh",
)

az.plot_trace(
    fit,
    var_names=("alpha", "home", "sd_att", "sd_def"),
    backend="bokeh",
)

fit = mcmc.get_samples()

# Attack and defence
quality = teams.copy()
quality = quality.assign(
    attack=fit["attack"].mean(axis=0),
    attacksd=fit["attack"].std(axis=0),
    defend=fit["defend"].mean(axis=0),
    defendsd=fit["defend"].std(axis=0),
)
quality = quality.assign(
    attack_low=quality["attack"] - quality["attacksd"],
    attack_high=quality["attack"] + quality["attacksd"],
    defend_low=quality["defend"] - quality["defendsd"],
    defend_high=quality["defend"] + quality["defendsd"],
)

plot_quality(quality)

# Predicted goals and table
predict = df[df["split"] == "predict"]

predictive = Predictive(model, fit, return_sites=["s1", "s2"])

predicted_score = predictive(
    random.PRNGKey(0),
    home_id=predict["Home_id"].values,
    away_id=predict["Away_id"].values,
)

predicted_full = predict.copy()
predicted_full = predicted_full.assign(
    score1=predicted_score["s1"].mean(axis=0).round(),
    score1error=predicted_score["s1"].std(axis=0),
    score2=predicted_score["s2"].mean(axis=0).round(),
    score2error=predicted_score["s2"].std(axis=0),
)

predicted_full = train.append(
    predicted_full.drop(columns=["score1error", "score2error"])
)

score_table(pl_data)
score_table(predicted_full)
