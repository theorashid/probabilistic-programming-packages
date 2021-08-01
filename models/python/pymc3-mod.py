"""Run Premier League prediction model using PyMC3
"""

import pandas as pd
import numpy as np
import pymc3 as pm
import theano.tensor as tt
import arviz as az
from utils import plot_quality, score_table

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
df = df.rename(columns={"i": "Home_id"}).drop("Team", axis=11)
df = pd.merge(df, teams, left_on="Away", right_on="Team", how="left")
df = df.rename(columns={"i": "Away_id"}).drop("Team", axis=11)

df["split"] = np.where(df.index + 1 <= ngob, "train", "predict")

train = df[df["split"] == "train"]

with pm.Model() as model:
    # priors
    mu_att = pm.Normal("mu_att", mu=0, sigma=1)
    sd_att = pm.HalfStudentT("sd_att", nu=3, sigma=2.5)
    mu_def = pm.Normal("mu_def", mu=0, sigma=1)
    sd_def = pm.HalfStudentT("sd_def", nu=3, sigma=2.5)

    home = pm.Normal("home", mu=0, sigma=1)  # home advantage

    # team-specific model parameters
    attack = pm.Normal("attack", mu=mu_att, sigma=sd_att, shape=nt)
    defend = pm.Normal("defend", mu=mu_def, sigma=sd_def, shape=nt)

    # data
    home_id = pm.Data("home_data", train["Home_id"])
    away_id = pm.Data("away_data", train["Away_id"])

    # likelihood
    theta1 = tt.exp(home + attack[home_id] - defend[away_id])
    theta2 = tt.exp(attack[away_id] - defend[home_id])

    s1 = pm.Poisson("s1", mu=theta1, observed=train["score1"])
    s2 = pm.Poisson("s2", mu=theta2, observed=train["score2"])

with model:
    fit = pm.sample(draws=10000, tune=5000, chains=2, cores=1, random_seed=1)

# Plot posterior
az.plot_forest(
    fit,
    var_names=("mu_att", "mu_def", "sd_att", "sd_def", "home"),
    backend="bokeh",
)

az.plot_trace(
    fit, var_names=("mu_att", "mu_def", "sd_att", "sd_def", "home"), backend="bokeh"
)

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

with model:
    pm.set_data({"home_data": predict["Home_id"]})
    pm.set_data({"away_data": predict["Away_id"]})

    predicted_score = pm.sample_posterior_predictive(
        fit, var_names=["s1", "s2"], random_seed=1
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
