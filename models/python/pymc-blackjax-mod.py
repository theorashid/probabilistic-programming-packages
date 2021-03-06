"""Run Premier League prediction model using PyMC + blackjax
"""

import argparse

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
from pymc.sampling_jax import sample_blackjax_nuts
from utils import plot_quality, score_table

__author__ = "Theo Rashid"
__email__ = "tar15@ic.ac.uk"


def load_data():
    pl_data = pd.read_csv("data/premierleague.csv")

    ng = len(pl_data)  # number of games
    npr = 50  # predict the last 5 rounds of games
    ngob = ng - npr  # number of games to train

    teams = pl_data["Home"].unique()
    teams = pd.DataFrame(teams, columns=["Team"])
    teams["i"] = teams.index

    df = pd.merge(pl_data, teams, left_on="Home", right_on="Team", how="left")
    df = df.rename(columns={"i": "Home_id"}).drop("Team", axis=1)
    df = pd.merge(df, teams, left_on="Away", right_on="Team", how="left")
    df = df.rename(columns={"i": "Away_id"}).drop("Team", axis=1)

    df["split"] = np.where(df.index + 1 <= ngob, "train", "predict")

    print(df.head())

    return teams, df


def main(args):
    print("Loading data...")
    teams, df = load_data()
    nt = len(teams)
    train = df[df["split"] == "train"]

    print("Starting inference...")
    with pm.Model() as model:
        # priors
        alpha = pm.Normal("alpha", mu=0, sigma=1)
        sd_att = pm.HalfStudentT("sd_att", nu=3, sigma=2.5)
        sd_def = pm.HalfStudentT("sd_def", nu=3, sigma=2.5)

        home = pm.Normal("home", mu=0, sigma=1)  # home advantage

        # team-specific model parameters
        attack = pm.Normal("attack", mu=0, sigma=sd_att, shape=nt)
        defend = pm.Normal("defend", mu=0, sigma=sd_def, shape=nt)

        # data
        home_id = pm.MutableData("home_data", train["Home_id"])
        away_id = pm.MutableData("away_data", train["Away_id"])

        # likelihood
        theta1 = pm.math.exp(alpha + home + attack[home_id] - defend[away_id])
        theta2 = pm.math.exp(alpha + attack[away_id] - defend[home_id])

        pm.Poisson("s1", mu=theta1, observed=train["score1"])
        pm.Poisson("s2", mu=theta2, observed=train["score2"])

    with model:
        fit = sample_blackjax_nuts(
            draws=args.num_samples,
            tune=args.num_warmup,
            chains=args.num_chains,
            random_seed=args.rng_seed,
        )

    print("Analyse posterior...")
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

    # Attack and defence
    quality = teams.copy()
    quality = quality.assign(
        attack=fit.posterior["attack"].mean(axis=(0, 1)),
        attacksd=fit.posterior["attack"].std(axis=(0, 1)),
        defend=fit.posterior["defend"].mean(axis=(0, 1)),
        defendsd=fit.posterior["defend"].std(axis=(0, 1)),
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
        score1=predicted_score.posterior_predictive["s1"]
        .mean(
            axis=(0, 1),
        )
        .round(),
        score1error=predicted_score.posterior_predictive["s1"].std(
            axis=(0, 1),
        ),
        score2=predicted_score.posterior_predictive["s2"]
        .mean(
            axis=(0, 1),
        )
        .round(),
        score2error=predicted_score.posterior_predictive["s2"].std(
            axis=(0, 1),
        ),
    )

    predicted_full = train.append(
        predicted_full.drop(columns=["score1error", "score2error"])
    )

    print(score_table(df))
    print(score_table(predicted_full))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="pymc model")
    parser.add_argument(
        "-n",
        "--num-samples",
        nargs="?",
        default=10000,
        type=int,
    )
    parser.add_argument("--num-warmup", nargs="?", default=5000, type=int)
    parser.add_argument("--num-chains", nargs="?", default=2, type=int)
    parser.add_argument(
        "--rng_seed", default=1, type=int, help="random number generator seed"
    )
    args = parser.parse_args()

    main(args)
