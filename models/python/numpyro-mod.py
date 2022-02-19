"""Run Premier League prediction model using numpyro
"""

import argparse

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


def load_data():
    pl_data = pd.read_csv("../../data/premierleague.csv")

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


def run_inference(model, home_id, away_id, score1, score2, rng_key, args):
    kernel = NUTS(model)
    mcmc = MCMC(
        kernel,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        progress_bar=True,
    )
    mcmc.run(rng_key, home_id, away_id, score1, score2)
    mcmc.print_summary()
    return mcmc


def main(args):
    print("Loading data...")
    teams, df = load_data()
    train = df[df["split"] == "train"]

    print("Starting inference...")
    rng_key = random.PRNGKey(args.rng_seed)
    mcmc = run_inference(
        model,
        train["Home_id"].values,
        train["Away_id"].values,
        train["score1"].values,
        train["score2"].values,
        rng_key,
        args,
    )

    fit = az.from_numpyro(mcmc)

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

    print(score_table(df))
    print(score_table(predicted_full))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="numpyro model")
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
        "--device",
        default="cpu",
        type=str,
        help='"cpu" or "gpu"?',
    )
    parser.add_argument(
        "--rng_seed", default=1, type=int, help="random number generator seed"
    )
    args = parser.parse_args()

    numpyro.set_platform(args.device)

    main(args)
