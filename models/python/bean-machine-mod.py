"""Run Premier League prediction model using Bean Machine
"""

import argparse

import arviz as az
import beanmachine.ppl as bm
import numpy as np
import pandas as pd
import torch
import torch.distributions as dist
from beanmachine.ppl.model import RVIdentifier

from utils import plot_quality, score_table

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


# tfp needs these as global variables due to function signature
teams, df = load_data()
train = df[df["split"] == "train"]
home_id = train["Home_id"].tolist()
away_id = train["Away_id"].tolist()
nt = len(teams)


@bm.random_variable
def alpha() -> RVIdentifier:
    return dist.Normal(0, 1)


@bm.random_variable
def home() -> RVIdentifier:
    return dist.Normal(0, 1)


@bm.random_variable
def sd_att() -> RVIdentifier:
    return dist.HalfNormal(1)


@bm.random_variable
def sd_def() -> RVIdentifier:
    return dist.HalfNormal(1)


@bm.random_variable
def attack() -> RVIdentifier:
    return dist.Normal(0, sd_att()).expand((nt,))


@bm.random_variable
def defend() -> RVIdentifier:
    return dist.Normal(0, sd_def()).expand((nt,))


@bm.functional
def theta1():
    return torch.exp(alpha() + home() + attack()[home_id] - defend()[away_id])


@bm.functional
def theta2():
    return torch.exp(alpha() + attack()[away_id] - defend()[home_id])


@bm.random_variable
def s1() -> RVIdentifier:
    return dist.Poisson(theta1())


@bm.random_variable
def s2() -> RVIdentifier:
    return dist.Poisson(theta2())


def main(args):
    print("Loading data...")
    teams, df = load_data()
    train = df[df["split"] == "train"]

    print("Starting inference...")
    samples = bm.GlobalNoUTurnSampler().infer(
        queries=[
            alpha(),
            home(),
            sd_att(),
            sd_def(),
            attack(),
            defend(),
        ],
        observations={
            s1(): torch.tensor(train["score1"].values),
            s2(): torch.tensor(train["score2"].values),
        },
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        num_adaptive_samples=args.num_warmup,
    )

    samples = samples.to_xarray()
    fit = az.InferenceData(posterior=samples)

    print("Analyse posterior...")
    az.plot_forest(
        fit,
        backend="bokeh",
    )

    az.plot_trace(
        fit,
        backend="bokeh",
    )

    # Attack and defence
    quality = teams.copy()
    quality = quality.assign(
        attack=samples[attack()].mean(axis=(0, 1)),
        attacksd=samples[attack()].std(axis=(0, 1)),
        defend=samples[defend()].mean(axis=(0, 1)),
        defendsd=samples[defend()].std(axis=(0, 1)),
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

    theta1 = (
        samples[alpha()].expand_dims("", axis=-1).values
        + samples[home()].expand_dims("", axis=-1).values
        + samples[attack()][:, :, predict["Home_id"]].values
        - samples[defend()][:, :, predict["Away_id"]].values
    )
    theta1 = torch.tensor(theta1.reshape(-1, theta1.shape[-1]))

    theta2 = (
        samples[alpha()].expand_dims("", axis=-1).values
        + samples[attack()][:, :, predict["Away_id"]].values
        - samples[defend()][:, :, predict["Home_id"]].values
    )
    theta2 = torch.tensor(theta2.reshape(-1, theta2.shape[-1]))

    score1 = np.array(dist.Poisson(torch.exp(theta1)).sample())
    score2 = np.array(dist.Poisson(torch.exp(theta2)).sample())

    predicted_full = predict.copy()
    predicted_full = predicted_full.assign(
        score1=score1.mean(axis=0).round(),
        score1error=score1.std(axis=0),
        score2=score2.mean(axis=0).round(),
        score2error=score2.std(axis=0),
    )

    predicted_full = train.append(
        predicted_full.drop(columns=["score1error", "score2error"])
    )

    print(score_table(df))
    print(score_table(predicted_full))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="bean machine model")
    parser.add_argument(
        "-n",
        "--num-samples",
        nargs="?",
        default=10000,
        type=int,
    )
    parser.add_argument("--num-warmup", nargs="?", default=5000, type=int)
    parser.add_argument("--num-chains", nargs="?", default=2, type=int)
    parser.add_argument("--num-cores", nargs="?", default=2, type=int)
    args = parser.parse_args()

    main(args)
