"""Run Premier League prediction model using TensorFlow Probability
"""

import argparse

import arviz as az
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
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


# tfp needs these as global variables due to function signature
teams, df = load_data()
train = df[df["split"] == "train"]
home_id = train["Home_id"].to_numpy()
away_id = train["Away_id"].to_numpy()
s1 = train["score1"].to_numpy()
s2 = train["score2"].to_numpy()
nt = len(teams)


def model(home_id, away_id, nt):
    def joint_dist():
        alpha = yield tfd.Normal(loc=0.0, scale=1.0)
        home = yield tfd.Normal(loc=0.0, scale=1.0)
        sd_att = yield tfd.HalfNormal(scale=1.0)
        sd_def = yield tfd.HalfNormal(scale=1.0)

        attack = yield tfd.Normal(loc=tf.zeros(nt), scale=sd_att)
        defend = yield tfd.Normal(loc=tf.zeros(nt), scale=sd_def)

        home_log_rate = (
            alpha
            + home
            + tf.gather(attack, home_id, axis=-1)
            - tf.gather(defend, away_id, axis=-1)
        )

        away_log_rate = (
            alpha
            + tf.gather(attack, away_id, axis=-1)
            - tf.gather(defend, home_id, axis=-1)
        )

        yield tfd.Poisson(log_rate=home_log_rate)
        yield tfd.Poisson(log_rate=away_log_rate)

    return tfd.JointDistributionCoroutineAutoBatched(joint_dist)


@tf.function
def target_log_prob(alpha, home, sd_att, sd_def, attack, defend):
    """Computes joint log prob pinned at `s1` and `s2`."""
    return model(home_id, away_id, nt).log_prob(
        [alpha, home, sd_att, sd_def, attack, defend, s1, s2]
    )


@tf.function(autograph=False, jit_compile=True)
def run_inference(num_chains, num_results, num_burnin_steps, nt):
    """Samples from the partial pooling model."""
    nuts = tfp.mcmc.NoUTurnSampler(
        target_log_prob_fn=target_log_prob,
        step_size=0.01,
    )

    initial_state = [
        tf.zeros([num_chains], name="init_alpha"),
        tf.zeros([num_chains], name="init_home"),
        tf.ones([num_chains], name="init_sd_att"),
        tf.ones([num_chains], name="init_sd_def"),
        tf.zeros([num_chains, nt], name="init_attack"),
        tf.zeros([num_chains, nt], name="init_defend"),
    ]

    unconstraining_bijectors = [
        tfp.bijectors.Identity(),  # alpha
        tfp.bijectors.Identity(),  # home
        tfp.bijectors.Exp(),  # sd_att
        tfp.bijectors.Exp(),  # sd_def
        tfp.bijectors.Identity(),  # attack
        tfp.bijectors.Identity(),  # defend
    ]

    kernel = tfp.mcmc.TransformedTransitionKernel(
        inner_kernel=nuts, bijector=unconstraining_bijectors
    )

    samples, _ = tfp.mcmc.sample_chain(
        num_results=num_results,
        num_burnin_steps=num_burnin_steps,
        current_state=initial_state,
        kernel=kernel,
    )

    return samples


def main(args):
    print("Loading data...")
    teams, df = load_data()
    train = df[df["split"] == "train"]
    nt = len(teams)

    print("Starting inference...")
    mcmc = run_inference(
        num_chains=args.num_chains,
        num_results=args.num_samples,
        num_burnin_steps=args.num_warmup,
        nt=nt,
    )

    samples = dict(
        zip(
            ["alpha", "home", "sd_att", "sd_def", "attack", "defend"],
            [np.swapaxes(sample, 0, 1) for sample in mcmc],
        )
    )

    fit = az.from_dict(samples)

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
        attack=samples["attack"].mean(axis=(0, 1)),
        attacksd=samples["attack"].std(axis=(0, 1)),
        defend=samples["defend"].mean(axis=(0, 1)),
        defendsd=samples["defend"].std(axis=(0, 1)),
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
        samples["alpha"].flatten()[..., np.newaxis]
        + samples["home"].flatten()[..., np.newaxis]
        + tf.gather(
            samples["attack"].reshape(-1, samples["attack"].shape[-1]),
            predict["Home_id"],
            axis=-1,
        )
        - tf.gather(
            samples["defend"].reshape(-1, samples["defend"].shape[-1]),
            predict["Away_id"],
            axis=-1,
        )
    )

    theta2 = (
        samples["alpha"].flatten()[..., np.newaxis]
        + tf.gather(
            samples["attack"].reshape(-1, samples["attack"].shape[-1]),
            predict["Away_id"],
            axis=-1,
        )
        - tf.gather(
            samples["defend"].reshape(-1, samples["defend"].shape[-1]),
            predict["Home_id"],
            axis=-1,
        )
    )

    s1 = np.array(tfd.Poisson(log_rate=theta1).sample())
    s2 = np.array(tfd.Poisson(log_rate=theta2).sample())

    predicted_full = predict.copy()
    predicted_full = predicted_full.assign(
        score1=s1.mean(axis=0).round(),
        score1error=s1.std(axis=0),
        score2=s2.mean(axis=0).round(),
        score2error=s2.std(axis=0),
    )

    predicted_full = train.append(
        predicted_full.drop(columns=["score1error", "score2error"])
    )

    print(score_table(df))
    print(score_table(predicted_full))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="tfp model")
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
