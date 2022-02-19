"""Run Premier League prediction model using TensorFlow Probability
"""

import numpy as np
import pandas as pd
import tensorflow as tf
# import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

# from tensorflow_probability import bijectors as tfb
# import arviz as az
# from utils import plot_quality, score_table

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

# %%
# Create copies of variables as Tensors.
Home_id = tf.convert_to_tensor(pl_data["Home_id"], dtype=tf.int32)
Away_id = tf.convert_to_tensor(pl_data["Away_id"], dtype=tf.int32)
s1 = tf.convert_to_tensor(pl_data["score1"], dtype=tf.int32)
s2 = tf.convert_to_tensor(pl_data["score2"], dtype=tf.int32)


# Specify GP model
def model(home_id, away_id, team, s1, s2):
    return tfd.JointDistributionNamed(
        dict(
            alpha=tfd.Normal(loc=0.0, scale=1.0),
            sd_att=tfd.HalfStudentT(df=3.0, loc=0.0, scale=2.5),
            sd_def=tfd.HalfStudentT(df=3.0, loc=0.0, scale=2.5),
            home=tfd.Normal(loc=0.0, scale=1.0),
            attack=lambda sd_att: tfd.MultivariateNormalDiag(
                loc=tf.gather(0, team, axis=-1),
                scale_identity_multiplier=sd_att,
            ),
            defend=lambda sd_def: tfd.MultivariateNormalDiag(
                loc=tf.gather(0, team, axis=-1),
                scale_identity_multiplier=sd_def,
            ),
            # theta1=tf.math.exp(
            #     alpha + home + attack[home_id] - defend[away_id]
            # ),
            # theta2=tf.math.exp(
            #     alpha + attack[away_id] - defend[home_id]
            # ),
            # s1=tfd.Independent(
            #     tfd.Poisson(rate=theta1),
            #     reinterpreted_batch_ndims=1
            # ),
            # s2=tfd.Independent(
            #     tfd.Poisson(rate=theta2),
            #     reinterpreted_batch_ndims=1
            # ),
        )
    )
