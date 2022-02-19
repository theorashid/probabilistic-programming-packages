"""Run Premier League prediction model using edward2
"""
# %%
import pandas as pd
import numpy as np
import edward2 as ed
import tensorflow as tf

# import arviz as az
# from bokeh.plotting import figure, show
# from bokeh.models import ColumnDataSource, Whisker
# from bokeh.models.tools import HoverTool

__author__ = "Theo Rashid"
__email__ = "tar15@ic.ac.uk"
# %%
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
    alpha = ed.Normal(loc=0.0, scale=1.0, name="alpha")
    sd_att = ed.StudenT(df=3.0, loc=0.0, scale=2.5, name="sd_att")
    sd_def = ed.StudenT(df=3.0, loc=0.0, scale=2.5, name="sd_def")

    home = ed.Normal(loc=0.0, scale=1.0, name="home")  # home advantage

    nt = len(np.unique(home_id))
    # team-specific model parameters
    attack = ed.Normal(loc=0, scale=sd_att, sample_shape=nt, name="attack")
    defend = ed.Normal(loc=0, scale=sd_def, sample_shape=nt, name="defend")

    # likelihood
    theta1 = tf.exp(alpha + home + attack[home_id] - defend[away_id])
    theta2 = tf.exp(alpha + attack[away_id] - defend[home_id])

    s1 = ed.Poisson(theta1, name="s1")
    s2 = ed.Poisson(theta2, name="s2")
    return s1, s2
