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
df = df.rename(columns={"i": "Home_id"}).drop("Team", 1)
df = pd.merge(df, teams, left_on="Away", right_on="Team", how="left")
df = df.rename(columns={"i": "Away_id"}).drop("Team", 1)

df["split"] = np.where(df.index + 1 <= ngob, "train", "predict")

train = df[df["split"] == "train"]


def model(home_id, away_id, score1_obs=None, score2_obs=None):
    # priors
    mu_att = ed.Normal(loc=0.0, scale=1.0, name="mu_att")
    sd_att = ed.StudenT(df=3.0, loc=0.0, scale=2.5, name="sd_att")
    mu_def = ed.Normal(loc=0.0, scale=1.0, name="mu_def")
    sd_def = ed.StudenT(df=3.0, loc=0.0, scale=2.5, name="sd_def")

    home = ed.Normal(loc=0.0, scale=1.0, name="home")  # home advantage

    nt = len(np.unique(home_id))
    # team-specific model parameters
    attack = ed.Normal(loc=mu_att, scale=sd_att, sample_shape=nt, name="attack")
    defend = ed.Normal(loc=mu_def, scale=sd_def, sample_shape=nt, name="defend")

    # likelihood
    theta1 = tf.exp(home + attack[home_id] - defend[away_id])
    theta2 = tf.exp(attack[away_id] - defend[home_id])

    s1 = ed.Poisson(theta1, name="s1")
    s2 = ed.Poisson(theta2, name="s2")
    return s1, s2


# nuts_kernel = NUTS(model)

# mcmc = MCMC(nuts_kernel, num_samples=10000, num_warmup=5000, num_chains=2)
# rng_key = random.PRNGKey(0)
# mcmc.run(
#     rng_key,
#     home_id=train["Home_id"].values,
#     away_id=train["Away_id"].values,
#     score1_obs=train["score1"].values,
#     score2_obs=train["score2"].values,
# )

# fit = az.from_numpyro(mcmc)

# Plot posterior
# az.plot_forest(
#     fit,
#     var_names=("mu_att", "mu_def", "sd_att", "sd_def", "home"),
#     backend="bokeh",
# )

# az.plot_trace(
#     fit, var_names=("mu_att", "mu_def", "sd_att", "sd_def", "home"), backend="bokeh"
# )

# fit = mcmc.get_samples()

# Attack and defence
# quality = teams.copy()
# quality = quality.assign(
#     attack=fit["attack"].mean(axis=0),
#     attacksd=fit["attack"].std(axis=0),
#     defend=fit["defend"].mean(axis=0),
#     defendsd=fit["defend"].std(axis=0),
# )
# quality = quality.assign(
#     attack_low=quality["attack"] - quality["attacksd"],
#     attack_high=quality["attack"] + quality["attacksd"],
#     defend_low=quality["defend"] - quality["defendsd"],
#     defend_high=quality["defend"] + quality["defendsd"],
# )

# source = ColumnDataSource(quality)
# p = figure(x_range=(-1, 1.5), y_range=(-1, 1.5))
# p.add_layout(
#     Whisker(
#         lower="attack_low",
#         upper="attack_high",
#         base="defend",
#         dimension="width",
#         source=source,
#     )
# )
# p.add_layout(
#     Whisker(
#         lower="defend_low",
#         upper="defend_high",
#         base="attack",
#         dimension="height",
#         source=source,
#     )
# )
# p.circle(x="attack", y="defend", source=source)

# p.title.text = "Team strengths"
# p.xaxis.axis_label = "Attacking"
# p.yaxis.axis_label = "Defending"

# hover = HoverTool()
# hover.tooltips = [
#     ("Team", "@Team"),
#     ("Attacking", "@attack"),
#     ("Attacking sd", "@attacksd"),
#     ("Defending", "@defend"),
#     ("Defending sd", "@defendsd"),
# ]
# p.add_tools(hover)

# show(p)

# # Predicted goals and table
# predict = df[df["split"] == "predict"]

# predictive = Predictive(model, fit, return_sites=["s1", "s2"])

# predicted_score = predictive(
#     random.PRNGKey(0),
#     home_id=predict["Home_id"].values,
#     away_id=predict["Away_id"].values,
# )

# predicted_full = predict.copy()
# predicted_full = predicted_full.assign(
#     score1=predicted_score["s1"].mean(axis=0).round(),
#     score1error=predicted_score["s1"].std(axis=0),
#     score2=predicted_score["s2"].mean(axis=0).round(),
#     score2error=predicted_score["s2"].std(axis=0),
# )

# predicted_full = train.append(
#     predicted_full.drop(columns=["score1error", "score2error"])
# )


# def score_table(df):
#     """Function to convert football match dataframe to a table

#     Keyword arguments:
#     df -- matches dataframe with columns Home, score1, score2, Away
#     """
#     df = df.assign(
#         HomePoints=np.select(
#             [df["score1"] > df["score2"], df["score1"] < df["score2"]],
#             [3, 0],
#             default=1,
#         ),
#         AwayPoints=np.select(
#             [df["score2"] > df["score1"], df["score2"] < df["score1"]],
#             [3, 0],
#             default=1,
#         ),
#         HomeGD=df["score1"] - df["score2"],
#         AwayGD=df["score2"] - df["score1"],
#     )

#     home_df = (
#         df[["Home", "HomePoints", "HomeGD"]].groupby("Home").sum().rename_axis("Team")
#     )
#     away_df = (
#         df[["Away", "AwayPoints", "AwayGD"]].groupby("Away").sum().rename_axis("Team")
#     )

#     df = pd.merge(home_df, away_df, left_index=True, right_index=True)

#     df = df.assign(
#         Points=df["HomePoints"] + df["AwayPoints"], GD=df["HomeGD"] + df["AwayGD"]
#     )

#     df = df[["Points", "GD"]]

#     df = df.sort_values(["Points", "GD"], ascending=[False, False])
#     return df


# score_table(pl_data)
# score_table(predicted_full)
