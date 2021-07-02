"""Run Premier League prediction model using pyro
"""
# %%
import pandas as pd
import numpy as np
import pyro
import pyro.distributions as dist
import torch

from torch.distributions import constraints
from pyro.infer import SVI, Trace_ELBO

# from pyro.contrib.autoguide import AutoDiagonalNormal
from pyro.optim import Adam

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
    mu_att = pyro.sample("mu_att", dist.Normal(0.0, 1.0))
    sd_att = pyro.sample("sd_att", dist.StudentT(3.0, 0.0, 2.5))
    mu_def = pyro.sample("mu_def", dist.Normal(0.0, 1.0))
    sd_def = pyro.sample("sd_def", dist.StudentT(3.0, 0.0, 2.5))

    home = pyro.sample("home", dist.Normal(0.0, 1.0))  # home advantage

    nt = len(np.unique(home_id))

    # team-specific model parameters
    with pyro.plate("plate_teams", nt):
        attack = pyro.sample("attack", dist.Normal(mu_att, sd_att))
        defend = pyro.sample("defend", dist.Normal(mu_def, sd_def))

    # likelihood
    theta1 = torch.exp(home + attack[home_id] - defend[away_id])
    theta2 = torch.exp(attack[away_id] - defend[home_id])

    with pyro.plate("data", len(home_id)):
        pyro.sample("s1", dist.Poisson(theta1), obs=score1_obs)
        pyro.sample("s2", dist.Poisson(theta2), obs=score2_obs)


# %%
def guide(home_id, away_id, score1_obs=None, score2_obs=None):
    # priors
    mu_locs = pyro.param("mu_loc", torch.tensor(0.0).expand(3))
    mu_scales = pyro.param(
        "mu_scale", torch.tensor(0.1).expand(3), constraint=constraints.positive
    )

    sd_dfs = pyro.param(
        "sd_df", torch.tensor(2.0).expand(3), constraint=constraints.positive
    )
    sd_scales = pyro.param(
        "sd_scale", torch.tensor(0.1).expand(3), constraint=constraints.positive
    )

    pyro.sample("mu_att", dist.Normal(mu_locs[0], mu_scales[0]))
    pyro.sample("mu_def", dist.Normal(mu_locs[1], mu_scales[1]))

    pyro.sample("sd_att", dist.StudentT(sd_dfs[0], torch.tensor(0.0), sd_scales[0]))
    pyro.sample("sd_def", dist.StudentT(sd_dfs[0], torch.tensor(0.0), sd_scales[1]))

    pyro.sample("home", dist.Normal(mu_locs[2], mu_scales[2]))  # home advantage

    nt = len(np.unique(home_id))

    mu_team_locs = pyro.param("mu_team_loc", torch.tensor(0.0).expand(2))
    mu_team_scales = pyro.param(
        "mu_team_scale", torch.tensor(0.1).expand(2), constraint=constraints.positive
    )

    # team-specific model parameters
    with pyro.plate("plate_teams", nt):
        pyro.sample("attack", dist.Normal(mu_team_locs[0], mu_team_scales[0]))
        pyro.sample("defend", dist.Normal(mu_team_locs[1], mu_team_scales[1]))


# guide = AutoDiagonalNormal(model)

# %%
svi = SVI(model=model, guide=guide, optim=Adam({"lr": 0.001}), loss=Trace_ELBO())

# %%
pyro.clear_param_store()  # clear global parameter cache
pyro.set_rng_seed(1)

num_iterations = 1000
advi_loss = []
for j in range(num_iterations):
    # calculate the loss and take a gradient step
    loss = svi.step(
        home_id=torch.tensor(train["Home_id"]),
        away_id=torch.tensor(train["Away_id"]),
        score1_obs=torch.tensor(train["score1"]),
        score2_obs=torch.tensor(train["score2"]),
    )
    advi_loss.append(loss)
    if j % 100 == 0:
        print("[iteration %4d] loss: %.4f" % (j + 1, loss))


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
# %%
guide.get_posterior().sample((1000,))
# fit = az.from_numpyro(mcmc)

# # Plot posterior
# az.plot_forest(
#     fit,
#     var_names=("mu_att", "mu_def", "sd_att", "sd_def", "home"),
#     backend="bokeh",
# )

# az.plot_trace(
#     fit, var_names=("mu_att", "mu_def", "sd_att", "sd_def", "home"), backend="bokeh"
# )

# # %%
# fit = mcmc.get_samples()

# # Attack and defence
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
