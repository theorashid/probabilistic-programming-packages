"""Convert Home, score1, score2, Away -> final table
"""

import pandas as pd
import numpy as np

__author__ = "Theo Rashid"
__email__ = "tar15@ic.ac.uk"


def score_table(df):
    """Function to convert football match dataframe to a table

    Keyword arguments:
    df -- matches dataframe with columns Home, score1, score2, Away
    """
    df = df.assign(
        HomePoints=np.select(
            [df["score1"] > df["score2"], df["score1"] < df["score2"]],
            [3, 0],
            default=1,
        ),
        AwayPoints=np.select(
            [df["score2"] > df["score1"], df["score2"] < df["score1"]],
            [3, 0],
            default=1,
        ),
        HomeGD=df["score1"] - df["score2"],
        AwayGD=df["score2"] - df["score1"],
    )

    home_df = (
        df[["Home", "HomePoints", "HomeGD"]].groupby("Home").sum().rename_axis("Team")
    )
    away_df = (
        df[["Away", "AwayPoints", "AwayGD"]].groupby("Away").sum().rename_axis("Team")
    )

    df = pd.merge(home_df, away_df, left_index=True, right_index=True)

    df = df.assign(
        Points=df["HomePoints"] + df["AwayPoints"], GD=df["HomeGD"] + df["AwayGD"]
    )

    df = df[["Points", "GD"]]

    df = df.sort_values(["Points", "GD"], ascending=[False, False])
    return df
