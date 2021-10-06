from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, CategoricalColorMapper
from bokeh.transform import factor_cmap
from bokeh.models.tools import HoverTool

simulation = np.arange(2000)
chelsea_goals = np.array(predicted_score["s1"][:, 20])
norwich_goals = np.array(predicted_score["s2"][:, 20])

GD = chelsea_goals - norwich_goals
win = np.where(GD > 0, "win", "lose")
winc = np.where(GD > 0, "#034694", "#FFF200")


df = pd.DataFrame(data=dict(simulation=simulation, GD=GD, win=win))
source = ColumnDataSource(df)

color_map = CategoricalColorMapper(
    factors=df["win"].unique(), palette=["#034694", "#FFF200"]
)

p = figure(
    title="Chelsea FC vs Norwich City FC",
    y_range=[0, 2000],
    x_range=[-5, 9],
    # tooltips = [("Team", "@win")]
)

p.segment(
    0,
    "simulation",
    "GD",
    "simulation",
    width=0.9,
    line_color={"field": "win", "transform": color_map},
    source=source,
)
p.xaxis.axis_label = "Simulated match goal difference"
p.xaxis.minor_tick_line_color = None  # turn off y-axis minor ticks
p.yaxis.minor_tick_line_color = None  # turn off y-axis minor ticks
p.yaxis.major_tick_line_color = None  # turn off y-axis major ticks
p.yaxis.major_label_text_color = (
    None  # note that this leaves space between the axis and the axis label
)

show(p)
# %%
# %%
