using CSV, DataFrames
using Random:seed!
using Distributions, Turing
using ArviZ
using PyPlot, Gadfly

seed!(1)

Turing.setadbackend(:forwarddiff)
# Turing.setadbackend(:zygote)
# Turing.setadbackend(:reversediff)

pl_data = DataFrame(CSV.File("data/premierleague.csv"))

ng   = nrow(pl_data)  # number of games
npr  = 50  # predict the last 5 rounds of games
ngob = ng - npr  # number of games to train

teams = unique(pl_data[!, :Home])
nt    = size(teams)[1, ]  # number of teams
teams = DataFrame(Team=teams, i=1:nt)

df = innerjoin(pl_data, teams, on=(:Home => :Team))
rename!(df, :i => :Home_id)
df = innerjoin(df, teams, on=(:Away => :Team))
rename!(df, :i => :Away_id)

df[!, :split] = map(df[!, :Round]) do r
    r <= 33 ? "train" : "predict"
end

train = df[df.split .== "train", :]

@model model(s1, s2, home_id, away_id; nt=length(unique(home_id))) = begin
    # priors
    α ~ Normal(0, 1)
    σ_att ~ truncated(LocationScale(0.0, 2.5, TDist(3)), 0, Inf)
    σ_def ~ truncated(LocationScale(0.0, 2.5, TDist(3)), 0, Inf)

    home ~ Normal(0, 1) # home advantage

    # team-specific model parameters
    attack ~ filldist(Normal(0, σ_att), nt)
    defend ~ filldist(Normal(0, σ_def), nt)

    # Likelihood
    θ_1 = @. exp(α + home + attack[home_id] - defend[away_id])
    θ_2 = @. exp(α + attack[away_id] - defend[home_id])

    s1 ~ arraydist(Poisson.(θ_1))
    s2 ~ arraydist(Poisson.(θ_2))
end;

m = model(
    train[!, :score1],
    train[!, :score2],
    train[!, :Home_id],
    train[!, :Away_id]
)

@time _ = sample(m, NUTS(), 1) # compile

@time fit = sample(
    m,
    NUTS(), # DynamicNUTS(),
    MCMCThreads(),
    1000, # 15000,
    1, # 2,
    # discard_initial=5000,
    thinning=1,
    progress=true
)

# Plot posterior
plot_forest(
    fit,
    var_names=("α", "home", "σ_att", "σ_def")
);
gcf()

plot_trace(
    fit,
    var_names=("α", "home", "σ_att", "σ_def")
);
gcf()

idata = from_mcmcchains(
    fit;
    library="Turing"
)

summary = summarystats(idata)

quality = copy(teams)
quality[!, :attack]   = filter(r -> any(occursin.(["attack"], r.variable)), summary)[!, :mean]
quality[!, :attacksd] = filter(r -> any(occursin.(["attack"], r.variable)), summary)[!, :sd]
quality[!, :defend]   = filter(r -> any(occursin.(["defend"], r.variable)), summary)[!, :mean]
quality[!, :defendsd] = filter(r -> any(occursin.(["defend"], r.variable)), summary)[!, :sd]

Gadfly.plot(
    quality,
    x=:attack, y=:defend,
    xmin=quality[!, :attack] - quality[!, :attacksd], xmax=quality[!, :attack] + quality[!, :attacksd],
    ymin=quality[!, :defend] - quality[!, :defendsd], ymax=quality[!, :defend] + quality[!, :defendsd],
    label=:Team,
    Geom.point,
    Geom.label(position=:centered),
    Geom.errorbar
)

# Predicted goals and table
pred = df[df.split .== "predict", :]
model_missing = model(
    Vector{Missing}(missing, npr),
    Vector{Missing}(missing, npr),
    pred[!, :Home_id],
    pred[!, :Away_id]
)

m_pred = DynamicPPL.Model{(:s1, :s2,)}(
    :m_pred,
    model_missing.f,
    model_missing.args,
    model_missing.defaults
)

predictions = predict(m_pred, fit)

summary = DataFrame(summarystats(predictions))
pred[!, :score1]   = @. convert(Int, round(summary[1:npr, :mean]))
pred[!, :score1sd] = summary[1:npr, :std]
pred[!, :score2]   = @. convert(Int, round(summary[npr + 1:end, :mean]))
pred[!, :score2sd] = summary[npr + 1:end, :std]

predicted_full = vcat(select(train, [:Round, :Home, :score1, :score2, :Away]), select(pred, [:Round, :Home, :score1, :score2, :Away]))

#  Final table – see how well the model predicts the final 50 games
include("utils.jl")
score_table(pl_data)
score_table(predicted_full)
