using CSV, DataFrames
using Random:seed!
using Distributions, Turing
using ArviZ, Pyplot

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
    μ_att ~ Normal(0, 1)
    σ_att ~ truncated(LocationScale(0.0, 2.5, TDist(3)), 0, Inf)
    μ_def ~ Normal(0, 1)
    σ_def ~ truncated(LocationScale(0.0, 2.5, TDist(3)), 0, Inf)

    home ~ Normal(0, 1) # home advantage

    # team-specific model parameters
    attack ~ filldist(Normal(μ_att, σ_att), nt)
    defend ~ filldist(Normal(μ_def, σ_def), nt)

    # Likelihood
    θ_1 = @. exp(home + attack[home_id] - defend[away_id])
    θ_2 = @. exp(attack[away_id] - defend[home_id])

    s1 ~ arraydist(Poisson.(θ_1))
    s2 ~ arraydist(Poisson.(θ_2))
end;

fit = sample(
    model(
        train[!,:score1],
        train[!,:score2],
        train[!,:Home_id],
        train[!,:Away_id]
    ),
    NUTS(), 
    MCMCThreads(),
    1000,
    1,
    discard_initial=500,
    thinning=1,
    progress=true
)

display(fit)

# JULIA IS SLOW?
# JULIA IS SLOW AND GETS QUICKER?
# PROGRESS BAR?
# BURN?
# PREDICTION

# using DynamicHMC
# fit = sample(gdemo(1.5, 2.0), DynamicNUTS(), 2000)

# Plot posterior
plot_forest(
    fit,
    var_names=("μ_att", "μ_def", "σ_att", "σ_def", "home")
);
gcf()

plot_trace(
    fit,
var_names=("μ_att", "μ_def", "σ_att", "σ_def", "home")
);
gcf()

idata = from_mcmcchains(
    fit;
    library="Turing"
)

summary = summarystats(idata)


