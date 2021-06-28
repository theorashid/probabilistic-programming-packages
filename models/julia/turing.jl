using CSV, DataFrames
using Random:seed!
using Distributions, Turing
using ArviZ, Pyplot

seed!(1)

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

@model model(s1, s2, home_id, away_id; nt=length(unique(home_id)), ng=length(home_id)) = begin
    # priors
    μ_att ~ Normal(0, 1)
    σ_att ~ truncated(Cauchy(0, 1), 0, Inf)
    μ_def ~ Normal(0, 1)
    σ_def ~ truncated(Cauchy(0, 1), 0, Inf)

    home ~ Normal(0, 1) # home advantage

    # team-specific model parameters
    attack = Vector(undef, nt)
    defend = Vector(undef, nt)
    for i = 1:nt
        attack[i] ~ Normal(μ_att, σ_att)
        defend[i] ~ Normal(μ_def, σ_def)
    end

    # Likelihood
    θ_1 = Vector(undef, ng)
    θ_2 = Vector(undef, ng)
    for i in 1:ng
        θ_1[i] = exp(home + attack[home_id[i]] - defend[away_id[i]])
        θ_2[i] = exp(attack[away_id[i]] - defend[home_id[i]])

        s1[i] ~ Poisson(θ_1[i])
        s2[i] ~ Poisson(θ_2[i])
    end
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
    2
)

# JULIA IS SLOW?
# JULIA IS SLOW AND GETS QUICKER?
# PROGRESS BAR?
# BURN?
# PREDICTION

# using Distributed
# addprocs(2)
# fit = sample(model(a, b, c, d), NUTS(), MCMCDistributed(), 1000, 2)

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


