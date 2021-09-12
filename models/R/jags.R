library(tidyverse)
library(R2jags)
library(bayesplot)

set.seed(1)

pl_data <- read_csv("data/premierleague.csv")
data <- pl_data

ng <- nrow(data) # number of games
nt <- data %>%
    pull(Home) %>%
    n_distinct() # number of teams
teams <- data %>%
    pull(Home) %>%
    unique() %>%
    sort() # names of the teams

data <- data %>%
    mutate(
        Home.id = data %>% group_by(Home) %>% group_indices(),
        Away.id = data %>% group_by(Away) %>% group_indices()
    )

np <- 50 # predict the last 5 rounds of games
ngob <- ng - np # number of games to train

data <- data %>%
    mutate(split = ifelse(row_number() <= ngob, "train", "predict"))

jags_data <- list(
    nt = nt,
    ng = ngob,
    ht = data %>% filter(split == "train") %>% pull(Home.id),
    at = data %>% filter(split == "train") %>% pull(Away.id),
    np = np,
    htnew = data %>% filter(split == "predict") %>% pull(Home.id),
    atnew = data %>% filter(split == "predict") %>% pull(Away.id),
    s1 = data %>% filter(split == "train") %>% pull(score1),
    s2 = data %>% filter(split == "train") %>% pull(score2)
)

jags_inits <- function() {
    list(
        "mu_att" = 0.09,
        "mu_def" = 0.00,
        "sd_att" = 0.30,
        "sd_def" = 0.19,
        "home"   = 0.24
    )
}

fit <- jags.parallel(
    data = jags_data,
    inits = jags_inits,
    parameters.to.save = c(
        "mu_att", "mu_def",
        "sd_att", "sd_def",
        "home", "att", "def",
        "s1new", "s2new"
    ),
    n.iter = 15000,
    n.burnin = 5000,
    n.chains = 2,
    n.thin = 1,
    model.file = "models/R/jags.bug"
)

samples <- fit$BUGSoutput$sims.matrix %>% as_tibble()

# Plot posterior
mcmc_intervals(
    samples,
    pars = c("mu_att", "mu_def", "sd_att", "sd_def", "home")
)
mcmc_trace(
    samples,
    pars = c("mu_att", "mu_def", "sd_att", "sd_def", "home"),
    facet_args = list(ncol = 1)
)

team_values <- samples %>% select(-c(mu_att, mu_def, sd_att, sd_def, home))

# Attack and defence
quality <- tibble(
    Team      = teams,
    attack    = team_values %>% select(matches("att")) %>% colMeans(),
    attacksd  = team_values %>% select(matches("att")) %>% apply(2, sd),
    defence   = team_values %>% select(matches("def")) %>% colMeans(),
    defencesd = team_values %>% select(matches("def")) %>% apply(2, sd)
)

quality %>%
    ggplot(aes(
        x = attack, y = defence,
        xmin = attack - attacksd, xmax = attack + attacksd,
        ymin = defence - defencesd, ymax = defence + defencesd,
        label = Team
    )) +
    geom_point(colour = "grey25") +
    geom_errorbar(colour = "grey25", alpha = 0.4) +
    geom_errorbarh(colour = "grey25", alpha = 0.4) +
    geom_text() +
    theme_minimal()

# Predicted goals and table
predicted <- data %>%
    filter(split == "predict") %>%
    mutate(score1true = score1, score2true = score2) %>%
    mutate(
        score1      = team_values %>% select(matches("s1new")) %>% colMeans(),
        score1error = team_values %>% select(matches("s1new")) %>% apply(2, sd),
        score2      = team_values %>% select(matches("s2new")) %>% colMeans(),
        score2error = team_values %>% select(matches("s2new")) %>% apply(2, sd),
    )

predicted_full <- bind_rows(
    data %>%
        filter(split == "train") %>%
        select(Round, Home, score1, score2, Away),
    predicted %>% select(Round, Home, score1, score2, Away)
) %>%
    mutate(score1 = round(score1), score2 = round(score2))

#  Final table – see how well the model predicts the final 50 games
source("models/R/utils.R")
score_table(pl_data)
score_table(predicted_full)