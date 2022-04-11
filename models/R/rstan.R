library(tidyverse)
library(rstan)
library(posterior)
library(bayesplot)

set.seed(1)
options(mc.cores = 2)

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

inputs <- list(
    nt = nt,
    ng = ngob,
    ht = data %>% filter(split == "train") %>% pull(Home.id),
    at = data %>% filter(split == "train") %>% pull(Away.id),
    s1 = data %>% filter(split == "train") %>% pull(score1),
    s2 = data %>% filter(split == "train") %>% pull(score2)
)

fit <- stan(
    file   = "models/R/stan.stan",
    data   = inputs,
    iter   = 15000,
    warmup = 5000,
    chains = 2,
    thin   = 1
)

# Plot posterior
posterior <- as_draws_array(fit)
dimnames(posterior)
mcmc_intervals(
    posterior,
    pars = c("alpha", "home", "sd_att", "sd_def")
)
mcmc_trace(
    posterior,
    pars = c("alpha", "home", "sd_att", "sd_def"),
    facet_args = list(ncol = 1)
)

# Extract samples
samples <- fit %>% extract()

# Attack and defence
quality <- tibble(
    Team      = teams,
    attack    = samples %>% pluck("att") %>% colMeans(),
    attacksd  = samples %>% pluck("att") %>% apply(2, sd),
    defence   = samples %>% pluck("def") %>% colMeans(),
    defencesd = samples %>% pluck("def") %>% apply(2, sd)
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
# Simulate from the posterior to get predicted scores
predicted <- data %>% filter(split == "predict")

s1 <- c()
s2 <- c()
for (i in 1:np) {
    h <- predicted[i, ] %>% pull(Home.id)
    a <- predicted[i, ] %>% pull(Away.id)

    theta1 <- exp(
        samples$alpha + samples$home + samples$att[, h] - samples$def[, a]
    )
    theta2 <- exp(
        samples$alpha + samples$att[, a] - samples$def[, h]
    )

    s1 <- c(s1, rpois(length(theta1), theta1))
    s2 <- c(s2, rpois(length(theta2), theta2))
}
s1 <- matrix(s1, ncol = np) #  iterations are rows, columns are parameters
s2 <- matrix(s2, ncol = np)

predicted <- data %>%
    filter(split == "predict") %>%
    mutate(score1true = score1, score2true = score2) %>%
    mutate(
        score1      = s1 %>% colMeans(),
        score1error = s1 %>% apply(2, sd),
        score2      = s2 %>% colMeans(),
        score2error = s2 %>% apply(2, sd),
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