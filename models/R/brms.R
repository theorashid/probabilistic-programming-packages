library(tidyverse)
library(brms)
library(bayesplot)

set.seed(1)
options(mc.cores = 2)

pl_data <- read_csv("data/premierleague.csv")
data    <- pl_data

ng    <- nrow(data) # number of games
nt    <- data %>% pull(Home) %>% n_distinct() # number of teams
teams <- data %>% pull(Home) %>% unique() %>% sort() # names of the teams

np   <- 50 # predict the last 5 rounds of games
ngob <- ng - np # number of games to train

data <- data %>%
    mutate(split = ifelse(row_number() <= ngob, "train", "predict"))

# reformulate as a univariate problem
data <- data %>% rowid_to_column("game.id")
data <- bind_rows(
    data %>% 
        select(Round, game.id, Home, Away, score1, split) %>% 
        rename(score = score1, attacking = Home, defending = Away) %>%
        mutate(Home = TRUE),
    data %>% 
        select(Round, game.id, Home, Away, score2, split) %>% 
        rename(score = score2, attacking = Away, defending = Home) %>%
        mutate(Home = FALSE),
) %>%
    arrange(Round, game.id)

fit <- get_prior(
    score ~ 1 + Home + (1|attacking) + (1|defending),
    family = poisson,
    prior  = prior(
        prior("normal(0, 1)", class="Intercept"),
        set_prior("normal(0, 1)", class="b"),
        #Create prior for scale parameters in hierarchical models
        set_prior("student_t(4,0,1)", class="sd"),
        prior("student_t(4,0,1)", class="sigma")
    ),
    data   = data %>% filter(split == "train"),
    iter   = 15000,
    warmup = 5000,
    chains = 2,
    thin   = 1
)

# SORT PRIORS AND QUALITY EXTRACTION

posterior <- as.array(fit)
mcmc_intervals(posterior, pars = c("sd_attacking__Intercept", "sd_defending__Intercept", "b_HomeTRUE"))
mcmc_trace(posterior, pars = c("mu_att", "mu_def", "sd_att", "sd_def", "home"), facet_args = list(ncol = 1))

samples <- fit %>% extract()


prior_brms <- c(
set_prior("normal(0, 1)", class="Intercept"),
set_prior("normal(0, 1)", class="b"),
#Create prior for scale parameters in hierarchical models
set_prior("student_t(4,0,1)", class="sd"),
set_prior("student_t(4,0,1)", class="sigma")
)

# Predicted goals and table
predicted_score <- predict(
    fit, 
    newdata = data %>% filter(split == "predict") %>% select(attacking, defending, Home)
)

predicted <- data %>%
    filter(split == "predict") %>%
    mutate(scoretrue = score) %>%
    mutate(
        score       = predicted_score[, "Estimate"],
        score_error = predicted_score[, "Est.Error"]
    )

predicted_full <- bind_rows(
    data %>% filter(split == "train"),
    predicted %>% select(Round, game.id, attacking, defending, score, split, Home)
)

predicted_full <- left_join(
    predicted_full %>%
        filter(Home) %>%
        select(-Home) %>%
        rename(Home = attacking, Away = defending, score1 = score),
    predicted_full %>%
        filter(!Home) %>%
        select(-Home) %>%
        rename(Away = attacking, Home = defending, score2 = score)
) %>%
    select(Round, Home, score1, score2, Away)

# Final table – see how well the model predicts the final 50 games
source("utils/score_table.R")
score_table(pl_data)
score_table(predicted_full)
