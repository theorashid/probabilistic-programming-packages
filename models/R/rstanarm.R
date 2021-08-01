library(tidyverse)
library(rstanarm)
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

fit <- stan_glmer(
    formula = score ~ 1 + Home + (1|attacking) + (1|defending),
    family = poisson,
    # prior_intercept  = default_prior_intercept(family),
    # prior_covariance = decov(),
    data   = data %>% filter(split == "train"),
    iter   = 15000,
    warmup = 5000,
    chains = 2,
    thin   = 1
)

posterior <- as.array(fit)
dimnames(posterior)
mcmc_intervals(posterior, pars = c("Sigma[attacking:(Intercept),(Intercept)]", "Sigma[defending:(Intercept),(Intercept)]", "HomeTRUE"))
mcmc_trace(posterior, pars = c("Sigma[attacking:(Intercept),(Intercept)]", "Sigma[defending:(Intercept),(Intercept)]", "HomeTRUE"), facet_args = list(ncol = 1))

samples <- as.data.frame(fit) %>% as_tibble()

# Attack and defence
quality <- tibble(
    Team      = teams,
    attack    = samples %>% select(matches(" attacking")) %>% colMeans(),
    attacksd  = samples %>% select(matches(" attacking")) %>% apply(2, sd),
    defence   = samples %>% select(matches(" defending")) %>% colMeans(),
    defencesd = samples %>% select(matches(" defending")) %>% apply(2, sd)
)

quality %>%
    ggplot(aes(
        x = attack, y = defence, 
        xmin = attack  - attacksd,  xmax = attack  + attacksd,
        ymin = defence - defencesd, ymax = defence + defencesd,
        label = Team
    )) +
    geom_point(colour = "grey25") +
    geom_errorbar(colour = "grey25", alpha = 0.4) +
    geom_errorbarh(colour = "grey25", alpha = 0.4) +
    geom_text() +
    theme_minimal()

# Predicted goals and table
predicted_score <- posterior_predict(
    fit, 
    newdata = data %>% filter(split == "predict") %>% select(attacking, defending, Home)
)

predicted <- data %>%
    filter(split == "predict") %>%
    mutate(scoretrue = score) %>%
    mutate(
        score       = predicted_score %>% colMeans(),
        score_error = predicted_score %>% apply(2, sd)
    )

predicted_full <- bind_rows(
    data %>% filter(split == "train"),
    predicted %>% select(Round, game.id, attacking, defending, score, split, Home)
) %>%
    mutate(score = round(score))

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
source("models/R/utils.R")
score_table(pl_data)
score_table(predicted_full)
