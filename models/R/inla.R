library(tidyverse)
library(INLA)
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

np <- 50 # predict the last 5 rounds of games
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

fit <- inla(
    score ~ 1 + Home +
        f(attacking, model = "iid") + f(defending, model = "iid"),
    family = "poisson",
    control.predictor = list(link = 1),
    data = data %>% mutate(score = ifelse(split == "predict", NA, score))
)

summary(fit)

fit$summary.fitted.values$sd[ngob:ng]

# Attack and defence
quality <- tibble(
    Team      = teams,
    attack    = fit$summary.random$attacking$mean,
    attacksd  = fit$summary.random$attacking$sd,
    defence   = fit$summary.random$defending$mean,
    defencesd = fit$summary.random$defending$sd
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
    mutate(
        score = ifelse(
            split == "predict",
            fit$summary.fitted.values$mean, score
        ),
        score_error = fit$summary.fitted.values$sd
    ) %>%
    filter(split == "predict")

predicted_full <- bind_rows(
    data %>% filter(split == "train"),
    predicted %>%
        select(Round, game.id, attacking, defending, score, split, Home)
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

#  Final table – see how well the model predicts the final 50 games
source("models/R/utils.R")
score_table(pl_data)
score_table(predicted_full)