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

fit_new <- stan_glmer(
    formula = score ~ 1 + Home + (1|attacking) + (1|defending),
    family = poisson,
    data   = data %>% filter(split == "train"),
    iter   = 15000,
    warmup = 5000,
    chains = 2,
    thin   = 1
)

