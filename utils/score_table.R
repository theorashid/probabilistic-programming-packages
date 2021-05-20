library(dplyr)

score_table <- function(data) {
    # Convert Home, score1, score2, Away -> final table
    data <- data %>%
        mutate(
            HomePoints = case_when(
                score1 >  score2 ~ 3,
                score1 <  score2 ~ 0,
                score1 == score2 ~ 1
            ),
            AwayPoints = case_when(
                score2 >  score1 ~ 3,
                score2 <  score1 ~ 0,
                score2 == score1 ~ 1
            ),
            HomeGD = score1 - score2,
            AwayGD = score2 - score1,
        )

    table <- left_join(
        data %>% 
            group_by(Home) %>%
            summarise(HomePoints = sum(HomePoints), HomeGD = sum(HomeGD)) %>%
            rename(Team = Home),
        data %>% 
            group_by(Away) %>%
            summarise(AwayPoints = sum(AwayPoints), AwayGD = sum(AwayGD)) %>%
            rename(Team = Away)
    ) %>%
        mutate(Points = HomePoints + AwayPoints, GD = HomeGD + AwayGD) %>%
        arrange(-Points, -GD) %>%
        select(Team, Points, GD)

    return(table)
}

