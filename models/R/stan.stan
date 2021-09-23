data {
    int<lower=0> nt; //number of teams
    int<lower=0> ng; //number of games

    int<lower=0> ht[ng]; //home team index
    int<lower=0> at[ng]; //away team index

    int<lower=0> s1[ng]; //score home team
    int<lower=0> s2[ng]; //score away team
}

parameters {
    real alpha; //overall intercept
    real home; //home advantage
    vector[nt] att; //attack ability of each team
    vector[nt] def; //defence ability of each team

    //hyper parameters
    real<lower=0> sd_att;
    real<lower=0> sd_def;
}

transformed parameters {
    vector[ng] theta1; //score probability of home team
    vector[ng] theta2; //score probability of away team

    theta1 = exp(alpha + home + att[ht] - def[at]);
    theta2 = exp(alpha + att[at] - def[ht]);
}

model {
    //hyper priors
    alpha ~ normal(0,1);
    sd_att ~ student_t(3,0,2.5);
    sd_def ~ student_t(3,0,2.5);

    //priors
    att  ~ normal(0, sd_att);
    def  ~ normal(0, sd_def);
    home ~ normal(0,1);

    //likelihood
    s1 ~ poisson(theta1);
    s2 ~ poisson(theta2);
}
