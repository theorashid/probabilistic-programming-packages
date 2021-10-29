# probabilistic-programming-packages
Trying out different probabilistic programming packages on the same statistical model

## Packages
Implemented:
- [`rstan`](https://mc-stan.org/rstan/)
- [`nimble`](https://r-nimble.org)
- [`JAGS`](https://mcmc-jags.sourceforge.io)
- [`rstanarm`](https://mc-stan.org/rstanarm/)
- [`brms`](https://paul-buerkner.github.io/brms/)
- [`INLA`](https://www.r-inla.org)
- [`PyMC3`](http://docs.pymc.io)
- [`pyro`](http://pyro.ai)
- [`numpyro`](http://num.pyro.ai/)
- [`Turing`](https://turing.ml/)

To do:
- [`blackjax`](https://github.com/blackjax-devs/blackjax)
- [`tensorflow_probability`](https://www.tensorflow.org/probability/)
- [`edward2`](https://github.com/google/edward2)
- [`Gen`](https://www.gen.dev)
- [`Soss`](https://cscherrer.github.io/Soss.jl/stable/)

## Model
We fit a hierarchical model to predict football results over a Premier League season. The idea is taken from the Stan's video on [Hierarchical Modelling in Stan: Predicting the Premier League](https://www.youtube.com/watch?v=dNZQrcAjgXQ), which itself is inspired by the paper [Bayesian hierarchical model for the prediction of
football results](https://discovery.ucl.ac.uk/id/eprint/16040/1/16040.pdf) by Baio and Blangiardo in 2010.

The [data](https://github.com/openfootball/england) are the 380 matches of the 2019/20 Premier League season, consisting of the home team, the away team and the scoreline.

The number of goals for the home or away team in a given match follows a Poisson distribution, with the rate modelled as a log-linear random effect model made up of an attacking parameter for each team, a defending parameter for each team, and a home advantage parameter in the home goals likelihood. The mean and precision of the attacking and defending parameters are each given their own hyperpriors, thus forming a hierarchical model.
