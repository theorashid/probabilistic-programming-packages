# probabilistic-programming-packages
Trying out different probabilistic programming languages on the same statistical model.

## _Current_ results
These are __very__ preliminary results based on the minimum effective sample size (ESS) across all parameters. All PPLs were run in Colab. More PPLs will be added, models and evaluation metrics will be improved by collaborating with devs, and the hardware will be swapped out for something more consistent.

Please get in touch if you'd like to help with this project, either [here](mailto:theoaorashid@gmail.com?subject=ppl%20project), [here](https://twitter.com/theorashid), or (even better) create a pull request. See [this post](https://theorashid.github.io/post/ppl-benchmark-help/) for more information.

PPL             | compile time (s) | CPU ESS/second | GPU ESS/second
--------------- | ---------------- | -------------- | --------------
stan            | 14.8             | 181.1          | –
nimble          | 7.7              | 46.2           | –
JAGS            | 5.0              | 405.9          | –
greta           | 9.3              | 12.1           | -
PyMC            | 13.0             | 56.2           | –
PyMC + blackjax | -                | -              | –
numpyro         | 7.2              | 293.8          | 10.9
tfp             | 22.9             | 192.0          | 2.4
beanmachine     | -                | 5.9            | –
Turing          | 14.9             | 16.3           | –

## Packages
Implemented:
- [`rstan`](https://mc-stan.org/rstan/)
- [`nimble`](https://r-nimble.org)
- [`JAGS`](https://mcmc-jags.sourceforge.io)
- [`rstanarm`](https://mc-stan.org/rstanarm/)
- [`brms`](https://paul-buerkner.github.io/brms/)
- [`greta`](https://greta-stats.org)
- [`INLA`](https://www.r-inla.org)
- [`PyMC`](http://docs.pymc.io)
- [`blackjax`](https://blackjax-devs.github.io/blackjax/)
- [`pyro`](http://pyro.ai)
- [`numpyro`](http://num.pyro.ai/)
- [`tensorflow_probability`](https://www.tensorflow.org/probability/)
- [`Bean Machine`](https://beanmachine.org/)
- [`Turing`](https://turing.ml/)

To do:
- [`edward2`](https://github.com/google/edward2)
- [`Gen`](https://www.gen.dev)
- [`Soss`](https://cscherrer.github.io/Soss.jl/stable/)
- [`MCX`](https://github.com/rlouf/mcx)
- [`aeppl`](https://github.com/aesara-devs/aeppl)

## Model
We fit a hierarchical model to predict football results over a Premier League season. The idea is taken from the Stan's video on [Hierarchical Modelling in Stan: Predicting the Premier League](https://www.youtube.com/watch?v=dNZQrcAjgXQ), which itself is inspired by the paper [Bayesian hierarchical model for the prediction of
football results](https://discovery.ucl.ac.uk/id/eprint/16040/1/16040.pdf) by Baio and Blangiardo in 2010.

The [data](https://github.com/openfootball/england) are the 380 matches of the 2019/20 Premier League season, consisting of the home team, the away team and the scoreline.

The number of goals for the home or away team in a match follows a Poisson distribution. The rate is modelled as a log-linear multilevel random effect model with an attacking parameter for each team, a defending parameter for each team, and a home advantage parameter in the home goals likelihood.
