# Trying to benchmark probabilistic programming languages
## And a cry for help.
### 2022-01-14

__Help me__ compare probabilistic programming languages.

I have a [repo](https://github.com/theorashid/probabilistic-programming-packages) for comparing various probabilistic programming languages (PPLs) in R, python and Julia in fitting a simple hierarchical model on a football dataset.

The current results are below, but they are very preliminary and there are plenty of other PPLs which I am yet to implement successfully. I am aware comparing MCMC samplers is a [flawed exercise](https://statmodeling.stat.columbia.edu/2021/11/17/its-so-hard-to-compare-the-efficiency-of-mcmc-samplers/), and that drawing grand conclusions about the best PPL based on its performance one very specific model is dishonest. But that's why I'd like help, __particularly from PPL developers__ who can tune their model and give their language the best shot at __ranking first__.

There are a list of current issues at the bottom. Please get in touch if you'd like to help with this project, either [here](mailto:theoaorashid@gmail.com?subject=ppl%20project), [here](https://twitter.com/theorashid), or (even better) create a pull request.

Finally, if you know of an appropriate outlet for this work, such as the [site of a PPL](https://www.pymc-labs.io/blog-posts/pymc-stan-benchmark/) or a statistical blog, contact me and we can write it up. Otherwise, it will remain open only to the minimal traffic of this site.

## Current results
These are __very__ preliminary results based on the minimum effective sample size (ESS) across all parameters. Each model was run for a single chain of 10000 iterations (with 1000 warmup). All PPLs were run in [Colab](https://colab.research.google.com).

PPL     | compile time (s) | CPU ESS/second | GPU ESS/second
------- | ---------------- | -------------- | --------------
stan    | 14.8             | 181.1          | –
nimble  | 7.7              | 24.2           | –
JAGS    | 5.0              | 286.3          | –
PyMC    | 8.8              | 63.8           | –
numpyro | 7.2              | 293.8          | 10.9
Turing  | 14.9             | 16.3           | –

## Help wanted
- __Models.__ A number of PPLs have __not__ yet been __implemented__ ([tensorflow probability](https://www.tensorflow.org/probability/), [Bean Machine](https://beanmachine.org/), [edward2](https://github.com/google/edward2), [Gen](https://www.gen.dev/), [Soss](https://cscherrer.github.io/Soss.jl/stable/), [MCX](https://github.com/rlouf/mcx), [blackjax](https://github.com/blackjax-devs/blackjax)). There are also plenty (I assume) of issues with my implementations. This is where PPL experts can come in and clean up.
- __Errors.__ The JAGS standard deviation parameters seem to be off. The Turing model is running extremely slowly. PyMC are releasing their [Aesara-backed v4](https://github.com/pymc-devs/pymc/releases/tag/v4.0.0b1), which will undoubtedly speed things up. And numpyro is running _slower_ on a **G**PU?
- __Comparing samplers.__ JAGS and NIMBLE are running their default MCMC. All the other PPLs are running NUTS. Are these comparable? Do the NUTS algorithms all need the same settings? 
- __Multiple chains.__ Here, a single chain was run. We should run multiple chains to explore which PPL works best in parallel.
- __Metrics.__ Is the minimum effective sample size per second the best metric? Can we also compare qualitatively how easy each PPL was to set up? (In my opinion, Turing code reads the best.)
- __Hardware.__ All the models were run on Colab, but this should be switched out for something better-performing and more consistent.
