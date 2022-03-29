<h1> <img style="height:5em;" alt="batty" src="https://raw.githubusercontent.com/philippeller/batty/main/batty_logo.svg"/> </h1> 

# BAT to Python (batty)

A small python interface to the Bayesian Analysis Toolkit (BAT.jl) https://github.com/bat/BAT.jl

# Quick Start

## Installation

There are two parts to an installation, one concerning the python side, and one the julia side:

* Python: `pip install batty`

* Julia: `import Pkg; Pkg.add.(["PyJulia", "DensityInterface", "Distributions", "ValueShapes", "TypedTables", "ArraysOfArrays", "BAT"])`

## Minimal Example

The code below is showing a minimal example:
* using a gaussian likelihood and a uniform prior
* generating samples via Metropolis-Hastings
* plotting the resulting sampes
* estimating the integral value via BridgeSampling


```python
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from batty import BAT_sampler, BAT, Distributions
```


```python
sampler = BAT_sampler(llh=lambda x : -0.5 * x**2, prior_specs=Distributions.Uniform(-3, 3))

sampler.sample();

sampler.corner();
```

    /mnt/c/Users/peller/work/batty/batty.py:6: UserWarning: Not able to use compiled modules, resulting in (very) slow import
     See https://pyjulia.readthedocs.io/en/latest/troubleshooting.html
      warnings.warn("Not able to use compiled modules, resulting in (very) slow import\n See https://pyjulia.readthedocs.io/en/latest/troubleshooting.html")



```python
sampler.integrate()
```




    0.416747877274376+/-0.00021004447293448817



# Usage

## Using Different Algotihms

There are a range of algorihtms available within BAT, and those can be further customized via arguments. Here are just a few examples:

### Integration:

* AHMI:


```python
sampler.integrate(strategy=BAT.AHMIntegration())
```




    0.4189844836634516+/-0.0015790934281673487



* Bridge Sampling:


```python
sampler.integrate(strategy=BAT.BridgeSampling())
```




    0.41675559744691054+/-0.0002100721115029865



### MCMC Sampling


```python
results = {}
```

* Metropolis-Hastings:


```python
results['Metropolis-Hastings'] = sampler.sample(strategy=BAT.MCMCSampling(nsteps=10_000, nchains=2))
```

* Metropolis-Hastings with Accept-Reject weighting:


```python
results['Accept-Reject Weighting'] = sampler.sample(strategy=BAT.MCMCSampling(mcalg=BAT.MetropolisHastings(weighting=BAT.ARPWeighting()), nsteps=10_000, nchains=2))
```

* Prior Importance Sampling:


```python
results['Prior Importance Sampling'] = sampler.sample(strategy=BAT.PriorImportanceSampler(nsamples=10_000))
```

* Sobol Sampler:


```python
results['Sobol Quasi Random Numbers'] = sampler.sample(strategy=BAT.SobolSampler(nsamples=10_000))
```

* Grid Sampler:


```python
results['Grid Points'] = sampler.sample(strategy=BAT.GridSampler(ppa=1000))
```


```python
fig = plt.figure(figsize=(9,6))
bins=np.linspace(-3, 3, 100)
for key, item in results.items():
    plt.hist(item.v, weights=item.weight, bins=bins, density=True, histtype="step", label=key);
plt.legend()
```




    <matplotlib.legend.Legend at 0x7f9650197dc0>




    
![png](https://raw.githubusercontent.com/philippeller/batty/main/README_files/README_21_1.png)
    


# Specifying Priors and Likelihoods

Priors are specified via Julia Distributions, multiple Dimensions can be defined via a dict, where the key is the dimension name


```python
s = np.array([[0.25, 0.4], [0.9, 0.75]])
prior_specs = {'a' : Distributions.Uniform(-3,3), 'b' : Distributions.MvNormal([1,1], s@s.T)}
```

The log-likelihood (llh) can be any python callable, that returns the log-likelihood values. The first argument to the function is the object with paramneter values, here `x`. If the prior is simple (i.e. like in the example in the beginning, `x` is directly the parameter value). If the prior is specified via a dict, then `x` contains a field per parameter with the value.
Any additional args to the llh can be specified in the sampler, such as here `d` for data:


```python
def llh(x, d):
    return -0.5 * ((x.b[0] - d[0])**2 + (x.b[1] - d[1])**2/4) - x.a
```


```python
d = [-1, 1]
```


```python
sampler = BAT_sampler(llh, prior_specs, llh_args=(d,), progress_bar=True)
```

Let us generate a few samples:


```python
sampler.sample(strategy=BAT.MCMCSampling(nsteps=10_000, nchains=2));
```

    llh at     2.0605: : 35132it [01:32, 306.96it/s]     

### Some interface to plotting tools are available

* The **G**reat **T**riangular **C**onfusion (GTC) plot:


```python
sampler.gtc(figureSize=8, customLabelFont={'size':14}, customTickFont={'size':10});
```

    findfont: Font family ['Arial'] not found. Falling back to DejaVu Sans.
    findfont: Font family ['Arial'] not found. Falling back to DejaVu Sans.



    
![png](https://raw.githubusercontent.com/philippeller/batty/main/README_files/README_31_1.png)
    


* The corner plot:


```python
sampler.corner(color='green');
```


    
![png](https://raw.githubusercontent.com/philippeller/batty/main/README_files/README_33_0.png)
    



```python
# does not work
#sampler.sample(strategy=BAT.MCMCSampling(nsteps=1000, nchains=2, mcalg=BAT.HamiltonianMC()));
```


```python
#takes way too long, something wrong
#sampler.sample(strategy=BAT.PartitionedSampling(npartitions=2, sampler=BAT.MCMCSampling(nchains=2, nsteps=100, strict=False), exploration_sampler=BAT.MCMCSampling(nchains=2, nsteps=100, strict=False)))
```
