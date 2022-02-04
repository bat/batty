from julia.core import UnsupportedPythonError
try:
    from julia import BAT, DensityInterface, Distributions, ValueShapes, Main
except UnsupportedPythonError:
    import warnings
    warnings.warn("Not able to use compiled modules, resulting in (very) slow import\n See https://pyjulia.readthedocs.io/en/latest/troubleshooting.html")
    from julia.api import Julia
    jl = Julia(compiled_modules=False)
    from julia import BAT, DensityInterface, Distributions, ValueShapes, Main

from collections import namedtuple

import numpy as np

BAT_result = namedtuple('BAT_result', ['samples', 'weight', 'logd'])

class BAT_sampler():
    
    def __init__(self, llh, prior_specs, llh_args=(), mcalg=BAT.MetropolisHastings()):
        '''BAT_sampler Python wrapper
        
        Paramaters:
        -----------
        llh : callable
            The likelihood function
        prior_specs : BAT prior, or dictionary thereof
            The prior specifications
        llh_args : tuple
            additional arguments to the llh function (optional)
        mcalg : BAT MC algorithm (optional)

        
        '''
        self.llh = DensityInterface.logfuncdensity(lambda x : llh(x, *llh_args))
        self.prior_specs = prior_specs
        if isinstance(self.prior_specs, dict):
            self.prior = ValueShapes.NamedTupleDist(**prior_specs)
            self.result_tuple = namedtuple('result_tuple', prior_specs.keys())
        else:
            self.prior = prior_specs
        self.posterior = BAT.PosteriorDensity(self.llh, self.prior)
        self.mcalg = mcalg
        self._result = None
        self.result = None
        
    def run(self, nsteps=1e5, **kwargs):
        '''
        Run the Sampler
        
        Paramaters:
        -----------
        nsteps : int
            Number of MCMC sampling steps
        kwargs : dict (optional)
            additional keyword arguments to the sampler
        
        Returns:
        --------
        result : namedtuple('BAT_result', ['samples', 'weight', 'logd'])
            Containing arrays of samples, weights, and llh values
        
        '''
        Main.result = BAT.bat_sample(self.posterior, BAT.MCMCSampling(mcalg=self.mcalg, nsteps=int(nsteps), **kwargs))
        self._result = Main.result
        
        # define viws of arrays
        if isinstance(self.prior_specs, dict):
            samples = {}
            for key in self.prior_specs.keys():
                if isinstance(Main.eval('result.result.v.%s[1]'%key), list):
                    samples[key] = Main.eval('convert(Array{Float64}, reduce(hcat, result.result.v.%s))'%key)                    
                else:
                    samples[key] = Main.eval('convert(Array{Float64}, result.result.v.%s)'%key)
            samples = self.result_tuple(**samples)
        else:
            if isinstance(Main.eval('result.result.v[1]'), list):
                samples = Main.eval('convert(Array{Float64}, reduce(hcat, result.result.v))')
            else:
                samples = Main.eval('convert(Array{Float64}, result.result.v)')
                    
        weight = Main.eval('convert(Array{Float64}, result.result.weight)')
        logd = Main.eval('convert(Array{Float64}, result.result.logd)')      
    
        self.result = BAT_result(samples, weight, logd)
        
        return self.result