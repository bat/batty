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
import awkward as ak

Main.eval("using TypedTables")
Main.eval("using ValueShapes")
Main.eval("using ArraysOfArrays")

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
        result : awkward.Array
            Containing arrays of v, weight, and logd
        
        '''
        Main.result = BAT.bat_sample(self.posterior, BAT.MCMCSampling(mcalg=self.mcalg, nsteps=int(nsteps), **kwargs))
        self._result = Main.result
        
        # define viws of arrays
        result = {}
        result['weight'] = Main.eval("Array(result.result.weight)")
        result['logd'] = Main.eval("Array(result.result.logd)")
        if isinstance(self.prior_specs, dict):
            v = Main.eval("Dict(pairs(map(c -> Array(flatview(unshaped.(c))), columns(result.result.v))))")
            result['v'] = ak.Array({key:np.squeeze(v[key].T) for key in v.keys()})
        else:
            result['v'] = Main.eval("Array(result.result.v)")

        self.result = ak.Array(result)
            
        return self.result