from julia.api import Julia
jl = Julia(compiled_modules=False)

from julia import BAT, DensityInterface, Distributions, ValueShapes

from collections import namedtuple

import numpy as np

BAT_result = namedtuple('BAT_result', ['samples', 'weight', 'logd'])

class BAT_sampler():
    
    def __init__(self, llh, prior_specs, llh_args=(), mcalg=BAT.MetropolisHastings(), **kwargs):
        self.llh = DensityInterface.logfuncdensity(lambda x : llh(x, *llh_args))
        self.prior_specs = prior_specs
        if isinstance(self.prior_specs, dict):
            self.prior = ValueShapes.NamedTupleDist(**prior_specs)
        else:
            self.prior = prior_specs
        self.posterior = BAT.PosteriorDensity(self.llh, self.prior)
        self.mcalg = mcalg
        self.kwargs = kwargs
        self._result = None
        self._result_digest = None
        
    def run(self, nsteps=1e5):
        self._result = BAT.bat_sample(self.posterior, BAT.MCMCSampling(mcalg=self.mcalg, nsteps=int(nsteps), **self.kwargs))
        self._result_digest = None
    
    @property
    def result(self,):
        if self._result_digest:
            return self._result_digest
        
        if self._result is None:
            return None
        
        if isinstance(self.prior_specs, dict):
            samples = {}
            for key in self.prior_dict.keys():
                samples[key] = np.array([getattr(s.v, key) for s in self._result.result]) 
        else:
            samples = np.array([s.v for s in self._result.result])
            
        weight = np.array([s.weight for s in self._result.result])
        logd = np.array([s.logd for s in self._result.result])
        self._result_digest = BAT_result(samples, weight, logd)
        
        return  self._result_digest