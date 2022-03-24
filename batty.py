from julia.core import UnsupportedPythonError
try:
    from julia import BAT, DensityInterface, Distributions, ValueShapes, Main
except UnsupportedPythonError:
    import warnings
    warnings.warn("Not able to use compiled modules, resulting in (very) slow import\n See https://pyjulia.readthedocs.io/en/latest/troubleshooting.html")
    from julia.api import Julia
    jl = Julia(compiled_modules=False)
    from julia import BAT, DensityInterface, Distributions, ValueShapes, Main

import uncertainties
import numpy as np
import awkward as ak

Main.eval("using TypedTables")
Main.eval("using ValueShapes")
Main.eval("using ArraysOfArrays")

class BAT_sampler():
    
    def __init__(self, llh, prior_specs, llh_args=()):
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
        self.result = None
        self.sampled_density = None
        
    def sample(self, strategy=BAT.MCMCSampling()):
        '''
        Run the Sampler
        
        Paramaters:
        -----------
        strategy : BAT sampling algorithm
        
        Returns:
        --------
        samples : awkward.Array
            Containing arrays of v, weight, and logd
        '''
        
        strategy
        
        Main.samples = BAT.bat_sample(self.posterior, strategy)
        self._samples = Main.samples
        
        Main.posterior = self.posterior
        self.sampled_density = Main.eval("BAT.SampledDensity(posterior, samples.result)")
        
        # define viws of arrays
        samples = {}
        samples['weight'] = Main.eval("Array(samples.result.weight)")
        samples['logd'] = Main.eval("Array(samples.result.logd)")
        if isinstance(self.prior_specs, dict):
            v = Main.eval("Dict(pairs(map(c -> Array(flatview(unshaped.(c))), columns(samples.result.v))))")
            samples['v'] = ak.Array({key:np.squeeze(v[key].T) for key in v.keys()})
        else:
            samples['v'] = Main.eval("Array(samples.result.v)")

        self.samples = ak.Array(samples)
            
        return self.samples
    
    def integrate(self, strategy=BAT.BridgeSampling(), use_samples=True):
        '''
        Run an integration algorithm on the posterior density
        
        Parameters:
        -----------
        strategy : BAT intergration algorithm
            such as BAT.AHMIntegration, BAT.BridgeSampling
        use_samples : bool
            whether to (try to) use existig samples from having run `.sample()`
            
        Returns:
        --------
        result : ufloat
            Integral estimate with uncertainty  
        '''
        if use_samples and self.sampled_density is not None:
            integral = BAT.bat_integrate(self.sampled_density, strategy)
        else:
            integral = BAT.bat_integrate(self.posterior, strategy)
        Main.integral = integral
        return uncertainties.ufloat(Main.eval("integral.result.val"), Main.eval("integral.result.err"))
        