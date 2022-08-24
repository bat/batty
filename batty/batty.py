from julia.core import UnsupportedPythonError

try:
    from julia import BAT, DensityInterface, Distributions, ValueShapes, Main
except UnsupportedPythonError:
    import warnings

    warnings.warn(
        "Not able to use compiled modules, resulting in (very) slow import\n See https://pyjulia.readthedocs.io/en/latest/troubleshooting.html"
    )
    from julia.api import Julia

    jl = Julia(compiled_modules=False)
    from julia import BAT, DensityInterface, Distributions, ValueShapes, Main

import uncertainties
import numpy as np
import os
import awkward as ak
import pygtc
import corner as corner_plot

Main.eval("using TypedTables")
Main.eval("using ValueShapes")
Main.eval("using ArraysOfArrays")

path = os.path.dirname(os.path.abspath(__file__))

Main.eval('include("%s/bat_pythoncall.jl")'%path)

class BAT_sampler:
    def __init__(self, prior_specs, llh, grad=None, llh_and_grad=None, llh_args=()):
        """BAT_sampler Python wrapper

        Paramaters:
        -----------
        prior_specs : BAT prior, or dictionary thereof
            The prior specifications
        llh : callable
            The likelihood function
        grad : callable (optional)
            function returning the gradients of llh
        llh_and_grad : callable (optional)
            function returning the llh and gradients in one go
        llh_args : tuple
            additional arguments to the llh function (optional)
        """

        assert llh is not None or llh_and_grad is not None, "Neither llh nor llh_and_grad were supplied"

        assert grad is None or llh_and_grad is None, "grad AND llh_and_grad were supplied, please choose only one or the other"

        if len(llh_args) == 0:
            if grad is not None:
                my_llh_and_grad = lambda x: (llh(x), grad(x))
                my_llh = llh
            elif llh_and_grad is not None:
                my_llh_and_grad = llh_and_grad
                my_llh = lambda x: llh_and_grad(x)[0]
            else:
                my_llh_and_grad = None
                my_llh = llh

            if my_llh_and_grad is None:
                self.llh = DensityInterface.logfuncdensity(my_llh)
            else:
                self.llh = Main.PyCallDensityWithGrad(my_llh, my_llh_and_grad)
        
        else:
            if grad is not None:
                my_llh_and_grad = lambda x: (llh(x, *llh_args), grad(x, *llh_args))
                my_llh = lambda x: llh(x, *llh_args)
            elif llh_and_grad is not None:
                my_llh_and_grad = lambda x: llh_and_grad(x, *llh_args)
                my_llh = lambda x: llh_and_grad(x, *llh_args)[0]
            else:
                my_llh_and_grad = None
                my_llh = lambda x: llh(x, *llh_args)

            if my_llh_and_grad is None:
                self.llh = DensityInterface.logfuncdensity(my_llh)
            else:
                self.llh = Main.PyCallDensityWithGrad(my_llh, my_llh_and_grad)

        self.prior_specs = prior_specs
        if isinstance(self.prior_specs, dict):
            self.prior = ValueShapes.NamedTupleDist(**prior_specs)
        else:
            self.prior = prior_specs
        self.posterior = BAT.PosteriorDensity(self.llh, self.prior)
        self.samples = None
        self._samples = None
        self.sampled_density = None

    def sample(self, strategy=BAT.MCMCSampling()):
        """
        Run the Sampler

        Paramaters:
        -----------
        strategy : BAT sampling algorithm

        Returns:
        --------
        samples : awkward.Array
            Containing arrays of v, weight, and logd
        """
        Main.samples = BAT.bat_sample(self.posterior, strategy)
        self._samples = Main.samples

        Main.posterior = self.posterior
        self.sampled_density = Main.eval(
            "BAT.SampledDensity(posterior, samples.result)"
        )

        # define viws of arrays
        samples = {}
        samples["weight"] = Main.eval("Array(samples.result.weight)")
        samples["logd"] = Main.eval("Array(samples.result.logd)")
        if isinstance(self.prior_specs, dict):
            v = Main.eval(
                "Dict(pairs(map(c -> Array(flatview(unshaped.(c))), columns(samples.result.v))))"
            )
            samples["v"] = ak.Array({key: np.squeeze(v[key].T) for key in v.keys()})
        else:
            samples["v"] = Main.eval("Array(samples.result.v)")

        self.samples = ak.Array(samples)

        return self.samples

    def integrate(self, strategy=BAT.BridgeSampling(), use_samples=True):
        """
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
        """
        if use_samples and self.sampled_density is not None:
            integral = BAT.bat_integrate(self.sampled_density, strategy)
        else:
            integral = BAT.bat_integrate(self.posterior, strategy)
        Main.integral = integral
        return uncertainties.ufloat(
            Main.eval("integral.result.val"), Main.eval("integral.result.err")
        )

    def get_arrays_for_plotting(self):
        if len(self.samples.v.fields) > 0:        
            s = [self.samples.v[var].to_numpy() for var in self.prior_specs.keys()]
            labels = []
            vs = []
            for v, n in zip(s, self.prior_specs.keys()):
                if v.ndim == 1:
                    labels.append(n)
                    vs.append(v[:, np.newaxis])
                elif v.ndim == 2:
                    for i in range(s[1].shape[1]):
                        labels.append("%s[%i]" % (n, i))
                    vs.append(v)
                else:
                    raise Exception("Cannot deal with ndim=%i array `%s`" % (v.ndim, n))
            vs = np.hstack(vs)

        else:
            vs = self.samples.v.to_numpy()
            if vs.ndim == 1:
                vs = vs[:, np.newaxis]
            labels = None

        return vs, labels

    def gtc(self, **kwargs):
        vs, labels = self.get_arrays_for_plotting()
        paramNames = kwargs.pop("paramNames", labels)

        return pygtc.plotGTC(
            chains=vs,
            weights=self.samples.weight.to_numpy(),
            paramNames=paramNames,
            **kwargs,
        )

    gtc.__doc__ = pygtc.plotGTC.__doc__

    def corner(self, **kwargs):
        vs, l = self.get_arrays_for_plotting()
        labels = kwargs.pop("labels", l)

        return corner_plot.corner(
            vs,
            weights=self.samples.weight.to_numpy(),
            labels=labels,
            **kwargs,
        )

    corner.__doc__ = corner_plot.corner.__doc__
