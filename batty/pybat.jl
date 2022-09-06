# This file is a modifyed version from pyBAT, licensed under the MIT License (MIT).
# The BAT.jl package is licensed under the MIT "Expat" License:
# 
# Copyright (c) 2017-2021:
# 
# Oliver Schulz oschulz@mpp.mpg.de and contributors
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

module PyBAT


using DensityInterface
using ChainRulesCore
using PythonCall
using BAT


const g_pythoncall_lock = ReentrantLock()


"""
    struct PyCallDensityWithGrad <: BAT.BATDensity

Constructors:

```julia
    PyCallDensityWithGrad(logf, valgradlogf)
```
"""

struct PyCallDensity <: BAT.BATDensity
    logf::PythonCall.Py
end

export PyCallDensity

struct PyCallDensityWithGrad <: BAT.BATDensity
    logf::PythonCall.Py
    valgradlogf::PythonCall.Py
end

export PyCallDensityWithGrad


function DensityInterface.logdensityof(density::Union{PyCallDensity, PyCallDensityWithGrad}, x::AbstractArray{<:Real})
    GC.@preserve x try
        lock(g_pythoncall_lock)
        py_x = Py(x).to_numpy()
        pyconvert(Float64, density.logf(py_x))::Float64
    finally
        unlock(g_pythoncall_lock)
    end
end

function DensityInterface.logdensityof(density::Union{PyCallDensity, PyCallDensityWithGrad}, x::Union{NamedTuple, Real})
    GC.@preserve x try
        lock(g_pythoncall_lock)
        py_x = Py(x)
        pyconvert(Float64, density.logf(py_x))::Float64
    finally
        unlock(g_pythoncall_lock)
    end
end


function ChainRulesCore.rrule(::typeof(DensityInterface.logdensityof), density::PyCallDensityWithGrad, x::Real)
    logd, gradlogd = GC.@preserve x try
        lock(g_pythoncall_lock)
        py_x = Py(x)
        pyconvert(Tuple{Float64,Float64}, density.valgradlogf(py_x))::Tuple{Float64,Float64}
    finally
        unlock(g_pythoncall_lock)
    end
    @assert logd isa Real

    function pcdwg_pullback(thunked_ΔΩ)
        ΔΩ = ChainRulesCore.unthunk(thunked_ΔΩ)
        @assert ΔΩ isa Real
        tangent = gradlogd * ΔΩ
        (NoTangent(), ZeroTangent(), tangent)
    end

    return logd, pcdwg_pullback
end

function ChainRulesCore.rrule(::typeof(DensityInterface.logdensityof), density::PyCallDensityWithGrad, x::AbstractArray{<:Real})
    logd, gradlogd = GC.@preserve x try
        lock(g_pythoncall_lock)
        py_x = Py(x).to_numpy()
        pyconvert(Tuple{Float64,Vector{Float64}}, density.valgradlogf(py_x))::Tuple{Float64,Vector{Float64}}
    finally
        unlock(g_pythoncall_lock)
    end
    @assert logd isa Real

    function pcdwg_pullback(thunked_ΔΩ)
        ΔΩ = ChainRulesCore.unthunk(thunked_ΔΩ)
        @assert ΔΩ isa Real
        tangent = gradlogd * ΔΩ
        (NoTangent(), ZeroTangent(), tangent)
    end

    return logd, pcdwg_pullback
end

function ChainRulesCore.rrule(::typeof(DensityInterface.logdensityof), density::PyCallDensityWithGrad, x::NamedTuple)
    logd, gradlogd = GC.@preserve x try
        lock(g_pythoncall_lock)
        py_x = Py(x)
        res = density.valgradlogf(py_x)
        pyconvert(Tuple{Float64,NamedTuple}, res)::Tuple{Float64,NamedTuple}
    finally
        unlock(g_pythoncall_lock)
    end
    @assert logd isa Real

    function pcdwg_pullback(thunked_ΔΩ)
        ΔΩ = ChainRulesCore.unthunk(thunked_ΔΩ)
        @assert ΔΩ isa Real
        tangent = map(x -> x * ΔΩ, gradlogd)
        (NoTangent(), ZeroTangent(), tangent)
    end

    return logd, pcdwg_pullback
end


BAT.vjp_algorithm(density::PyCallDensityWithGrad) = BAT.ZygoteAD()


end ## module PyBAT

using .PyBAT
