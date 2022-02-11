# Monte Carlo to check thatquantiles of the chain give valid
# confidence intervals when the likelihood is well-specified
# ("calibrated") but not when the likelihood is incorrect 
# (unweighted vanilla sum of squares)
using Turing, Statistics, Distributions, Random, LinearAlgebra, StatsPlots
using AdvancedMH

# the data generating process
function dgp(θ, n)
    [rand.(Exponential(θ[1]),n) rand.(Poisson(θ[2]),n)] 
end

# summary statistics for estimation
function moments(y)
    n = size(y,1)
    sqrt(n) .* [mean(y, dims=1)[:]; std(y, dims=1)[:]]
end

@model function abc(z, S, n, calibrated)
    # create the prior: the product of the following array of marginal priors
    θ  ~ arraydist([LogNormal(1.,1.); LogNormal(1.,1.)])
    # sample from the model, at the trial parameter value, and compute statistics
    y = zeros(n,2)
    zs = zeros(S, size(z,1))
    @inbounds for i = 1:S
        y .= dgp(θ, n)
        zs[i,:] .= moments(y) # simulated summary statistics
    end
    # the asymptotic Gaussian distribution of the statistics
    m = mean(zs, dims=1)[:]
    calibrated ? Σ = Symmetric((1. + 1/S)*cov(zs)) : Σ = I
    z ~ MvNormal(m, Σ)
end;

function mcloop(reps=1000, calibrated=true)
    inci = zeros(reps,2)
    for rep = 1:reps
        θ⁰ = [2.; 3.] # true parameters
        n = 100 # sample size
        S = 100 # number of simulation draws
        # get data and statistics
        y = dgp(θ⁰, n)
        z = moments(y)
        # sample chains, 4 chains of length 1000, with 100 burnins dropped
        length = 2000
        burnin = 500
        chain = sample(abc(z, S, n, calibrated), 
            MH(:θ => AdvancedMH.RandomWalkProposal(MvNormal(zeros(2), 0.25*I))),
            MCMCThreads(), length+burnin, 4)
        chain = chain[burnin+1:end,:,:]
        @show display(chain)
        chain = Array(chain)
        q1 = quantile(chain[:,1], [0.025, 0.975])
        q2 = quantile(chain[:,2], [0.025, 0.975])
        inci[rep,1] = q1[1] <= θ⁰[1] && q1[2] >= θ⁰[1]  
        inci[rep,2] = q2[1] <= θ⁰[2] && q2[2] >= θ⁰[2]
    end
println("proportion of times true parameter is in 95% CI")
@show mean(inci, dims=1)
inci
end

#inci = mcloop(100, true) # when false, uncalibrated, when true, calibrated
inci = mcloop(100, false) # when false, uncalibrated, when true, calibrated

