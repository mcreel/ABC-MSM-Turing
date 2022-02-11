# first example draws data from exponential
# and fits using two statistics
using Turing, Statistics, Distributions, Random, LinearAlgebra, StatsPlots
using AdvancedMH

@views function main()
θ = 2.0
n = 100 # sample size
S = 100 # number of simulation draws
y = rand.(Exponential(θ),n) # the data generating process
# summary statistics for estimation
function moments(y)
    [sqrt(n)*mean(y), sqrt(n)*std(y)] 
end

z = moments(y)
zs = zeros(S,2)

@model function abc(z)
    θ ~ LogNormal(-1.,1.) # the prior for the parameters
    # sample from the model, at the trial parameter value, and compute statistics
    @inbounds for i = 1:S
        y .= rand(Exponential(θ), n)
        zs[i,:] .= moments(y) # simulated summary statistics
    end
    # the asymptotic Gaussian distribution of the statistics
    m = mean(zs, dims=1)[:]
    Σ = Symmetric((1. + 1/S)*cov(zs))
    z ~ MvNormal(m, Σ)
end;

# sample chains, 8 chains of length 1000, with 100 burnins dropped
chain = sample(abc(z), MH(:θ => AdvancedMH.RandomWalkProposal(Normal(0, 0.25))), MCMCThreads(), 1100, 8)
display(chain)
p = plot(chain[101:end])
display(p)
end
main()
