# ABC-MSM-Turing

update: please see https://github.com/mcreel/SimulatedNeuralMoments.jl for a better-developed package that does this.

Basic scripts doing Approximate Bayesian Computation (ABC) / Method of simulated moments (MSM) using Turing.jl

* SingleParameter.jl is a very basic script that was used to get the basic things working
* ParameterVector.jl has two parameters, and allows exploring correctly and incorrectly calibrated ABC
* CheckCIs.jl goes more into the calibration issue, by Monte Carlo.
