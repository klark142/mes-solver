using LinearAlgebra
using LinearSolve
using NumericalIntegration
using QuadGK
using Pkg
Pkg.add("Plots")


struct Solution
    start_domain::Float64
    end_domain::Float64
    coefficients::Vector{Float64}
end

function E(x::Float64)
    return x <= 1.0 ? 3.0 : 5.0
end

function basis(discretization::Int, dom::Float64, i::Int, x::Float64)
    discretization = convert(Float64, discretization)
    i = convert(Float64, i)
    h = dom / discretization

    center = i * h
    left = center - h
    right = center + h

    if x < left || x > right
        return 0.0
    elseif x <= center
        return (x - left) / h
    else
        return (right - x) / h
    end
end

function basis_derivative(discretization::Int, dom::Float64, i::Int, x::Float64)
    discretization = convert(Float64, discretization)
    i = convert(Float64, i)
    h = dom / discretization

    center = i * h
    left = center - h
    right = center + h

    if x < left || x > right
        return 0.0
    elseif x < center
        return 1.0 / h
    else
        return - 1.0 / h
    end
end


function createBMatrix(discretization::Int, dom::Float64)
    bMatrix = zeros(discretization, discretization)

    for i in 1:discretization
        for j in 1:discretization           
            x = collect(0 : 1e-6 : dom)
            f(x) = E(x) * basis_derivative(discretization, dom, i-1, x) * basis_derivative(discretization, dom, j-1, x)
            y = f.(x)
            value = integrate(x, y)
            value = value - E(0.0) * basis(discretization, dom, i-1, 0.0) * basis(discretization, dom, j-1, 0.0)

            bMatrix[i, j] = value
        end
    end

    return bMatrix
end


function createLVector(discretization::Int, dom::Float64)
    lVector = zeros(discretization)
    lVector[1] = -10 * E(0.0) * basis(discretization, dom, 0, 0.0)
    return lVector
end


function solveMES(discretization::Int)
    if discretization < 2
        throw(ArgumentError("discretization must be > 2"))
    end

    dom = 2.0
    bMatrix = createBMatrix(discretization, dom)
    lVector = createLVector(discretization, dom)
    coefficients = solve(LinearProblem(bMatrix, lVector), KrylovJL_GMRES()).u

    result = vcat(coefficients, 0.0)

    return Solution(0.0, dom, result)
end

using Plots
gr()

function plotSolution(solution::Solution)
    x_values = range(solution.start_domain, stop=solution.end_domain, length=length(solution.coefficients))
    
    plot(x_values, solution.coefficients, title="Metoda elementów skończonych", xlabel="Dziedzina", ylabel="Rozwiązanie", legend=false)
end

solution = solveMES(50)
plt = plotSolution(solution)
savefig(plt, "wykres.png")