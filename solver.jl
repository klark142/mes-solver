using LinearAlgebra
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
    h = dom / discretization
    hInv = discretization / dom

    center = dom * i / discretization
    left = center - h
    right = center + h

    if x < left || x > right
        return 0.0
    elseif x <= center
        return (x - left) * hInv
    else
        return (right - x) * hInv
    end
end

function basis_derivative(discretization::Int, dom::Float64, i::Int, x::Float64)
    h = dom / discretization
    hInv = discretization / dom

    center = dom * i / discretization
    left = center - h
    right = center + h

    if x < left || x > right
        return 0.0
    elseif x <= center
        return hInv
    else
        return -hInv
    end
end


function createBMatrix(discretization::Int, dom::Float64)
    bMatrix = zeros(discretization, discretization)

    for i in 1:discretization
        for j in 1:discretization
            value = 0.0

            integrateFrom = dom * max(max(i, j) - 1, 0) / discretization
            integrateTo = dom * min(min(i, j) + 1, discretization) / discretization

            value, _ = quadgk(x -> E(x) * basis_derivative(discretization, dom, i, x) * basis_derivative(discretization, dom, j, x), integrateFrom, integrateTo)
            
            bMatrix[i, j] = -E(0.0) * basis(discretization, dom, i, 0.0) * basis(discretization, dom, j, 0.0) + value
        end
    end

    return bMatrix
end


function createLVector(discretization::Int, dom::Float64)
    lVector = zeros(discretization)
    lVector[1] = -10 * E(0.0) * basis(discretization, dom, 0, 0.0)
    return lVector
end


function solve(discretization::Int)
    if discretization < 2
        throw(ArgumentError("discretization must be > 2"))
    end

    dom = 2.0
    bMatrix = createBMatrix(discretization, dom)
    print(bMatrix)
    lVector = createLVector(discretization, dom)
    print(lVector)

    coefficients = lu(bMatrix) \ lVector

    result = vcat(coefficients, 0.0)

    return Solution(0.0, dom, result)
end

using Plots
gr()

function plotSolution(solution::Solution)
    # Generate x values from the domain range
    x_values = range(solution.start_domain, stop=solution.end_domain, length=length(solution.coefficients))
    
    # Plotting the solution
    plot(x_values, solution.coefficients, title="Finite Element Solution", xlabel="Domain", ylabel="Solution", legend=false)
end

# Assuming you have a `solution` from your finite element solver:
solution = solve(50)
plt = plotSolution(solution)
display(plt)
savefig(plt, "solution_plot.png")