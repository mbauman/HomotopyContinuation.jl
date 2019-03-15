export NewtonResult, newton, newton!, NewtonCache


"""
    NewtonResult{T}

Structure holding information about the outcome of the `newton` function. The fields are.
* `retcode` The return code of the compuation. `converged` means that `accuracy ≤ tol`.
* `accuracy::T` |xᵢ-xᵢ₋₁| for i = iters and x₀,x₁,…,xᵢ₋₁,xᵢ are the Newton iterates.
* `iters::Int` The number of iterations used.
* `digits_lost::Float64` Estimate of the (relative) lost digits in the linear algebra.
"""
struct NewtonResult{T}
    retcode::ReturnCode
    accuracy::T
    iters::Int
    digits_lost::Float64
    ω₀::Float64
    ω::Float64
    norm_Δx₀::T
end

Base.show(io::IO, ::MIME"application/prs.juno.inline", r::NewtonResult) = r
Base.show(io::IO, result::NewtonResult) = print_fieldnames(io, result)


"""
    isconverged(result::Result)

Returns whether the correction was successfull.
"""
isconverged(result::NewtonResult) = result.retcode == converged


"""
    NewtonCache(F::AbstractSystem, x)

Cache for the [`newton`](@ref) function.
"""
struct NewtonCache{T, Fac<:LinearAlgebra.Factorization}
    Jac::Jacobian{T, Fac}
    rᵢ::Vector{T}
    Δxᵢ::Vector{T}
end

function NewtonCache(F::AbstractSystem, x)
    Jac = Jacobian(jacobian(F, x))
    rᵢ = evaluate(F, x)
    Δxᵢ = copy(rᵢ)

    NewtonCache(Jac, rᵢ, Δxᵢ)
end


"""
    newton(F::AbstractSystem, x₀; tol=1e-6, maxiters=3, simplified_last_step=true)

An ordinary Newton's method. If `simplified_last_step` is `true`, then for the last iteration
the previously Jacobian will be used. This uses an LU-factorization for square systems
and a QR-factorization for overdetermined.
"""
function newton(F::AbstractSystem, x₀; tol=1e-6, maxiters=3, simplified_last_step=true)
    newton!(copy(x₀), F::AbstractSystem, x₀, tol, maxiters, simplified_last_step, NewtonCache(F::AbstractSystem, x₀))
end

"""
    newton!(out, F::AbstractSystem, x₀, tol, maxiters::Integer, simplified_last_step::Bool, cache::NewtonCache)

In-place version of [`newton`](@ref). Needs a [`NewtonCache`](@ref) as input.
"""
function newton!(out, F::AbstractSystem, x₀, tol, maxiters::Integer, simplified_last_step::Bool, cache::NewtonCache)
    Jac, rᵢ, Δxᵢ = cache.Jac, cache.rᵢ, cache.Δxᵢ
    Jᵢ = Jac.J
    copyto!(out, x₀)
    xᵢ₊₁ = xᵢ = out # just alias to make logic easier
    rᵢ₊₁ = rᵢ

    # initialize variables
    T = real(eltype(xᵢ))
    Θ₀ = Θᵢ₋₁ = norm_Δxᵢ₋₁ = norm_Δxᵢ = norm_Δx₀ = zero(T)
    accuracy = T(Inf)
    digits_lost = 0.0
    ω₀ = ω = 0.0
    for i ∈ 0:(maxiters)
        if i == maxiters && simplified_last_step
            evaluate!(rᵢ, F, xᵢ)
        else
            evaluate_and_jacobian!(rᵢ, Jᵢ, F, xᵢ)
            updated_jacobian!(Jac)
        end
        if i == 0
            digits_lost = solve_with_digits_lost!(Δxᵢ, Jac, rᵢ)
        else
            solve!(Δxᵢ, Jac, rᵢ)
        end

        norm_Δxᵢ₋₁ = norm_Δxᵢ
        norm_Δxᵢ = euclidean_norm(Δxᵢ)
        @inbounds for k in eachindex(xᵢ)
            xᵢ₊₁[k] = xᵢ[k] - Δxᵢ[k]
        end

        if i == 0
            accuracy = norm_Δx₀ = norm_Δxᵢ₋₁ = norm_Δxᵢ
            if norm_Δx₀ ≤ tol
                return NewtonResult(converged, norm_Δx₀, i + 1, digits_lost, ω₀, ω, norm_Δx₀)
            end

        else
            Θᵢ₋₁ = norm_Δxᵢ / norm_Δxᵢ₋₁
            ωᵢ₋₁ = 2Θᵢ₋₁ / norm_Δxᵢ₋₁
            if i == 1
                ω = ω₀ = ωᵢ₋₁
            else
                ω = @fastmath max(ω, ωᵢ₋₁)
            end

            if Θᵢ₋₁ > 0.5
                return NewtonResult(terminated, accuracy, i + 1, digits_lost, ω₀, ω, norm_Δx₀)
            end

            accuracy = norm_Δxᵢ / (1 - 2Θᵢ₋₁^2)
            if accuracy ≤ tol
                return NewtonResult(converged, accuracy, i + 1, digits_lost, ω₀, ω, norm_Δx₀)
            end
        end
    end

    return NewtonResult(maximal_iterations, accuracy, maxiters, digits_lost, ω₀, ω, norm_Δx₀)
end