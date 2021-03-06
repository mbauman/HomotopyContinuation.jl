export AffineSubspaceHomotopy, set_subspaces!

"""
    AffineSubspaceHomotopy(F::System, V::AffineSubspace, W::AffineSubspace)
    AffineSubspaceHomotopy(F::AbstractSystem, V::AffineSubspace, W::AffineSubspace)

Creates a homotopy ``H(x,t) = (F ∩ γ(t))(x)`` where ``γ(t)`` is a family of affine subspaces
such that ``H(x,1) = (F ∩ V)(x)`` and ``H(x,0) = (F ∩ W)(x)``.
Here ``γ(t)`` is the geodesic between `V` and `W` in the affine Grassmanian, i.e.,
it is the curve of minimal length connecting `V` and `W`.
See also [`AffineSubspace`](@ref) and [`geodesic`](@ref) and the references therein.
"""
Base.@kwdef struct AffineSubspaceHomotopy{S<:AbstractSystem} <: AbstractHomotopy
    system::S

    start::AffineSubspace{ComplexF64}
    target::AffineSubspace{ComplexF64}

    J::Matrix{ComplexF64}
    Q::Matrix{ComplexF64}
    Q_cos::Matrix{ComplexF64}
    Θ::Vector{Float64}
    U::Matrix{ComplexF64}
    γ1::Matrix{ComplexF64}

    x::Vector{ComplexF64}
    ẋ::Vector{ComplexF64}
    x_high::Vector{ComplexDF64}
    t_cache::Base.RefValue{ComplexF64}
    # For AD
    taylor_t_cache::Base.RefValue{ComplexF64}
    taylor_γ::NTuple{4,Matrix{ComplexF64}}
    v::Vector{ComplexF64}
    tx³::TaylorVector{4,ComplexF64}
    tx²::TaylorVector{3,ComplexF64}
    tx¹::TaylorVector{2,ComplexF64}
end

## Implementation details

# The computation is performed using (implicit) Stiefel coordinates. If `size(F) = m, n`
# and `V` and `W` are affine subspaces of dimension `k, then the computation is performed in
# the affine Grassmanian Graff(k,n) embedded in the Grassmanian(k+1,n+1) using the system
# ``[F(γ(t)v); (γ(t)v)[end] - 1]``. Here the `(γ(t)v)[k+1] - 1` condition ensures that
# we are in the affine Grassmanian and that `(γ(t)v)[1:k]` is the correct value in \C^n.

AffineSubspaceHomotopy(F::ModelKit.System, start::AffineSubspace, target::AffineSubspace) =
    AffineSubspaceHomotopy(ModelKitSystem(F), start, target)

function AffineSubspaceHomotopy(
    system::AbstractSystem,
    start::AffineSubspace,
    target::AffineSubspace,
)

    Q, Θ, U = geodesic_svd(target, start)
    Q_cos = target.intrinsic.Y * U
    tx³ = TaylorVector{4}(ComplexF64, size(Q, 1))

    AffineSubspaceHomotopy(
        system = system,
        start = copy(start),
        target = copy(target),
        J = zeros(ComplexF64, size(system) .+ (1, 1)),
        Q = Q,
        Q_cos = Q_cos,
        Θ = Θ,
        U = U,
        γ1 = Q_cos * LA.diagm(0 => cos.(Θ)) + Q * LA.diagm(0 => sin.(Θ)),
        x = zeros(ComplexF64, size(Q, 1)),
        ẋ = zeros(ComplexF64, size(Q, 1)),
        x_high = zeros(ComplexDF64, size(Q, 1)),
        t_cache = Ref(complex(NaN, NaN)),
        taylor_t_cache = Ref(complex(NaN, NaN)),
        taylor_γ = tuple((similar(Q) for i = 0:3)...),
        v = zeros(ComplexF64, size(Q, 2)),
        tx³ = tx³,
        tx² = TaylorVector{3}(tx³),
        tx¹ = TaylorVector{2}(tx³),
    )
end
Base.size(H::AffineSubspaceHomotopy) = (size(H.system)[1] + 1, dim(H.start) + 1)

"""
    set_subspaces!(H::AffineSubspaceHomotopy, start::AffineSubspace, target::AffineSubspace)

Update the homotopy `H` to track from the affine subspace `start` to `target`.
"""
function set_subspaces!(
    H::AffineSubspaceHomotopy,
    start::AffineSubspace,
    target::AffineSubspace,
)
    Q, Θ, U = geodesic_svd(target, start)
    copy!(H.start, start)
    copy!(H.target, target)
    H.Q .= Q
    H.Θ .= Θ
    H.U .= U
    LA.mul!(H.Q_cos, target.intrinsic.Y, U)
    LA.mul!(H.γ1, Q, LA.diagm(0 => sin.(Θ)))
    LA.mul!(H.γ1, H.Q_cos, LA.diagm(0 => cos.(Θ)), true, true)
    H
end

function γ!(H::AffineSubspaceHomotopy, t::Number)
    H.t_cache[] != t || return first(H.taylor_γ)
    if isreal(t)
        _γ!(H, real(t))
    else
        _γ!(H, t)
    end
    H.t_cache[] = t

    first(H.taylor_γ)
end
@inline function _γ!(H::AffineSubspaceHomotopy, t::Number)
    @unpack Q, Q_cos, Θ = H
    γ = first(H.taylor_γ)
    n, k = size(γ)
    @inbounds for j = 1:k
        Θⱼ = Θ[j]
        s, c = sincos(t * Θⱼ)
        for i = 1:n
            γ[i, j] = Q_cos[i, j] * c + Q[i, j] * s
        end
    end
    γ
end

γ̇!(H::AffineSubspaceHomotopy, t::Number) = isreal(t) ? _γ̇!(H, real(t)) : _γ̇!(H, t)
@inline function _γ̇!(H::AffineSubspaceHomotopy, t::Number)
    @unpack Q, Q_cos, Θ = H
    _, γ̇ = H.taylor_γ
    n, k = size(γ̇)
    @inbounds for j = 1:k
        Θⱼ = Θ[j]
        s, c = sincos(t * Θⱼ)
        ċ = -s * Θⱼ
        ṡ = c * Θⱼ
        for i = 1:n
            γ̇[i, j] = Q_cos[i, j] * ċ + Q[i, j] * ṡ
        end
    end
    γ̇
end

function set_solution!(u::Vector, H::AffineSubspaceHomotopy, x::AbstractVector, t)
    (length(x) == length(H.x) - 1) ||
        throw(ArgumentError("Cannot set solution. Expected extrinsic coordinates."))
    for i = 1:length(x)
        H.x[i] = x[i]
    end
    H.x[end] = 1

    if isone(t)
        LA.mul!(u, H.γ1', H.x)
    elseif iszero(t)
        LA.mul!(u, H.Q_cos', H.x)
    else
        LA.mul!(u, γ!(H, t), H.x)
    end
end

function get_solution(H::AffineSubspaceHomotopy, u::AbstractVector, t)
    if isone(t)
        (@view H.γ1[1:end-1, :]) * u
    elseif iszero(t)
        (@view H.Q_cos[1:end-1, :]) * u
    else
        γ = γ!(H, t)
        (@view γ[1:end-1, :]) * u
    end
end

function evaluate!(u, H::AffineSubspaceHomotopy, v::AbstractVector, t)
    γ = γ!(H, t)
    n = first(size(H.system))
    if eltype(v) isa ComplexDF64
        LA.mul!(H.x_high, γ, v)
        evaluate!(u, H.system, H.x_high)
        u[n+1] = H.x_high[end] - 1.0
    else
        LA.mul!(H.x, γ, v)
        evaluate!(u, H.system, H.x)
        u[n+1] = H.x[end] - 1.0
    end
    u
end

function evaluate_and_jacobian!(u, U, H::AffineSubspaceHomotopy, v::AbstractVector, t)
    γ = γ!(H, t)

    LA.mul!(H.x, γ, v)
    evaluate_and_jacobian!(u, H.J, H.system, H.x)
    LA.mul!(U, H.J, γ)

    n = first(size(H.system))
    u[n+1] = H.x[end] - 1

    m = length(v)
    for j = 1:m
        U[n+1, j] = γ[end, j]
    end

    nothing
end

function taylor!(u, ::Val{1}, H::AffineSubspaceHomotopy, v, t)
    γ = γ!(H, t)
    γ̇ = γ̇!(H, t)

    # apply chain rule
    #    d/dt [F(γ(t)v); (γ(t)v)[end] - 1] = [J_F(γ(t)v)* γ̇(t)*v;  (γ̇(t)v)[end]]
    LA.mul!(H.x, γ, v)
    LA.mul!(H.ẋ, γ̇, v)
    evaluate_and_jacobian!(u, H.J, H.system, H.x)
    LA.mul!(u, H.J, H.ẋ)
    M = size(H, 1)
    u[M] = H.ẋ[end]

    u
end

function _taylor_γ!(H::AffineSubspaceHomotopy, t::Number)
    @unpack Q, Q_cos, Θ, U, taylor_γ = H

    γ, γ¹, γ², γ³ = taylor_γ
    n, k = size(γ)
    @inbounds for j = 1:k
        Θⱼ = Θ[j]
        s, c = sincos(t * Θⱼ)
        c¹ = -s * Θⱼ
        s¹ = c * Θⱼ
        Θⱼ_2 = 0.5 * Θⱼ^2
        s² = -s * Θⱼ_2
        c² = -c * Θⱼ_2
        Θⱼ_3 = Θⱼ_2 * Θⱼ / 3
        s³ = -c * Θⱼ_3
        c³ = s * Θⱼ_3
        for i = 1:n
            γ[i, j] = Q_cos[i, j] * c + Q[i, j] * s
            γ¹[i, j] = Q_cos[i, j] * c¹ + Q[i, j] * s¹
            γ²[i, j] = Q_cos[i, j] * c² + Q[i, j] * s²
            γ³[i, j] = Q_cos[i, j] * c³ + Q[i, j] * s³
        end
    end

end

function taylor_γ!(H::AffineSubspaceHomotopy, t::Number)
    H.taylor_t_cache[] != t || return H.taylor_γ

    if isreal(t)
        _taylor_γ!(H, real(t))
    else
        _taylor_γ!(H, t)
    end
    H.taylor_t_cache[] = t

    H.taylor_γ
end


function taylor!(u, v::Val{2}, H::AffineSubspaceHomotopy, tv::TaylorVector, t, incr::Bool)
    γ, γ¹, γ², γ³ = taylor_γ!(H, t)
    x, x¹, x² = vectors(H.tx²)
    v, v¹ = vectors(tv)

    if !incr
        LA.mul!(x, γ, v)
        LA.mul!(x¹, γ¹, v)
    end

    H.v .= v¹
    LA.mul!(H.x, γ, v¹)
    x¹ .+= H.x

    LA.mul!(H.x, γ¹, H.v)
    H.v .= v
    LA.mul!(H.x, γ², H.v, true, true)
    x² .= H.x

    taylor!(u, Val(2), H.system, H.tx²)

    n = first(size(H.system))
    u[n+1] = x²[end]

    u
end

function taylor!(u, v::Val{3}, H::AffineSubspaceHomotopy, tv::TaylorVector, t, incr::Bool)
    γ, γ¹, γ², γ³ = taylor_γ!(H, t)
    x, x¹, x², x³ = vectors(H.tx³)
    v, v¹, v² = vectors(tv)

    if !incr
        LA.mul!(x, γ, v)
        LA.mul!(x¹, γ¹, v)
        LA.mul!(x¹, γ, v¹, true, true)
        LA.mul!(x², γ², v)
        LA.mul!(x², γ¹, v¹, true, true)
    end

    H.v .= v²
    LA.mul!(H.x, γ, H.v)
    x² .+= H.x

    LA.mul!(H.x, γ¹, H.v)
    H.v .= v
    LA.mul!(H.x, γ³, H.v, true, true)
    H.v .= v¹
    LA.mul!(H.x, γ², H.v, true, true)
    x³ .= H.x

    taylor!(u, Val(3), H.system, H.tx³)

    n = first(size(H.system))
    u[n+1] = x³[end]

    u
end
