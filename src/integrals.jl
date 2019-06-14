# Author: Stefano Battaglia
# 2018


using GSL: sf_hyperg_1F1
import Base: exponent


"""Abstract type defining a basis function."""
abstract type BasisFunction end


############## GTO ###############


"""Concrete type defining a Cartesian GTO, i.e. a CGF."""
struct GTO <: BasisFunction
    A::Vector{Float64}        # center of the GTO
    α::Float64                # exponent
    l::Tuple{Int,Int,Int}     # Cartesian quantum numbers
end


### some getter functions ###
center(gto::GTO)   = gto.A
exponent(gto::GTO) = gto.α
shell(gto::GTO)    = gto.l

Ax(gto::GTO) = gto.A[1]
Ay(gto::GTO) = gto.A[2]
Az(gto::GTO) = gto.A[3]

lx(gto::GTO) = gto.l[1]
ly(gto::GTO) = gto.l[2]
lz(gto::GTO) = gto.l[3]

ltot(gto::GTO) = sum(gto.l)
###


"""Returns the self-overlap integral <Ga|Ga> of a `GTO` Ga."""
function overlap(Ga::GTO)
    α = exponent(Ga)
    num = dfactorial(2*lx(Ga)-1)*dfactorial(2*ly(Ga)-1)*dfactorial(2*lz(Ga)-1)
    den = (4.0*α)^ltot(Ga)
    return ((π/(2.0*α))^1.5 * num)/den
end


"""Returns the normalization factor of a `GTO` Ga."""
function normalization(Ga::GTO)
    α = exponent(Ga)
    num = (2.0*α/π)^0.75 * (4.0*α)^(ltot(Ga)/2.0)
    den = sqrt(dfactorial(2*lx(Ga)-1)*dfactorial(2*ly(Ga)-1)*dfactorial(2*lz(Ga)-1))
    return num/den
end


"""Returns the overlap integral <Ga|Gb> of two Cartesian GTOs."""
function overlap(Ga::GTO, Gb::GTO)

    # precomputing all required quantities
    α = exponent(Ga); β = exponent(Gb)
    A = center(Ga); B = center(Gb)
    μ = (α * β)/(α + β)
    P = (A*α + B*β)./(α + β)
    RAB = A - B; RPA = P - A; RPB = P - B
    Kαβ = exp.(-μ*RAB.^2)

    # calculate overlaps in the 3 Cartesian directions
    @inbounds Sx = Etij(0, lx(Ga), lx(Gb), Kαβ[1], RPA[1], RPB[1], α, β)
    @inbounds Sy = Etij(0, ly(Ga), ly(Gb), Kαβ[2], RPA[2], RPB[2], α, β)
    @inbounds Sz = Etij(0, lz(Ga), lz(Gb), Kαβ[3], RPA[3], RPB[3], α, β)

    return Sx * Sy * Sz * (π / (α + β) )^1.5
end


"""Returns the overlap integral <Ga|Gb> of two Cartesian Gaussian functions."""
function overlap(α::Real, ikm::Tuple{Int,Int,Int}, A::Vector{T},
                 β::Real, jln::Tuple{Int,Int,Int}, B::Vector{T}) where {T<:Real}

    # precomputing all required quantities
    μ = (α * β)/(α + β)
    P = (A*α + B*β)./(α + β)
    RAB = A - B; RPA = P - A; RPB = P - B
    Kαβ = exp.(-μ*RAB.^2)

    # calculate overlaps in the 3 Cartesian directions
    @inbounds Sx = Etij(0, ikm[1], jln[1], Kαβ[1], RPA[1], RPB[1], α, β)
    @inbounds Sy = Etij(0, ikm[2], jln[2], Kαβ[2], RPA[2], RPB[2], α, β)
    @inbounds Sz = Etij(0, ikm[3], jln[3], Kαβ[3], RPA[3], RPB[3], α, β)

    return Sx * Sy * Sz * (π / (α + β) )^1.5
end



"""Returns the kinetic energy integral -0.5*<Ga|∇^2|Gb> of two GTOs."""
function kinetic(Ga::GTO, Gb::GTO)

    # extract info from GTOs
    α = exponent(Ga); β = exponent(Gb)
    A = center(Ga); B = center(Gb)
    (i,k,m) = shell(Ga); (j,l,n) = shell(Gb)

    # precompute common terms
    dαSab = 2*α*overlap(α, (i,k,m), A, β, (j,l,n), B)
    fα2 = 4*α^2

    # Dij^2 * Skl * Smn
    Dij = i*(i-1) * overlap(α, (i-2,k,m), A, β, (j,l,n), B) -
          (2*i+1) * dαSab +
              fα2 * overlap(α, (i+2,k,m), A, β, (j,l,n), B)

    # Sij * Dkl^2 * Smn
    Dkl = k*(k-1) * overlap(α, (i,k-2,m), A, β, (j,l,n), B) -
          (2*k+1) * dαSab +
              fα2 * overlap(α, (i,k+2,m), A, β, (j,l,n), B)

    # Sij * Skl * Dmn^2
    Dmn = m*(m-1) * overlap(α, (i,k,m-2), A, β, (j,l,n), B) -
          (2*m+1) * dαSab +
              fα2 * overlap(α, (i,k,m+2), A, β, (j,l,n), B)

    return -0.5 * (Dij + Dkl + Dmn)
end


"""Returns the nuclear attraction energy integral of the distribution Ωab from center C."""
function nuclear(Ga::GTO, Gb::GTO, C::Vector{T}) where {T<:Real}

    # precomputing all required quantities
    α = exponent(Ga); β = exponent(Gb)
    (i,k,m) = shell(Ga); (j,l,n) = shell(Gb)
    A = center(Ga); B = center(Gb)
    p = α + β
    P = (α.*A + β.*B) .* (1.0/(α + β))
    RAB = A - B; RPA = P - A; RPB = P - B; RPC = P - C
    μ = (α * β)/(α + β)
    Kαβ = exp.(-μ .* RAB.^2)

    vne = 0.0
    for t = 0:i+j
        Eij = Etij(t, i, j, Kαβ[1], RPA[1], RPB[1], α, β)
        for u = 0:k+l
            Ekl = Etij(u, k, l, Kαβ[2], RPA[2], RPB[2], α, β)
            for v = 0:m+n
                Emn = Etij(v, m, n, Kαβ[3], RPA[3], RPB[3], α, β)
                vne +=  Eij * Ekl * Emn * Rtuv(t, u, v, 0, p, RPC)
            end
        end
    end

    return 2.0*π*vne/p
end


"""Returns the two-electron integral gabcd = < Ga(r1) Gb(r1) | 1/r12 | Gc(r2) Gd(r2) >."""
function repulsion(Ga::GTO, Gb::GTO, Gc::GTO, Gd::GTO)

    # extract info from gaussians for electron 1
    α = exponent(Ga); β = exponent(Gb)
    (i1,k1,m1) = shell(Ga); (j1,l1,n1) = shell(Gb)
    A = center(Ga); B = center(Gb); RAB = A-B
    # precompute quantities for electron 1
    P = (α.*A + β.*B) ./ (α + β)
    p = α + β
    Kαβ = exp.(-α*β/(α+β).*RAB.^2)
    RPA = P-A; RPB = P-B

    # extract info from gaussians for electron 2
    γ = exponent(Gc); δ = exponent(Gd)
    (i2,k2,m2) = shell(Gc); (j2,l2,n2) = shell(Gd)
    C = center(Gc); D = center(Gd); RCD = C-D
    # precompute quantities for electron 2
    Q = (γ.*C + δ.*D) ./ (γ + δ)
    q = γ + δ
    Kγδ = exp.(-γ*δ/(γ+δ).*RCD.^2)
    RQC = Q-C; RQD = Q-D

    # precompute quantities for auxiliary integral R
    RPQ = P-Q
    ξ = p*q/(p+q)

    vee = 0.0
    for t = 0:i1+j1
        Eij1 = Etij(t, i1, j1, Kαβ[1], RPA[1], RPB[1], α, β)
        for u = 0:k1+l1
            Ekl1 = Etij(u, k1, l1, Kαβ[2], RPA[2], RPB[2], α, β)
            for v = 0:m1+n1
                Emn1 = Etij(v, m1, n1, Kαβ[3], RPA[3], RPB[3], α, β)
                for τ = 0:i2+j2
                    Eij2 = Etij(τ, i2, j2, Kγδ[1], RQC[1], RQD[1], γ, δ)
                    for ν = 0:k2+l2
                        Ekl2 = Etij(ν, k2, l2, Kγδ[2], RQC[2], RQD[2], γ, δ)
                        for ϕ = 0:m2+n2
                            Emn2 = Etij(ϕ, m2, n2, Kγδ[3], RQC[3], RQD[3], γ, δ)
                            vee += Eij1 * Ekl1 * Emn1 * Eij2 * Ekl2 * Emn2 *
                                   Rtuv(t+τ, u+ν, v+ϕ, 0, ξ, RPQ) * (-1)^(τ+ν+ϕ)
                        end
                    end
                end
            end
        end
    end

    return (vee*2.0*π^2.5)/(p*q*sqrt(p+q))
end


################ CGTO ###################


"""Contracted Gaussian-type orbital."""
struct CGTO
    funcs::Vector{GTO}          # primitive GTOs
    coefs::Vector{Float64}      # contraction coefs
    norms::Vector{Float64}      # normalization factors
end


"""Construct a `CGTO` centered on `A` with angular momenutm `l`."""
function CGTO(A::AbstractVector{T}, l::Tuple{Int,Int,Int},
              α::AbstractVector{T}, d::AbstractVector{T}) where {T<:Real}

    # number of coefs and exponents must match
    @assert(length(d) == length(α))
    n = length(d)

    # initialize data vector
    funcs = Vector{GTO}(undef,n)
    norms = Vector{Float64}(undef,n)

    for i in 1:n
        funcs[i] = GTO(A,α[i],l)
        norms[i] = normalization(funcs[i])
    end

    temp_cgto = CGTO(funcs, d, norms)
    norms .*= 1.0/sqrt(overlap(temp_cgto))

    return CGTO(funcs, d, norms)
end


### some getter functions ###
primitives(cgto::CGTO) = cgto.funcs
nprims(cgto::CGTO) = length(cgto.funcs)
center(cgto::CGTO) = cgto.funcs[1].A
Ax(cgto::CGTO) = Ax(cgto.funcs[1])
Ay(cgto::CGTO) = Ay(cgto.funcs[1])
Az(cgto::CGTO) = Az(cgto.funcs[1])
shell(cgto::CGTO) = shell(cgto.funcs[1])
ltot(cgto::CGTO) = ltot(cgto.funcs[1])
lx(cgto::CGTO) = lx(cgto.funcs[1])
ly(cgto::CGTO) = ly(cgto.funcs[1])
lz(cgto::CGTO) = lz(cgto.funcs[1])
exponents(cgto::CGTO) = map(x->exponent(x),primitives(cgto))
coefs(cgto::CGTO) = cgto.coefs
norms(cgto::CGTO) = cgto.norms
###


"""Returns the self-overlap integral of a contracted GTO."""
function overlap(μ::CGTO)
    return overlap(μ,μ)
end


"""Returns the overlap integral between two contracted GTOs."""
function overlap(μ::CGTO, ν::CGTO)
    Gμ = primitives(μ); dμ = coefs(μ); Nμ = norms(μ)
    Gν = primitives(ν); dν = coefs(ν); Nν = norms(ν)
    S = 0.0
    for a in 1:nprims(μ)
        for b in 1:nprims(ν)
            S += Nμ[a] * Nν[b] * dμ[a] * dν[b] * overlap(Gμ[a],Gν[b])
        end
    end
    return S
end


"""Returns the kinetic energy integral between two contracted GTOs."""
function kinetic(μ::CGTO, ν::CGTO)
    Gμ = primitives(μ); dμ = coefs(μ); Nμ = norms(μ)
    Gν = primitives(ν); dν = coefs(ν); Nν = norms(ν)
    T = 0.0
    for a in 1:nprims(μ)
        for b in 1:nprims(ν)
            T += Nμ[a] * Nν[b] * dμ[a] * dν[b] * kinetic(Gμ[a],Gν[b])
        end
    end
    return T
end


"""
Returns the nuclear attraction integral between two contracted GTOs and
the nucleus centered at `C`.
"""
function nuclear(μ::CGTO, ν::CGTO, C::AbstractVector{T}) where {T<:Real}
    Gμ = primitives(μ); dμ = coefs(μ); Nμ = norms(μ)
    Gν = primitives(ν); dν = coefs(ν); Nν = norms(ν)
    V = 0.0
    for a in 1:nprims(μ)
        for b in 1:nprims(ν)
            V += Nμ[a] * Nν[b] * dμ[a] * dν[b] * nuclear(Gμ[a],Gν[b],C)
        end
    end
    return V
end


"""Returns the two-electron repulsion integral over four contracted GTOs."""
function repulsion(μ::CGTO, ν::CGTO, λ::CGTO, σ::CGTO)
    Gμ = primitives(μ); dμ = coefs(μ); Nμ = norms(μ)
    Gν = primitives(ν); dν = coefs(ν); Nν = norms(ν)
    Gλ = primitives(λ); dλ = coefs(λ); Nλ = norms(λ)
    Gσ = primitives(σ); dσ = coefs(σ); Nσ = norms(σ)
    V = 0.0
    for a in 1:nprims(μ)
        for b in 1:nprims(ν)
            for c in 1:nprims(λ)
                for d in 1:nprims(σ)
                    V += Nμ[a] * Nν[b] * Nλ[c] * Nσ[d] *
                         dμ[a] * dν[b] * dλ[c] * dσ[d] *
                         repulsion(Gμ[a],Gν[b],Gλ[c],Gσ[d])
                end
            end
        end
    end
    return V
end


### auxiliary functions ###


"""Returns the double factorial n!! of an integer number."""
function dfactorial(n::Int)
    if n == 0
        return 1.0
    elseif iseven(n) && n > 0
        return reduce(*,n:-2:2)[end]
    elseif isodd(n) && n > 0
        return reduce(*,n:-2:1)[end]
    elseif isodd(n) && n < 0
        return 1/reduce(*,n+2:2:1)[end]
    else
        error("n!! undefined for even negative n values")
    end
end


"""
Returns the Hermite expansion coefficients for a 1D Cartesian overlap distribution
using a two-term recursion relation.
"""
function Etij(t::Int ,i::Int, j::Int, Kαβx::Real, XPA::Real, XPB::Real, α::Real, β::Real)

    # compute overlap exponent
    p = α + β

    # enter recursion
    if t < 0 || t > i+j
        return 0.0
    elseif t == 0
        if i == j == 0
            return Kαβx
        elseif j == 0
            return XPA * Etij(0, i-1, j, Kαβx, XPA, XPB, α, β) +
                         Etij(1, i-1, j, Kαβx, XPA, XPB, α, β)
        else
            return XPB * Etij(0, i, j-1, Kαβx, XPA, XPB, α, β) +
                         Etij(1, i, j-1, Kαβx, XPA, XPB, α, β)
        end
    else
        return (1/(2*p*t)) * (i * Etij(t-1, i-1, j, Kαβx, XPA, XPB, α, β) +
                              j * Etij(t-1, i, j-1, Kαβx, XPA, XPB, α, β) )
    end
end


"""Returns the integral of an Hermite Gaussian divided by the Coulomb operator."""
function Rtuv(t::Int, u::Int, v::Int, n::Int, p::Real, RPC::Vector{T}) where {T<:Real}
    if t == u == v == 0
        return (-2.0*p)^n * boys(n,p*abs(sum(RPC.*RPC)))
    elseif u == v == 0
        if t > 1
            return  (t-1)*Rtuv(t-2, u, v, n+1, p, RPC) +
                   RPC[1]*Rtuv(t-1, u, v, n+1, p, RPC)
        else
            return RPC[1]*Rtuv(t-1, u, v, n+1, p, RPC)
        end
    elseif v == 0
        if u > 1
            return  (u-1)*Rtuv(t, u-2, v, n+1, p, RPC) +
                   RPC[2]*Rtuv(t, u-1, v, n+1, p, RPC)
        else
            return RPC[2]*Rtuv(t, u-1, v, n+1, p ,RPC)
        end
    else
        if v > 1
            return  (v-1)*Rtuv(t, u, v-2, n+1, p, RPC) +
                   RPC[3]*Rtuv(t, u, v-1, n+1, p, RPC)
        else
            return RPC[3]*Rtuv(t, u, v-1, n+1, p, RPC)
        end
    end
end


"""Returns the Boys fucntion F_n(x)."""
function boys(n::Int, x::Real)
    return sf_hyperg_1F1(n+0.5, n+1.5, -x) / (2.0*n+1.0)
end

