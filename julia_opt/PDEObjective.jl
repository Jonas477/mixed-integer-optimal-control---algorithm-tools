# PDEObjective.jl
# This file defines an abstract type for PDE-constrained optimal control problems of a special structure.
module PDEObjective

using FEMBundle
using OptBundle

using LinearAlgebra
using Printf

export AbstractPDEObjective

# Evaluation routines (to be implemented)
export G, G_t, Gy!, Gu!

# Helper routines
export i2t, t2i, trange0, trange

@doc raw"""
	abstract type AbstractPDEObjective{T} <: AbstractObjectiveLazy{T}

An abstract PDE-based optimization objective.

# Fields
- `Nglobal_dofs`::UInt64: Degrees of freedom on `mesh` with elements `fe` (which should also be fields)
- `T0`::Float64: Start time.
- `T1`::Float64: End time.
- `nt`::UInt64: Number of time steps.
- `nx`::UInt64: Number of controls.
- `tau`::Float64: Time step.
- `x`::Matrix{Float64}: Optimization variable (control).
- `state`::Matrix{Float64}: Matrix to store the state at each time step.
- `adjoint`::Matrix{Float64}: Matrix to store the adjoint variables.
- `f`::Base.RefValue{Float64}: Cached objective value.
- `df`::Matrix{Float64}: Gradient of the objective w.r.t. `x`.
- `df_valid`::Base.RefValue{Bool}: Indicates whether `df` is up-to-date.
- `f_evals`::Base.RefValue{Int}: Counter for objective evaluations.
- `df_evals`::Base.RefValue{Int}: Counter for gradient evaluations.

It is assumed that the optimization problem has the following structure: 
		Min ∫∫_Ω G(`x`,y) dA dt + ∫ G_t(`x`) dt
		s.t. ∂y/∂t + ∑ᵢⱼ aᵢⱼ DᵢDⱼy + ∑ⱼ bⱼ Dⱼy + c y = ∑ᵢfᵢ`xᵢ`  on Ω×(`T0`,`T1`)
			 y(`T0`) = `state0`
			 ∂y/∂n + g y = α  								 on Γ×(`T0`,`T1`)
Here, y corresponds to `state` and Ω is described by `mesh`. The outer integral is over [`T0`, `T1`]. 
This range is discretized by an equidistant grid {tᵢ} with tᵢ = `T0` + i⋅`tau` for i = 0,...,`nt`.

Several functions need to be implemented:
	function G( obj, i, x_i )
		Evaluates G at tᵢ and returns the value.
	function Gy!( obj, Gyval, i, x_i )
		Evaluates derivative of G w.r.t. the state y at tᵢ and stores it in Gyval.
	function Gu!( obj, Guval, i, x_i ) end
		Evaluates derivative of G w.r.t. the control x at tᵢ and stores it in Guval.

Furthermore, in the objective preamble, several matrices need to be assembled / precalculated:
- `A`: contains contribution to weak form from coefficient functions aᵢⱼ, bⱼ, c aswell as from the boundary condition (function α)
- `M`: mass matrix
- `F`: contains contributions from the right hand side (functions fᵢ independent of time) to the weak formulation
	   aswell as from the boundary condition (function g)
- `state0`: y at t=`T0` (of dimension `Nglobal_dofs`), will be entered into `state[:,1]`
- `M_invA`: precalculated Matrix M⁻¹A
- `M_invF`: precalculated Matrix M⁻¹F
- `SMatLU`: precalculated LU Decomposition of I + `tau`⋅M⁻¹A
- `AMatLU`: precalculated LU Decomposition of I + `tau`⋅(M⁻¹A)ᵀ
"""
abstract type AbstractPDEObjective{T} <: AbstractObjectiveLazy{T}
end

@doc raw"""
		function i2t( obj::AbstractPDEObjective, i )
Convert a time step index to the corresponding time.

# Arguments
- `obj::AbstractPDEObjective`: The PDE objective object.
- `i`: Time step index.

# Returns
- Time corresponding to the time step index `i`.
"""
function i2t( obj::AbstractPDEObjective, i )
	return obj.T0 + i*obj.tau
end

@doc raw"""
		function t2i( obj::AbstractPDEObjective, t )
Convert a time to the corresponding time step index.

# Arguments
- `obj::AbstractPDEObjective`: The PDE objective object.
- `t`: Time.

# Returns
- Time step index corresponding to the time `t`.
"""
function t2i( obj::AbstractPDEObjective, t )
	return UInt64( (t - obj.T0) / obj.tau )
end

@doc raw"""
		function trange0( obj::AbstractPDEObjective )
Returns the range from `T0` to `T1` with `nt+1` equidistant steps of distance `tau`.

# Arguments
- `obj::AbstractPDEObjective`: The PDE objective object.

# Returns
- A range covering time from `T0` to `T1`.
"""
function trange0( obj::AbstractPDEObjective )
	return range(obj.T0,obj.T1,length=obj.nt+1)
end

@doc raw"""
		function trange( obj::AbstractPDEObjective )
Returns the range from `T0 + tau` to `T1` with `nt` equidistant steps steps of distance `tau`.

# Arguments
- `obj::AbstractPDEObjective`: The PDE objective object.

# Returns
- A range covering time from `T0 + tau` to `T1`.
"""
function trange( obj::AbstractPDEObjective )
	return range(obj.T0+obj.tau,obj.T1,length=obj.nt)
end

# Special implicit euler method to calculate the state of the discretized time-dependent PDE
function impleuler_state!(obj,x)
	obj.state[:,1] .= obj.state0
	# Use precalculated LU Decomposition of state matrix
	MMat = obj.SMatLU

	for i=2:obj.nt+1
		# Use precalculated Matrix for right hand side
		obj.state[:,i] .= MMat \ (obj.state[:,i-1] + obj.tau*obj.M_invF*x[:,i-1])
	end

end

# Implements eval_f_helper for AbstractODEObjectives
function OptBundle.eval_f_helper( obj::AbstractPDEObjective, x, cache_me::Val{X}) where X

	## Solve time-dependent ODE with own implicit Euler
	impleuler_state!(obj,hcat(x,x[:,end]))

	# Calculate Objective value, assume area integral is already evaluated in G, G_t
	fval = .5 * (G(obj,1,x[:,1]) + G_t(obj, 1, x[:,1]))
	for i in 2:obj.nt
			@views fval += G(obj,i,x[:,i]) + G_t(obj, i, x[:,i])
	end
	@views fval += .5 * (G(obj,obj.nt+1,x[:,end]) + G_t(obj, obj.nt+1, x[:,end]))
	fval *= obj.tau

	return fval
end

# Special implicit euler method to calculate the adjoint of the discretized time-dependent PDE
function impleuler_adjoint!(obj)
	# Initialize Gy vector
	Gyval = zeros(Float64,obj.Nglobal_dofs)

	obj.adjoint[:,obj.nt+1] .= 0

	# Use precalculated LU Decomposition of A
	MMat = obj.AMatLU
	for i = obj.nt:-1:1
		Gy!(obj,Gyval,i)
		obj.adjoint[:,i] .= MMat \ (obj.adjoint[:,i+1] + obj.tau*Gyval)
	end 

end

function OptBundle.eval_df_helper( obj::AbstractPDEObjective )

	# Calculate Fyᵀ, Fuᵀ using precalculated matrices M⁻¹A, M⁻¹F
	FyvalT = -obj.M_invA'
	FuT = obj.M_invF'

	## Solve ODE with implicit Euler
	impleuler_adjoint!(obj)
	obj.df .= 0.

	# Calculate df
	for i = 1:obj.nt
		obj.df[:,i] += FuT*obj.adjoint[:,i]
	end

	# Initialize Gu
	Guval = zeros(Float64, obj.nx)

	# i = 0
	@views Gu!( obj, Guval, 1)
	for i = 2:obj.nt
		@views Gu!( obj, Guval, i)
		obj.df[:,i] += Guval
	end

end


## Some routines to test the derivatives

function G end
function G_t end
function Gy! end
function Gu! end

end

