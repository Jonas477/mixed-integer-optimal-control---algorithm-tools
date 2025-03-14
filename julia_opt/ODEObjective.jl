# ODEObjective.jl
# This file defines an abstract type for ODE-constrained optimal control problems.

module ODEObjective

using OptBundle

using LinearAlgebra
using Printf

export AbstractODEObjective

# Evaluation routines to be implemented
export G, Gy!, Gu!, F!, Fy!, Fu!

# Helper routines
export i2t, t2i, trange0, trange


@doc raw"""
	abstract type AbstractODEObjective{T} <: AbstractObjectiveLazy{T}

An abstract ODE-based optimization objective. 

# Fields
- `T0`::Float64: Start time.
- `T1`::Float64: End time.
- `nt`::UInt64: Number of time steps.
- `nx`::UInt64: Number of controls.
- `ny`::UInt64: Dimension of the state vector.
- `tau`::Float64: Time step.
- `x`::Matrix{Float64}: Optimization variable (control).
- `state`::Matrix{Float64}: Matrix to store the state at each time step.
- `adjoint`::Matrix{Float64}: Matrix to store the adjoint variables.
- `state0`::Vector{Float64}: Initial state.
- `f`::Base.RefValue{Float64}: Cached objective value.
- `df`::Matrix{Float64}: Gradient of the objective w.r.t. `x`.
- `df_valid`::Base.RefValue{Bool}: Indicates whether `df` is up-to-date.
- `f_evals`::Base.RefValue{Int}: Counter for objective evaluations.
- `df_evals`::Base.RefValue{Int}: Counter for gradient evaluations.

It is assumed that the optimization problem has the following structure: 
		Min ∫_Ω G(`x`,y) dt
		s.t. y' = F(`x`,y)
			 y(`T0`) = `state0`
Here, y corresponds to `state` and Ω = [`T0`,`T1`] is discretized by an equidistant grid {tᵢ} with tᵢ = `T0` + i⋅`tau` for i = 0,...,`nt`.

Several functions need to be implemented:
	function G( obj, i, state, x_i )
		Evaluates G at tᵢ and returns the value.
	function Gy!( obj, Gyval, i, state_i, x_i )
		Evaluates derivative of G w.r.t. the state y at tᵢ and stores it in Gyval.
	function Gu!( obj, Guval, i, state_i, x_i ) end
		Evaluates derivative of G w.r.t. the control x at tᵢ and stores it in Guval.
	function F!( obj, Fval, i, state_i, x_i )
		Evaluates the right-hand side of the ODE F at tᵢ and stores the value in Fval.
	function Fy!( obj, Fyval, i, state_i, x_i)
		Evaluates the Jacobian of F w.r.t. the state y at tᵢ and stores it in Fyval.
	function Fu!( obj, Fuval, i, state_i, x_i)
		Evaluates the Jacobian of F w.r.t. the control x at tᵢ and stores it in Fuval.
"""
abstract type AbstractODEObjective{T} <: AbstractObjectiveLazy{T}
end

@doc raw"""
		function i2t( obj::AbstractODEObjective, i )
Convert a time step index to the corresponding time.

# Arguments
- `obj::AbstractODEObjective`: The ODE objective object.
- `i`: Time step index.

# Returns
- Time corresponding to the time step index `i`.
"""
function i2t( obj::AbstractODEObjective, i )
	return obj.T0 + i*obj.tau
end

@doc raw"""
		function t2i( obj::AbstractODEObjective, t )
Convert a time to the corresponding time step index.

# Arguments
- `obj::AbstractODEObjective`: The ODE objective object.
- `t`: Time.

# Returns
- Time step index corresponding to the time `t`.
"""
function t2i( obj::AbstractODEObjective, t )
	return UInt64( (t - obj.T0) / obj.tau )
end


@doc raw"""
		function trange0( obj::AbstractODEObjective )
Returns the range from `T0` to `T1` with `nt+1` equidistant steps of distance `tau`.

# Arguments
- `obj::AbstractODEObjective`: The ODE objective object.

# Returns
- A range covering time from `T0` to `T1`.
"""
function trange0( obj::AbstractODEObjective )
	return range(obj.T0,obj.T1,length=obj.nt+1)
end

@doc raw"""
		function trange( obj::AbstractODEObjective )
Returns the range from `T0 + tau` to `T1` with `nt` equidistant steps steps of distance `tau`.

# Arguments
- `obj::AbstractODEObjective`: The ODE objective object.

# Returns
- A range covering time from `T0 + tau` to `T1`.
"""
function trange( obj::AbstractODEObjective )
	return range(obj.T0+obj.tau,obj.T1,length=obj.nt)
end

# Implements eval_f_helper for AbstractODEObjectives
function OptBundle.eval_f_helper( obj::AbstractODEObjective, x, cache_me::Val{X}) where X
	Fval = Vector{Float64}(undef, obj.ny)
	state = copy(obj.state0)

	# Evaluate objective, trapezoidal rule
	fval = .5 * G(obj, 0, obj.state0, x[:,1] )

	# Forward Euler
	for i in 0:obj.nt-1
		F!( obj, Fval, i, state, x[:,i+1])
		@. state += obj.tau * Fval

		if X == true
			obj.state[:,i+1] = state
		end

		# Evaluate objective, trapezoidal rule
		if i < obj.nt-1
			@views fval += G(obj, i+1, state, x[:,i+2])
		else
			@views fval += .5 * G(obj, obj.nt-1, state, x[:,obj.nt] )
		end
	end

	fval *= obj.tau
end

# Implements eval_df_helper for AbstractODEObjectives
function OptBundle.eval_df_helper( obj::AbstractODEObjective )

	Fyval = Matrix{Float64}(undef, obj.ny, obj.ny)
	Gyval = Vector{Float64}(undef, obj.ny)
	Fuval = Matrix{Float64}(undef, obj.ny, obj.nx)
	Guval = Vector{Float64}(undef, obj.nx)
	Gyval .= 0.
	Guval .= 0.
	Fyval .= 0.
	Fuval .= 0.

	# Terminal condition
	@views Gy!( obj, Gyval, obj.nt, obj.state[:,obj.nt], obj.x[:,obj.nt])
	@. obj.adjoint[:,obj.nt] = -.5 * obj.tau * Gyval
	
	# Forward Euler for the adjoint
	for i = obj.nt-1:-1:1
		@views Gy!( obj, Gyval, i, obj.state[:,i], obj.x[:,i+1])
		@views Fy!( obj, Fyval, i, obj.state[:,i], obj.x[:,i+1])
		@views obj.adjoint[:,i] .= obj.adjoint[:,i+1] + obj.tau * (Fyval'*obj.adjoint[:,i+1] - Gyval)
	end

	obj.df .= 0.

	for i = 1:obj.nt
		state = (i==1) ? obj.state0 : obj.state[:,i-1]
		@views Fu!(obj, Fuval, i-1, state, obj.x[:,i])
		@views Gu!(obj, Guval, i-1, state, obj.x[:,i])
		obj.df[:,i] .-= Fuval' * obj.adjoint[:,i]
		obj.df[:,i] .+= Guval
	end
end

function test_Fy!( obj )
	println("Testing Fy")
	ny = obj.ny

	y = randn(ny)
	dy = randn(ny)
	u = randn(obj.nx)
	i = rand(1:obj.nt)

	Fval = zeros(ny)
	Fyval = zeros(ny,ny)
	Fvalt = zeros(ny)

	F!( obj, Fval, i, y, u )
	Fy!( obj, Fyval, i, y, u)

	t = 10 .^ range(-16, 0, length=17)
	err = zero(t)

	for j in eachindex(t)
		F!( obj, Fvalt, i, y + t[j]*dy, u )

		temp_vec =  (Fvalt - Fval) ./ t[j] - Fyval*dy
		err[j] = norm( temp_vec )

		@printf("%.2e %e\n", t[j], err[j] )
	end
end

function test_Fu!( obj )
	println("Testing Fu")

	y = randn(obj.ny)
	h = randn(obj.nx)
	i = rand(1:obj.nt)
	u = randn(obj.nx)

	Fval = zeros(obj.ny)
	Fvalt = zeros(obj.ny)
	Fuval = zeros(obj.ny,obj.nx)

	F!( obj, Fval, i, y, u )
	Fu!( obj, Fuval, i, y, u )

	t = 10 .^ range(-16, 0, length=17)
	err = zero(t)

	for j in eachindex(t)
		F!( obj, Fvalt, i, y, u + t[j]*h )

		temp_vec =  (Fvalt - Fval)./t[j] - (Fuval*h)
		err[j] = norm( temp_vec )

		@printf("%.2e %e\n", t[j], err[j] )
	end
end

function G end
function Gy! end
function Gu! end
function F! end
function Fy! end
function Fu! end

end
