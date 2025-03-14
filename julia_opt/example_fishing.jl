@doc raw"""
An implementation of the Lotka-Volterra multimode fishing problem, see 
<https://mintoc.de/index.php/Lotka_Volterra_Multimode_fishing_problem>.
"""
module example_fishing

using OptBundle, ODEObjective

using Parameters
using Plots

export LVMObj

@with_kw struct LVMObj <: AbstractODEObjective{ Matrix{Float64} }
	
	# Problem specification: Interval and discretization, number of controls and states
	T0::Float64 = 0.
	T1::Float64 = 12.
	nt::UInt64 = 1200

	# Admissible control values for each control
	𝓥::Vector{Vector{Int64}} = [[0, 1], [0, 1], [0, 1]]
	# Iterator containing admissible control combinations at any timestep
	iterator = OptBundle.bounded_sum_iterator(𝓥, 1, 1) # Exactly one active control at each timestep

	# Initial value for ODE
	state0::Vector{Float64} = [0.5, 0.7]

	# Parameters
	alpha::Float64 = 1.
	beta:: Float64 = 1.
	gamma::Float64 = 1.
	delta::Float64 = 1.
	c1::Float64 = 1.
	c2::Float64 = 1.
    v1 = [.2, 0.4, 0.01]
	v2 = [.1,.2,.1]

	# Fields for ODEObjective, do not modify
	nu::UInt64 = 0 # Number of non-integer Controls
	nv::UInt64 = length(𝓥) # Number of integer Controls
	ny::UInt64 = length(state0) # Number of states
	nx::UInt64 = nu + nv # Total number of controls
	tau::Float64 = (T1 - T0) / nt
	x::Matrix{Float64} = Matrix{Float64}(undef, nx, nt)
	f::Base.RefValue{Float64} = Ref(0.)
	f_evals::Base.RefValue{Int} = Ref(0)
	df::Matrix{Float64} = Matrix{Float64}(undef, nx, nt)
	df_valid::Base.RefValue{Bool} = Ref(false)
	df_evals::Base.RefValue{Int} = Ref(0)
	state::Matrix{Float64} = Matrix{Float64}(undef, ny, nt)
	adjoint::Matrix{Float64} = Matrix{Float64}(undef, ny, nt)
end


function ODEObjective.F!( obj::LVMObj, Fval, i, y, x )
	Fval[1] = y[1] * ( obj.alpha - obj.beta * y[2] - obj.c1 * sum(x.*obj.v1) )
	Fval[2] = y[2] * ( -obj.gamma + obj.delta * y[1] - obj.c2 * sum(x.*obj.v2) )

	return nothing
end

function ODEObjective.Fy!( obj::LVMObj, Fyval, i, y, x)
	Fyval[1,1] = ( obj.alpha - obj.beta * y[2] - obj.c1 * sum(x.*obj.v1) )
	Fyval[1,2] = y[1] * - obj.beta
	Fyval[2,1] = y[2] * obj.delta
	Fyval[2,2] = ( -obj.gamma + obj.delta * y[1] - obj.c2 * sum(x.*obj.v2) )
	
	return nothing
end

function ODEObjective.Fu!( obj::LVMObj, Fuval, i, y, x )
	Fuval[1,:] = y[1] * -obj.c1 * obj.v1
    Fuval[2,:] = y[2] * -obj.c2 * obj.v2
	return nothing
end

# Objective from Sager
function ODEObjective.G( obj::LVMObj, i, y, x )
	return .5*(y[1]-1.)^2 + .5*(y[2]-1.)^2
end

function ODEObjective.Gy!( obj::LVMObj, Gyval, i, y, x )
	Gyval[1] = y[1] - 1.
	Gyval[2] = y[2] - 1.

	return nothing
end

function ODEObjective.Gu!( obj::LVMObj, Guval, i, y, x)
	# Do nothing.
end

function test_df()
	# Specify problem
	obj = LVMObj()
	
	x = obj.x
	x .= .5
	h = randn(size(x))

	# Evaluate derivative in direction h
	fval = eval_f!( obj )
	eval_df!(obj)
	df = obj.df
	dfh = 0.

	for i=1:obj.nt
		dfh += obj.tau*df[:,i]'*h[:,i]
	end

	# Compare with finite differences
	t = 10 .^ range(-10, 0, length=11)
	err = zero(t)

	for j in eachindex(t)
		fnew = eval_f(obj,x.+t[j]*h)
		temp = (fnew-fval)/t[j] - dfh		
		err[j] = abs(temp)
	
		println(t[j],", ",err[j])
	end
end

# End module
end
