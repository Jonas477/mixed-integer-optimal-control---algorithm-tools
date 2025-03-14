@doc raw"""
An implementation of the Van der Pol Oscillator problem, see <https://mintoc.de/index.php/Van_der_Pol_Oscillator_(binary_variant)>.
The ODE in the problem is unstable, which may lead to errors for small values of `nt`.
"""
module example_vanderpol

using OptBundle, ODEObjective

using Parameters
using Plots

export VPOObj

@with_kw struct VPOObj <: AbstractODEObjective{ Matrix{Float64} }
	T0::Float64 = 0.
	T1::Float64 = 20.
	nt::UInt64 = 2000

	# Admissible control values for each control
	𝓥::Vector{Vector{Int64}} = [[0, 1], [0, 1], [0, 1]]
	# Iterator containing admissible control combinations at any timestep
	iterator = OptBundle.bounded_sum_iterator(𝓥, 1, 1) # Exactly one active control at each timestep
	
	# Initial value for ODE
	state0::Vector{Float64} = [1.,0.]

	# Parameters
	k1::Float64 = 2.
    k2::Float64 = 3.
    c = [-1.,.75,-2.]
	
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

function ODEObjective.F!( obj::VPOObj, Fval, i, y, x )
	Fval[1] = y[2]
	Fval[2] = (1-y[1]^2)*y[2]*sum(obj.c'*x) - y[1]
	return nothing
end

function ODEObjective.Fy!( obj::VPOObj, Fyval, i, y, x )
	Fyval[1,1] = 0
	Fyval[1,2] = 1
	Fyval[2,1] = -2*y[1]*y[2]*sum(obj.c'*x) - 1
	Fyval[2,2] = (1-y[1]^2)*sum(obj.c'*x) 

	return nothing
end

function ODEObjective.Fu!( obj::VPOObj, Fuval, i, y, x )
	Fuval .= [0 0 0; obj.c' * (1-y[1]^2)*y[2]]
	return nothing
end

# Objective from Sager
function ODEObjective.G( obj::VPOObj, i, y, x )
	return y[1]^2+y[2]^2
end

function ODEObjective.Gy!( obj::VPOObj, Gyval, i, y, x )
	Gyval[1] = 2*y[1]
	Gyval[2] = 2*y[2]
	return nothing
end

function ODEObjective.Gu!( obj::VPOObj, Guval, i, y, x )
	# Do nothing.
end

function test_df()
	# Specify problem
	obj = VPOObj()
	
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

end
