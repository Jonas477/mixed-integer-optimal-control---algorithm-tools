@doc raw"""
An implementation of the Double tank multimode problem, see <https://mintoc.de/index.php/Double_Tank_multimode_problem>.
"""
module example_doubletank

using OptBundle, ODEObjective

using Parameters
using Plots
using LinearAlgebra

export DTMObj

@with_kw struct DTMObj <: AbstractODEObjective{ Matrix{Float64} }
	T0::Float64 = 0.
	T1::Float64 = 10.
	nt::UInt64 = 1000

	# Admissible control values for each control
	𝓥::Vector{Vector{Int64}} = [[0, 1], [0, 1], [0, 1]]
	# Iterator containing admissible control combinations at any timestep
	iterator = OptBundle.bounded_sum_iterator(𝓥, 1, 1) # Exactly one active control at each timestep

	# Initial value for ODE
	state0::Vector{Float64} = [2.,2.]

	# Parameters
	k1::Float64 = 2.
    k2::Float64 = 3.
    c = [1.,.5,2.]

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

function ODEObjective.F!( obj::DTMObj, Fval, i, y, x )

	Fval[1] = obj.c'*x - sqrt(y[1])
	Fval[2] = sqrt(y[1]) - sqrt(y[2])
	return nothing
end

function ODEObjective.Fy!( obj::DTMObj, Fyval, i, y, x)
	Fyval[1,1] = -1/(2*sqrt(y[1]))
	Fyval[1,2] = 0
	Fyval[2,1] = 1/(2*sqrt(y[1]))
	Fyval[2,2] = -1/(2*sqrt(y[2]))
	
	return nothing
end

function ODEObjective.Fu!( obj::DTMObj, Fuval, i, y, x )
	Fuval .= [obj.c';0 0 0]
	return nothing
end

# Objective from Sager
function ODEObjective.G( obj::DTMObj, i, y, x )
	return obj.k1*(y[2]-obj.k2)^2
end

function ODEObjective.Gy!( obj::DTMObj, Gyval, i, y, x )
	Gyval[1] = 0.
	Gyval[2] = 2*obj.k1*(y[2] - obj.k2)
	return nothing
end

function ODEObjective.Gu!( obj::DTMObj, Guval, i, y, x)
	# Do nothing.
end

function test_df()
	# Specify problem
	obj = DTMObj()
	
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

# Seperate test functions needed, as F is not defined for all y
function test_Fy!( obj::DTMObj )
	println("Testing Fy")
	ny = obj.ny

	y = abs.(randn(ny))
	dy = randn(ny)
	# Make sure that the state is ≥ 0
	for i in eachindex(y)
		if y[i] < -dy[i]
			dy[i] = -y[i]
		end
	end

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

		println(t[j]," ", err[j])
	end
end

function test_Fu!( obj )
	println("Testing Fu")

	y = abs.(randn(obj.ny))
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

		println(t[j]," ", err[j])
	end
end

end
