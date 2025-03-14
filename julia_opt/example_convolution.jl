@doc raw"""
An implementation of the signal reconstruction problem in Section 6.2 of *integer optimal control
problems with total variation regularization* by Jonas Marko and Gerd Wachsmuth, see
<https://www.esaim-cocv.org/articles/cocv/abs/2023/01/cocv220118/cocv220118.html>.
"""
module example_convolution

using OptBundle

using Parameters
using Plots
using LinearAlgebra

export ConvObj

# Convolution Problem from Leyffer-Manns Paper
@with_kw struct ConvObj <: AbstractObjectiveLazy{ Matrix{Float64} }
	# Problem specification: Interval and discretization, number of controls and states
	T0::Float64 = -1.
	T1::Float64 = 1.
	nt::UInt64 = 2048

	# Admissible control values for each control
	𝓥::Vector{Vector{Int64}} = [[-2, -1, 0, 1, 2]]
	iterator = OptBundle.product_iterator(𝓥)

	# Necessary fields, do not modify
	nu::UInt64 = 0 # Number of non-integer Controls
	nv::UInt64 = length(𝓥) # Number of integer controls
	nx::UInt64 = nu + nv # Total number of controls
	tau::Float64 = (T1 - T0) / nt
	x::Matrix{Float64} = Matrix{Float64}(undef, nx, nt)
	f::Base.RefValue{Float64} = Ref(0.)
	df::Matrix{Float64} = Matrix{Float64}(undef, nx, nt)
	df_valid::Base.RefValue{Bool} = Ref(false)
	f_evals::Base.RefValue{Int} = Ref(0)
	df_evals::Base.RefValue{Int} = Ref(0)

	# Intervals in which (T0,T1) is divided when approximating an integral with Gauß Legendre
	# Use for kernel where antiderivative is unknown
	#approx_quality::Int64 = 50

	# Parameters
	ω₀::Float64 = Float64(pi)
	target = t -> .4*cos(2*pi*t)
	# Convert target function to a vector over specified grid
	fvec = f_to_vec(nt, T0, tau, target)

	# Kernel function
	k = t -> begin
		if t < 0 
			return 0
		else
			a = ω₀*(t-1)/sqrt(2)
			return -(sqrt(2)/10)*ω₀*exp(-a)*sin(a)
		end
	end
	
	# Antiderivative of k
	int_k = t -> begin
		a = ω₀*(t-1)/sqrt(2)
		return 1/10*exp(-a)*(sin(a)+cos(a))
	end	

	## Matrices for fast calculation of objective value and derivative
	# K - contains specific integrals over kernel
	K = toeplitz(T0, nt, tau, int_k)
	# M - contains specific integrals over hat functions for interpolation
	M = Mmat(nt, tau)
end

# Convert target function to vector
function f_to_vec( nt::Number, T0::Float64, tau::Float64, target::Function )

	fvec = zeros(Float64, nt+1)
	for i=1:nt+1
		fvec[i] = target(T0 + tau*i)
	end

	return fvec
end

# Matrix M containing integrals of interpolation hat functions as specified in 
# "integer optimal control problems with total variation regularization", Section 6.2 by Marko, Wachsmuth
function Mmat( nt::Number, tau::Float64)

	M = zeros(Float64, nt+1, nt+1)
	
	M[1,1] = tau/3
	M[nt+1,nt+1] = tau/3
	for i=2:nt
		M[i,i] = 2/3*tau
	end
	for i=1:nt
		M[i,i+1] = tau/6
		M[i+1,i] = tau/6
	end
	
	return M
end

# Generate Toeplitz matrix with integrals over kernel as specified in 
# "integer optimal control problems with total variation regularization", Section 6.2 by Marko, Wachsmuth
function toeplitz(T0::Float64, nt::Number, tau::Float64, int_k::Function)

	K = zeros(Float64, nt+1, nt)
	tj = T0
	Δt = tau#/approx_quality

	for i=2:nt+1
		ti = T0 + (i-1)*tau
		#= Use if antiderivative is unknown
		val = 0
		for j=1:approx_quality
			val += GaußLegendre5(k, ti-tj-tau +(j-1)*Δt, ti-tj-tau+j*Δt)
		end
		=#
		val = int_k(ti-tj) - int_k(ti-tj-tau)
		for j=i:nt+1
			K[j,j-i+1] = val
		end
	end

	return K
end

# Need to implement objective calculation for AbstractObjectiveLazy - Object
function OptBundle.eval_f_helper( obj::ConvObj,  u::Matrix{Float64}, cache)
	
	Ku = obj.K*u'
	v = Ku-obj.fvec
	val = .5*v'*obj.M*v
	# val is a 1x1 Matrix
	return val[1]
end

# Need to implement derivative calculation for AbstractObjectiveLazy - Object
function OptBundle.eval_df_helper( obj::ConvObj )
	
	obj.df .= (obj.K'*(obj.M*(obj.K*obj.x' - obj.fvec)))'
end

# Gauß-Legendre Formula of 5-th Order to approximate integral of `f` over (a,b)
function GaußLegendre5(f::Function, a::Number, b::Number)

	w = [ 0.236926885056189, 0.478628670499366, 0.568888888888889, 0.478628670499366, 0.236926885056189]
	x = [ -0.906179845938664, -0.538469310105683, 0, 0.538469310105683, 0.906179845938664]

    
	y = (b-a)/2 *x .+ (a+b)/2
	val = (b-a)/2*dot(w, f.(y))

	return val
end

function test_df()
	# Specify problem
	obj = ConvObj()
	
	x = obj.x
	x .= 1.
	h = randn(size(x))

	# Evaluate derivative in direction h
	fval = eval_f!( obj )
	eval_df!(obj)
	df = obj.df
	dfh = 0.

	for i=1:obj.nt
		dfh += df[:,i]'*h[:,i]
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
