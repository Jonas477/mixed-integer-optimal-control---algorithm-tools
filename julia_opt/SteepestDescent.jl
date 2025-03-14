@with_kw struct SteepestDescent{ T, TObj <: AbstractObjective{T}, TLS <: AbstractLineSearchMethod } <: AbstractLineSearchOptimizer{T, TObj, TLS }
	obj::TObj
	ls::TLS

	# Some parameters
	maxiter::Int = 4000
	tol::Float64 = 1e-8

	# The optimizer state

	# Search direction
	d::T = zero(obj.x)

	# Preconditioned derivative, i.e., gradient
	g::T = zero(obj.x)

	iter::Base.RefValue{Int} = Ref(0)
end

function opt_init( opt::SteepestDescent{T}, obj, x0::Union{Nothing,T} = nothing ) where {T}
	if typeof(x0) != Nothing
		obj.x .= x0
	end

	opt.iter[] = 0
end

function  opt_compute_direction( opt::SteepestDescent, obj )
	opt.d .= -opt.g
end
