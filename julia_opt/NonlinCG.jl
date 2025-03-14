@with_kw struct NonlinCG{ T, TObj <: AbstractObjective{T}, TLS <: AbstractLineSearchMethod } <: AbstractLineSearchOptimizer{T, TObj, TLS }
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

	old_g::T = zero(obj.x)
	old_df::T = zero(obj.x)

	iter::Base.RefValue{Int} = Ref(0)
end

function opt_init( opt::NonlinCG{T}, obj, x0::Union{Nothing,T} = nothing ) where {T}
	if typeof(x0) != Nothing
		obj.x .= x0
	end

	opt.iter[] = 0
end



function  opt_compute_direction( opt::NonlinCG, obj )
	if opt.iter[] == 0
		opt.d .= -opt.g
	else

		# Use the two vectors temporarily
		y = opt.old_df
		yz = opt.old_g

		@. y = obj.df - opt.old_df
		# Preconditioned y
		@. yz = opt.g - opt.old_g

		# Compute next beta using the formula by Hager/Zhang.
		# beta = (yz - 2*d*(yz'*y)/(y'*d))'*r/(y'*d);

		#= @. yz = yz - 2*opt.d*$dot(yz,y)/$dot(y,opt.d)
		beta = dot( yz, obj.df)/dot(y,opt.d) =#

		beta = (dot(yz,obj.df) - 2*dot(opt.d,obj.df)*dot(yz,y)/dot(y,opt.d))/dot(y,opt.d)

		opt.d .= -opt.g + beta * opt.d 
	end

	opt.old_g .= opt.g
	opt.old_df .= obj.df
end
