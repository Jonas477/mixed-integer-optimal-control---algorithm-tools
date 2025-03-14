using LinearAlgebra: dot

abstract type AbstractLineSearchOptimizer{T, TObj <: AbstractObjective{T}, TLS <: AbstractLineSearchMethod }
end

function  opt_do_step( opt::AbstractLineSearchOptimizer, obj )
	apply_ls!( obj, opt.d, opt.ls )
end

function opt_converged( opt::AbstractLineSearchOptimizer, obj )
	# Preconditioned norm of derivative
	norm_g = dot(opt.g, obj.df)

	return sqrt(norm_g) < opt.tol
end

function opt_compute_gradient( opt::AbstractLineSearchOptimizer, obj )
	if opt.iter[] == 0
		# In the first iteration, we have to evaluate the objective.
		eval_fdf!( obj )
	else
		# The line search has already computed the function value,
		# we only need to compute the gradient.
		eval_df!( obj )
	end

	# The gradient, i.e., preconditioned derivative
	opt.g .= obj.df
end

function opt_optimize( opt::AbstractLineSearchOptimizer{T}, obj, x0::Union{Nothing,T} = nothing ) where {T}
	opt_init( opt, obj, x0 )

	opt_compute_gradient( opt, obj )

	while opt.iter[] < opt.maxiter && !opt_converged( opt, obj )
		opt_compute_direction( opt, obj )
		opt_do_step( opt, obj )

		opt_compute_gradient( opt, obj )

		opt.iter[] += 1
	end
end

