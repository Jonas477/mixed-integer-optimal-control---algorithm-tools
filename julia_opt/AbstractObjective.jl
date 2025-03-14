# AbstractObjective.jl
# This file defines the abstract types and base classes for the optimal control problems.

"""
Abstract type for optimization problems, where T is the type of the optimization variable
"""
abstract type AbstractObjective{T} end


"""
Abstract type for optimization problems, where the function and gradient are evaluated "all at once". User needs to implement
	function eval_fdf_helper( obj, x, df )
which calculates the objective value as well as the derivative.
"""
abstract type AbstractObjectiveAAO{T} <: AbstractObjective{T} end

# Evaluates objective at x
function eval_f( obj::AbstractObjectiveAAO{T}, x::T ) where {T}
	obj.fdf_evals[] += 1
	f = eval_fdf_helper( obj, x, nothing )
	return f
end

# Evaluates objective at obj.x
function eval_f!( obj::AbstractObjectiveAAO, )
	f = eval_f( obj, obj.x )

	# Todo: Actually, we could directly compute df as well...?

	obj.f[] = f
	# obj.x = x
	obj.df_valid[] = false

	return f
end

# Evaluates gradient at obj.x
function eval_df!( obj::AbstractObjectiveAAO )
	if !obj.df_valid[]
		obj.fdf_evals[] += 1
		eval_fdf_helper( obj, obj.x, obj.df )

		obj.df_valid[] = true
	end

	return nothing
end

# Evaluates objective and gradient at obj.x
function eval_fdf!( obj::AbstractObjectiveAAO )
	obj.fdf_evals[] += 1
	f = eval_fdf_helper( obj, obj.x, obj.df )

	obj.f[] = f

	obj.df_valid[] = true

	return f
end

# Needs to be implemented for concrete problem, returns objective value at x and changes field obj.df to correct gradient at obj.x.
function eval_fdf_helper end


"""
Abstract type, where the objective value and gradient are calculated seperately. User needs to implement
	function eval_f_helper( obj, x, cache_me )
	function eval_df_helper( obj )
"""
abstract type AbstractObjectiveLazy{T} <: AbstractObjective{T}
end

# Evaluates obj at x; does not store anything; only increases counter
function eval_f( obj::AbstractObjectiveLazy{T}, x::T ) where {T}
	obj.f_evals[] += 1

	return eval_f_helper( obj, x, Val(false) )
end

# Evaluates obj at obj.x; modifies obj (f, cache, counter)
function eval_f!( obj::AbstractObjectiveLazy )
	obj.f_evals[] += 1

	fval = eval_f_helper( obj, obj.x, Val(true) )

	# Cache the results
	obj.f[] = fval
	obj.df_valid[] = false

	return fval
end

# Evaluates derivative at obj.x; assumes that eval_f! has been called previously for this obj.x
function eval_df!( obj::AbstractObjectiveLazy )
	if !obj.df_valid[]
		obj.df_evals[] += 1
		eval_df_helper( obj )

		obj.df_valid[] = true
	end
	return nothing
end

# Evaluates function and derivative at obj.x
function eval_fdf!( obj::AbstractObjectiveLazy )
	f = eval_f!(obj)
	eval_df!(obj)

	return f
end

# Needs to be implemented for concrete problem, first function returns objective value at x and second function changes field obj.df to correct gradient at obj.x.
function eval_f_helper end
function eval_df_helper end