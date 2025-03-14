using Parameters
using StaticArrays

abstract type AbstractLSInitial end;

"""
Always returns a fixed alpha.
"""
@with_kw struct LSInitialStatic <: AbstractLSInitial
	alpha0::Float64 = 1.0
end

function (lsis::LSInitialStatic)()
	return lsis.alpha0
end

function set_last_alpha!(lsis::LSInitialStatic, alpha)
	# lsis.alpha0 = alpha * 100.0
	return nothing
end

"""
Returns the last alpha, multiplied by a fixed beta
"""
@with_kw struct LSInitialLastInc <: AbstractLSInitial
	alpha0::Base.RefValue{Float64} = Ref(1.0)
	beta::Float64 = 2.0
end

function (lsis::LSInitialLastInc)()
	return lsis.alpha0[]
end

function set_last_alpha!(lsis::LSInitialLastInc, alpha)
	lsis.alpha0[] = alpha * lsis.beta
end


abstract type AbstractLineSearchMethod end

@with_kw struct ArmijoLS{X, LSI} <: AbstractLineSearchMethod
	# Parameters
	beta::Float64 = 0.5
	sigma::Float64 = 0.1
	lsi::LSI = LSInitialStatic()

	# How to conveniently construct x_old??
	# Construct the object with ArmijoLS( zero(obj.x) )??
	x_old::X

	@assert beta > 0.0
	@assert beta < 1.0
	@assert sigma > 0.0
	@assert sigma < 1.0
end

"""
Perform linesearch,
starting at obj.x
in direction d;

Prerequirement:
eval_g!(obj) has been called already
Requirement:
eval_f!(obj) is called after last update to obj.x
"""
function apply_ls!( obj::AbstractObjective, d, a_obj::ArmijoLS )
	# Initial step length
	alpha = a_obj.lsi()

	beta = a_obj.beta
	sigma = a_obj.sigma

	fref::Ref{Float64} = obj.f
	fval::Float64 = fref[]
	gd = dot( obj.df, d )

	@assert gd < 0.

	x = obj.x
	# x_old = deepcopy(x) # Is this good? Should we allocate memory in advance?
	a_obj.x_old .= x

	@. x = a_obj.x_old + alpha * d

	while eval_f!( obj ) > fval + alpha*sigma*gd
		alpha *= beta

		@. x = a_obj.x_old + alpha * d

		if alpha < 1e-10
			throw(ErrorException("Armijo line search failed"))
		end
	end

	set_last_alpha!(a_obj.lsi, alpha)
	return alpha
end

@with_kw struct WolfeLS{X, LSI} <: AbstractLineSearchMethod
	# Parameters

	sigma::Float64  = 1e-2
	beta::Float64   = .5
	tau::Float64    = 1e-1
	gamma::Float64  = 2.
	gamma1::Float64 = .01
	gamma2::Float64 = .01

	maxiter_phase1::Int = 20
	maxiter_phase2::Int = 40

	lsi::LSI = LSInitialStatic(1.0)

	# How to conveniently construct x_old??
	# Construct the object with WolfeLS( zero(obj.x) )??
	x_old::X

	# Keep sure that parameters satisfy the corresponding conditions.
	@assert( 0. < sigma && sigma < tau && tau < 1. )
	@assert( 1. < gamma )
	@assert( 0. < gamma1 && gamma1 <= .5 )
	@assert( 0. < gamma2 && gamma2 <= .5 )
	@assert( maxiter_phase1 > 0 )
	@assert( maxiter_phase2 > 0 )
end

"""
Perform linesearch,
starting at obj.x
in direction d;

Prerequirement:
eval_g!(obj) has been called already
Requirement:
eval_f!(obj) is called after last update to obj.x
"""
function apply_ls!( obj::AbstractObjective, d, w_obj::WolfeLS )
	# Initial step length
	alpha = w_obj.lsi()

	sigma  = w_obj.sigma
	beta   = w_obj.beta
	tau    = w_obj.tau
	gamma  = w_obj.gamma
	gamma1 = w_obj.gamma1
	gamma2 = w_obj.gamma2

	maxiter_phase1 = w_obj.maxiter_phase1
	maxiter_phase2 = w_obj.maxiter_phase2

	f0 = obj.f[]
	df0 = obj.df
	df0d = dot( df0, d )

	@assert df0d < 0.

	sdf0d = sigma * df0d

	# Often, the function values are very noise near the optimum.
	# Therefore, all function values which are equal up to f_eps are considered equal
	# to increase the numerical stability.
	f_eps = (1e-12)*(1+abs(f0));

	# Reference to obj.x and a copy of the old x
	x = obj.x
	# x_old = deepcopy(x) # Is this good? Should we allocate memory in advance?
	w_obj.x_old .= x


	# Evaluation of auxiliary function psi and its derivative.
	function psi(t)
		@. x = w_obj.x_old + t * d
		ft = eval_fdf!( obj )

		psi_val = ft - (f0 + t*sdf0d);
		psi_der = dot(obj.df, d) - sdf0d;

		return psi_val, psi_der
	end

	# Test the strong wolfe conditions.
	function strong_wolfe_conditions( psi_val, psi_der )
		satisfied = ( psi_val <= f_eps && abs( psi_der + sdf0d ) <= tau * abs(df0d) );
	end

	# Phase 1: Find b satisfying strong Wolfe conditions or psi(b) >= 0 or ( psi(b)<= 0 and psi'(b)>=0 )
	k = 1
	a = 0.
	psi_a_val = 0.
	psi_a_der = (1. - sigma)*df0d

	b = alpha
	psi_b_val, psi_b_der = psi(b)

	while k < maxiter_phase1 &&
		!( strong_wolfe_conditions(psi_b_val, psi_b_der) ) &&
		!( psi_b_val >= f_eps || psi_b_der >= 0 )

		# Increase b
		a = b
		b = gamma*b

		# Remember the function values for a.
		psi_a_val, psi_a_der = psi_b_val, psi_b_der

		# Evaluate function and derivative at b.
		psi_b_val, psi_b_der = psi(b);

		k = k + 1;
	end

	if k == maxiter_phase1
		error("Strong Wolfe line search failed in Phase 1.")
	end

	k = 0

	if strong_wolfe_conditions( psi_b_val, psi_b_der )
		#= if visualize
		display('Phase 1 was sufficient.')
		end =#
		t = b

		# Fall through the next loop.
		# Do not return here, because of the visualization at the end.
		k = maxiter_phase2 + 2;
	else
		#= if visualize
		if psi_b_val >= f_eps
		display('Phase 1: detected point violating Armijo condition.');
		else
		display('Phase 1: detected point with positive derivative.')
		end
		end =#
	end

	# Phase 2: Find point satisfying strong Wolfe conditions.
	while k < maxiter_phase2
		# Check that the conditions on the interval are satisfied:
		@assert( (psi_a_val <= f_eps) && (psi_a_der < 0) && (psi_b_val >= f_eps || psi_b_der >= 0) )

		if ( psi_b_val > 1e30 )
			# This step seems to be way to large
			t = (a+b)/2.;
		elseif (psi_a_val < - f_eps || psi_b_val > f_eps )
			# Function values are not noisy.

			# Use cubic interpolation to compute a good guess.
			A = @SMatrix [1 a a^2 a^3; 0 1 2*a 3*a^2; 1 b b^2 b^3; 0 1 2*b 3*b^2]
			B = @SVector [psi_a_val, psi_a_der, psi_b_val, psi_b_der]

			# Solve linear system to obtain the coefficients of the Hermite polynomial.
			X::MVector = A\B

			if abs(X[4]) > 1e-10
				# The polynomial is really cubic.
				if psi_b_der > sigma*abs(df0d)
					# In this case, we can use the minimizer of 
					#      psi_I - sigma*(df0'*d),
					# which corresponds to a minimizer of  f  (and not of  psi)
					X[2] = X[2] + sdf0d
				end

				@assert( (4*X[3]^2-12*X[2]*X[4])/(36*X[4]^2) > 0 )

				# The stationary points are:
				t1 = -X[3]/(3*X[4]) - sqrt((4*X[3]^2-12*X[2]*X[4])/(36*X[4]^2))
				t2 = -X[3]/(3*X[4]) + sqrt((4*X[3]^2-12*X[2]*X[4])/(36*X[4]^2))

				# The first point has precedence.
				if( a <= t1 && t1 <= b )
					t = t1
				else
					t = t2
				end
			else
				# The polynomial is essentially quadratic.
				# Reinterpolate.
				A2 = @SMatrix [1 a a^2; 0 1 2*a; 1 b b^2; 0 1 2*b]
				B2 = @SVector [psi_a_val, psi_a_der, psi_b_val, psi_b_der]

				# Solve linear system to obtain the coefficients of the quadratic polynomial.
				X2::MVector = A2\B2

				if psi_b_der > sigma*abs(df0d)
					# In this case, we can use the minimizer of 
					#      psi_I - sigma*(df0'*d),
					# which corresponds to a minimizer of  f  (and not of  psi)
					X2[2] = X2[2] + sdf0d
				end

				# The stationary point is:
				t = -.5*X2[2]/X2[3]
			end
		else
			# In this case the function values are very noisy.
			# Do *not* use them for interpolation,
			# but use a linear interpolation of the derivative.
			t = a - psi_a_der * (b-a) / (psi_b_der - psi_a_der)
		end

		# Our guess should lie in [a,b].
		@assert a <= t && t <= b

		# Clip t to get a guaranteed reduction of the interval.
		t = max(t, a + gamma1*(b-a) )
		t = min(t, b - gamma2*(b-a) )

		# Evaluate psi
		psi_val, psi_der = psi(t)

		#= if visualize
		tried_t = [tried_t; t];
		end =#

		# Check for the strong Wolfe conditions.
		if strong_wolfe_conditions( psi_val, psi_der )
			break
		else
			# Otherwise, set up new interval.
			if psi_val <= f_eps
				if psi_der < 0
					a = t
					psi_a_val, psi_a_der = psi_val, psi_der
				else
					b = t
					psi_b_val, psi_b_der = psi_val, psi_der
				end
			else
				b = t
				psi_b_val, psi_b_der = psi_val, psi_der
			end
		end

		k = k+1

		if k == maxiter_phase2
			error("Strong Wolfe line search failed in Phase 2.")
		end

	end

	set_last_alpha!(w_obj.lsi, t)

	return t

end

# Todo: Barzilai-Borwein?

