
@doc raw"""
An implementation of the heat distribution problem in Section 6.2 of *Vector-Valued 
Integer Optimal Control with TV Regularization* by Jonas Marko and Gerd Wachsmuth, see
<https://arxiv.org/abs/2411.06856>
"""
module example_heat

using OptBundle
using FEMBundle
using PDEObjective

using Parameters
using Plots
using LinearAlgebra
using SparseArrays

export HeatObj


# ∂ₜ y + α Δₓy = f₁(t,x)u₁(t) + f₂(t,x)u₂(t) on Ω × [0,T] = [-1,1]² × [0,10] with boundary condition
# ∂y/∂n + g y = α
@with_kw struct HeatObj <: AbstractPDEObjective{ Matrix{Float64} }
	
	## FEM Data

	# Construct mesh using function below
	mesh = construct_mesh()
	# Designate finite element type
	fe = FE_Lagrange(2)

	# Quadrature order, A - Area integrals, B - boundary integrals
	QuadOrderA::Int64 = 3
	QuadOrderB::Int64 = 1

	# Problem data
	T0::Float64 = 0.
	T1::Float64 = 10.
	nt::UInt64 = 500
	
	# Admissible control values for each integer control
	𝓥::Vector{Vector{Int64}} = [[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]]
	# Iterator containing admissible control combinations at any timestep
	iterator = OptBundle.product_iterator(𝓥) # No restrictions on integer control

	## Parameters
	# Coefficient for Heat Dispersion
	alpha = 1.
	# Coefficients for Heater, c1 -> range of Heater (Large values mean smaller range),
	# c2 -> strength of heating
	c1 = [10.,10.]
	c2 = [20.,20.]
	# Coefficient for boundary condition
	κ = .12
	# Outside temperature
	Tout = 0.
	temp0::Float64 = 10.
	tempT::Float64 = 20.
	# Weight for cost of heating
	γ::Float64 = 10.

	# Placement of Heaters
	x1 = [-1,0]
	x2 = [1,0]

	# Designate functions describing coefficient functions aᵢⱼ, bⱼ, c, g, alpha from
	# ∂y/∂t + ∑ᵢⱼ aᵢⱼ DᵢDⱼy + ∑ⱼ bⱼ Dⱼy + c y = ∑ᵢfᵢ`xᵢ`(t) on Ω×(`T0`,`T1`)
	# ∂y/∂n + g y = α on Γ×(`T0`,`T1`)

    h_A = x -> alpha*[1. 0.; 0. 1.]
    h_b = nothing
    h_c = nothing
	h_g = κ*Tout
	h_alpha = κ
	# State at t=0
	y0 = x -> temp0
	# Functions on right hand side
	rhs = [x -> c2[1]*exp.(-c1[1]*sum((x.-x1).^2, dims=1)),
		 x -> c2[2]*exp.(-c1[2]*sum((x.-x2).^2, dims=1))]

	# Fields for PDEObjective, do not modify
	#------------------------------------------------------------------
	quad = quadrature_unit_triangle_area(QuadOrderA)
	bquad = edge -> quadrature_unit_triangle_bdry(edge,QuadOrderB)
	Nglobal_dofs = ndofs(fe,mesh)
	nquad = length(quad)
	nu::UInt64 = 0 # Number of non-integer controls
	nv = length(𝓥)
	nx = nu + nv
	tau::Float64 = (T1 - T0) / nt
	x::Matrix{Float64} = Matrix{Float64}(undef, nx, nt)
	state::Matrix{Float64} = Matrix{Float64}(undef, Nglobal_dofs, nt+1)
	adjoint::Matrix{Float64} = Matrix{Float64}(undef, Nglobal_dofs, nt+1)
	f::Base.RefValue{Float64} = Ref(0.)
	df::Matrix{Float64} = Matrix{Float64}(undef, nx, nt)
	df_valid::Base.RefValue{Bool} = Ref(false)
	f_evals::Base.RefValue{Int} = Ref(0)
	df_evals::Base.RefValue{Int} = Ref(0)
	#------------------------------------------------------------------
	
	# Assemble target temperature distribution
	yd = assemble_yd(mesh,fe,quad,Nglobal_dofs,nt,tempT)

	# Assemble PDE Matrices using routines below, should not be necessary to modify them
	A = assemble_stiffness(mesh,fe,quad,bquad,h_A,h_b,h_c,h_alpha)
	M = assemble_mass(mesh,fe,quad)
	F = assemble_rhs(mesh,fe,quad,bquad,h_g,rhs,nx)	
	state0 = assemble_state0(mesh,fe,quad,M,y0)

	# Precalculations for implicit Euler, do not modify
	M_invA = calculate_M_invA(M,A)
	M_invF = calculate_M_invF(M,F)
	StateMat = spdiagm(ones(Nglobal_dofs)) +tau*M_invA
	SMatLU = lu(StateMat)
	AMatLU = lu(StateMat')
	
end

# Get nice mesh by refining
function construct_mesh()
	mesh = mesh_library("squareg", 1.)
	for i=1:3
		mesh = refine_all_cells(mesh)
	end

	return mesh
end

# Desired temperature = tempT
function assemble_yd(mesh,fe,quad,Nglobal_dofs,nt,tempT)
	return tempT*ones(Float64,Nglobal_dofs,nt+1)
end

# Returns value of G(y,u,t) where state vector resulting from FEM is already calculated
function PDEObjective.G( obj::HeatObj, i, x)

	vec = obj.state[:,i] - obj.yd[:,i]

	return .5*vec'*obj.M*vec
end

# γ ∑ᵢ ∫ uᵢ(t) dt 
function PDEObjective.G_t( obj::HeatObj, i, x)
	return obj.γ*sum(x)
end

# Derivative w.r.t. obj.state =: U, meaning w.r.t. coefficient vector on mesh.
# We have dG/dUⱼ = 0.5* d/dUⱼ ∑_K ∫ (∑_i (Uᵢ-ydᵢ)φᵢ(x))^2 dx, differentiation 
# and pulling the sum out of the integral leads to ∑_i (Uᵢ-ydᵢ) ∑_K ∫_K φᵢφⱼ dx, 
# where the last sum amounts to the mass matrix.
# Use obj.x[:,i] if control is involved
function PDEObjective.Gy!( obj::HeatObj, Gyval, i)
	Gyval .= obj.M*(obj.state[:,i] - obj.yd[:,i])
	return nothing
end

# Use obj.x[:,i], obj.state[:,i]
function PDEObjective.Gu!( obj::HeatObj, Guval, i)
	Guval .= obj.γ*ones(Float64,obj.nx)
	return nothing
end

function test_f(n = 100)
	t1 = time()
	obj = HeatObj(nt = n)

	println("Global dofs: ",obj.Nglobal_dofs)
	time_assembly = (time() - t1)
	println("Assembly:       $time_assembly","s")
	u = obj.x
	u[:,1:Int(obj.nt/2)] .= 3
	u[1,Int(obj.nt/2):end] .= 2
	u[2,Int(obj.nt/2):end] .= 4

	fval = eval_f(obj,u)
	time_final = (time() - t1)
	println("Total Time:     $time_final","s")
	println("---------------------------------")
	println("Objective value: $fval\n")
	color_min = minimum(obj.state)
    color_max = maximum(obj.state)
    println("Minimum: $color_min, Maximum: $color_max")
	animate_solution(obj.mesh,obj.state,obj.tau; v = u)
end

function test_df(n = 100)
	obj = HeatObj(nt = n)
	u = obj.x
	u .= 1	

	falt = eval_f(obj,u)

	t1 = time()

	eval_df!(obj)
	dfval = obj.df

	time_final = (time() - t1)

	println("Total Time:     $time_final","s")
	println("---------------------------------")

	# Calculate directional derivative
	h = randn(size(u))
	dfh = obj.tau*dot(dfval[:,1],h[:,1])
	for i=2:obj.nt-1
		dfh += obj.tau*dot(dfval[:,i],h[:,i])
	end
	dfh += obj.tau*dot(dfval[:,end],h[:,end])

	println("dfh: $dfh, falt: $falt")
	# Compare with finite differences
	t = 10 .^ range(-10, 0, length=11)
	err = zero(t)
	f_diff = zero(t)
	for j in eachindex(t)
		fneu = eval_f(obj,u.+t[j]*h)
		f_diff[j] = (fneu-falt)/t[j]		
		err[j] = norm(f_diff[j] - dfh)
	
		println(t[j],", ",err[j])
	end
end


## Do not change the following routines
#-------------------------------------------------------------------------------
function assemble_stiffness(mesh,fe,quad,bquad,h_A,h_beta,h_c0,h_alpha)
	# Stiffness Matrix
	A, _ = area_integrator(mesh, fe, quad, h_A, h_beta, h_c0, nothing)
	Q, _ = bdry_integrator(mesh,fe,bquad,h_alpha,nothing)
	return A+Q
end

function assemble_mass(mesh,fe,quad)
	# Stiffness Matrix
	M, _ = area_integrator(mesh, fe, quad, nothing, nothing, 1., nothing)
	return M
end

# M⁻¹F
function calculate_M_invF(M,F)
	MM = cholesky(M)
	MinvF = zeros(Float64, size(F))
	_, cols = size(F)

	for i=1:cols
		MinvF[:,i] .= MM\F[:,i]
	end
	return MinvF
end

# M⁻¹A
function calculate_M_invA(M,A)
	MM = cholesky(M)
	MinvA = zeros(Float64, size(A))
	_, cols = size(A)
	for i=1:cols
		MinvA[:,i] .= MM\A[:,i]
	end
	return MinvA
end

# If rhs is Σ fᵢ(x)⋅uᵢ(t), evaluates the vector [f₁(x),…,fₙ(x)]
function assemble_rhs(mesh,fe,quad,bquad,h_g,f,nx)

	# Right hand side functions, every column is a function in the sum
	F = zeros(Float64, ndofs(fe,mesh), nx)
	_,G = bdry_integrator(mesh,fe,bquad,nothing, h_g)

	for i = 1:nx
		_, F[:,i] = area_integrator(mesh, fe, quad, nothing, nothing, nothing, f[i])
		F[:,i] += G
	end

	return F
end

# Value of y at t=0
function assemble_state0(mesh,fe,quad,M,y0)
	_, Y0 = area_integrator(mesh, fe, quad, nothing, nothing, nothing, y0)
	return M \ Y0
end

end
