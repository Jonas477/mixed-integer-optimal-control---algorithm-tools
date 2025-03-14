using .FEMBundle

using LinearAlgebra
using StaticArrays

function simple_test_FEM(; hmax = 0.01, dirichlet = false, geometry = "squareg" )

	h_A = @SMatrix [1. 0.;0. 1.]
	h_beta = nothing
	h_c = nothing
	# h_f = nothing
	h_f = 1.0

	h_alpha = 1.
	h_g = 1.

	mesh, U = FEM(h_A,h_beta,h_c,h_f,h_alpha,h_g; fe_type = "Lagrange_3",hmax = hmax, geometry = geometry, dirichlet = dirichlet, QuadOrderA = 3, QuadOrderB = 3)
	return mesh, U
end

function FEM(h_A,h_beta,h_c,h_f,h_alpha,h_g; fe_type = "Lagrange_2", hmax = .01, geometry = "squareg", 
			vertices = nothing, dirichlet = false, QuadOrderA = 2, QuadOrderB = 1)

# Select geometry
if isnothing(vertices)
	mesh = mesh_library(geometry, hmax)
else
	mesh = init_mesh(vertices, hmax)
end

# Select finite elements
if fe_type == "Lagrange_1"
	fe = FE_Lagrange{1}()
elseif fe_type == "Lagrange_2"
	fe = FE_Lagrange{2}()
elseif fe_type == "Lagrange_3"
	fe = FE_Lagrange{3}()
elseif fe_type == "Hermite"
	error("Finite element-type not implemented yet, sorry!")
	return
else
	error("Finite Element unknown.")
	return
end

# Generate mesh

quad_area = quadrature_unit_triangle_area(QuadOrderA)
quad_bdry = edge -> quadrature_unit_triangle_bdry(edge,QuadOrderB)

# Assembly
@time A,F = area_integrator(mesh,fe,quad_area,h_A,h_beta,h_c,h_f)
@time Q,G = bdry_integrator(mesh,fe,quad_bdry,h_alpha,h_g)

A += Q
F += G

# For debugging, print trace of matrix.
println(tr(A))
println(sum(F))

@time if dirichlet
	D = dirichlet_constraints(fe,mesh)
	O = spzeros(size(D,1),size(D,1))
	# Solve saddle-point formulation
	US = [A D'; D O] \ [F;zeros(size(D,1))]

	# Extract solution
	U = US[1:ndofs(fe,mesh)]
else
	# Solve Problem
	U = A \ F
end

# Save .vtk of solution data - Refine and calculate new vertices first if higher order elements are used. Also plot with GLMakie
if fe_type == "Lagrange_1"
	write_vtk("Solution-" * fe_type, mesh, U)
	plot_solution(mesh,U,name(fe))
elseif fe_type == "Lagrange_2"
	# Dirty hack
	rmesh = refine_all_cells(mesh)
	write_vtk("Solution-" * fe_type, rmesh, U)
	plot_solution(rmesh,U,name(fe))
elseif fe_type == "Lagrange_3"
	rmesh = refine_all_cells(mesh)
	P1 = prolongation(mesh,rmesh,fe)
	rmesh2 = refine_all_cells(rmesh)
	P2 = prolongation(rmesh,rmesh2,fe)
	U2 = P2*P1*U
	write_vtk("Solution-" * fe_type, rmesh2, U2[1:rmesh2.np])
	plot_solution(rmesh2,U2[1:rmesh2.np],name(fe))
end

return mesh, U
end

function test_FE()

	# Select geometry
	geometry = "squareg";
	# geometry = "lshapeg"
	# geometry = "regulartriangleg"
	# geometry = "unittriangle"
	# geometry = "slitg"

	# Generate mesh
	mesh = mesh_library(geometry, .5)

	# Select finite elements
	fes = [FE_Lagrange{1}(), FE_Lagrange{2}(),FE_Lagrange{3}()#=,FE_Hermite_3_red=#]
	
	fe = fes[1]
	print("\nShape:\n")
	lambda =[0.5 1.;
			0.5 0.;
			0. 0.]
	val = Float64[[0.] [0.]]
	val_x = Float64[[0.] [0.]]
	val_y = Float64[[0.] [0.]]
	val_H = Float64[[0.] [0.]]
	val, val_x,val_y,val_H = shape(fe,lambda,val, val_x, val_y, val_H)
	print(val)
	print("\n")
	print(val_x)
	print("\n")
	print(val_y)
	print("\n")
	print(val_H)
	print("\n")
	
	print("\nDofmap:\n")
	print(dofmap(fe, mesh, 2))

	print("\nDof:\n")
	f(x,y) = x+y
	print(dof(fe, mesh, 3, f))

	print("\nLocal_dofs:\n")
	f(x,y,z) = x+y+z.*z
	print(local_dofs(fe, f))
	
	print("\nDirichlet:\n")
	print(dirichlet_constraints(fe,mesh))
end

function torus_test_FEM(R = 3., r = 1., N = 300, n = 100)
	mesh = torus_mesh(R, r, N, n)

	fe = FE_Lagrange{1}()

	quad_area = quadrature_unit_triangle_area(2)

	@time A,F = area_integrator(mesh,fe,quad_area,1.,nothing,1e-3,nothing)

	# Put a Dirac here.
	F[1] = 1

	# Solve problem
	U = A \ F

	# Write solution
	write_vtk("torus", mesh, U)

	(mesh, U)
end

function moebius_test_FEM(R = 3., w = 3., N = 300, n = round(Int, N*w/(2*pi*R)))
	mesh = moebius_mesh(R, w, N, n)

	fe = FE_Lagrange{1}()

	quad_area = quadrature_unit_triangle_area(2)

	@time A,F = area_integrator(mesh,fe,quad_area,1.,nothing,1e-3,nothing)

	# Put a Dirac here.
	F[round(Int,n/2)] = 1

	# Solve problem
	U = A \ F

	# Write solution
	write_vtk("moebius", mesh, U)

	(mesh, U)
end
