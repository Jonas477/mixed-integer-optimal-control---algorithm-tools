using Printf

include("mesh.jl")
include("FE.jl")
include("assembly.jl")
include("quadrature_unit_triangle.jl")


function benchmark_mesh(refs)
	vertices = [-1 -1; 1 -1; 1 1; -1 1]
	mesh = init_mesh(vertices, 1.0)

	@printf("First mesh:\n%d vertices\n%d triangles\n\n", mesh.np, mesh.nt)

	for i = 1:refs
		mesh = @time refine_all_cells(mesh);
	end

	@printf("Final mesh:\n%d vertices\n%d triangles\n\n", mesh.np, mesh.nt)

	mesh
end

function init()
	mesh = benchmark_mesh(1)

	fe = FE_Lagrange(1)

	quad = quadrature_unit_triangle_area(2)

	A, f = area_integrator(mesh, fe, quad, 1., nothing, 1., 1.)

	u = A \ f
end

function benchmark()
	@time begin
		mesh = benchmark_mesh(9)
		println("Mesh generation")
	end

	fe = FE_Lagrange(1)

	quad = quadrature_unit_triangle_area(2)

	println("Assembly")
	@time A, f = area_integrator(mesh, fe, quad, 1., nothing, 1., 1.)

	println("Solve")
	@time u = A \ f

	println("Solve indefinit")
	A[1,1] = -1
	@time u = A \ f

	println("Solve unsymmetric")
	A[1,2] = 1
	@time u = A \ f

	nothing
end

function clear()
	print("\033c")
end

function run()
	init()
	clear()
	benchmark()
end


run()
