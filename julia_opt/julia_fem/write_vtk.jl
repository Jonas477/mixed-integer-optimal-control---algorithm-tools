function write_vtk(filename, mesh::Mesh{GD}, data=nothing, pvd=nothing, time=nothing) where GD

	if GD == 2
		# VTK does not like two-dimensional points in the mesh...
		points = [@SVector[mesh.p[i][1], mesh.p[i][2], 0.] for i = 1:mesh.np]
	else
		points = mesh.p
	end
	triangles = [MeshCell(VTKCellTypes.VTK_TRIANGLE, mesh.t[i]) for i = 1:mesh.nt]

	vtk_grid(filename, points, triangles; append=false) do vtk
		if isnothing(data) || isa(data, Array) && length(data) == 0
			# Nothing to write here.
			return
		end
		# Next line is a dirty hack. How to do it correctly?
		if isa(data, Array) && isa(data[1], Pair)
			# Write multiple variables with given names.
			for datapoint in data
				vtk[ datapoint.first ] = datapoint.second
			end
		elseif isa(data, Pair)
			# Write a single variable with given name.
			vtk[ data.first ] = data.second
		else
			# Write a single variable which will be named "u"
			vtk[ "u" ] = data
		end
		if !isnothing(pvd) && !isnothing(time)
			pvd[time] = vtk
		end
	end
end

function pvd_append(pvd, time, mesh, data)
	n = length(pvd.timeSteps) + 1
	filename = replace(pvd.path, r".pvd$" => "_$n")

	write_vtk(filename, mesh, data, pvd, time)
end

function test_vtk()
	filename = "test"

	vertices = [-1 -1; 1 -1; 1 1; -1 1]
	mesh = init_mesh(vertices, 0.01)

	u = randn(size(mesh.p,2))
	c = randn(mesh.nt)
	v = randn(3, mesh.np)

	# Write three variables with given names.
	write_vtk(filename * "1", mesh, ["u" => u, "c" => c, "v" => v])

	# Write a single variable with given name.
	write_vtk(filename * "2", mesh, "v" => v)

	# Write a single variable which will be named "u"
	write_vtk(filename * "3", mesh, u)
end
