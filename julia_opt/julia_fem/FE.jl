abstract type FE{T} end
# The parameter T is the value of the shape functions, i.e., the shape functions are T-valued.


function dofmap(fe::FE, mesh, idx)
	# Returns the connectivity matrix (called C_K) of the element on
	# the cell K with index 'idx', i.e., the entries (C_K)_{ji} are the numbers
	# $c^{K,i}_j$.

	global_dofs, i, j, s = flat_dofmap( fe, mesh, idx )
	return sparse( global_dofs[i], j, s, ndofs(fe, mesh), nlocaldofs(fe) )
end



# A continuous Lagrange element of polynomial degree k on triangles
struct FE_Lagrange{k} <: FE{Float64} end

# Convenience constructor
FE_Lagrange(k) = FE_Lagrange{k}()

# defining all methods of FE_Lagrange{k} using multiple dispatch.

function ndofs( ::FE_Lagrange{k}, mesh ) where k
	# The number of global degrees of freedom

	return mesh.np + mesh.ne * (k-1) + mesh.nt * ((k-1) * (k-2)) ÷ 2
end

function nlocaldofs( ::FE_Lagrange{k} ) where k
	# The number of local degrees of freedom

	return (k+1)*(k+2) ÷ 2
end

function vnlocaldofs( ::Type{FE_Lagrange{k}} ) where k
	# The number of local degrees of freedom, returned as type Val{}

	return Val((k+1)*(k+2) ÷ 2)
end

function flat_dofmap(fe::FE_Lagrange{1}, mesh, idx)
	# Returns the information from which the connectivity matrix can be constructed
	global_dofs = mesh.t[idx]
	i = 1:3
	j = 1:3
	s = @SVector [1.; 1.; 1.]

	return global_dofs, i, j, s
end

@doc raw"""
    shape( fe::FE_Lagrange{1}, lambda, ::Val{return_d} = Val(false), ::Val{return_H} = Val(false)) where {return_d, return_H}

This function returns as a vector the values of all local shape functions on
the reference cell evaluated at the point specified by the barycentric
coordinates `lambda[1,:]`, `lambda[2,:]`, `lambda[3,:]`. The latter can be row vectors so that
the evaluation is done simultaneously at all points, and a matrix is
returned.  The function also returns the elements of the gradient of the
shape functions and the elements of the Hessian.
"""
function shape( fe::FE_Lagrange{1}, lambda, ::Val{return_d} = Val(false), ::Val{return_H} = Val(false)) where {return_d, return_H}

	leng = size(lambda, 2)
	val = [SVector{3,Float64}(lambda[:,i]) for i in 1:leng]

	if !return_d
		return val
	end

	Z = zeros(Float64,1,leng)
	E = ones(Float64,1,leng)

	# Compute also the gradient
	dpdLambda = [E Z Z;
							 Z E Z;
							Z Z E]

	I = Matrix{Float64}(LinearAlgebra.I, leng, leng)
	O = zeros(leng,leng)

	val_x = dpdLambda * [-I; I; O]
	val_y = dpdLambda * [-I; O; I]

	dshape = [SMatrix{2,3}([val_x[:,nquad] val_y[:,nquad]]') for nquad in 1:leng]

	if !return_H
		return val, dshape
	end

	# Each subarray [i,j,:,:] in this array
	# is the Hessian of shape function i
	# in the point j (of the input)
	val_H = zeros(Float64,3,leng,2,2)

	return val,dshape,val_H
end

# Evaluates global dof 'i' on the mesh 'mesh' at the function 'f' (which is assumed to be given in usual Cartesian coordinates)
function dof(fe::FE_Lagrange{1},mesh, i, f)
	coords = mesh.p[i]
	return f(coords[1], coords[2])
end

# Evaluates local dofs at the function 'f' (which is assumed to be given in barycentric coordinates with 3 outputs)
function local_dofs(fe::FE_Lagrange{1},f)
	coordinates = [1 0 0;
				   0 1 0;
				   0 0 1]
	
	val, _, _ = f(coordinates')

	return val
end
	
function dirichlet_constraints(fe::FE_Lagrange{1},mesh)
	nbe = length(mesh.be)
	dirichlet_nodes = Vector{Int}(undef, 2*nbe)
	for i = 1:nbe
		dirichlet_nodes[2*i - 1] = mesh.e[mesh.be[i][1]][1]
		dirichlet_nodes[2*i    ] = mesh.e[mesh.be[i][1]][2]
	end
	unique!(dirichlet_nodes)

	i = 1:length(dirichlet_nodes)
	j = dirichlet_nodes
	s = ones(size(i))
	
	return sparse(i,j,s, length(i), ndofs(fe,mesh))
end

function name(fe::FE_Lagrange{1})
	return "Linear Lagrange"
end

function flat_dofmap(fe::FE_Lagrange{2}, mesh, idx)
	global_dofs = @SVector [mesh.t[idx][1]; mesh.t[idx][2]; mesh.t[idx][3];
		mesh.np + mesh.cell_to_edge[idx][1];mesh.np + mesh.cell_to_edge[idx][2];mesh.np + mesh.cell_to_edge[idx][3]]
	i = 1:6
	j = 1:6
	s = @SVector [1.; 1.; 1.; 1.; 1.; 1.]
	return global_dofs, i, j, s
end

function shape(fe::FE_Lagrange{2}, lambda, ::Val{return_d}=Val(false), 
		::Val{return_H}=Val(false)) where {return_d,return_H}


    leng = size(lambda, 2)
    val = zeros(6, leng)

    val = Vector{SVector{6,Float64}}(undef, leng)

    for i=1:leng
        val[i] = @SVector[
                          lambda[1, i] .* (2 * lambda[1, i] .- 1),
                          lambda[2, i] .* (2 * lambda[2, i] .- 1),
                          lambda[3, i] .* (2 * lambda[3, i] .- 1),
                          4 * lambda[2, i] .* lambda[3, i],
                          4 * lambda[3, i] .* lambda[1, i],
                          4 * lambda[1, i] .* lambda[2, i]
                         ]
    end

    if !return_d
        return val
    end

    Z = zeros(Float64, 1, leng)

    # Compute also the gradient
    dpdLambda = [((2 * lambda[1, :] .- 1) + 2 * lambda[1, :])' Z Z;
        Z ((2 * lambda[2, :] .- 1) + 2 * lambda[2, :])' Z;
        Z Z ((2 * lambda[3, :] .- 1) + 2 * lambda[3, :])';
        Z 4*lambda[3, :]' 4*lambda[2, :]';
        4*lambda[3, :]' Z 4*lambda[1, :]';
        4*lambda[2, :]' 4*lambda[1, :]' Z]

    I = Matrix{Float64}(LinearAlgebra.I, leng, leng)
    O = zeros(leng, leng)

    val_x = dpdLambda * [-I; I; O]
    val_y = dpdLambda * [-I; O; I]

    dshape = [SMatrix{2,6}([val_x[:, nquad] val_y[:, nquad]]') for nquad in 1:leng]

    if !return_H
        return val, dshape
    end
	
	# Each subarray [i,j,:,:] in this array
    # is the Hessian of shape function i
    # in the point j (of the input)
    val_H = zeros(Float64, 6, leng, 2, 2)

    K = [-1 -1; 1 0; 0 1]
    for j = 1:leng
        val_H[1, j, :, :] .= K' * [4 0 0; 0 0 0; 0 0 0] * K
        val_H[2, j, :, :] .= K' * [0 0 0; 0 4 0; 0 0 0] * K
        val_H[3, j, :, :] .= K' * [0 0 0; 0 0 0; 0 0 4] * K
        val_H[4, j, :, :] .= K' * [0 0 0; 0 0 4; 0 4 0] * K
        val_H[5, j, :, :] .= K' * [0 0 4; 0 0 0; 4 0 0] * K
        val_H[6, j, :, :] .= K' * [0 4 0; 4 0 0; 0 0 0] * K
    end

    return val, dshape, val_H
end

function dof(fe::FE_Lagrange{2},mesh, i, f)		#brauchen i nicht bei Lagrange 2?
	# Coordinates of vertices
	p = mesh.p;
	# Coordinates of midpoints of the edges
	pe =(mesh.p[mesh.e[1,:]] + mesh.p[mesh.e[2,:]] ) / 2
	
	p_all = hcat(p,pe);
	return f(p_all[1,:], p_all[2,:])
end

# Evaluates local dofs at the function 'f' (which is assumed to be given in barycentric coordinates with 3 outputs)
function local_dofs(fe::FE_Lagrange{2},f)
	coordinates = [
			1 0 0;
			0 1 0;
			0 0 1;
			0 0.5 0.5;
			0.5 0 0.5;
			0.5 0.5 0;
			]
	
	val, _, _ = f(coordinates')

	return val
end
	
function dirichlet_constraints(fe::FE_Lagrange{2},mesh)
	nbe = length(mesh.be)
	dirichlet_nodes = Vector{Int}(undef, 2*nbe)
	for i = 1:nbe
		dirichlet_nodes[2*i - 1] = mesh.e[mesh.be[i][1]][1]
		dirichlet_nodes[2*i    ] = mesh.e[mesh.be[i][1]][2]
	end
	unique!(dirichlet_nodes)
	
	# Now, find all edges on the boundaries for which the end points are Dirichlet nodes (Boundary edges)
	dirichlet_edges = [mesh.be[i][1] for i in 1:nbe]

	i = 1:(length(dirichlet_nodes)+length(dirichlet_edges))
	j = vcat(dirichlet_nodes,mesh.np .+ dirichlet_edges)
	s = ones(size(i))

	return sparse(i,j,s, length(i), ndofs(fe,mesh))
end

function name(fe::FE_Lagrange{2})
	return "Quadratic Lagrange"
end

function flat_dofmap(fe::FE_Lagrange{3}, mesh, idx)

	edge_dof_rows = @MVector zeros(Int,6)
	for i = 1:3
		if mesh.t[idx][mod(i,3)+1] == mesh.e[mesh.cell_to_edge[idx][i]][1]
			# Global edge has same alignment as triangle edge
			flip = 0
		else
			# Global edge has different alignment as triangle edge
			flip = 1
		end
		edge_dof_rows[2*i-1] = mesh.np + 2*mesh.cell_to_edge[idx][i] - 1 + flip
		edge_dof_rows[2*i] = mesh.np + 2*mesh.cell_to_edge[idx][i]     - flip
	end


	global_dofs = SVector(mesh.t[idx][1], mesh.t[idx][2], mesh.t[idx][3], edge_dof_rows..., mesh.np + 2*mesh.ne + idx)
	i = 1:10
	j = 1:10
	s = @SVector [1.; 1.; 1.; 1.; 1.; 1.; 1.; 1.; 1.; 1.]
	# The dofmap is sparse( global_dofs[i], j, s, ndofs(fe, mesh), nlocaldofs(fe) )
	return global_dofs, i, j, s
end

function shape( fe::FE_Lagrange{3}, lambda, ::Val{return_d} = Val(false), 
		::Val{return_H} = Val(false)) where {return_d, return_H}
	
	leng = length(lambda[1,:])

	alpha0 = 3*lambda[1,:].-1
	alpha1 = 3*lambda[2,:].-1
	alpha2 = 3*lambda[3,:].-1
	beta0 = 3*lambda[1,:].-2
	beta1 = 3*lambda[2,:].-2
	beta2 = 3*lambda[3,:].-2

	val = Vector{SVector{10,Float64}}(undef, leng)
	
	for i=1:leng
		val[i] = @SVector[
											1/2*lambda[1,i].*(alpha0[i]).*(beta0[i]),
											1/2*lambda[2,i].*(alpha1[i]).*(beta1[i]),
											1/2*lambda[3,i].*(alpha2[i]).*(beta2[i]),

											9/2*lambda[2,i].*(alpha1[i]).*lambda[3,i],
											9/2*lambda[3,i].*(alpha2[i]).*lambda[2,i],

											9/2*lambda[3,i].*(alpha2[i]).*lambda[1,i],
											9/2*lambda[1,i].*(alpha0[i]).*lambda[3,i],

											9/2*lambda[1,i].*(alpha0[i]).*lambda[2,i],
											9/2*lambda[2,i].*(alpha1[i]).*lambda[1,i],

										 27*lambda[1,i].*lambda[2,i].*lambda[3,i],
										 ]
	end

	if !return_d
		return val
	end
	
	Z = zeros(Float64,1, leng);

	# Compute also the gradient
	dpdLambda = [
			(1/2*(alpha0).*(beta0) + 1/2*lambda[1,:].*(beta0)*3 + 1/2*lambda[1,:].*(alpha0)*3)'    Z                      Z; 
			Z              (1/2*(alpha1).*(beta1) + 1/2*lambda[2,:].*(beta1)*3 + 1/2*lambda[2,:].*(alpha1)*3)'            Z; 
			Z              Z                      	(1/2*(alpha2).*(beta2) + 1/2*lambda[3,:].*(beta2)*3 + 1/2*lambda[3,:].*(alpha2)*3)'; 
			Z                      	(9/2*lambda[2,:].*3 .*lambda[3,:] + 9/2*(alpha1).*lambda[3,:])'                   (9/2*lambda[2,:].*(alpha1))'; 
			Z                      	(9/2*lambda[3,:].*(alpha2))'                   (9/2*(alpha2).*lambda[2,:] + 9/2*lambda[3,:].*3 .*lambda[2,:])'; 
			(9/2*lambda[3,:].*(alpha2))'  	Z                      					(9/2*(alpha2).*lambda[1,:] + 9/2*lambda[3,:].*3 .*lambda[1,:])'; 
			(9/2*(alpha0).*lambda[3,:] + 9/2*lambda[1,:].*3 .*lambda[3,:])'                      Z                      (9/2*lambda[1,:].*(alpha0))'; 
			(9/2*(alpha0).*lambda[2,:] + 9/2*lambda[1,:].*3 .*lambda[2,:])'                      (9/2*lambda[1,:].*(alpha0))'                      Z; 
			(9/2*lambda[2,:].*(alpha1))'                      (9/2*(alpha1).*lambda[1,:] + 9/2*lambda[2,:].*3 .*lambda[1,:])'                      Z; 
			(27*lambda[2,:].*lambda[3,:])'                      (27*lambda[1,:].*lambda[3,:])'                      (27*lambda[1,:].*lambda[2,:])']

	I = Matrix{Float64}(LinearAlgebra.I, leng, leng)
	O = zeros(leng,leng)

	val_x = dpdLambda * [-I; I; O]
	val_y = dpdLambda * [-I; O; I]
	
	dshape = [SMatrix{2,10}([val_x[:,nquad] val_y[:,nquad]]') for nquad in 1:leng]

	if !return_H
		return val, dshape
	end

	# Each subarray [i,j,:,:] in this array
	# is the Hessian of shape function i
	# in the point j (of the input)
	val_H = zeros(Float64,10,leng,2,2)

	K = [-1 -1; 1 0; 0 1];
	for j=1:leng
		val_H[1,j,:,:] .= K' * [27*lambda[1,j]-9 0 0; 0 0 0; 0 0 0] * K
		val_H[2,j,:,:] .= K' * [0 0 0; 0 27*lambda[2,j]-9 0; 0 0 0] * K
		val_H[3,j,:,:] .= K' * [0 0 0; 0 0 0; 0 0 27*lambda[3,j]-9] * K
		val_H[4,j,:,:] .= K' * [
			0 0                 0                 ;
			0 27*lambda[3,j]     27*lambda[2,j]-9/2 ;
			0 27*lambda[2,j]-9/2 0                 ;
			] * K
		val_H[5,j,:,:] .= K' * [
			0 0                 0                 ;
			0 0                 27*lambda[3,j]-9/2 ;
			0 27*lambda[3,j]-9/2 27*lambda[2,j]    ;
			] * K
		val_H[6,j,:,:] .= K' * [
			0                 0 27*lambda[3,j]-9/2 ;
			0                 0 0                 ;
			27*lambda[3,j]-9/2 0 27*lambda[1,j]     ;
			] * K
		val_H[7,j,:,:] .= K' * [
			27*lambda[3,j]     0 27*lambda[1,j]-9/2 ;
			0                 0 0                 ;
			27*lambda[1,j]-9/2 0 0                 ;
			] * K
		val_H[8,j,:,:] .= K' * [
			27*lambda[2,j]    27*lambda[1,j]-9/2 0 ;
			27*lambda[1,j]-9/2 0                 0 ;
			0                 0                 0 ;
			] * K
		val_H[9,j,:,:] .= K' * [
			0                 27*lambda[2,j]-9/2 0 ;
			27*lambda[2,j]-9/2 27*lambda[1,j]     0 ;
			0                 0                 0 ;
			] * K
		val_H[10,j,:,:] .= 27 * K' * [
			0          lambda[3,j] lambda[2,j] ;
			lambda[3,j] 0          lambda[1,j] ;
			lambda[2,j] lambda[1,j] 0          ;
			] * K
	end

	return val,dshape,val_H
end

function dof(fe::FE_Lagrange{3},mesh, i, f)
	if i <= mesh.np
		# Vertex dof
	else
		# Edge dof
	end
end

function local_dofs(fe::FE_Lagrange{3},f)
	coordinates = [
	1 0 0;
	0 1 0;
	0 0 1;
	0 2/3 1/3;
	0 1/3 2/3;
	1/3 0 2/3;
	2/3 0 1/3;
	2/3 1/3 0;
	1/3 2/3 0;
	1/3 1/3 1/3;
	]

	val, _, _ = f(coordinates')

	return val
end

function dirichlet_constraints(fe::FE_Lagrange{3},mesh)
	dirichlet_nodes = unique(reshape(mesh.e[1:2,mesh.be[1,:]],1,:))
	
	# Now, find all edges on the boundaries for which the end points are Dirichlet nodes
	dirichlet_edges = mesh.be[1,:]

	i = 1:(length(dirichlet_nodes)+2*length(dirichlet_edges))
	j = vcat(dirichlet_nodes, mesh.np .+ 2*dirichlet_edges , mesh.np .+ 2*dirichlet_edges .- 1)
	s = ones(size(i));

	return sparse(i,j,s, length(i), ndofs(fe,mesh))
end

function name(fe::FE_Lagrange{3})
	return "Cubic Lagrange"
end

function plot_shape_functions(fe, refs=3, mesh::Mesh{GD}=triangle_mesh(), adaptive=false) where GD
	n = ndofs(fe, mesh)

	rmesh = mesh
	for i = 1:refs
		if adaptive
			rmesh = refine_adaptively(rmesh, (1:rmesh.nt)[rand(Bool,rmesh.nt)])
		else
			rmesh = refine_all_cells(rmesh)
		end
	end
	P = prolongation(mesh, rmesh, fe, FE_Lagrange(1))

	paraview_collection(name(fe)) do pvd
		for i = 1:n
			U = Vector(P[:,i])
			pvd_append(pvd, i-1, rmesh, U)
		end
	end

end
