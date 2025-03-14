@with_kw struct Mesh{GD,GD2}
    geometry::Matrix{Float64}
    p::Vector{SVector{GD,Float64}}
    # - p                 ... the vertices of the mesh
    #   * each column contains the coordinates of one vertex
    e::Vector{SVector{4,Int}}
    # - e                 ... the edges of the mesh
    #   * each column contains the index (in p) of the incident vertices (pos. 1-2), triangle on right (3), triangle on left (4)
    t::Vector{SVector{3,Int}}
    # - t                 ... the triangles of the mesh
    #   * the first three entries of each column contain the indices (in p) of the incident vertices
    np::Int
    ne::Int
    nt::Int
    # - np, ne, nt        ... number of vertices, edges, triangles
    be::Vector{SVector{2,Int}}
    # - be                ... the boundary edges of the mesh
    #   * pos 1: index of edge in e. pos. 2: number of the part of the boundary this edge belongs to.
    cell_to_edge::Vector{SVector{3,Int}}
    # - cell_to_edge      ... incidence structure of the mesh
    #   * column i contains the indices of the edges (in edges) incident to triangle i (in p)
    #     note that edge j is opposite to vertex j
    affine_matrix::Vector{SMatrix{GD,2,Float64,GD2}}
    # - affine_matrix     ... contains the transformation matrices B_K
    affine_vector::Vector{SVector{GD,Float64}}
    # - affine_vector     ... contains the transformation vectors b_K
    affine_invmatrixT::Vector{SMatrix{GD,2,Float64,GD2}}
    # - affine_invmatrixT ... contains the matrices B_K^{-T}

    function Mesh(geometry,p::Vector{SVector{GD,Float64}},be,t::Vector{SVector{3,Int}},segmentmarkerlist; align_triangles = false) where GD
        np = length(p)
        nt = length(t)

        if align_triangles
            align_triangles!(t, p)
        end

        #we will now compute everything.
        e, ne = compute_e_and_ne(t, nt, np, Val(GD))
        cell_to_edge = compute_cell_to_edge(t,nt,e,ne)
        # affine_matrix, affine_vector, affine_invmatrixT = compute_matrices_and_vector(p, t, nt)
        be = change_be(e,ne,be,segmentmarkerlist)

        affine_matrix, affine_vector, affine_invmatrixT = compute_transformation_matrices(p,t,nt)
        return new{GD,2*GD}(geometry,p,e,t,np,ne,nt,be,cell_to_edge,affine_matrix,affine_vector,affine_invmatrixT)
    end
end

# Some predefined geometries
function mesh_library(geometry, hmax)

    if geometry == "squareg"
		vertices = [-1 -1; 1 -1; 1 1; -1 1]
	elseif geometry == "lshapeg"
		vertices = [-1 -1; 1 -1; 1 1; 0 1; 0 0;	-1 0]
	elseif geometry == "regulartriangleg"
		vertices = [cos(0*2*pi/3) sin(0*2*pi/3); cos(1*2*pi/3) sin(1*2*pi/3); cos(2*2*pi/3) sin(2*2*pi/3)]
	elseif geometry == "unittriangle"
		vertices = [0 0; 1 0; 0 1]
	elseif geometry == "slitg"
		vertices = [-1 -1; 1 -1; 1 0; 0 0; 1 1e-2; 1 1;	-1 1]
	else
		println("Geometry not recognized, please designate `vertices` of your region Ω such that Ω is the convex hull of `vertices` and use the `init_mesh` function!")
		return
	end

    return init_mesh(vertices, hmax)
end

function align_triangles!(t::Vector{SVector{3,Int}}, p::Vector{SVector{GD,Float64}}) where GD

    # Check that the longest edge is the first edge and that the triangles are oriented counterclockwise
    for i = 1:length(t)
        ti = t[i]
        idx1 = ti[1]
        idx2 = ti[2]
        idx3 = ti[3]
        p1 = p[idx1]
        p2 = p[idx2]
        p3 = p[idx3]

        e1 = norm( p2 - p3 )
        e2 = norm( p3 - p1 )
        e3 = norm( p1 - p2 )

        if GD == 2
            o = ((p1[2]+p2[2])*(p1[1]-p2[1]) + (p2[2]+p3[2])*(p2[1]-p3[1]) + (p3[2]+p1[2])*(p3[1]-p1[1])) > 0
        else
            # No orientation check possible
            o = true
        end

        if e1 >= e2 && e1 >= e3
            # First edge longest
            if o
                t[i] = @SVector[idx1, idx2, idx3]
            else
                t[i] = @SVector[idx1, idx3, idx2]
            end
        elseif e2 >= e3
            # Second edge longest
            if o
                t[i] = @SVector[idx2, idx3, idx1]
            else
                t[i] = @SVector[idx2, idx1, idx3]
            end
        else
            # Third edge longest
            if o
                t[i] = @SVector[idx3, idx1, idx2]
            else
                t[i] = @SVector[idx3, idx2, idx1]
            end
        end

    end
end

function compute_e_and_ne(tlist,nt,np,::Val{GD}) where GD
    e = Vector{SVector{4, Int}}(undef, 3*nt)

    #we first put all the edges (some of them are doubled) in the array.
    @inbounds for i in 1:nt
        tri = tlist[i]
        v1 = tri[1]
        v2 = tri[2]
        v3 = tri[3]

        if v1 < v2
            e[3*i-2] = @SVector [v1, v2, 0, i] #information on vertices, on the side that triangle lies on, and triangle number.
        else
            e[3*i-2] = @SVector [v2, v1, i, 0]
        end
        if v2 < v3
            e[3*i-1] = @SVector [v2, v3, 0, i]
        else
            e[3*i-1] = @SVector [v3, v2, i, 0]
        end
        if v3 < v1
            e[3*i] = @SVector [v3, v1, 0, i]
        else
            e[3*i] = @SVector [v1, v3, i, 0]
        end
    end

    #We want to sort this using 1st row as first criterion, 2nd row as a tie breaker and 3rd row as the next tie breaker.
    sort!(e)

    #Merge edges
    #i is the index where to copy to and j is the searching index.
    i = 1
    j = 2
    @inbounds while j <= nt*3
        if e[i][1] == e[j][1] && e[i][2] == e[j][2] #merging

            # Todo: Improve this. Use @Reset from transportation_simplex.jl??? ei = e[i]
            # e[i][3] = e[j][3]
            ei = e[i]
            if GD == 2
                # In two dimensions, all triangles are oriented counterclockwise
                @reset ei[3] = e[j][3]
            else
                # In higher dimension, we cannot rely on a consistent orientation (Möbius strip)
                if ei[4] == 0
                    @assert e[j][4] == 0
                    @reset ei[4] = e[j][3]
                elseif e[j][4] == 0
                    @reset ei[3] = e[j][3]
                else
                    @reset ei[3] = e[j][4]
                end
            end
            e[i] = ei

            if j+1 <= nt*3
                e[i+1] = e[j+1]
                i += 1
            end
            j += 2
        else #merging not needed as it is a boundary edge.
            e[i+1] = e[j]
            i += 1
            j += 1
        end
    end
    @inbounds e = e[1:i]

    #It happens that some columns of e are [v1,v2,0,trianglenumber]. We change it such that 0 ist last entry.
    @inbounds for k = 1:i
        if e[k][3] == 0
            e[k] = @SVector [e[k][1], e[k][2], e[k][4], 0]
        end
    end

    return e, i
end

function compute_cell_to_edge(tlist,nt,elist,ne)
    cell_to_edge = Vector{SVector{3,Int}}(undef, nt)

    @inbounds for i = 1:ne
        tri = elist[i][3] #number of triangle
        index = find_index_of_third_vertex_in_triangle(tlist,elist[i][1],elist[i][2],tri)

        cte = cell_to_edge[tri]
        @reset cte[index] = i
        cell_to_edge[tri] = cte

        #Inner edge => add the edge to the other triangle too.
        if elist[i][4] != 0
            tri = elist[i][4]
            index = find_index_of_third_vertex_in_triangle(tlist,elist[i][1],elist[i][2],tri)

            cte = cell_to_edge[tri]
            @reset cte[index] = i
            cell_to_edge[tri] = cte
        end
    end


    return cell_to_edge
end

function find_index_of_third_vertex_in_triangle(tlist, v1, v2, tri)
    vertices = tlist[tri] #vertices of triangle
    if (vertices[1] != v1) && (vertices[1] != v2)
        return 1
    elseif (vertices[2] != v1) && (vertices[2] != v2)
        return 2
    else
        return 3
    end
end

function compute_transformation_matrices(plist::Vector{SVector{GD,Float64}}, tlist, nt) where GD
    affine_matrix = Vector{SMatrix{GD,2,Float64}}(undef,nt)
    affine_vector = Vector{SVector{GD,Float64}}(undef,nt)
    affine_invmatrixT = Vector{SMatrix{GD,2,Float64}}(undef,nt)
    for i = 1:nt
        #we get the vertex numbers and their coordinates.
        tri = tlist[i]

        v1 = plist[tri[1]]
        v2 = plist[tri[2]]
        v3 = plist[tri[3]]

        #if the vertices are v1,v2,v3 => B_K = [v2-v1, v3-v1], b_K = v1. The inverse transposed matrix can be computed easily.
        b_K = v1
        B_K = hcat(v2-v1, v3-v1)
        if GD == 2
            B_K_minusT = inv(transpose(B_K))
        else
            B_K_minusT = B_K * inv(transpose(B_K)*B_K)
        end

        #add matrices and vector to the arrays.
        affine_matrix[i] = B_K
        affine_vector[i] = b_K
        affine_invmatrixT[i] = B_K_minusT
    end

    return affine_matrix, affine_vector, affine_invmatrixT
end

function change_be(edgelist,ne,be,segmentmarkerlist)
    nbe = size(be,2)

    newbe = Vector{SVector{2,Int}}(undef, nbe)
    for i = 1:nbe
        edge = sort(@SVector [be[1,i], be[2,i]])
        newbe[i] = @SVector[ find_index_of_edge_in_edgelist(edge, edgelist, ne), segmentmarkerlist[i] ]
    end

    return newbe
end

function find_index_of_edge_in_edgelist(edge, edgelist,ne)
    #binary search using the fact that the edge definitely exists inside the edgelist.
    from = 1
    until = ne

    while from < until
        middle = floor(Int,(from+until)/2)
        if edgelist[middle][@SVector[1,2]] > edge
            until = middle - 1
        elseif edgelist[middle][@SVector[1,2]] < edge
            from = middle +1
        else
            return middle
        end
    end

    return from
end

function init_mesh(vertices::Matrix, maxarea)
    #Builds edges, triangle-edge connectivity and data for the affine transformation.

    number = size(vertices,1)

    # Only for two-dimensional meshes.
    @assert size(vertices, 2) == 2

    segments = zeros(Int8,2,number)
    for i = 1:2
        for j = 1:number
            segments[i,j] = i + j - 1;
        end
    end
    segments[2,number] = 1

    triin=Triangulate.TriangulateIO()
    triin.pointlist=Matrix{Cdouble}(vertices')
    triin.segmentlist=Matrix{Cint}(segments)
    triin.segmentmarkerlist=Vector{Cint}([i for i = 1:number])
    area=@sprintf("%.15f",maxarea)
    (triout, vorout)=triangulate("pa$(area)Qq", triin)

    @assert Cdouble == Float64
    p = copy(reinterpret(SVector{2,Float64}, vec(triout.pointlist)))

    nt = size(triout.trianglelist, 2)
    t = [SVector{3,Int64}(@view triout.trianglelist[:,i]) for i in 1:nt]
    
    # Initialize the mesh.
    return Mesh(vertices, p, triout.segmentlist, t, triout.segmentmarkerlist)
end

function refine_all_cells(mesh::Mesh{GD}) where GD
    #Idea:
    #1. Create points on each edge. coordinates as mean values of coordinates of vertices of edge. Save a mapping: edge -> number of new point.
    #2. Write down triangles using this mapping.
    #3. Write down new segments using this mapping: go through be and create new edges using mapping. Segmentmarker is second row in be.
    #example: be[:,1] = [5; 1] -> e[:,5] = [2;7]; edge no. 5 is mapped on new point 100 -> segmentlist = [[2;100],[100,7], ...], segmentmarkerlist = [1, 1, ...]

    #new points list: positions 1:np are for the already existing points.
    newp = Vector{SVector{GD,Float64}}(undef, mesh.np + mesh.ne)
    newp[1:mesh.np] = mesh.p

    for i = 1:mesh.ne
        #get vertices of the edge.
        vind1 = mesh.e[i][1]
        vind2 = mesh.e[i][2]
        v1 = mesh.p[vind1]
        v2 = mesh.p[vind2]
        #calculate coordinates of middle point.
        newp[mesh.np + i] = (v1 + v2) / 2
    end

    newt = Vector{SVector{3,Int}}(undef, 4*mesh.nt)
    for i = 1:mesh.nt
        #use get indices of triangle vertices and middle points.
        p1 = mesh.t[i][1]
        p2 = mesh.t[i][2]
        p3 = mesh.t[i][3]
        p23 = mesh.cell_to_edge[i][1] + mesh.np
        p31 = mesh.cell_to_edge[i][2] + mesh.np
        p12 = mesh.cell_to_edge[i][3] + mesh.np
        #triangles are such that triangle lies on the left.
        newt[4*i-3] = @SVector [p1;p12;p31]
        newt[4*i-2] = @SVector [p2;p23;p12]
        newt[4*i-1] = @SVector [p3;p31;p23]
        newt[4*i] = @SVector [p12;p23;p31]

    end

    nbe = length(mesh.be)
    newsegmentlist = zeros(Int,2,2*nbe)
    newsegmentmarkerlist = zeros(Int,1,2*nbe)
    for i = 1:nbe
        #pos 1: index of edge in e. pos. 2: number of the part of the edge this part of the edge it belongs to.
        p1 = mesh.e[mesh.be[i][1]][1]
        p2 = mesh.e[mesh.be[i][1]][2]
        p12 = mesh.np + mesh.be[i][1]

        newsegmentlist[:,2*i-1] = @SVector [p1;p12]
        newsegmentmarkerlist[1,2*i-1] = mesh.be[i][2]
        newsegmentlist[:,2*i] = @SVector [p12;p2]
        newsegmentmarkerlist[1,2*i] = mesh.be[i][2]
    end

    return Mesh(mesh.geometry, newp, newsegmentlist, newt, newsegmentmarkerlist, align_triangles=false)
end

function _coord_trafo(mesh::Mesh{GD}, rmesh::Mesh{GD}, idx, ridx) where GD
	A = SMatrix{3,GD+1,Float64}(i <= GD ? rmesh.p[rmesh.t[ridx][j]][i] : 1. for j in 1:3, i in 1:GD+1)
	B = SMatrix{3,GD+1,Float64}(i <= GD ? mesh.p[mesh.t[idx][j]][i] : 1. for j in 1:3, i in 1:GD+1)
	C = A / B
end

# Compute Prolongation operator P if rmesh is a refinement of mesh
# such that P*U is a coefficient vector on rmesh (with FE rfe)
# for a coefficient vector U (with FE fe) on mesh
function prolongation(mesh, rmesh, fe::FE, rfe::RFE = fe,
		::Val{N} = vnlocaldofs(FE), ::Val{RN} = vnlocaldofs(RFE)) where {FE, RFE, N, RN}

	II = Vector{Int}(undef, rmesh.nt * RN * N)
	JJ = Vector{Int}(undef, rmesh.nt * RN * N)
	SS = Vector{Float64}(undef, rmesh.nt * RN * N)
	entries = 0

	coun = zeros(Int,ndofs(rfe,rmesh))

	# Set up cache
	rdof_to_shape_cache = Vector{SMatrix{N,RN,Float64,N*RN}}(undef, 0)
	hashes = Vector{UInt64}(undef, 0)
	sizehint!(rdof_to_shape_cache, 10)
	sizehint!(hashes, 10)

	# digits for cache
	digits_for_cache = 8

	if !( FE <: FE_Lagrange )
		cdofmap = @MMatrix zeros(N,N)
	end
	if !( RFE <: FE_Lagrange )
		rdofmap = @MMatrix zeros(RN,RN)
	end
	W = @MMatrix zeros(RN,N)
	V = @MMatrix zeros(RN,N)

	# Index of parent cell
	idx = 1

	# Create A in global scope. Is this needed???
	A = _coord_trafo( mesh, rmesh, 1, 1)

	for i = 1:rmesh.nt
		# First, we need to find the parent cell.
		# We hope that the children are numbered according to their parents

		while idx <= mesh.nt
			A = _coord_trafo( mesh, rmesh, idx, i)
			# If A has non-negative entries, we have found the parent
			if all( x -> x >= -1e-10, A)
				break
			end
			idx += 1
		end
		if idx > mesh.nt
			throw(ErrorException("Did not found parent of cell $i"))
		end

		# Check whether we can reuse the cache. We round A to a binary fraction
		B = round.(Int, 2^digits_for_cache .* A )
		C = B ./ 2^digits_for_cache

		hashA = hash(B)
		hash_idx = findfirst( x -> x==hashA, hashes )

		if isnothing(hash_idx)
			transformed_shape1 = lambda -> shape(fe, C'*lambda, Val(true))
			# we need to transform the derivatives via rmesh.affine_matrix[i] and mesh.affine_matrix[idx]!

			E = rmesh.affine_matrix[i]'*mesh.affine_invmatrixT[idx]
			transformed_shape = lambda -> transform_derivative( E, transformed_shape1, lambda)

			rdof_to_shape_calc = local_dofs(rfe,transformed_shape)

			temp = @MMatrix zeros(N,RN)
			for j=1:RN
				temp[:,j] .= rdof_to_shape_calc[j]
			end

			push!(rdof_to_shape_cache, temp)
			push!(hashes, hashA)

			rdof_to_shape = rdof_to_shape_cache[end]
		else
			@inbounds rdof_to_shape = rdof_to_shape_cache[hash_idx]
		end

		# This matrix can be used to convert the global dofs on mesh to local dofs on the current cell
		# We use it only implicitely below.
		# l_dofs = dofmap(fe, mesh, idx) * sparse(rdof_to_shape)

		# We have to convert back to the global dofs.
		# We assume that all global dofs that participate in the local dofs on this cell
		# are uniquely determined by the local dofs on this cell.

		rdofg, rdofi, rdofj, rdofs = flat_dofmap(rfe, rmesh, i)
		if RFE <: FE_Lagrange
			rdofmap = I
			rdofmap2 = I
		else
			rdofmap .= 0.
			for i in eachindex(rdofi)
				rdofmap[rdofi[i],rdofj[i]] += rdofs[i]
			end
			# Solving with an SMatrix below does not allocate.
			rdofmap2 = SMatrix{RN,RN,Float64,RN*RN}(rdofmap)
		end
		# rdofmap = sparse(rdofi, rdofj, rdofs)


		# Solve the normal equation
		# V = (rdofmap*rdofmap') \ (rdofmap*l_dofs')
		# Since rdofmap is square, the normal equation is not needed
		W .= rdofmap \ rdof_to_shape'

		dofg::SVector{N,Int}, dofi, dofj, dofs = flat_dofmap(fe, mesh, idx)
		if FE <: FE_Lagrange
			cdofmap = I
		else
			cdofmap .= 0.
			for i in eachindex(dofi)
				cdofmap[dofi[i],dofj[i]] += dofs[i]
			end
		end
		V .= W * cdofmap'

		# The generation of P and the following update of II, JJ, SS is equivalent to:
		# P(dofg,:) = P(dofg,:) + V;

		for i=1:RN, j=1:N
			if V[i,j] != 0.
				II[entries+1] = rdofg[i]
				JJ[entries+1] = dofg[j]
				SS[entries+1] = V[i,j]
				entries += 1
			end
		end
		coun[rdofg] .+=  1
	end

	resize!(II, entries)
	resize!(JJ, entries)
	resize!(SS, entries)

	P = sparse(II, JJ, SS, ndofs(rfe,rmesh), ndofs(fe,mesh))

	for i in eachindex(P.rowval)
		row = P.rowval[i]
		P.nzval[i] /= coun[row]
	end

	return P
end

# Needed for prolongation matrix
function transform_derivative( B, t_shape, lambda )
	val, dshape = t_shape(lambda)
	val_x_transf = zeros(Float64, length(dshape), size(dshape[1],2))
	val_y_transf = copy(val_x_transf)

	for i=1:length(dshape)
		val_x_transf[i,:] = B[1,1]*dshape[i][1,:] + B[1,2]*dshape[i][2,:]
		val_y_transf[i,:] = B[2,1]*dshape[i][1,:] + B[2,2]*dshape[i][2,:]
	end

	return val, val_x_transf, val_y_transf
end

function refine_adaptively(mesh, marker::AbstractArray{Int})
    # Create a (vector) copy of marker and apply the inplace routine
    refine_adaptively!(mesh, copy(vec(marker)))
end

function refine_adaptively!(mesh::Mesh{GD}, marker::AbstractVector{Int}) where GD
    # Works inplace on the vector marker

    # https://lyc102.github.io/ifem/afem/bisect/

    isCutEdge = zeros(Int, mesh.ne)
    nce = 0
    while length(marker)>0
        for i in eachindex(marker)
            idx = marker[i]

            # The refinement edge of the current cell will be cut
            edge = mesh.cell_to_edge[idx][1]

            if isCutEdge[edge] == 0
                nce += 1
                isCutEdge[edge] = nce
            end

            # Find the neighbor of the current cell incident to the splitting edge.

            if mesh.e[edge][4] == 0
                #there is no neighbor for boundary edges.
                marker[i] = idx
            else
                if mesh.e[edge][3] == idx
                    marker[i] = mesh.e[edge][4]
                else
                    marker[i] = mesh.e[edge][3]
                end
            end
        end
        
        # Filter out the marked cells whose splitting edge is already being cut
        filter!(marker) do idx
            0 == isCutEdge[ mesh.cell_to_edge[idx][1] ]
        end
    end

    # Next, we are going to cut all edges from isCutEdge

    # We introduce the mid points of the cut edges
    newp = Vector{SVector{GD,Float64}}(undef, mesh.np + nce)
    newp[1:mesh.np] = mesh.p

    new_nt = mesh.nt
    for i = 1:mesh.ne
        if isCutEdge[i] > 0
            idx1 = mesh.e[i][1]
            idx2 = mesh.e[i][2]
            newp[mesh.np + isCutEdge[i]] = ( mesh.p[idx1] + mesh.p[idx2] ) / 2.
            
            # Each cut edge induces a new triangle in the incident triangles
            if mesh.e[i][4] == 0
                new_nt += 1
            else
                new_nt += 2
            end
        end
    end

    # Compute new triangles
    newt = Vector{SVector{3,Int}}(undef, new_nt)
    next_t = 1
    for i = 1:mesh.nt
        ip1 = mesh.t[i][1]
        ip2 = mesh.t[i][2]
        ip3 = mesh.t[i][3]
        ie1 = mesh.np + isCutEdge[mesh.cell_to_edge[i][1]]
        ie2 = mesh.np + isCutEdge[mesh.cell_to_edge[i][2]]
        ie3 = mesh.np + isCutEdge[mesh.cell_to_edge[i][3]]

        if ie1 > mesh.np
            # Cell i is cut. Check whether it is cut again...
            if ie2 > mesh.np
                # We have to add two children.
                newt[next_t  ] = @SVector[ie2, ie1, ip3 ]
                newt[next_t+1] = @SVector[ie2, ip1, ie1 ]
                next_t += 2
            else
                # We have to add one children.
                newt[next_t] = @SVector[ie1, ip3, ip1 ]
                next_t += 1
            end
            if ie3 > mesh.np
                # We have to add two children.
                newt[next_t  ] = @SVector[ie3, ie1, ip1 ]
                newt[next_t+1] = @SVector[ie3, ip2, ie1 ]
                next_t += 2
            else
                # We have to add one children.
                newt[next_t] = @SVector[ie1, ip1, ip2 ]
                next_t += 1
            end
        else
            # Cell is unchanged
            newt[next_t] = mesh.t[i]
            next_t += 1
        end
    end
    @assert next_t == new_nt + 1

    new_nbe = length(mesh.be)
    for i = 1:length(mesh.be)
        if isCutEdge[mesh.be[i][1]] > 0
            new_nbe += 1
        end
    end

    newbe = zeros(Int, 2, new_nbe)
    newsegmentmarkerlist = zeros(Int, new_nbe)
    next_idx = 1
    for i = 1:length(mesh.be)
        edge = mesh.e[mesh.be[i][1]]
        if isCutEdge[mesh.be[i][1]] > 0
            newbe[1,next_idx  ] = edge[1]
            newbe[2,next_idx  ] = mesh.np + isCutEdge[mesh.be[i][1]]
            newbe[1,next_idx+1] = mesh.np + isCutEdge[mesh.be[i][1]]
            newbe[2,next_idx+1] = edge[2]
            newsegmentmarkerlist[next_idx  ] = mesh.be[i][2]
            newsegmentmarkerlist[next_idx+1] = mesh.be[i][2]
            next_idx += 2
        else
            newbe[1,next_idx] = edge[1]
            newbe[2,next_idx] = edge[2]
            newsegmentmarkerlist[next_idx] = mesh.be[i][2]
            next_idx += 1
        end
    end

    Mesh(mesh.geometry, newp, newbe, newt, newsegmentmarkerlist, align_triangles=false)
end

function triangle_mesh()
	p = Vector{SVector{2,Float64}}(undef, 3)
	t = Vector{SVector{3,Int}}(undef, 1)

	p[1] = @SVector[0., 0.]
	p[2] = @SVector[1., 0.]
	p[3] = @SVector[0., 1.]

	t[1] = @SVector[1, 2, 3]

	mesh = Mesh(zeros(0,0), p, zeros(0,0), t, zeros(0,0))
end

function torus_mesh(R, r, N = 10, n = round(Int, N*r/R))
	p = Vector{SVector{3,Float64}}(undef, N * n)
	t = Vector{SVector{3,Int}}(undef, 2 * N * n)

	dphi = 2 * pi / N
	dpsi = 2 * pi / n

	for I = 1:N, i = 1:n
		phi = (I-1)*dphi
		psi = (i-1)*dpsi
		p[(I-1)*n + i] = [
											(r * cos(psi) + R) * sin(phi),
											(r * cos(psi) + R) * cos(phi),
											r * sin(psi)
											]

		Ip1 = mod1(I+1, N)
		ip1 = mod1(i+1, n)
		t[ 2*((I-1)*n + i) - 1 ] = @SVector[(I-1)*n + i, (Ip1-1)*n + i  , (Ip1-1)*n + ip1]
		t[ 2*((I-1)*n + i)     ] = @SVector[(I-1)*n + i, (I  -1)*n + ip1, (Ip1-1)*n + ip1]
	end

	geometry = reshape([R, r],2,1)

	mesh = Mesh(geometry, p, zeros(0,0), t, zeros(0,0))
end

function moebius_mesh(R, w, N, n = round(Int, N*w/(2*pi*R)))
	p = Vector{SVector{3,Float64}}(undef, N * (n+1))
	t = Vector{SVector{3,Int}}(undef, 2 * N * n)

	dphi = 2 * pi / N
	dv = w / n

	for I = 1:N, i = 1:(n+1)
		phi = (I-1)*dphi
		v   = -w/2 + (i-1)*dv
		p[(I-1)*(n+1) + i] = [
													(R + v*cos(phi/2)) * cos(phi),
													(R + v*cos(phi/2)) * sin(phi),
													v*sin(phi/2)
													]

		if i < n+1
			if I < N
				t[ 2*((I-1)*n + i) - 1 ] = @SVector[(I-1)*(n+1) + i, (I+1-1)*(n+1) + i  , (I+1-1)*(n+1) + i+1]
				t[ 2*((I-1)*n + i)     ] = @SVector[(I-1)*(n+1) + i, (I+1-1)*(n+1) + i+1, (I  -1)*(n+1) + i+1]
			else
				ii = n + 1 - i
				t[ 2*((I-1)*n + i) - 1 ] = @SVector[(I-1)*(n+1) + i + 1, (1  -1)*(n+1) + ii  , (1  -1)*(n+1) + ii+1]
				t[ 2*((I-1)*n + i)     ] = @SVector[(I-1)*(n+1) + i, (1  -1)*(n+1) + ii+1, (I  -1)*(n+1) + i+1]
			end
		end
	end

	nbe = 2*N
	be = zeros(2,nbe)
	seg = ones(nbe)

	for I = 1:N-1
		be[:,2*I - 1] = [(I-1)*(n+1) +     1, I*(n+1) + 1    ]
		be[:,2*I    ] = [(I-1)*(n+1) + n + 1, I*(n+1) + n + 1]
	end
	be[:,2*N - 1] = [(N-1)*(n+1) +     1, n + 1]
	be[:,2*N    ] = [(N-1)*(n+1) + n + 1, 1    ]

	geometry = reshape([R, w],2,1)

	mesh = Mesh(geometry, p, be, t, seg)
end

function klein_bottle_mesh(N, n = round(Int, N/6)*2)
	p = Vector{SVector{3,Float64}}(undef, N * n)
	t = Vector{SVector{3,Int}}(undef, 2 * N * n)

	if !iseven(n)
		throw(DomainError("Second argument n has to be even"))
	end

	# Parametrization inspired from
	# The Klein Bottle:
	# Variations on a Theme
	# Gregorio Franzoni

	# Parameter vector
	a, b, c, d, e, f, g = (20., 12., 11/2, 4., 1.5, 4., 3.8)

	# Helper function
	h1 = s -> b*exp(-e*(s - g)^2)
	h1p = s -> b*exp(-e*(s - g)^2)*2*e*-(s - g)

	# Correct h1 such that h2(0) = h2(2pi) = 0 and h2p(0) = h2p(2pi) = 0
	h2 = s -> h1(s) - h1(0) - (h1(2pi)-h1(0))*(s/2pi)
	h2p = s -> h1p(s) - h1p(0) - (h1p(2pi)-h1p(0))*(s/2pi)

	# Central line
	gamma = s -> @SVector [a*(1-cos(s)), h2(s),0]
	gammap = s -> @SVector [a*sin(s), h2p(s),0]

	# Radius
	h = s -> atan(e*sin(s + 1.5*exp(-(s - 2.5)^2/2.5)))/atan(e)
	r = s -> c + d*( h(s) - (h(2*pi)-h(0))*(s-pi)/2pi)

	# Discretization
	ds = 2 * pi / N
	dtheta = 2 * pi / n

	k = @SVector[0., 0., 1.]

	for I = 1:N, i = 1:n
		s = (I-1)*ds
		theta = (i-1)*dtheta

		if s > 0
			T = gammap(s) ./ norm(gammap(s))
		else
			T = @SVector[1., 0., 0.]
		end
		M = cross(k, T)


		p[(I-1)*n + i] = gamma(s) + r(s)*(M*cos(theta)+k*sin(theta))

		Ip1 = mod1(I+1, N)
		ip1 = mod1(i+1, n)
		if I < N
			t[ 2*((I-1)*n + i) - 1 ] = @SVector[(I-1)*n + i, (Ip1-1)*n + i  , (Ip1-1)*n + ip1]
			t[ 2*((I-1)*n + i)     ] = @SVector[(I-1)*n + i, (I  -1)*n + ip1, (Ip1-1)*n + ip1]
		else
			# Looks like magic, but somehow it works...
			shift = -(div(n,2)-1)
			ii = mod1(n + 1 - i + shift, n)
			iim1 = mod1(ii - 1, n)
			t[ 2*((I-1)*n + i) - 1 ] = @SVector[(I-1)*n + i, (Ip1-1)*n + ii, (Ip1-1)*n + iim1 ]
			t[ 2*((I-1)*n + i)     ] = @SVector[(I-1)*n + i, (I  -1)*n + ip1, (Ip1-1)*n + iim1 ]
		end
	end

	geometry = zeros(0,0)

	mesh = Mesh(geometry, p, zeros(0,0), t, zeros(0,0))
end


function test()
    # For testing.
    vertices = [-1. -1.; 1. -1.; 1. 1.; -1. 1.]
    mesh = init_mesh(vertices, 0.5)

    print(mesh)
    print("\n")

    mesh
end

function benchmark()
    # For testing.
    vertices = [-1 -1; 1 -1; 1 1; -1 1]
    mesh = init_mesh(vertices, 0.000005)
end

function testrefineall()
    # For testing.
    vertices = [0.0 0.0; 1.0 0.0; 0.0 1.0] #Dreieck
    mesh = init_mesh(vertices, 0.5)

    print(mesh)
    print("\n")

    rmesh = refine_all_cells(mesh)

    print(rmesh)
end

function testrefineadaptively(refs=16)
    # For testing.
    vertices = [-1 -1; 1 -1; 1 1; -1 1]
    mesh = init_mesh(vertices, 0.5)

    for i = 1:refs
        marker = collect(1:mesh.nt)
        @time mesh = refine_adaptively!(mesh, marker)

        println("$(mesh.nt) triangles after refinement")
    end

    mesh
end

function sanity_check(mesh::Mesh{GD}) where GD
    # Compute some sanity checks for a given mesh.
    @unpack_Mesh mesh
    @assert np == length(p)
    @assert ne == length(e)
    @assert nt == length(t)
    nbe = length(be)
    @assert nt == length(cell_to_edge)

    @assert 2*ne - nbe == 3*nt

    @printf("Euler-characteristic: %d\n", np - ne + nt)

    area = 0.
    for i = 1:nt
        i1 = t[i][1]
        i2 = t[i][2]
        i3 = t[i][3]
        v1 = p[i1]
        v2 = p[i2]
        v3 = p[i3]

        if GD == 2
            da = ((v1[2]+v2[2])*(v1[1]-v2[1]) + (v2[2]+v3[2])*(v2[1]-v3[1]) + (v3[2]+v1[2])*(v3[1]-v1[1])) / 2.
            @assert da ≈ abs(det(mesh.affine_matrix[i]))/2
        else
            da = sqrt(det(mesh.affine_matrix[i]'*mesh.affine_matrix[i]))/2
        end

        area += abs(da)
    end

    @printf("Area: %.15f\n", area)

    circum = 0.
    for i = 1:nbe
        i1 = e[be[i][1]][1]
        i2 = e[be[i][1]][2]
        v1 = p[i1]
        v2 = p[i2]

        circum += norm(v2-v1)
    end

    @printf("Circumference: %.15f\n", circum)
end