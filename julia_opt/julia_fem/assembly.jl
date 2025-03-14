CoeffType = Union{Nothing,AbstractArray,Function,Real}

@doc raw"""
    area_integrator(mesh::Mesh{GD}, fe::FE, quadrature, h_A::CoeffType, h_beta::CoeffType, h_c0::CoeffType, h_f::CoeffType, v::Val{NLD} = vnlocaldofs(FE)) where {GD, FE, NLD}

Assembly of the contributions to the weak form coming from the area integrals. These are (integrals over Ω)
Aᵢⱼ = ∫ ∇φᵢᵀ A(x) ∇φⱼ dx
Bᵢⱼ = ∫ φᵢ β(x) ⋅ ∇φⱼ dx
Cᵢⱼ = ∫ φᵢ c₀(x) φⱼ dx
Fᵢ = ∫ f(x) φᵢ dx
"""
function area_integrator(mesh::Mesh{GD}, fe::FE, quadrature, h_A::CoeffType, h_beta::CoeffType, h_c0::CoeffType, h_f::CoeffType, v::Val{NLD} = vnlocaldofs(FE)) where {GD, FE, NLD}

    # Calculate the # of local and global dofs
    Nlocal_dofs = NLD
    Nglobal_dofs = ndofs(fe,mesh)

    # Get quadrature points and weights 
    n_quad = length(quadrature)
    Lambda = zeros(3,n_quad)
    weights = zeros(n_quad)
    
    for i=1:n_quad
        Lambda[:,i] = quadrature[i].coords
        weights[i] = quadrature[i].weight
    end

    # Evaluate the shape functions and their derivatives on the reference
    # cell in all quadrature points
    shapef, dshape = shape(fe, Lambda, Val(true))

    # Assemble the A term
    # ------------------------------------------------------------------------

    # Initialize index vectors
    i = Vector{Int64}(undef, Nlocal_dofs^2 * mesh.nt)
    j = Vector{Int64}(undef, Nlocal_dofs^2 * mesh.nt)
    s = Vector{Float64}(undef, Nlocal_dofs^2 * mesh.nt)

    entries = 0

    # More initialization
    ii = MMatrix{Nlocal_dofs,Nlocal_dofs,Int64}(undef)
    jj = MMatrix{Nlocal_dofs,Nlocal_dofs,Int64}(undef)
    ss = MMatrix{Nlocal_dofs,Nlocal_dofs,Float64}(undef)

    # Initialize the local matrices
    AK = MMatrix{Nlocal_dofs,Nlocal_dofs,Float64}(undef)
    FK = MVector{Nlocal_dofs,Float64}(undef)

    # Initialize global vector
    F = zeros(Float64,Nglobal_dofs)

    # Initialize coefficient matrices if A, beta, c0, f are constant
    if isnothing(h_A)
        coeff_A = nothing
    elseif isa(h_A, Real)
        coeff_A = h_A
    elseif isa(h_A, AbstractArray)
        coeff_A = SizedMatrix{2,2}(reshape(h_A, 2, 2))
    end
    if isnothing(h_beta)
        coeff_beta = nothing
    elseif isa(h_beta, Real)
        error("The coefficient beta cannot be a real number.")
    elseif isa(h_beta, AbstractArray)
        coeff_beta = SizedVector{2}(h_beta)
    end
    if isnothing(h_c0)
        coeff_c0 = nothing
    elseif isa(h_c0, Real)
        coeff_c0 = h_c0
    end
    if isnothing(h_f)
        coeff_f = nothing
    elseif isa(h_f, Real)
        coeff_f = h_f
    end

    # Loop over all cells
    for idx = 1:mesh.nt

        # Evaluate the coeffient at quadrature points transformed to the
        # world cell
        if isa(h_A, Function)
            coeff_A = h_A(affine_transformation(mesh, Lambda, idx))
        end
        if isa(h_beta, Function)
            coeff_beta = h_beta(affine_transformation(mesh, Lambda, idx))
        end
        if isa(h_c0, Function)
            coeff_c0 = h_c0(affine_transformation(mesh, Lambda, idx))
        end
        if isa(h_f, Function)
            coeff_f = h_f(affine_transformation(mesh, Lambda, idx))
        end

        # Zero the local matrices
        AK .= 0.0
        FK .= 0.0

        # Loop over all quadrature points
        for nquad = 1:n_quad

            # Build the local contribution matrix AK
            BKmT_dshape_dx = mesh.affine_invmatrixT[idx] * dshape[nquad]

            if isa(coeff_A, AbstractVector)
                AK .+= weights[nquad] * BKmT_dshape_dx' * coeff_A[nquad] * BKmT_dshape_dx
            elseif !isnothing(h_A)
                AK .+= weights[nquad] * BKmT_dshape_dx' * coeff_A * BKmT_dshape_dx
            end
            if isa(coeff_beta, AbstractVector)
                AK .+= weights[nquad] * shapef[nquad] * (coeff_beta[nquad]' * BKmT_dshape_dx)
            elseif !isnothing(h_beta)
                AK .+= weights[nquad] * shapef[nquad] * (coeff_beta[:, nquad]' * BKmT_dshape_dx)
            end
            if isa(coeff_c0, Real)
                AK .+= weights[nquad] * shapef[nquad] * coeff_c0 * shapef[nquad]'
            elseif !isnothing(h_c0)
                AK .+= weights[nquad] * shapef[nquad] * coeff_c0[nquad] * shapef[nquad]'
            end
            if isa(coeff_f, Real)
                FK .+= weights[nquad] * coeff_f * shapef[nquad]
            elseif !isnothing(h_f)
                FK .+= weights[nquad] * coeff_f[nquad] * shapef[nquad]
            end
        end

        # Finally, scale the local matrices by the determinant of the Jacobian
        if GD == 2
            scale = abs(det(mesh.affine_matrix[idx]))
        else
            scale = sqrt(det(mesh.affine_matrix[idx]'*mesh.affine_matrix[idx]))
        end
        AK .*= scale
        FK .*= scale

        # Get the connectivity matrix
        global_dofs_k, ik, jk, sk = flat_dofmap(fe, mesh, idx)

        k = Nlocal_dofs^2

        for i1 = 1:Nlocal_dofs, i2 = 1:Nlocal_dofs
            ii[i1, i2] = global_dofs_k[i1]
            jj[i1, i2] = global_dofs_k[i2]
        end
        
        ss .= 0.0

        for i1 = 1:length(ik), i2 = 1:length(ik)
            ss[ik[i1], ik[i2]] += sk[i1]*sk[i2]*AK[jk[i1],jk[i2]]
        end

        @views F[global_dofs_k[ik]] .+= sk .* FK[jk]
    

        i[entries+1:entries+k] = ii
        j[entries+1:entries+k] = jj
        s[entries+1:entries+k] = ss
        entries += k
    end

    if entries != length(i)
        i = i[1:entries]
        j = j[1:entries]
        s = s[1:entries]
    end

    # Build the full matrix
    A = sparse(i, j, s, Nglobal_dofs, Nglobal_dofs)

    return A, F
end


@doc raw"""
    bdry_integrator(mesh, fe::FE, h_bdry_quadrature, h_alpha, h_g, v::Val{NLD} = vnlocaldofs(FE)) where {NLD, FE}

Assembly of the contributions to the weak form coming from the boundary integrals. These are (integrals over Γ)
Q = ∫ φᵢ α(s) φⱼ ds
G = ∫ g(s) φᵢ ds
"""
function bdry_integrator(mesh, fe::FE, h_bdry_quadrature, h_alpha, h_g, v::Val{NLD} = vnlocaldofs(FE)) where {NLD, FE}

	# Get quadrature points and weights for all three possible edges 
	QuadP1 = h_bdry_quadrature(1)
	QuadP2 = h_bdry_quadrature(2)
	QuadP3 = h_bdry_quadrature(3)

    # First Dimension designates edge, second dimension the barycentric coordinates, third dimension the index of the quadrature point of the edge
    n_quad = length(QuadP1)
	Lambda = zeros(3,3,n_quad)
	weights = zeros(3,n_quad)
	shapef = zeros(3,3,n_quad)

	shapef = Matrix{SVector{NLD,Float64}}(undef, 3, n_quad)

	for i = 1:n_quad
		Lambda[1,:,i] = QuadP1[i].coords
		Lambda[2,:,i] = QuadP2[i].coords
		Lambda[3,:,i] = QuadP3[i].coords
		weights[1,i] = QuadP1[i].weight
		weights[2,i] = QuadP2[i].weight
		weights[3,i] = QuadP3[i].weight
	end

	# Evaluate the shape functions on the reference
	# cell in all quadrature points (separately for each edge)
	vec1 = shape(fe,Lambda[1,:,:]) 
	vec2 = shape(fe,Lambda[2,:,:]) 
	vec3 = shape(fe,Lambda[3,:,:]) 

	for i = 1:n_quad
		shapef[1,i] = vec1[i]
		shapef[2,i] = vec2[i]
		shapef[3,i] = vec3[i]
	end
	
	# Calculate the # of local and global dofs
	Nglobal_dofs = ndofs(fe,mesh)
	Nlocal_dofs = NLD
	# Nglobal_dofs = ndofs(fe,mesh)

	if isa(h_alpha,Real)
		coeff_alpha = h_alpha
	end
	if isa(h_g,Real)
		coeff_g = h_g
	end

	nbe = length(mesh.be)

	# Initialize index vectors
    i = Vector{Int64}(undef, Nlocal_dofs^2 * nbe)
    j = Vector{Int64}(undef, Nlocal_dofs^2 * nbe)
    s = Vector{Float64}(undef, Nlocal_dofs^2 * nbe)

    entries = 0

    # More initialization
    ii = MMatrix{Nlocal_dofs,Nlocal_dofs,Int64}(undef)
    jj = MMatrix{Nlocal_dofs,Nlocal_dofs,Int64}(undef)
    ss = MMatrix{Nlocal_dofs,Nlocal_dofs,Float64}(undef)

    # Initialize the local matrices
    QK = MMatrix{Nlocal_dofs,Nlocal_dofs,Float64}(undef)
    GK = MVector{Nlocal_dofs,Float64}(undef)

    # Initialize global vector
    G = zeros(Float64,Nglobal_dofs)

	# Loop over all boundary edges
	for bi in 1:nbe
		bedge = mesh.be[bi][1]

		# Find cell index
		idx = mesh.e[bedge][3] != 0 ? mesh.e[bedge][3] : mesh.e[bedge][4]

		# Find (indices of) boundary edges of current cell, from 1 to 3
		edges = mesh.cell_to_edge[idx]
		nedge = findfirst(edges .== bedge)

		# Evaluate the coeffient at quadrature points transformed to the
		# world cell
		if !isa(h_alpha,Real) && !isnothing(h_alpha)
			coeff_alpha = h_alpha(affine_transformation(mesh,Lambda[nedge,:,:],idx))
		end
		if !isa(h_g,Real) && !isnothing(h_g)
			coeff_g = h_g(affine_transformation(mesh,Lambda[nedge,:,:],idx))
		end

		# Initialize the local matrix
		QK .= 0.
		GK .= 0.

		# Loop over all quadrature points
		for nquad = 1:n_quad
			# Build the local contribution matrix QK
			if isa(h_alpha,Real)
				QK .+= weights[nedge,nquad] * shapef[nedge,nquad] * coeff_alpha * shapef[nedge,nquad]'
			elseif !isnothing(h_alpha)
				QK .+= weights[nedge,nquad] * shapef[nedge,nquad] * coeff_alpha[nquad] * shapef[nedge,nquad]'
			end
			# Build the local contribution vector GK
			if isa(h_g,Real)
				GK .+= weights[nedge,nquad] * coeff_g * shapef[nedge,nquad]
			elseif !isnothing(h_g)
				GK .+= weights[nedge,nquad] * coeff_g[nquad] * shapef[nedge,nquad]
			end
		end

		# Finally, scale the local matrix by the length of the edge
		v1 = mesh.p[mesh.e[bedge][1]]
		v2 = mesh.p[mesh.e[bedge][2]]
		len = norm(v1 - v2)
		QK .*= len
		GK .*= len

		# Get the connectivity matrix
		global_dofs_k, ik, jk, sk = flat_dofmap(fe, mesh, idx)
		k = Nlocal_dofs^2

		for i1 = 1:Nlocal_dofs, i2 = 1:Nlocal_dofs
			ii[i1, i2] = global_dofs_k[i1]
			jj[i1, i2] = global_dofs_k[i2]
		end

		ss .= 0.0

		for i1 = 1:length(ik), i2 = 1:length(ik)
			ss[ik[i1], ik[i2]] += sk[i1]*sk[i2]*QK[jk[i1],jk[i2]]
		end

		@views G[global_dofs_k[ik]] .+= sk .* GK[jk]

		i[entries+1:entries+k] = ii
		j[entries+1:entries+k] = jj
		s[entries+1:entries+k] = ss
		entries += k
	end

	if entries != length(i)
		i = i[1:entries]
		j = j[1:entries]
		s = s[1:entries]
	end

	# Build the full matrix
	Q = sparse(i,j,s,Nglobal_dofs,Nglobal_dofs)

    return Q, G
end


@doc raw"""
    affine_transformation(mesh, lambda, ncell)

Maps the point(s) specified by the barycentric coordinates `lambda` into the corresponding world
coordinates of the given cell `ncell`.
"""
function affine_transformation(mesh, lambda, ncell)
    return mesh.affine_matrix[ncell] * [lambda[2,:]  lambda[3,:]]' + repeat(mesh.affine_vector[ncell],1,length(lambda[1,:]))
end
