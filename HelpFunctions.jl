# Auxilliary functions used in trust region method (and beyond)

using StatsBase
using Random
using DSP
using Plots

@doc raw"""
    bellman_TRM!(∇f, u_old, B, β, Δt, nu, U, Φ, iterator)

Generates Matrices `U` and `Φ`, where `U(:,b,l,i,)` represents the best Index set for `nu` 
with the cost `b` at the discrete time-step `i`, assuming the value of the optimal 
solution at the previous timestep `i-1` is `nu(l)`. `Φ` saves the 
corresponding target values for the trust-region subproblem specified by `∇f` with 
the TV weight-parameter `β`. The distance of the discrete time-steps `Δt` and the
trust-region `Δ` are used to calculate the Budget `B`, which dictates the maximum
deviation of solutions in `U` from the previous solution `u_old`. Finally, `iterator`
denotes all allowed combinations of control levels.
"""
function bellman_TRM!(∇f, u_old, B, β, p, Δt, nu, U, Φ, iterator)

    # Get dimensions
    M, n = size(u_old)
	Inds = CartesianIndices(size(Φ)[1:end-1])

    # Initialize Φ
    Φ[Inds, (n+1)%2+1] .= Inf

	@inbounds for l in iterator
		
        # Calculate ν_l = [nu[m][l[m]] for m=1:M] and b = norm(ν_l - u_old[:, n],1) with few allocations
		b = Int64(0)
        temp_val_1 = 0.
        for m = 1:M
            numl = nu[m][l[m]]
            temp_val_1 += Δt * ∇f[m,n] * numl
            b += convert(Int64, abs(numl - u_old[m, n]))
        end

        if b <= B
            Φ[b+1,l..., (n+1)%2+1] = temp_val_1
        end
    end

    @inbounds for i = n-1:-1:1

        Φ[Inds, (i+1)%2+1] .= Inf

        for l in iterator

            # The next contribution is independent of j and b
			temp_val_1 = 0.
            b̃ = Int64(0)
            for m = 1:M
                numl = nu[m][l[m]]
                temp_val_1 += Δt * ∇f[m,i] * numl
                b̃ += convert(Int64, abs(numl - u_old[m, i]))
            end

            for j in iterator

                # The next contribution is independent of b
				temp_val_2 = 0.
                for m = 1:M
                    temp_val_2 += abs( nu[m][j[m]] - nu[m][l[m]] )^p
                end
                temp_val_2 = temp_val_1 + β * temp_val_2^(1/p)

                for b = 0:B-b̃

                    val = temp_val_2 + Φ[b+1, j..., i%2+1]
					
                    if Φ[b+b̃+1, l..., (i+1)%2+1] > val
                        U[:,b+b̃+1, l..., i] .= j
                        Φ[b+b̃+1, l..., (i+1)%2+1] = val
                    end

                end

            end
        end
    end
end

@doc raw"""
    eval_u_TRM!(u,u_old,U,Φ,B,nu)

Computes a discrete solution `u` from the trust-region subproblem using the Matrices `U` and `Φ` generated in the function `bellman_TRM`.

# Arguments
- `u`: Vector where the solution will be stored.
- `u_old`: Vector representing the previous solution with the same length as `u`.
- `U`: Matrix generated by `bellman_TR`
- `Φ`: Matrix generated by `bellman_TR`
- `B`: Budget representing the maximum total deviation of `u` and `u_old`
- `nu`: Vector containing the possible integervalues of `u`.
"""
function eval_u_TRM!(u, u_old, U, Φ, B, nu)

    # calculate u with given U and Phi

    M, n = size(u_old)
	
	Inds = CartesianIndices(size(Φ)[2:end-1])

    index = argmin(@view Φ[1:B+1,Inds,1]) 

    l = [index[m+1] for m=1:M]
    b = index[1] - 1
    @inbounds for m = 1:M
        u[m, 1] = nu[m][l[m]]
    end


    @inbounds for i = 1:n-1
        l = U[:,b+1, l..., i]
        for m = 1:M
            u[m, i+1] = nu[m][l[m]]
        end

        b = convert(Int64, b - norm(u[:,i] - u_old[:,i],1))
    end

end

@doc raw"""
    rand_func(obj, jumps::Int=Int(floor(obj.nt/10)), σ = 100.)

Generates a random admissible control `x0` for mixed-integer optimal control problems specified in `obj`.
For the non-integer part, a Gaussian kernel with width `σ` is used to construct a continuous function 
from random noise by convolution. For the integer part, the `jumps`-many switching times and values inbetween 
are generated randomly from the feasible set specified in the structure `obj`. 

Optionally specify a seed `rng` for randomization.
"""
function rand_func(obj;rng=rand(1:10^10), jumps::Int=Int(floor(obj.nt/10)), σ = 100.)
    
    x0 = zeros(Float64, obj.nx, obj.nt) 

    if obj.nu > 0
        x0[1:obj.nu,:] .= rand_func_cont(obj;rng = rng,σ=σ)
    end
    if obj.nv > 0
        x0[obj.nu+1:obj.nx, :] .= rand_func_int(obj;rng = rng,jumps = jumps)
    end
    
    return x0
end

@doc raw"""
    rand_func_cont(obj; rng=rand(1:10^10), σ = 100.)

Generates a random continuous function `u0` of dimension `obj.nu` that is admissible, i.e. satisfies `obj.umin[i,j]` ≤ `u0[i,j]` ≤ `obj.umax[i,j]`
for i = 1,...,`obj.nu`, j = 1,...,`obj.nt`. A Gaussian kernel with width `σ` is used to construct a continuous function from random noise by convolution.  

Optionally specify a seed `rng` for randomization.
"""
function rand_func_cont(obj; rng=rand(1:10^10), σ = 100.)
    r = MersenneTwister(Int(rng))
	# Generate noise
	ξ = randn(r, Float64, (Int64(obj.nu),Int64(obj.nt)))
	# Generate Gaussian kernel
	Gaussian = [exp(-((i-obj.nt/2)^2)/(2*σ^2)) for i in 1:obj.nt]
	Gaussian .= Gaussian ./ sum(Gaussian)
	# Convolute Noise and Kernel
	u0 = zeros(Float64, obj.nu, obj.nt)
	for i = 1:obj.nu
		full_conv = conv(ξ[i,:],Gaussian)
		start_idx = Int(floor((length(full_conv)-obj.nt)/2) + 1)
		end_idx = start_idx + obj.nt - 1
		u0[i,:] = full_conv[start_idx:end_idx]
	end

    # Normalize to range [min_range, max_range]
    min_range = minimum(obj.umin, dims = 2)
    max_range = maximum(obj.umax, dims = 2)
    u0 = min_range .+ (max_range - min_range) * (u0 .- minimum(u0, dims = 2)) ./ (maximum(u0, dims = 2) - minimum(u0, dims = 2))
	
    # Now, there might be some points where u0 is not in [umin,umax], since these are pointwise bounds.
    # We project u0 to a function that satisfies these pointwise bounds
    for i = 1:obj.nt
        for j = 1:obj.nu
           if u0[j, i] > obj.umax[j,i]
              u0[j, i] = obj.umax[j,i]
           end
           if u0[j, i] < obj.umin[j,i]
              u0[j, i] = obj.umin[j,i]
           end
        end
    end

	return u0
end

@doc raw"""
    rand_func_int(obj; rng=rand(1:10^10), jumps::Int=Int(floor(obj.nt/10)))

Generates a piecewise constant function of dimension `obj.nv`. The amount of switching times is 
specified by the variable `jumps`. The values between the switching times are generated randomly 
from the feasible set specified in the structure `obj`. 

Optionally specify a seed `rng` for randomization.
"""
function rand_func_int(obj; rng=rand(1:10^10), jumps::Int=Int(floor(obj.nt/10)))
    
    r = MersenneTwister(Int(rng))
    # jump times
    t = sample(r, range(2, obj.nt, step=1), jumps, replace=false, ordered=true) 
	
	# Initialize start function u
    v0 = zeros(Float64, obj.nv, obj.nt) 
	j = 1

	# Assign new values at jumps
	l = rand(r,collect(obj.iterator))
    for i = 1:obj.nt
        if j ≤ jumps && i ≥ t[j]
            j += 1
			l = rand(r,collect(obj.iterator))
        end
        v0[:,i] = [obj.𝓥[m][l[m]] for m = 1:obj.nv]
    end

    return v0
end

@doc raw"""
    TV_p(u::AbstractMatrix{T},p::Int64) where T

Computes a measure of variation of a function represented by the vector `u` using the `p`-Norm. 
`p = Inf` corresponds to the `∞`-Norm.

# Examples

```julia-repl
julia> u = [1 -1 1; 3 3 0; 2 2 1]
3×3 Matrix{Int64}:
 1  -1  1
 3   3  0
 2   2  1
julia> TV_p(u,1)
8

julia> TV_p(u,2)
5.741657386773941

julia> TV_p(u,Inf)
5
```
"""
function TV_p(u::AbstractMatrix{T},p) where T
	_, n = size(u)
	val = zero(T)

	if p == Inf
		for i = 2:n
			val += maximum(abs.(u[:,i] - u[:,i-1]))
		end
	elseif p > 0
		for i = 2:n
			val += sum(@. abs(u[:,i] - u[:,i-1])^p)^(1/p)
		end
	else
		error("Only positive integer valued `p` are accepted!")
	end
	
	return val
end

# Catch case that there is no integer control with multiple dispatch
function TV_p(u::Nothing,p)
	return 0.
end

@doc raw"""
    plot_results(obj, x, state, df)

Plot control `x`, state `state` and scaled derivative `df` corresponding to problem specified in `obj`.
"""
function plot_results(obj)

    x = obj.x
    N = obj.nu
    M = obj.nv
    n = obj.nt
    t_range = range(obj.T0,obj.T1,length=Int(n))

    if typeof(obj) <: AbstractODEObjective
        state = [obj.state0 obj.state[:,1:end-1]]
        if M > 0 && N > 0
            layout = @layout [ 
                grid(max(N, M), 2)
                b
                ]
            fig = plot(layout = layout)
            plot!(fig, collect(t_range), state', subplot = max(N,M)*2+1, title = "States", lw = 2)
        else
            numplots = max(N,M) + 1
            fig = plot(layout = (numplots,1))
            plot!(fig, collect(t_range), state', subplot = numplots, title = "States", lw = 2)

        end
        
        # Save results in pgfplots-Format
        for i=1:obj.ny
            save_latex_format(t_range,state[i,:],"y($i)")
        end
    elseif typeof(obj) <: AbstractPDEObjective
        println("Animating Solution, this could take a few seconds")
        # Determine minimal and maximal u,v for better plot
        if N > 0
            umin = minimum(x[1:N,:])
            umax = maximum(x[1:N,:])
        else 
            umin = Inf
            umax = -Inf
        end

        if M > 0
            vmin = minimum(obj.𝓥[1])
            vmax = maximum(obj.𝓥[1])
            for i=2:obj.nv
                if minimum(obj.𝓥[i]) < vmin
                    vmin = minimum(obj.𝓥[i])
                end
                if maximum(obj.𝓥[i]) > vmax
                    vmax = maximum(obj.𝓥[i])
                end
            end
        else
            vmin = Inf
            vmax = -Inf
        end
        
        # Run solution animation function from FEMBundle
        animate_solution(obj.mesh,obj.state,obj.tau,"final-state"; u = x[1:N,:], v = x[N+1:N+M,:], 
                            u_range = [umin,umax], v_range = [vmin,vmax])
        
        if M > 0 && N > 0
            fig = plot(layout = (max(N,M),2))
        else
            fig = plot(layout = (max(N,M),1))
        end
    else
        if M>0 && N>0
            fig = plot(layout = (max(N,M),2))
        else
            fig = plot(layout = (max(N,M),1))
        end
    end

    # Plot u, df
    # Scale ∇f, i.e. divide by largest entry
    plot_∇f = abs.(copy(obj.df))
    max_∇f = maximum(plot_∇f)
    if max_∇f == 0
        max_∇f = 1
    end
    @. plot_∇f = obj.df / max_∇f

    # Plot all components of u, v, ∇f seperately
    for i = 1:N
        if M>0
            subplotnum = 2*i-1
        else
            subplotnum = i
        end
        plot!(fig, collect(t_range), x[i,:], label = "u$i", subplot = subplotnum, lw = 2, lc=:green)
        plot!(fig, collect(t_range),plot_∇f[i,:], label = "∇fᵤ$i", subplot = subplotnum, lw = 2, lc=:red)         
    end
    for i = 1:M
        if N>0
            subplotnum = 2*i
        else
            subplotnum = i
        end
        plot!(fig, collect(t_range), x[N+i,:], label = "v$i", subplot = subplotnum, lw = 2, lc=:green, seriestype=:steppost)
        plot!(fig, collect(t_range),plot_∇f[N+i,:], label = "∇fᵥ$i", subplot = subplotnum, lw = 2, lc=:red) 
    end

    plot!(legend=:outerright)
    display(fig)
    
    # Save results in pgfplots-Format
    for i=1:N
        save_latex_format(t_range[1:end],x[i,:],"u($i)")
        save_latex_format(t_range[1:end],plot_∇f[i,:],"nabla_f_u($i)")
    end
    for i=1:M
        save_latex_format(t_range[1:end],x[N+i,:],"v($i)")
        save_latex_format(t_range[1:end],plot_∇f[N+i,:],"nabla_f_v($i)")
    end
end


@doc raw"""
         function save_latex_format(x,y,name)

Save the function with value `y[i]` for input `x[i]` for given Vectors `x`, `y` into a file `name`.dat in a format that can be imported by pgfplots in Latex.
"""
function save_latex_format(x,y,name)
   open("data_files/"*name*".dat", "w") do io
      println(io, "x    y")
      for i in eachindex(x)
		print(io, x[i])
		print(io, " ")
		println(io, y[i])
      end
   end
end

@doc raw"""
         import_from_latex_format(name)

Given a .dat file in a format accepted by pgfplots in Latex, recovers the function (or the representative vector) `u` of the file `name`.dat.
"""
function import_from_latex_format(name)
   file = open("data_files/"*name*".dat", "r")
   x = Float64[]
   u = Float64[]
   
   # Read and process the file line by line
   for line in eachline(file)
      # Split each line into columns using whitespace as the delimiter
      columns = split(line)
      
      # Check if the line has at least two columns (x and y)
      if length(columns) >= 2
         # Parse the second column (y) as a Float64 and append it to the vector u
         try 
			xi = parse(Float64,columns[1])
            ui = parse(Float64, columns[2])
			push!(x, xi)
            push!(u, ui)
         catch
			error("Could not parse entries of the following line to Float64:\n"*line)
         end
      end
   end

   # Close the file
   close(file)
   return x, u
end