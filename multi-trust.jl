# multi-trust.jl
# Contains a trust-region based algorithm for integer optimal control problems.

using Printf
using Parameters
using LinearAlgebra

push!(LOAD_PATH, "./julia_opt/julia_fem/")
push!(LOAD_PATH, "./julia_opt/")
push!(LOAD_PATH, "./")

using OptBundle, FEMBundle
using ODEObjective, PDEObjective

# Import all examples
example_files = filter(f -> startswith(f, "example_"), readdir("./julia_opt/"))
examples = [replace(f, ".jl" => "") for f in example_files]
for mod in examples
   @eval using $(Symbol(mod))
end

# Auxilliary functions used in trust region method
include("HelpFunctions.jl")

# Algorithmic parameters
@with_kw struct TRM_parameters
   β::Float64 = .001 # Weight for TVp - influence
   p = 1 # Parameter of TVp, p = Inf marks p = ∞ as norm
   Δ⁰::Float64 = 1 # Initial trust region radius
   σ::Float64 = .5 # Desired rate of actual reduction w.r.t. predicted reduction
   kmax::UInt64 = 40 # Maximum number inner iterations (i.e., reductions of Δᵏ)
   maxiter::UInt64 = 1000 # maximum number of outer iterations
   log::Bool = false # Show information about process during runtime
end

@doc raw"""
    function TRM(obj::AbstractObjective, par::TRM_parameters = TRM_parameters(); x0 = rand_func(obj))

Computes an approximated solution for the problem specified in `obj` using a Trust-Region algorithm and 
the Bellman-principle. The algorithm uses parameters stored in the struct `par`. Setting `par.log = true` 
enables the output of monitor-parameters. It is possible to specify a starting integer control `x₀` using 
`x0 = x₀`.

# Example
```julia-repl
julia> include("multi-trust.jl")
julia> obj = LVMObj()
julia> J = TRM(obj)
julia> println(J)
0.9398946251530471
```
"""
function TRM(obj::AbstractObjectiveLazy, par::TRM_parameters = TRM_parameters(); x0 = rand_func(obj))

# Define Parameters
n = obj.nt
M = obj.nx
Δt = obj.tau
iterator = obj.iterator

@unpack β,Δ⁰,σ,p,kmax,maxiter = par

# Define Vectors and Matrices for algorithm
nu = obj.𝓥
u = obj.x
u .= x0
u_old = copy(u)

B = Int64(floor(Δ⁰/Δt))

dims = zeros(Int64,M+3)
dims[1] = M
dims[2] = B+1
dims[3:M+2] .= [length(nu[m]) for m = 1:M]
dims[M+3] = n-1
U = zeros(Int64,dims...)
Φ = zeros(Float64,dims[2:M+2]...,2)

# Define starting values for algorithm
J = Inf
iter = 1
stop = false
J_old = eval_f!(obj)

# Initialize table
if par.log
   @printf " Iter |   k |   Δᵏ   |      J      |   pred   |   ared   |       step            \n"
   @printf "---------------------------------------------------------------------------------\n"
   @printf "%5i |%4i | %6.2f | %.5e | %8.4f | %8.4f | %s   \n" 0 0 Δ⁰ J_old+β*TV_p(u,p) 0 0 "Initial Value"
end

while !stop && (iter ≤ maxiter)

   Δᵏ = Δ⁰
   k = 1
   ared = 0.
   pred = 1.
   halved = false
   TV_old = TV_p(u,p)
   
   # Calculate ∇f
   eval_df!(obj)
   ∇f = obj.df

   while (ared < σ*pred) && (k ≤ kmax)
      
      # Solve trustregion-subproblem
      if halved
         B_new = convert(Int64,floor(Δᵏ/Δt))
         eval_u_TRM!(u,u_old,U,Φ,B_new,nu)
      else
         bellman_TRM!(∇f, u_old, B, β, p, Δt, nu, U, Φ, iterator)
         eval_u_TRM!(u,u_old,U,Φ,B,nu)
      end

      # Calculate pred and ared
      int_val = 0.
      for j = 1:n
        int_val += ∇f[:,j]' * (u_old[:,j] - u[:,j])
      end
      int_val *= Δt

      TV_new = TV_p(u,p)
      J_new = eval_f!(obj)

      pred = int_val + β*(TV_old-TV_new)
      ared = J_old - J_new + β*(TV_old-TV_new)

      # Adapt Δᵏ or terminate
      if pred ≤ 0

         J = J_old
         stop = true

         if par.log
            @printf "%5i |%4i | %6.2f | %.5e | %8.4f | %8.4f | %s   \n" iter k Δᵏ J+β*TV_old pred ared "optimal solution found"
         end
         break

      elseif ared < σ*pred

         if par.log
            @printf "%5i |%4i | %6.2f | %.5e | %8.4f | %8.4f | %s   \n" iter k Δᵏ J_old+β*TV_old pred ared "bad step, Δᵏ halved"
         end
         Δᵏ = Δᵏ/2
         halved = true

      else

         u_old .= u
         J_old = J_new
         TV_old = TV_new
         J = J_new
         if par.log
            @printf "%5i |%4i | %6.2f | %.5e | %8.4f | %8.4f | %s   \n" iter k Δᵏ J+β*TV_new pred ared "good step"
         end

      end
      k += 1 
   end

   iter += 1
end

# Evaluate final derivative for plotting purposes
eval_df!(obj)
∇f = obj.df

return J+β*TV_p(u,p)
end

@doc raw"""
    main(problem = "fishing"; do_plot = true, n = 1024 )

Solve a predefined optimization problem specified by `problem` using `TRM`.
Choose if solution should be plotted with the boolean `do_plot` and choose the discretization by specifying the
number of timesteps `n`.
"""
function main(problem = "fishing"; do_plot = true, n = 1024 )

	if problem == "fishing"
		obj = LVMObj( nt = n )
      parameters = TRM_parameters( β = .0001, Δ⁰ = 2,  log = true, p = Inf)
   elseif problem == "doubletank"
      obj = DTMObj( nt = n )
      parameters = TRM_parameters( β = .00001, Δ⁰ = 2,  log = true, p = Inf)
   elseif problem == "vanderpol"
      obj = VPOObj(nt = n)
      parameters = TRM_parameters( β = .1, Δ⁰ = 1,  log = true, p = Inf)
   elseif problem == "convolution"
      obj = ConvObj(nt = n)
      parameters = TRM_parameters( β = .0001, Δ⁰ = .125,  log = true, p = 1)
   elseif problem == "heat"
      obj = HeatObj(nt = n)
      parameters = TRM_parameters( β = .001, Δ⁰ = 2,  log = true, p = 2)
   else
		error("I do not know the problem \"$problem\".")
	end

   @time J = TRM( obj, parameters)
   println("Objective Value: J = $J")

   if do_plot
      plot_results(obj)
   end

end