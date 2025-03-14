# Predefined Plot properties
plot_cols = [:darkblue,:orangered,:darkred,:turquoise,:yellowgreen,:violet,:yellow,:blue,:orange,:darkgreen] # Add more if needed
labels_u = ["u₁","u₂","u₃","u₄","u₅","u₆","u₇","u₈","u₉","u₁₀"] # Add more if needed
labels_v = ["v₁","v₂","v₃","v₄","v₅","v₆","v₇","v₈","v₉","v₁₀"]


@doc raw"""
        plot_solution(mesh,U,name = "finite"; show_mesh = false)

Plots the function given by the coefficient vector `U` on `mesh`.   
"""
function plot_solution(mesh,U,name = "finite"; show_mesh = false)
    
    # Assume that the first np entries of U are the vertice values
    Makie_verts = zeros(Float64, 3, mesh.np)
    Makie_triangles = zeros(Int64,mesh.nt,3)
    # Designate vertices of Makie mesh, needs 3D-mesh -> assign U as z-value
    for i = 1:mesh.np
        Makie_verts[:,i] = [mesh.p[i];U[i]]
    end
    for i = 1:mesh.nt
        Makie_triangles[i,:] = mesh.t[i]
    end

    colors = U[1:mesh.np]
    name = "Solution - "*name*" elements"

    # Create a figure
    fig = Figure()

    # Create a mesh plot with the vertex colors
    ax = fig[1, 1] = Axis3(fig,title = name)
    m = mesh!(ax, Makie_verts, Makie_triangles, color=colors)

	cb = Colorbar(fig[1,1,Right()], m)
    cb.alignmode = Mixed(right = 0)

	# Show mesh plots all mesh lines
    if show_mesh
        for i = 1:mesh.nt
            p1 = Makie_verts[:, Makie_triangles[i,1]]
            p2 = Makie_verts[:, Makie_triangles[i,2]]
            p3 = Makie_verts[:, Makie_triangles[i,3]]
            lines!(ax, [p1 p2], color=:black, linewidth=.5)
            lines!(ax, [p2 p3], color=:black, linewidth=.5)
            lines!(ax, [p3 p1], color=:black, linewidth=.5)
        end
    end
    # Display the figure
	display(fig)
end

@doc raw"""
        animate_solution(mesh,state,tau,name = ""; u = nothing, v = nothing, u_range = nothing, v_range = nothing, x_ticks = nothing)

Animates the state matrix `state` to a time-dependent PDE on the mesh `mesh` with time step length `tau`. 
If `u` or `v` are specified as the matrices corresponding to non-integer/integer controls influencing `state`, a live plot of them
will be created. The vectors `u_range` and `v_range` designate the limits of the y-axis (first entry lower limit, second entry upper limit). If not specified, these will be taken from `u` and `v`.
Use `x_ticks` to specify which ticks on the x-axis should be displayed.
"""
function animate_solution(mesh,state,tau,name = ""; u = nothing, v = nothing, u_range = nothing, v_range = nothing, x_ticks = nothing)
    
    # Initialize time scale and values for plotting
    timesteps = length(state[1,:])
    t_range = 0:tau:tau*(timesteps-1)
    if isnothing(x_ticks)
        x_ticks = 0:1:t_range[end]
    end

    # Designate vertices of Makie mesh, needs 3D-mesh -> assign state as z-value
    Makie_verts = zeros(Float64,3, mesh.np)
    Makie_triangles = zeros(Int64,mesh.nt,3)
    for t=1:timesteps
        for i = 1:mesh.np
            Makie_verts[:,i] = [mesh.p[i];state[i,1]]
        end
    end
    for i = 1:mesh.nt
        Makie_triangles[i,:] = mesh.t[i]
    end

    # Values for plot settings
    grid_min_x = minimum(Makie_verts[1,:])
    grid_max_x = maximum(Makie_verts[1,:])
    grid_min_y = minimum(Makie_verts[2,:])
    grid_max_y = maximum(Makie_verts[2,:])

    color_min = minimum(state)
    color_max = maximum(state)
    colors = state[1:mesh.np,:]

    # Initiate plot
    fig = Figure(;size = (800,800),backgroundcolor = RGBf(0.98, 0.98, 0.98))
    figl = fig[1,1] = GridLayout()

    if !isnothing(u) || !isnothing(v)
        # Setup control figure 
        figu = fig[2,1] = GridLayout()
        Label(figu[1, 1, Top()], "Controls", valign = :bottom,font = :bold)
        width_legend = 60

        # Handle input cases
        if !isnothing(u)
            sizeu = size(u)
            if length(sizeu) == 1
                num_plots_u = 1
                u = u'
            else
                num_plots_u = sizeu[1]
            end
        else
            num_plots_u = 0
        end
        if !isnothing(v)
            sizev = size(v)
            if length(sizev) == 1
                num_plots_v = 1
                v = v'
            else
                num_plots_v = sizev[1]
            end
        else
            num_plots_v = 0
        end
        # y-axis for control
        if isnothing(u_range)
            if !isnothing(u)
                u_min = minimum(u)-1
                u_max = maximum(u)+1
            else
                u_min = Inf
                u_max = -Inf
            end
        else
            u_min = u_range[1]
            u_max = u_range[2]
        end
        if isnothing(v_range)
            if !isnothing(v)
                v_min = minimum(v)-1
                v_max = maximum(v)+1
            else
                v_min = Inf
                v_max = -Inf
            end
        else
            v_min = v_range[1]
            v_max = v_range[2]
        end
        y_ticks = floor(min(u_min,v_min)):1:ceil(max(u_max,v_max))

    end

    # Initialize state plot
    time = Observable("State at t = 0.00")
    ax = Axis(figl[1, 1],width = 550, height = 550, title = time)
    xlims!(ax, low = grid_min_x, high = grid_max_x)
    ylims!(ax, low = grid_min_y, high = grid_max_y)

    # Plot first timestep
    m = mesh!(ax, Makie_verts, Makie_triangles, color = colors[:,1], colormap = :thermal,colorrange = (color_min, color_max))    
    cb = Colorbar(figl[1,2], m)
    colgap!(figl, 10)

    # Initialize control plot
    if !isnothing(u) || !isnothing(v)
        ax2 = Axis(figu[1,1], xticks = (x_ticks, string.(x_ticks)), yticks = (y_ticks, string.(y_ticks)),
                width = 700-width_legend, height = 100)

        # Make sure user is not confused
        if num_plots_u + num_plots_v > 10
            error("Plotting only supported for 10 or less controls as of now! To change this, edit constant arrays at the top of this file!")
        end
        dots = []
        for i=1:num_plots_u
            lines!(ax2, t_range, append!(u[i,:],u[i,end]), color = plot_cols[i], 
            label = labels_u[i])
            push!(dots,scatter!([0], [u[i,1]], color= plot_cols[i], markersize=10))
        end
        for i=1:num_plots_v
            stairs!(ax2, t_range, append!(v[i,:],v[i,end]), color = plot_cols[num_plots_u+i], 
                    step=:post, label = labels_v[i])
           push!(dots,scatter!([0], [v[i,1]], color= plot_cols[num_plots_u + i], markersize=10))
        end

        # Define variables changing in frames
        ax2.limits = ((0, t_range[end]),(min(u_min,v_min),max(u_max,v_max)))
        leg = Legend(figu[1,2], ax2)
        leg.width = width_legend
        leg.height = 100
    end

    # Updates each frame
    function update_frame(t)
        for i = 1:mesh.np
            Makie_verts[:, i] = [mesh.p[i]; state[i, t]]
        end
        m[1] = Makie_verts  # Update mesh vertices
        m.color = colors[:, t]  # Update mesh colors

        rounded_time = string(round(100*t_range[t])/100)
        if length(split(rounded_time, ".")[2]) == 1  # If only one decimal place, add a 0 to the string
            rounded_time *= "0"
        end

        # Update time
        time[] = "State at t = "*rounded_time

        # Update moving dot
        if !isnothing(u) || !isnothing(v)
            if t >= timesteps
                for i=1:num_plots_u
                    dots[i][1][] = [Point2f(t_range[end],u[i,end])]
                end
                for i=1:num_plots_v
                    dots[num_plots_u+i][1][] = [Point2f(t_range[end],v[i,end])] 
                end
            else
                for i=1:num_plots_u
                    dots[i][1][] = [Point2f(t_range[t],u[i,t])]
                end
                for i=1:num_plots_v
                    dots[num_plots_u+i][1][] = [Point2f(t_range[t],v[i,t])] 
                end
            end
        end
    end
    
    # Save animation in a file
    record(fig, name == "" ? "pde_solution_animation.mp4" : name*".mp4", 1:timesteps; framerate = 1/tau) do t
        update_frame(t)
    end
end

