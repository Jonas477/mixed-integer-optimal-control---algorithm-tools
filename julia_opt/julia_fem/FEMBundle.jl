module FEMBundle

using Accessors
using LinearAlgebra
using Parameters
using Printf
using SparseArrays
using StaticArrays
using Triangulate
using WriteVTK
using CairoMakie

export Mesh, mesh_library, init_mesh, refine_adaptively, refine_adaptively!,refine_all_cells, prolongation
export FE, FE_Lagrange, ndofs, nlocaldofs, vnlocaldofs, dofmap, shape, flat_dofmap, dof, local_dofs, dirichlet_constraints, name
export area_integrator, bdry_integrator
export quadrature_unit_triangle_area, quadrature_unit_triangle_bdry
export write_vtk, pvd_append
export plot_solution, animate_solution

include("mesh.jl")
include("FE.jl")
include("assembly.jl")
include("quadrature_unit_triangle.jl")
include("write_vtk.jl")
include("plot_solution.jl")


end
