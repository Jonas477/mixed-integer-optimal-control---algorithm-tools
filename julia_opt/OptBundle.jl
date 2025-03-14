module OptBundle

# Objectives
export AbstractObjective, AbstractObjectiveAAO, AbstractObjectiveLazy

# Optimizers
export AbstractLineSearchOptimizer, opt_optimize, NonlinCG, SteepestDescent

# Line searches
export AbstractLineSearchMethod, ArmijoLS, WolfeLS

# Evaluation routines
export eval_f, eval_f!, eval_df, eval_df!, eval_fdf!

# Iterators over admissible controls
export product_iterator, bounded_sum_iterator

# The actual implementations
include("AbstractObjective.jl")
include("AdmissibleIterators.jl")

include("LineSearches.jl")
include("AbstractLineSearchOptimizer.jl")

include("NonlinCG.jl")
include("SteepestDescent.jl")

end
