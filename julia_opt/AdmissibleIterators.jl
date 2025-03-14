# AdmissibleIterators.jl
# Contains methods returning iterators over admissible integer controls

@doc raw"""
      product_iterator( nu )

Returns an iterator cycling through all tuples of indices of the ragged array `nu`.
"""
function product_iterator( nu )
    nx = length(nu)

    # Get dimensions of ragged array nu
    sizes = [length(nu[i]) for i in 1:nx]
    range_vec = [1:sizes[i] for i in 1:nx]

    # Iterator containing all possible combinations of indices belonging to nu
    return Iterators.product(range_vec...)
end

@doc raw"""
      bounded_sum_iterator( nu, lower_bound, upper_bound)

Returns an iterator cycling through all tuples of indices `l` of the ragged array `nu` where the 
sum of all values in `nu[i][l[i]]` is in the range `[lower_bound, upper_bound]`.
"""
function bounded_sum_iterator( nu, lower_bound, upper_bound)   
    nx = length(nu)

    # Get product iterator
    prod_iterator = product_iterator( nu ) 

    # Filter iterator s.t. check_sum is satisfied
    return (l for l in prod_iterator if check_sum(l,nu,nx,lower_bound,upper_bound))
end

@doc raw"""
      check_sum(l,nu,ctrlsum)

Check if the sum of the entries of `nu(l)` are in the bounds `lb`, `ub`. 
"""
function check_sum(l,nu,nx,lb,ub)

    val = 0
    for i=1:nx
       val += nu[i][l[i]]
    end

    return val ≥ lb && val ≤ ub
end