struct QuadraturePoint
	coords::SVector{3,Float64}
	weight::Float64

	function QuadraturePoint(l1,l2,l3,w)
		# We provide an interior constructor to enforce that the barycentric
		# coordinates are valid
		@assert 0 <= l1
		@assert 0 <= l2
		@assert 0 <= l3
		@assert abs( 1 - (l1 + l2 + l3) ) < 1e-15
		return new(SVector{3,Float64}(l1,l2,l3), w)
	end
end

@doc raw"""
    quadrature_unit_triangle_area(exactness_order)

This function returns quadrature points in barycentric coordinates and their
respective weights of a quadrature formula of the desired exactness order on the unit triangle.
A quadrature rule has exactness order `r` if it is exact for all polynomials of degree at most `r`.
"""
function quadrature_unit_triangle_area(exactness_order)

	if (exactness_order == 1)
		return [QuadraturePoint(1/3,1/3,1/3,1/2)]

	elseif (exactness_order == 2)
		return [QuadraturePoint(1/2,0/2,1/2,1/6), QuadraturePoint(1/2,1/2,0/2,1/6), QuadraturePoint(0/2,1/2,1/2,1/6)]

	elseif (exactness_order == 3)
		return [QuadraturePoint(1/3,1/3,1/3,9/40),
				QuadraturePoint(1/2,0/2,1/2,1/15), QuadraturePoint(1/2,1/2,0/2,1/15), QuadraturePoint(0/2,1/2,1/2,1/15),
				QuadraturePoint(1,0,0,1/40), QuadraturePoint(0,1,0,1/40), QuadraturePoint(0,0,1,1/40)]

	elseif (exactness_order == 4)

		a1 = 0.445948490915965;
		a2 = 0.091576213509771;
		w1 = 0.223381589678010;
		w2 = 0.109951743655322;

		return [QuadraturePoint(a1,a1,1-2*a1,w1/2), QuadraturePoint(a1,1-2*a1,a1,w1/2), QuadraturePoint(1-2*a1,a1,a1,w1/2),
				QuadraturePoint(a2,a2,1-2*a2,w2/2), QuadraturePoint(a2,1-2*a2,a2,w2/2), QuadraturePoint(1-2*a2,a2,a2,w2/2)]

	elseif (exactness_order == 5)

		a1 = (6-sqrt(15))/21;
		a2 = (6+sqrt(15))/21;
		w1 = (155-sqrt(15))/2400;
		w2 = (155+sqrt(15))/2400;

		return [QuadraturePoint(1/3,1/3,1/3,9/80),
				QuadraturePoint(a1,a1,1-2*a1,w1), QuadraturePoint(a1,1-2*a1,a1,w1), QuadraturePoint(1-2*a1,a1,a1,w1),
				QuadraturePoint(a2,a2,1-2*a2,w2), QuadraturePoint(a2,1-2*a2,a2,w2), QuadraturePoint(1-2*a2,a2,a2,w2)]

	elseif (exactness_order == 6)

		a1 = 0.063089014491502;
		a2 = 0.249286745170910;
		a  = 0.310352451033785;
		b  = 0.053145049844816;

		w1 = 0.050844906370206;
		w2 = 0.116786275726378;
		w3 = 0.082851075618374;

		return [QuadraturePoint(a1,a1,1-2*a1,w1/2), QuadraturePoint(a1,1-2*a1,a1,w1/2), QuadraturePoint(1-2*a1,a1,a1,w1/2),
		QuadraturePoint(a2,a2,1-2*a2,w2/2), QuadraturePoint(a2,1-2*a2,a2,w2/2), QuadraturePoint(1-2*a2,a2,a2,w2/2),
		QuadraturePoint(a,b,1-a-b,w3/2), QuadraturePoint(a,1-a-b,b,w3/2), QuadraturePoint(b,a,1-a-b,w3/2),
		QuadraturePoint(b,1-a-b,a,w3/2), QuadraturePoint(1-a-b,a,b,w3/2), QuadraturePoint(1-a-b,b,a,w3/2)]

	else

		error("Quadrature formula of exactness order $(exactness_order) not implemented.")

	end
end

@doc raw"""
    quadrature_unit_triangle_bdry(edge, exactness_order)

This function returns the barycentric coordinates and weights of a
quadrature formula of the desired exactness order on the edge with the given
number (1,2,3) of the unit triangle 
"""
function quadrature_unit_triangle_bdry(edge, exactness_order)

    # We program the quadrature formula first as though it were for edge # 3
    if (exactness_order == 1)

        # see Ern & Guermond, p.359
        l1 = 1 / 2
        l2 = 1 - l1
        l3 = 0
        w = 1

    elseif (exactness_order == 3)

        # see Ern & Guermond, p.359
        l1 = [0.5 + 0.5 * sqrt(3) / 3, 0.5 - 0.5 * sqrt(3) / 3]
        l2 = 1 .- l1
        l3 = [0.0 0.0]
        w = [0.5 0.5]

    elseif (exactness_order == 5)

        # see Ern & Guermond, p.359
        l1 = [0.5 + 0.5 * sqrt(3 / 5), 0.5, 0.5 - 0.5 * sqrt(3 / 5)]
        l2 = 1 .- l1
        l3 = [0.0 0.0 0.0]
        w = [5 / 18 8 / 18 5 / 18]

    else
        error("Quadrature formula of exactness order $exactness_order not implemented.\n\n")
    end

    # If an edge different than # 3 is requested, we simply permute the
    # evaluation points and adjust the weight
    if (edge == 1)
        l1, l2, l3 = l3, l1, l2
    elseif (edge == 2)
        l1, l2, l3 = l2, l3, l1
    end

    if exactness_order == 1
        return [QuadraturePoint(l1, l2, l3, w)]
    elseif exactness_order == 3
        return [QuadraturePoint(l1[1], l2[1], l3[1], w[1]), QuadraturePoint(l1[2], l2[2], l3[2], w[2])]
    else
        return [QuadraturePoint(l1[1], l2[1], l3[1], w[1]), QuadraturePoint(l1[2], l2[2], l3[2], w[2]),
            QuadraturePoint(l1[3], l2[3], l3[3], w[3])]
    end
end


