using Tensors
using Ferrite
using MaterialModels
using ForwardDiff

function element_routine(fe, material, reference_coords::Vector{Vec{3,Float64}}, h::Float64, ae::Vector{T}) where T

    ip = Lagrange{2,RefCube, 1}()
    qr_inp = QuadratureRule{2,RefCube}(2)
    qr_oop = QuadratureRule{1,RefCube}(2)

    udofs = 1:4*3
    u = reinterpret(Vec{3,T}, ae[udofs])
    coords = reference_coords .+ u
    fibers = normal_vector(ip, coords)

    fe = zeros(T, 24)
    ke = zeros(T, 24, 24)

    for iqp in 1:length(qr_inp.points)

        ξ, η = qr_inp.points[iqp]
        ξη = qr_inp.points[iqp]

        N = zeros(Float64, getnbasefunctions(ip))
        dNdξ = zeros(Vec{2, Float64}, getnbasefunctions(ip))
        for i in 1:getnbasefunctions(ip)
            N[i] = Ferrite.value(ip, i, ξη)
            dNdξ[i] = Tensors.gradient(x -> Ferrite.value(ip, i, x), ξη)
        end

        
        for iqp_oop in 1:length(qr_oop.points)
            
            ζ = qr_oop.points[iqp_oop][1]
            dζdξ = Vec{3}((0.0, 0.0, 1.0))

            R_lam = lamina_coordsys(dNdξ, ζ, coords, fibers, h)
            ξqp = Vec((ξ, η, ζ))

            J = getjacobian(N, dNdξ, ζ, dζdξ, coords, fibers, h)
            
            dV = det(J) * qr_oop.weights[iqp_oop] * qr_inp.weights[iqp]

            Jinv = inv(J)
            dζdx = Vec{3}((0.0, 0.0, 1.0)) ⋅ Jinv
            dNdx = [Vec{3}((dNdξ[i][1], dNdξ[i][2], 0.0)) ⋅ Jinv for i in 1:4]

            L = velocity_gradient(N, dNdx, ζ, dζdx, fibers, h, ae)
            D = symmetric(L)

            D_lam = R_lam' ⋅ D ⋅ R_lam
            D_lam = symmetric(D_lam)

            material = LinearElastic(;E=100.0, ν = 0.3)
            state = initial_material_state(material)
            σ_lam, dσdε_lam, new_state = MaterialModels.material_response(material, D_lam, state)

            σ = R_lam ⋅ σ_lam ⋅ R_lam'
            dσdε = symmetric(otimesu(R_lam,R_lam) ⊡ dσdε_lam ⊡ otimesu(R_lam',R_lam'))

            δL_matrix = ForwardDiff.jacobian(ae -> velocity_gradient(N, dNdx, ζ, dζdx, fibers, h, ae), ae)
            for i in 1:24
                δDi = symmetric( fromvoigt(Tensor{2,3,T,9}, @view(δL_matrix[:,i])) )
                fe[i] += σ ⊡ δDi * dV
                for j in 1:24
                    δDj = symmetric( fromvoigt(Tensor{2,3,T,9}, @view(δL_matrix[:,j])) )
                    ke[i,j] += δDj ⊡ dσdε ⊡ δDi * dV
                end
            end
        end
    end

    k = 1.0
    for i in 1:4
        n = fibers[i]
        kww = k*(n ⊗ n)
        for j in 1:3
            A = 12 + (i-1)*3 + j
            ke[A,A] += kww[j,j]
        end
    end

    return ke, fe
end

function velocity(ξ, directors, ae)
    offset = getnbasefunctions(cv)÷2
    for a in 1:getnbasefunctions(cv)
        ξη= Vec{2}(ξ[i])
        ζ = ξ[3]
        N = Ferrite.value(ip, a, ξη)
        
        va = Vec{3}(ae[a+i-1])
        wa = Vec{3}(ae[a+i-1+offset])
        v += N*va + h/2*N*ζ*Tensors.cross(wa, directors[i])
    end
end

function velocity_gradient(N, dNdx, ζ, dζdx, directors, h, ae)
    offset = 4*3
    L = zero(Tensor{2,3,Float64})
    for a in 1:4
        va = Vec{3}(i ->ae[(a-1)*3 + i])
        wa = Vec{3}(i ->ae[(a-1)*3 + i + offset])
        L += dNdx[a] ⊗ va + h/2*(ζ*dNdx[a] + dζdx*N[a]) ⊗ Tensors.cross(wa, directors[a])
    end
    return L
end

function lamina_coordsys(dNdξ, ζ, x, p, h)

    e1 = zero(Vec{3})
    e2 = zero(Vec{3})

    for i in 1:4
        e1 += dNdξ[i][1] * x[i] + 0.5*h*ζ * dNdξ[i][1] * p[i]
        e2 += dNdξ[i][2] * x[i] + 0.5*h*ζ * dNdξ[i][1] * p[i]
    end

    e1 /= norm(e1)
    e2 /= norm(e2)

    ez = Tensors.cross(e1,e2)
    ez /= norm(ez)

    a = 0.5*(e1 + e2)
    a /= norm(a)

    b = Tensors.cross(ez,a)
    b /= norm(b)

    ex = sqrt(2)/2 * (a - b)
    ey = sqrt(2)/2 * (a + b)

    return Tensor{2,3}(hcat(ex,ey,ez))
end;

function getjacobian(N, dNdξ, ζ, dζdξ, coords::Vector{Vec{3,T}}, fibers, h) where T

    J = zeros(T,3,3)
    for a in 1:4
        for i in 1:3, j in 1:3
            _dNdξ = (j==3) ? 0.0 : dNdξ[a][j]
            _dζdξ = (j==3) ? dζdξ[j] : 0.0
            _N = N[a]

            J[i,j] += _dNdξ * coords[a][i]  +  (_dNdξ*ζ + _N*_dζdξ) * h/2 * fibers[a][i]
        end
    end

    return Tensor{2,3}(J)
end;

function normal_vector(ip, coords::Vector{Vec{3,T}}) where T

    normals = zeros(Vec{3,T}, 4)
    for j in 1:4
        ξ = Ferrite.reference_coordinates(ip)[j]
       
        dNdξ = zeros(Vec{2, Float64}, getnbasefunctions(ip))
        for i in 1:getnbasefunctions(ip)
            dNdξ[i] = Tensors.gradient(x -> Ferrite.value(ip, i, x), ξ)
        end

        dxdξ = zero(Vec{3,T})
        dxdη = zero(Vec{3,T})
        for i in 1:4
            dxdξ += dNdξ[i][1]*coords[i]
            dxdη += dNdξ[i][2]*coords[i]
        end
        normals[j] = Tensors.cross(dxdξ,dxdη)
    end
    return normals

end

coords = [
    Vec((-1.0, -1.0, 0.0)),
    Vec((1.0, -1.0, 0.0)),
    Vec((1.0, 1.0, 0.0)),
    Vec((-1.0, 1.0, 0.0)),
]

h = 0.1

fe = zeros(4*6)
material = 2
ae = zeros(Float64, 4*6)
element_routine(fe, material, coords, h, ae)

#ke = ForwardDiff.jacobian(ae -> element_routine(fe, material, coords, h, ae), ae)
ke, fe = element_routine(fe, material, coords, h, ae)

