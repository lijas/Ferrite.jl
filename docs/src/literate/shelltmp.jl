



function element_routine(ke, fe, material)

    udofs = 1:4*3
    u = reinterpret(Vec{3,Float64}, ae[udofs])
    x = coords .+ u

    for i in 1:N
        n = normal_vector(ip, x)
        fibers[i] = n
    end

    for iqp in 1:length(qr_inp.points)

        ξ, η = qr_inp.points[iqp]
        R_lam = lamina_coordsys()
        
        dNdξ, N = Ferrite.value(ip, ξ)

        for iqp_oop in 1:length(qr_oop.points)
            
            ζ = qr_oop.points[iqp_oop][1]
            dζdξ = Vec{3}((0.0, 0.0, 1.0))
            ξqp = Vec((ξ, η, ζ))

            J = getjacobian(N, dNdξ, ζ, dζdξ, X, p, h)
            
            dζdx = Vec{3}((0.0, 0.0, 1.0)) ⋅ Jinv
            dNdx = [Vec{3}((dNdξ[i][1], dNdξ[i][2], 0.0)) ⋅ Jinv for i in 1:nnodes]

            L = velocity_gradient(N, dNdx, ζ, dζdx, directors, ae)
            D = symmetric(L)
            
            D_lam = R_lam' ⋅ D ⋅ R_lam

            σ_lam, dσdε_lam, new_state = material_response(ShellState(), material, D_lam, state)

            σ = R_lam ⋅ σ_lam ⋅ R_lam'
            dσdε = otimesu(R_lam, R_lam) ⋅ dσdε_lam ⋅  otimesu(R_lam', R_lam')

            δv_matrix = ForwardDiff.jacobian(ae -> velocity_gradient(ξ, directors, ae), ae)
            for i in 1:getnbasefunctions(cv)
                δD = fromvoigt(Tensor{2,3,Float64,9}, @view(δv_matrix[i,:]))
                f[i] += σ ⋅ δD * dV
            end

        end
    end
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

function velocity_gradient(N, dNdx, ζ, dζdx, directors, ae)
    offset = getnbasefunctions(cv)÷2
    L = zero(Tensors{2,3,Float64})
    for a in 1:getnbasefunctions(cv)
        va = Vec{3}(ae[a+i-1])
        wa = Vec{3}(ae[a+i-1+offset])
        L += dNdx[i]*va + h/2*(ζ*dNdx[i] + dζdx*N)*Tensors.cross(wa, directors[i])
    end
    return L
end

function lamina_coordsys(dNdξ, ζ, x, p, h)

    e1 = zero(Vec{3})
    e2 = zero(Vec{3})

    for i in 1:length(dNdξ)
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

function getjacobian(N, dNdξ, ζ, X, p, h)

    J = zeros(3,3)
    for a in 1:length(N)
        for i in 1:3, j in 1:3
            _dNdξ = (j==3) ? 0.0 : dNdξ[a][j]
            _dζdξ = (j==3) ? 1.0 : 0.0
            _N = N[a]

            J[i,j] += _dNdξ * X[a][i]  +  (_dNdξ*ζ + _N*_dζdξ) * h/2 * p[a][i]
        end
    end

    return Tensor{2,3,Float64}(J)
end;

function position()

end