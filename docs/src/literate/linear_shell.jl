# # Linear shell
#
# ![](linear_shell.png)
#-
# ## Introduction
#
# In this example we show how shell elements can be analyzed in Ferrite.jl. The shell implemented here comes from the book 
# "The finite elment method - Linear static and dynamic finite element analysis" by Hughes (1987), and a brief description of it is 
# given at the end of this tutorial.  The first part of the tutorial explains how to set up the problem.

# ## Setting up the problem
using Ferrite
using ForwardDiff
using MaterialModels

function main() #wrap everything in a function...

# First we generate a flat rectangular mesh. There is currently no built-in function for generating
# shell meshes in Ferrite, so we have to create our own simple mesh generator (see the  
# function `generate_shell_grid` further down in this file).
#+
nels = (10,10)
size = (10.0, 10.0)
grid = generate_shell_grid(nels, size)

# Here we define the bi-linear interpolation used for the geometrical description of the shell.
# We also create two quadrature rules for the in-plane and out-of-plane directions. Note that we use 
# under integration for the inplane integration, to avoid shear locking. 
#+
ip = Lagrange{2,RefCube,1}()
qr_inplane = QuadratureRule{2,RefCube}(1)
qr_ooplane = QuadratureRule{1,RefCube}(2)
cv = CellScalarValues(qr_inplane, ip)

# Next we distribute displacement dofs,`:u = (x,y,z)` and rotational dofs, `:θ = (θ₁,  θ₂)`.
#+
dh = DofHandler(grid)
push!(dh, :u, 3, ip)
push!(dh, :θ, 3, ip)
close!(dh)

# In order to apply our boundary conditions, we first need to create some edge- and vertex-sets. This 
# is done with `addedgeset!` and `addvertexset!` (similar to `addfaceset!`)
#+
addedgeset!(grid, "left",  (x) -> x[1] ≈ 0.0)
addedgeset!(grid, "right", (x) -> x[1] ≈ size[1])
addvertexset!(grid, "corner", (x) -> x[1] ≈ 0.0 && x[2] ≈ 0.0 && x[3] ≈ 0.0)

# Here we define the boundary conditions. On the left edge, we lock the displacements in the x- and z- directions, and all the rotations.
#+
ch = ConstraintHandler(dh)
add!(ch,  Dirichlet(:u, getedgeset(grid, "left"), (x, t) -> (0.0, 0.0), [1,3])  )
add!(ch,  Dirichlet(:θ, getedgeset(grid, "left"), (x, t) -> (0.0, 0.0), [1,2])  )

# On the right edge, we also lock the displacements in the x- and z- directions, but apply a precribed roation.
#+
add!(ch,  Dirichlet(:u, getedgeset(grid, "right"), (x, t) -> (0.01, 0.0), [1,3])  )
#add!(ch,  Dirichlet(:θ, getedgeset(grid, "right"), (x, t) -> (0.0, pi/10), [1,2])  )

# In order to not get rigid body motion, we lock the y-displacement in one fo the corners.
#+
add!(ch,  Dirichlet(:u, getvertexset(grid, "corner"), (x, t) -> (0.0), [2])  )

close!(ch)
update!(ch, 0.0)

# Next we define relevant data for the shell, such as shear correction factor and stiffness matrix for the material. 
# In this linear shell, plane stress is assumed, ie $\\sigma_{zz} = 0 $. Therefor, the stiffness matrix is 5x5 (opposed to the normal 6x6).
#+
κ = 5/6 # Shear correction factor
E = 210.0
ν = 0.3
a = (1-ν)/2
C = E/(1-ν^2) * [1 ν 0   0   0;
                ν 1 0   0   0;
                0 0 a*κ 0   0;
                0 0 0   a*κ 0;
                0 0 0   0   a*κ]


data = (thickness = 1.0, C = C); #Named tuple

# We now assemble the problem in standard finite element fashion
#+
nnodes = getnbasefunctions(ip)
ndofs_shell = ndofs_per_cell(dh)

K = create_sparsity_pattern(dh)
f = zeros(Float64, ndofs(dh))

celldofs = zeros(Int, ndofs_shell)
cellcoords = zeros(Vec{3,Float64}, nnodes)

Δa = zeros(Float64, ndofs(dh))
a  = zeros(Float64, ndofs(dh))

apply!(a, ch)
apply!(Δa, ch)

for inewton in 1:4

    assembler = start_assemble(K, f)
    for cellid in 1:getncells(grid)

        celldofs!(celldofs, dh, cellid)
        getcoordinates!(cellcoords, grid, cellid)

        ae = a[celldofs]
        Δae = Δa[celldofs]

        #Call the element routine
        fe, ke = element_routine(cellcoords, h, ae, Δae)

        assemble!(assembler, celldofs, fe, ke)
    end

    # Apply BC and solve.
    #+
    apply!(K, f, ch)
    δa = K\f

    residual = norm(f)
    a += δa
    Δa += δa

    @show residual

end
# Output results.
#+
vtk_grid("linear_shell", dh) do vtk
    vtk_point_data(vtk, dh, a)
end

end; #end main functions

# Below is the function that creates the shell mesh. It simply generates a 2d-quadrature mesh, and appends
# a third coordinate (z-direction) to the node-positions. 
function generate_shell_grid(nels, size)
    _grid = generate_grid(Quadrilateral, nels, Vec((0.0,0.0)), Vec(size))
    nodes = [(n.x[1], n.x[2], 0.0) |> Vec{3} |> Node  for n in _grid.nodes]
    cells = [Quadrilateral3D(cell.nodes) for cell in _grid.cells]

    grid = Grid(cells, nodes)

    return grid
end;


function element_routine(ccoords::Vector{Vec{3,Float64}}, h::Float64, ae::Vector{T}, Δae::Vector{T}) where T

    material = LinearElastic(;E=100.0, ν = 0.3)
    ip = Lagrange{2,RefCube, 1}()
    qr_inp = QuadratureRule{2,RefCube}(2)
    qr_oop = QuadratureRule{1,RefCube}(2)

    udofs = 1:4*3
    u = reinterpret(Vec{3,T}, ae[udofs])
    coords = ccoords .+ u
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

            L = velocity_gradient(N, dNdx, ζ, dζdx, fibers, h, Δae)
            D = symmetric(L)

            D_lam = R_lam' ⋅ D ⋅ R_lam
            D_lam = symmetric(D_lam)

            state = initial_material_state(material)
            σ_lam, dσdε_lam, new_state = MaterialModels.material_response(material, D_lam, state)

            σ = R_lam ⋅ σ_lam ⋅ R_lam'
            dσdε = symmetric(otimesu(R_lam,R_lam) ⊡ dσdε_lam ⊡ otimesu(R_lam',R_lam'))

            δL_matrix = ForwardDiff.jacobian(Δae -> velocity_gradient(N, dNdx, ζ, dζdx, fibers, h, Δae), Δae)
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

function velocity_gradient(N, dNdx, ζ, dζdx, directors, h, Δae)
    offset = 4*3
    L = zero(Tensor{2,3,Float64})
    for a in 1:4
        va = Vec{3}(i ->Δae[(a-1)*3 + i])
        wa = Vec{3}(i ->Δae[(a-1)*3 + i + offset])
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


# Run everything:
main()
