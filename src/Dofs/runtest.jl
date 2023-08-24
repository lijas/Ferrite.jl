using Ferrite
using MaterialModels

function integrate_element!(ke::AbstractMatrix, C::SymmetricTensor{4,2}, cv)
    n_basefuncs = getnbasefunctions(cv)

    δɛ = [zero(SymmetricTensor{2,2,Float64}) for i in 1:n_basefuncs]
    for q_point in 1:getnquadpoints(cv)

        for i in 1:n_basefuncs
            δɛ[i] = symmetric(shape_gradient(cv, q_point, i))
        end

        dΩ = getdetJdV(cv, q_point)
        for i in 1:n_basefuncs
            for j in 1:n_basefuncs
                ke[i, j] += (δɛ[i] ⊡ C ⊡ δɛ[j]) * dΩ
            end
        end
    end
end;

grid = generate_grid(Quadrilateral, (10,10))

ip = Lagrange{RefQuadrilateral,1}()^2
dh = DofHandler(grid) 
add!(dh, :u, ip)
close!(dh)

cdofs = [1,]
n = ndofs_per_cell(dh)
Ge = zeros(n,length(cdofs))
ge = zeros(length(cdofs))

qr = QuadratureRule{RefQuadrilateral}(2)
fqr = FaceQuadratureRule{RefQuadrilateral}(2)
cv = CellValues(qr, ip)
fv = FaceValues(fqr, ip)



material = LinearElastic(; E=100.0, ν=0.3)
_, Cmat, _ = material_response(PlaneStrain(), material, rand(SymmetricTensor{2,2}), initial_material_state(material))

assem = start_assemble()
fcv = cv
for facedata in CellIterator(dh)#FaceIterator(dh, union(getfaceset(grid, "top"), getfaceset(grid, "bottom"), getfaceset(grid, "right"), getfaceset(grid, "left")))
    
    Ge = fill!(Ge, 0.0)
    ge = fill!(ge, 0.0)
    
    reinit!(fcv, facedata)
    
    e = basevec(Vec{2})
    for iqp in 1:getnquadpoints(fcv)
        dV = getdetJdV(fcv, iqp)
        u = Vec((0.0,0.0))#function_value(cv, iqp, ue)
        
        for d in cdofs
            ge[d] += e[d]⋅u * dV 
        end
        
        for i in 1:getnbasefunctions(fcv)
            Ni = shape_value(fcv, iqp, i)
            for d in cdofs
                Ge[i,d] += (e[d]⋅Ni) * dV
            end
        end
    end
    
    assemble!(assem, celldofs(facedata), cdofs, Ge)
    
end


function integrate_traction_force!(fe::AbstractVector, t, fv)
    n_basefuncs = getnbasefunctions(fv)
    
    for q_point in 1:getnquadpoints(fv)
        dA = getdetJdV(fv, q_point)
        for i in 1:n_basefuncs
            δu = shape_value(fv, q_point, i)
            fe[i] += t ⋅ δu * dA
        end
    end
end;



C = finish_assemble(assem)
g = zeros(eltype(C), size(C,2))

a =  Ferrite.lu(C')

ch = ConstraintHandler(dh)
#add!(ch, Dirichlet(:u, getfaceset(grid, "top"), x -> (0.0,), [2]))
add!(ch, Ferrite.MatrixConstrant(C,g))
close!(ch)

K = create_sparsity_pattern(dh)
K = create_sparsity_pattern(dh, ch)
f = zeros(ndofs(dh))

n = ndofs_per_cell(dh)
dofs = zeros(Int, n)
ke = zeros(n, n)
fe = zeros(n)

assembler = start_assemble(K, f)
for cellid in 1:length(grid.cells)
    fill!(ke, 0.0)
    coords = getcoordinates(grid, cellid)
    reinit!(cv, coords)
    celldofs!(dofs, dh, cellid)

    integrate_element!(ke, Cmat, cv)

    assemble!(assembler, dofs, ke)
end

t = Vec((1.0, 0.0))
for faceid in getfaceset(grid,"bottom")
    fill!(fe, 0.0)
    cellid, lfaceid = faceid

    coords = getcoordinates(grid, cellid)
    celldofs!(dofs, dh, cellid)
    reinit!(fv, coords, lfaceid)

    integrate_traction_force!(fe, t, fv)
    f[dofs] += fe
end

apply!(K, f, ch)
a = K\f
apply!(a, ch)

vtk_grid("matrixc", grid) do vtk
    vtk_point_data(vtk, dh, a)
end

