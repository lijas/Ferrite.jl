using Ferrite
using FerriteGmsh
using MaterialModels

grid = togrid("docs/src/literate-tutorials/plate_hole.geo")

grid = Grid(
    Ferrite.AbstractCell[grid.cells...],
    grid.nodes,
    facetsets = grid.facetsets,
    cellsets = grid.cellsets
)
rigidbody_node = Node(Vec((5.0, 5.0)))
rigidbody_cellid = getncells(grid) + 1
push!(grid.nodes, rigidbody_node)
push!(grid.cells, Ferrite.Point((getnnodes(grid),)))

addcellset!(grid, "rigidbody", [getncells(grid)])
addvertexset!(grid, "rigidvertex", x -> x ≈ rigidbody_node.x)

ip_u = Lagrange{RefTriangle, 1}()^2
ip_rb_u = Lagrange{Ferrite.RefPoint, 1}()^2
ip_rb_θ = Lagrange{Ferrite.RefPoint, 1}()

qr = QuadratureRule{RefTriangle}(2)
cellvalues = CellValues(qr, ip_u)

fqr = FacetQuadratureRule{RefTriangle}(2)
facetvalues = FacetValues(fqr, ip_u)

dh = DofHandler(grid)
sdh = SubDofHandler(dh, getcellset(grid, "PlateWithHole"))
add!(sdh, :u, ip_u)

sdh = SubDofHandler(dh, getcellset(grid, "rigidbody"))
add!(sdh, :u, ip_rb_u)
add!(sdh, :θ, ip_rb_θ)

npoints = length(getfacetset(grid, "HoleInterior")) 
Ferrite.add_global_dofs!(dh, :λ, npoints*2)
close!(dh)

ch = ConstraintHandler(dh)
#=rb = Ferrite.RigidConnector(;
    rigidbody_cellid = rigidbody_cellid,
    facets = getfacetset(grid, "HoleInterior"),
)
add!(ch, rb)=#
#add!(ch, Dirichlet(:u, getfacetset(grid, "PlateRight"), (x,t) -> (-0.01*t*0, 0.0)))
#add!(ch, Dirichlet(:u, getfacetset(grid, "PlateLeft"), x -> (0.01*t*0, 0.0)))
add!(ch, Dirichlet(:u, getvertexset(grid, "rigidvertex"), x -> (0.0, 0.0)))
add!(ch, Dirichlet(:θ, getvertexset(grid, "rigidvertex"), x -> (pi/20*t)))
close!(ch)

E = 200.0e3 # Young's modulus [MPa]
ν = 0.3        # Poisson's ratio [-]
material = NeoHook(μ=E/(2(1+ν)), λ=E*ν/((1+ν)*(1-2ν)))

function assemble_cell!(ke, cellvalues, C)
    for q_point in 1:getnquadpoints(cellvalues)
        ## Get the integration weight for the quadrature point
        dΩ = getdetJdV(cellvalues, q_point)
        for i in 1:getnbasefunctions(cellvalues)
            ## Gradient of the test function
            ∇Nᵢ = shape_gradient(cellvalues, q_point, i)
            for j in 1:getnbasefunctions(cellvalues)
                ## Symmetric gradient of the trial function
                ∇ˢʸᵐNⱼ = shape_symmetric_gradient(cellvalues, q_point, j)
                ke[i, j] += (∇Nᵢ ⊡ C ⊡ ∇ˢʸᵐNⱼ) * dΩ
            end
        end
    end
    return ke
end

function assemble_cell_largedef!(ke, re, cv, material, ue)
    ndofs = getnbasefunctions(cv)
    for qp in 1:getnquadpoints(cv)
        dΩ = getdetJdV(cv, qp)
        # Compute deformation gradient F and right Cauchy-Green tensor C
        ∇u = function_gradient(cv, qp, ue)
        F = one(∇u) + ∇u
        C = tdot(F) # F' ⋅ F
        C3d = SymmetricTensor{2,3}((C[1,1], C[2,1],0.0,C[2,2], 0.0, 1.0))
        
        # Compute stress and tangent
        S3D, ∂S∂C3D, _ = material_response(material, C3d, MaterialModels.NeoHookState())
        S = MaterialModels.reduce_dim(S3D, PlaneStrain())
        ∂S∂C = MaterialModels.reduce_dim(∂S∂C3D, PlaneStrain())
        P = F ⋅ S
        I = one(S)
        ∂P∂F = otimesu(I, S) + 2 * F ⋅ ∂S∂C ⊡ otimesu(F', I)

        # Loop over test functions
        for i in 1:ndofs
            # Test function and gradient
            #δui = shape_value(cv, qp, i)
            ∇δui = shape_gradient(cv, qp, i)
            # Add contribution to the residual from this test function
            re[i] += (∇δui ⊡ P) * dΩ

            ∇δui∂P∂F = ∇δui ⊡ ∂P∂F # Hoisted computation
            for j in 1:ndofs
                ∇δuj = shape_gradient(cv, qp, j)
                # Add contribution to the tangent
                ke[i, j] += (∇δui∂P∂F ⊡ ∇δuj) * dΩ
            end
        end
    end
end

function assemble_global!(K, r, dh, cellvalues, a, material, cellset)
    ## Allocate the element stiffness matrix
    n_basefuncs = getnbasefunctions(cellvalues)
    ke = zeros(n_basefuncs, n_basefuncs)
    re = zeros(n_basefuncs)
    ## Create an assembler
    assembler = start_assemble(K, r)
    ## Loop over all cells
    for cell in CellIterator(dh, cellset)
        ## Update the shape function gradients based on the cell coordinates
        reinit!(cellvalues, cell)
        ## Reset the element stiffness matrix
        fill!(ke, 0.0)
        fill!(re, 0.0)
        ue = a[cell.dofs]
        ## Compute element contribution
        assemble_cell_largedef!(ke, re, cellvalues, material, ue)
        ## Assemble ke into K
        assemble!(assembler, celldofs(cell), ke, re)
    end
end

function assemble_constraint!(K, residual, dh, facetvalues, a, c)

    (; dh) = ch
    grid = Ferrite.get_grid(dh)

    xdof, ydof, θdof = rb_dofs = celldofs(dh, c.rigidbody_cellid)

    λdofs_all = Ferrite.global_dof_range(dh, :λ)

    Ferrite._check_same_celltype(grid, c.facets)
    @assert getcells(grid, c.rigidbody_cellid) isa Ferrite.Point
    R = getcoordinates(grid, c.rigidbody_cellid)[1]
    ur = Vec{2}((a[xdof], a[ydof]))
    θ = a[θdof]
    cellid = first(c.facets)[1]
    sdh_index = dh.cell_to_subdofhandler[cellid]
    sdh = dh.subdofhandlers[sdh_index]

    field_idx = Ferrite.find_field(sdh, :u) #Hardcoded
    CT = getcelltype(sdh)
    ip = Ferrite.getfieldinterpolation(sdh, field_idx)
    ip_geo = Ferrite.geometric_interpolation(CT)
    offset = Ferrite.field_offset(sdh, field_idx)
    n_comp = Ferrite.n_dbc_components(ip)

    local_facet_dofs, local_facet_dofs_offset =
        Ferrite._local_facet_dofs_for_bc(ip.ip, n_comp, 1:n_comp, offset)

    #Values for rigid body
    A = Tensor{2,2}((cos(θ), sin(θ), -sin(θ), cos(θ)))
    dA = Tensor{2,2}((-sin(θ), cos(θ), -cos(θ), -sin(θ)))
    ddA = Tensor{2,2}((-cos(θ), -sin(θ), sin(θ), -cos(θ)))
    #uperp = rotate(ū, pi / 2)
    r = R + ur

    fv = Ferrite.BCValues(ip.ip, ip_geo, FacetIndex)
    #fv = facetvalues
    cc = FacetCache(dh, UpdateFlags(; nodes = false, coords = true, dofs = true))
    npoints = 0
    dofvisited = Set{Int}()
    lambdacounter = 0
    for fid in c.facets
        (cellidx, entityidx) = fid
        reinit!(cc, fid)
        # no need to reinit!, enough to update current_entity since we only need geometric shape functions M
        #reinit!(fv, cc)
        fv.current_entity = entityidx

        # local dof-range for this facet
        erange = local_facet_dofs_offset[entityidx]:(local_facet_dofs_offset[entityidx + 1] - 1)
        #dofs[1] in dofsadded && continue
        ae = a[cc.dofs]

        c = 0
        for ipoint in 1:getnquadpoints(fv)
            femdofs = cc.dofs[local_facet_dofs[erange[(1:2) .+ 2*(ipoint-1)]]]

            if femdofs[1] in dofvisited
                continue
            else
                push!(dofvisited, femdofs[1])
            end

            λdofs = λdofs_all[(1:2) .+ lambdacounter]
            λ1, λ2 = λvals = a[λdofs]
            lambdacounter += 2

            npoints += 1
            X = spatial_coordinate(fv, ipoint, getcoordinates(cc))

            uvec = reinterpret(Vec{2,Float64}, ae)
            u = spatial_coordinate(fv, ipoint, uvec) #function_value(fv, ipoint, ae)
            
            # r + A*ū == x
            x = X + u
            ū = X - R
            
            I = [1 0; 0 1]

            #Values
            g = x - (r + A⋅ū)
            #Jacobian
            G = [I -I -(dA⋅ū)]
            #Hessian
            d²Audθ² = ddA⋅ū
            H1 = zeros(5,5); H1[5,5] = -d²Audθ²[1]
            H2 = zeros(5,5); H2[5,5] = -d²Audθ²[2] 
        
            Ke = H1*λ1 + H2*λ2

            dofs = [femdofs..., rb_dofs...]
            
            tmp = G'*λvals

            #assemble
            for (i, I) in pairs(dofs)
                residual[I] += tmp[i] #plus or minus
                for (j, J) in pairs(dofs)
                    K[I, J] += Ke[i, j]
                end
            end

            #assemble
            for (i, I) in pairs(λdofs)
                residual[I] += g[i]
                for (j, J) in pairs(dofs)
                    K[I, J] += G[i, j]
                    K[J, I] += G[i, j]
                end
            end

        end
    end
    return

end
#K = allocate_matrix(dh, ch)

sp = init_sparsity_pattern(dh)
add_cell_entries!(sp, dh)

for λdof in Ferrite.global_dof_range(dh, :λ)
    for faceindex in getfacetset(grid, "HoleInterior"), meshdof in celldofs(dh, faceindex[1])
        Ferrite.add_entry!(sp, λdof, meshdof)
    end
end
for rbdof in celldofs(dh, rigidbody_cellid)
    for faceindex in getfacetset(grid, "HoleInterior"), meshdof in celldofs(dh, faceindex[1])
        Ferrite.add_entry!(sp, rbdof, meshdof)
    end
end
add_constraint_entries!(sp, ch)
K = allocate_matrix(sp)

a = zeros(ndofs(dh))
r = zeros(ndofs(dh))
f_ext = zeros(Float64, ndofs(dh))

constraint_def = (;
    rigidbody_cellid = rigidbody_cellid,
    facets = getfacetset(grid, "HoleInterior"),
    displacement_field = :u,
    rotation_field = :θ,
    lagrange_field = :λ
)

t = 0.0
dt = 1.0
for i in 1:5
    @info "step $i"
    global t += dt
    update!(ch, t)
    apply!(a, ch)
    for inewton in 1:10
        fill!(K, 0.0)
        fill!(r, 0.0)
        assemble_global!(K, r, dh, cellvalues, a, material, getcellset(grid, "PlateWithHole"));
        assemble_constraint!(K, r, dh, facetvalues, a, constraint_def)
        apply_zero!(K, r, ch)
        @show norm(r)
        if norm(r) < 1e-8
            break
        end 
        da = -(K \ r);
        apply_zero!(da, ch)
        global a += da
    end
end


function calculate_stresses(grid, dh, cv, u, C, cellset)
    qp_stresses = [
        [zero(SymmetricTensor{2, 2}) for _ in 1:getnquadpoints(cv)]
            for _ in 1:getncells(grid)
    ]
    avg_cell_stresses = tuple((zeros(length(cellset)) for _ in 1:3)...)
    for cell in CellIterator(dh, cellset)
        reinit!(cv, cell)
        cell_stresses = qp_stresses[cellid(cell)]
        for q_point in 1:getnquadpoints(cv)
            ε = function_symmetric_gradient(cv, q_point, u, celldofs(cell))
            cell_stresses[q_point] = C ⊡ ε
        end
        σ_avg = sum(cell_stresses) / getnquadpoints(cv)
        avg_cell_stresses[1][cellid(cell)] = σ_avg[1, 1]
        avg_cell_stresses[2][cellid(cell)] = σ_avg[2, 2]
        avg_cell_stresses[3][cellid(cell)] = σ_avg[1, 2]
    end
    return qp_stresses, avg_cell_stresses
end

qp_stresses, avg_cell_stresses = calculate_stresses(grid, dh, cellvalues, a, C, getcellset(grid, "PlateWithHole"));

# We now use the the L2Projector to project the stress-field onto the piecewise linear
# finite element space that we used to solve the problem.
proj = L2Projector(grid)
add!(proj, getcellset(grid, "PlateWithHole"), ip_u; qr_rhs = qr)
close!(proj)

projected = project(proj, qp_stresses)

VTKGridFile("rigid_con", grid) do vtk
    write_solution(vtk, dh, a)
    write_projection(vtk, proj, projected, "stress")
end
