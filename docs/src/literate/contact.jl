using Ferrite
using FerriteGmsh
using ForwardDiff

include("contact_search.jl")

struct NeoHooke
    μ::Float64
    λ::Float64
end

function Ψ(C, mp::NeoHooke)
    μ = mp.μ
    λ = mp.λ
    Ic = tr(C)
    J = sqrt(det(C))
    return μ / 2 * (Ic - 3) - μ * log(J) + λ / 2 * log(J)^2
end

function constitutive_driver(C, mp::NeoHooke)
    # Compute all derivatives in one function call
    ∂²Ψ∂C², ∂Ψ∂C = Tensors.hessian(y -> Ψ(y, mp), C, :all)
    S = 2.0 * ∂Ψ∂C
    ∂S∂C = 2.0 * ∂²Ψ∂C²
    return S, ∂S∂C
end;

function construct_grid()

    # Initialize gmsh
    gmsh.initialize()
    gmsh.clear()
    gmsh.initialize()
    gmsh.model.add("model")

    gmsh.option.set_number("General.Verbosity", 1)

    body1 = gmsh.model.occ.add_rectangle(0.0, 0.0, 0.0, 2.0, 2.0)
    body2 = gmsh.model.occ.add_rectangle(0.1, 2.01, 0.0, 2.0, 2.0)
    
    _, body1tags = gmsh.model.occ.getCurveLoops(body1)
    _, body2tags = gmsh.model.occ.getCurveLoops(body2)

 #   @show body1tags
#    @show body2tags

    # Synchronize the model
    gmsh.model.occ.synchronize()

    # Create the physical domains
    gmsh.model.add_physical_group(1, body1tags[1], -1, "contactsurfaces")
    #gmsh.model.add_physical_group(1, body2tags[1], -1, "contactnodes")
    #gmsh.model.add_physical_group(1, [body1tags[1][3]], -1, "contactsurfaces")
    gmsh.model.add_physical_group(1, [body2tags[1][1]], -1, "contactnodes")
    gmsh.model.add_physical_group(1, [body2tags[1][3]], -1, "body2_top")
    gmsh.model.add_physical_group(1, [body1tags[1][1]], -1, "body1_bottom")
    gmsh.model.add_physical_group(2, [body1, body2])

    meshsize = 0.5
    ov = gmsh.model.getEntities(-1);
    gmsh.model.mesh.setSize(ov, meshsize);

    # Generate a 2D mesh
    gmsh.model.mesh.generate(2)

    # Save the mesh, and read back in as a Ferrite Grid
    grid = mktempdir() do dir
        path = joinpath(dir, "mesh.msh")
        gmsh.write(path)
        togrid(path)
    end

    #grid = togrid()
    #gmsh.fltk.run()

    # Finalize the Gmsh library
    gmsh.finalize()

    return grid
end

function collect_dofs!(facedofs::Vector, dh, faceid::FaceIndex)

    cellid, lfaceidx = faceid
    nfields = length(dh.field_names)

    local_face_dofs = Int[]
    for ifield in 1:nfields
        field_name = dh.field_names[ifield]
        offset = Ferrite.field_offset(dh, field_name)
        field_dim = Ferrite.getfielddim(dh, field_name)
        field_ip = dh.field_interpolations[ifield]
        face = Ferrite.faces(field_ip)[lfaceidx]
        
        for fdof in face, d in 1:field_dim
            push!(local_face_dofs, (fdof-1)*field_dim + d + offset)
        end
    end
    
    _celldofs = zeros(Int, ndofs_per_cell(dh, cellid))
    celldofs!(_celldofs, dh, cellid)
    
    for i in 1:length(local_face_dofs)
        j = dh.cell_dofs_offset[cellid] + local_face_dofs[i] - 1
        facedofs[i] = dh.cell_dofs[j]
    end

   # return _celldofs[local_face_dofs] 
    
end

struct NodeId
    id::Int
end

function mygetcoordinates!(coords, grid::Grid, faceid::FaceIndex)

    cellid, lfaceid = faceid
    cell = getcells(grid, cellid)

    ip = Ferrite.default_interpolation(typeof(cell))

    for (j,i) in enumerate(Ferrite.faces(ip)[lfaceid])
        coords[j] = grid.nodes[cell.nodes[i]].x
    end

end

function collect_dofs!(nodedofs::Vector, dh, faceid::NodeId)

    cellid, lfaceidx = faceid
    nfields = length(dh.field_names)

    local_face_dofs = Int[]
    for ifield in 1:nfields
        field_name = dh.field_names[ifield]
        offset = Ferrite.field_offset(dh, field_name)
        field_dim = Ferrite.getfielddim(dh, field_name)
        field_ip = dh.field_interpolations[ifield]
        face = Ferrite.faces(field_ip)[lfaceidx]
        
        for fdof in face, d in 1:field_dim
            push!(local_face_dofs, (fdof-1)*field_dim + d + offset)
        end
    end
    
    _celldofs = zeros(Int, ndofs_per_cell(dh, cellid))
    celldofs!(_celldofs, dh, cellid)
    
    for i in 1:length(local_face_dofs)
        j = dh.cell_dofs_offset[cellid] + local_face_dofs[i] - 1
        facedofs[i] = dh.cell_dofs[j]
    end

   @assert facedofs == _celldofs[local_face_dofs] 
    
end

function ndofs_per_face(dh, faceid = FaceIndex(1,1))

    cellid, lfaceidx = faceid
    nfields = length(dh.field_names)

    local_face_dofs = Int[]
    for ifield in 1:nfields
        field_name = dh.field_names[ifield]
        offset = Ferrite.field_offset(dh, field_name)
        field_dim = Ferrite.getfielddim(dh, field_name)
        field_ip = dh.field_interpolations[ifield]
        face = Ferrite.faces(field_ip)[lfaceidx]
        
        for fdof in face, d in 1:field_dim
            push!(local_face_dofs, (fdof-1)*field_dim + d + offset)
        end
    end


    return length(local_face_dofs)
end

function gapfunction(node::ContactNode{sdim}, surface, ae::Vector{T}, ξ0) where {sdim,T}

    r1 = (1:length(node.dofs))
    r2 = (1:length(surface.dofs)) .+ length(node.dofs)

    us = Vec{sdim,T}(i -> ae[r1[i]])
    xs = node.X + us

    disps = reinterpret(Vec{sdim,T}, view(ae, r2))
    coords = surface.X .+ disps

   # d2ddξ2, dddξ, _ = hessian(ξ -> distance2(xs, surface.ip, coords, ξ), ξ0, :all)
    #dξ = -d2ddξ2\dddξ
    ξ = ξ0#+ dξ

    xm = function_value(surface.ip, ξ, coords)
    surface_tangents = function_derivatives(surface.ip, ξ, coords)

    n = compute_normal(surface_tangents...)

    g = (xs - xm)⋅n

    return g
end

function assemble_contact_constraints!(K, f, dh, contact_data, a, ρ)

    assembler = start_assemble()
    fc = zero(f)
    for data in contact_data
        dofs = [data.node.dofs..., data.surface.dofs...]
        ae = a[dofs]
        gap_f(ae) = gapfunction(data.node, data.surface, ae, data.ξ)
        g = gap_f(ae)
        ∂g = ForwardDiff.gradient(gap_f, ae)
        ∂∂g = ForwardDiff.hessian(gap_f, ae)
        Ke = ρ*∂g*∂g' + ρ*g*∂∂g
        fe = ρ*g*∂g
        fc[dofs] += fe
        assemble!(assembler, dofs, Ke)
    end

    assemble!(assembler, [ndofs(dh)], zeros(Float64,1,1))
    Kc = end_assemble(assembler)
    return Kc, fc
end

function assemble_contact_constraints2!(K, f, dh, contact_data::Vector{ContactData}, a, ρ)

    assembler = start_assemble()
    fc = zero(f)
    for data in contact_data
        n = data.n
        fe[data.node.dofs] .= n
        for i in 1:ndofs_node
            fe .= -data.shape_value[i]*n
            for j in 1:ndofs_n

            end
            for j in 1:ndofs_segment

            end
        end

        for i in 1:ndofs_segment
            fe[i] .= -data.shape_value[i] ⋅ n
        end

        nξ(ξ) = normal(ip, coords, ae, ξ)
        na(ae) = normal(ip, coords, ae, data.ξ)

        Ke = ρ*∂g*∂g' + ρ*g*∂∂g
        fe = ρ*g*∂g
        fc[dofs] += fe
        assemble!(assembler, dofs, Ke)
    end

    assemble!(assembler, [ndofs(dh)], zeros(Float64,1,1))
    Kc = end_assemble(assembler)
    return Kc, fc
end

function integrate_cell!(ke, fe, cv, coords, material, ue)
    
    # Reinitialize cell values, and reset output arrays
    reinit!(cv, coords)
    ndofs = getnbasefunctions(cv)

    for qp in 1:getnquadpoints(cv)
        dΩ = getdetJdV(cv, qp)
        # Compute deformation gradient F and right Cauchy-Green tensor C
        ∇u = function_gradient(cv, qp, ue)
        F = one(∇u) + ∇u
        C = tdot(F)
        # Compute stress and tangent
        S, ∂S∂C = constitutive_driver(C, material)
        P = F ⋅ S
        I = one(S)
        ∂P∂F =  otimesu(I, S) + 2 * otimesu(F, I) ⊡ ∂S∂C ⊡ otimesu(F', I)

        # Loop over test functions
        for i in 1:ndofs
            # Test function and gradient
            δui = shape_value(cv, qp, i)
            ∇δui = shape_gradient(cv, qp, i)
            # Add contribution to the residual from this test function
            fe[i] += ( ∇δui ⊡ P ) * dΩ

            ∇δui∂P∂F = ∇δui ⊡ ∂P∂F # Hoisted computation
            for j in 1:ndofs
                ∇δuj = shape_gradient(cv, qp, j)
                # Add contribution to the tangent
                ke[i, j] += ( ∇δui∂P∂F ⊡ ∇δuj ) * dΩ
            end
        end
    end
end

function normal(ip, coords, ae, ξ)
    surface_tangents = function_derivatives(ip, ξ, coords)
    n = contact_normal(surface_tangents)
    n /= norm(n)
    return n
end

function assemble_cells!(K, f, dh, cv, material, a)
    n = ndofs_per_cell(dh)
    ke = zeros(n, n)
    fe = zeros(n)

    # start_assemble resets K and f
    assembler = start_assemble(K, f)

    # Loop over all cells in the grid
    for cell in CellIterator(dh)
        fill!(ke, 0.0)
        fill!(fe, 0.0)

        global_dofs = celldofs(cell)
        coords = getcoordinates(cell)
        ue = a[global_dofs] # element dofs
        integrate_cell!(ke, fe, cv, coords, material, ue)
        assemble!(assembler, global_dofs, fe, ke)
    end
end

function go()

    dim = 2
    grid = construct_grid()
    celltype = getcelltype(grid)
    
    addcellset!(grid, "contactsurfaces", first.(getfaceset(grid, "contactsurfaces")))
    addcellset!(grid, "contactnodes", first.(getfaceset(grid, "contactnodes")))

    dh = DofHandler(grid)
    push!(dh, :u, dim)
    close!(dh)

    ch = ConstraintHandler(dh)
    dcb = Dirichlet(:u, getfaceset(grid, "body2_top"), (x, t) -> (0.0, -1.0*t), 1:dim)
    add!(ch, dcb)
    dcb = Dirichlet(:u, getfaceset(grid, "body1_bottom"), (x, t) -> (0.0, 0.0), 1:dim)
    add!(ch, dcb)
    close!(ch)
    update!(ch, 0.0)

    ip = Ferrite.default_interpolation(celltype)
    faceip = Ferrite.getlowerdim(ip)
    qr = QuadratureRule{dim,RefTetrahedron}(2) # default_quadraturerule(ip)

    cv = CellVectorValues(qr, ip)
    N = getnbasefunctions(faceip)*dim

    contactnodes = ContactNode{dim}[]
    contactsurfaces = ContactSurface{dim,typeof(faceip),N,N÷2}[]

    dofs = zeros(Int, ndofs_per_face(dh))
    coords = zeros(Vec{dim,Float64}, getnbasefunctions(faceip))
    for faceid in getfaceset(grid, "contactsurfaces")
        collect_dofs!(dofs, dh, faceid)
        mygetcoordinates!(coords, grid, faceid)
        cs = ContactSurface(faceip, Tuple(dofs), Tuple(coords))
        push!(contactsurfaces, cs)
    end
    
    for faceid in getfaceset(grid, "contactnodes")
        collect_dofs!(dofs, dh, faceid)
        
        for i in 1:getnbasefunctions(faceip)
            r = (1:dim) .+ (i-1)*dim
            mygetcoordinates!(coords, grid, faceid)
            cs = ContactNode(Tuple(dofs[r]), coords[i])
            push!(contactnodes, cs)
        end
    end
    unique!(contactnodes)

    contact = Node2SurfaceContact(contactnodes, contactsurfaces)

    ρ = 1e3
    E = 10.0
    ν = 0.3
    μ = E / (2(1 + ν))
    λ = (E * ν) / ((1 + ν) * (1 - 2ν))
    material = NeoHooke(μ, λ)

    K = create_sparsity_pattern(dh)
    f = zeros(Float64, ndofs(dh))
    a = zeros(Float64, ndofs(dh))
    δa = zeros(Float64, ndofs(dh))

    contact_data = search_contact!(contact, a)
    active_set = collect_active_set(contact_data)

    previous_active_set = copy(active_set)
    ntimesteps = 4
    t = 0
    Δt = 0.02
    TOL = 1e-10
    for istep in 0:ntimesteps
        @show istep
        t = Δt*istep
        update!(ch, t)
        apply!(a, ch)

        nnewton_itr = 0
        while true #Newton iterations
            fill!(K, 0.0)
            fill!(f, 0.0)

            nnewton_itr += 1

            assemble_cells!(K, f, dh, cv, material, a)
            Kc, fc = assemble_contact_constraints!(K, f, dh, contact_data, a, ρ)
            
            r = f - fc
            J = K - Kc

            apply_zero!(J, r, ch)
            δa .= -( J\r )
            apply_zero!(δa, ch)
            a += δa

            @show norm(r)
            if norm(r) < TOL
                contact_data = search_contact!(contact, a)
                active_set = collect_active_set(contact_data)
                @show active_set
                if active_set == previous_active_set
                    previous_active_set = active_set
                    break
                else
                    previous_active_set = active_set
                    @info "new active set"
                    continue
                end
            end

        end


    end

    vtk_grid("whatsupp", grid) do vtk
        vtk_point_data(vtk, dh, a)
        vtk_cellset(vtk, grid)
    end

    return contact_data
end

go()
