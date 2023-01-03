using Ferrite
using FerriteGmsh
using ForwardDiff

include("contact_search.jl")

function construct_grid()

    # Initialize gmsh
    gmsh.initialize()
    gmsh.clear()
    gmsh.initialize()
    gmsh.model.add("model")

    #gmsh.option.set_number("General.Verbosity", 2)

    body1 = gmsh.model.occ.add_rectangle(0.0, 0.0, 0.0, 2.0, 2.0)
    body2 = gmsh.model.occ.add_rectangle(0.1, 1.99, 0.0, 2.0, 2.0)
    
    _, body1tags = gmsh.model.occ.getCurveLoops(body1)
    _, body2tags = gmsh.model.occ.getCurveLoops(body2)

    # Synchronize the model
    gmsh.model.occ.synchronize()

    # Create the physical domains
    gmsh.model.add_physical_group(1, body1tags[1], -1, "contactsurfaces")
    gmsh.model.add_physical_group(1, body2tags[1], -1, "contactnodes")
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

    #d2ddξ2, dddξ, _ = hessian(ξ -> distance2(xs, surface.ip, coords, ξ), ξ0, :all)
    #dξ = -d2ddξ2\dddξ
    ξ = ξ0# + dξ

    xm = function_value(surface.ip, ξ, coords)
    surface_tangents = function_derivatives(surface.ip, ξ, coords)

    n = compute_normal(surface_tangents...)

    g = (xs - xm)⋅n

    return g
end

function apply_contact_constraint(contactdata, a)

    for data in contactdata
        ae = a[[data.node.dofs..., data.surface.dofs...]]
        f(ae) = gapfunction(data.node, data.surface, ae, data.ξ)
        ∂g = ForwardDiff.gradient(f, ae)
        #∂∂g = ForwardDiff.hessian(f, ae)
        @show ∂g# ∂∂g
    end

end

function go()

    dim = 2
    grid = construct_grid()

    addcellset!(grid, "contactsurfaces", first.(getfaceset(grid, "contactsurfaces")))
    addcellset!(grid, "contactnodes", first.(getfaceset(grid, "contactnodes")))

    dh = DofHandler(grid)
    push!(dh, :u, dim)
    close!(dh)

    ip = Lagrange{dim,RefCube,1}()
    faceip = Ferrite.getlowerdim(ip)
    N = getnbasefunctions(faceip)*dim

    contactnodes = ContactNode{dim}[]
    contactsurfaces = ContactSurface{dim,typeof(faceip),N,N÷2}[]

    dofs = zeros(Int, ndofs_per_face(dh))
    coords = zeros(Vec{dim,Float64}, getnbasefunctions(faceip))
    for faceid in getfaceset(grid, "contactsurfaces")
        collect_dofs!(dofs, dh, faceid)
        mygetcoordinates!(coords, grid, faceid)
        @show coords
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
    x = zeros(Float64, ndofs(dh))
    update_contact!(contact, x)
    contact_data = search1!(contact, x)
    
    apply_contact_constraint(contact_data, x)

    
    vtk_grid("whatsupp", grid) do vtk
        vtk_cellset(vtk, grid)
    end

    return contact_data
end

go()