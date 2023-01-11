#https://www.osti.gov/servlets/purl/10175733

struct ContactSurface{dim,I,N,N2}
    ip::I
    dofs::NTuple{N,Int}
    X::NTuple{N2,Vec{dim,Float64}}
end

function ContactSurface(ip::Interpolation{pdim}, dofs::NTuple{N,Int}, coords::NTuple{N2,Vec{sdim,Float64}}) where {pdim,sdim,N,N2}
    ContactSurface{sdim,typeof(ip),N,N2}(ip, dofs, coords)
end

struct ContactNode{dim}
    dofs::NTuple{dim,Int}
    X::Vec{dim,Float64}
end

struct AABB{dim,T}
    corner::Vec{dim,T}
    sidelength::Vec{dim,T}
end

getdim(::Type{ContactNode{dim}}) where dim = dim
getdim(::ContactNode{dim}) where dim = dim

function getAABB(a::ContactSurface{2,I,N,N2}, x::Vector{T}) where {I,T,N,N2} 
    maxx = maxy = T(-Inf)
    minx = miny = T(Inf)
    dim = 2
    for n in 1:(N2)
        r = (1:dim) .+ (n-1)*dim
        coord = a.X[n] + Vec{dim,Float64}( i -> x[a.dofs[r[i]]] )
        
        maxx = coord[1] > maxx ? coord[1] : maxx
        minx = coord[1] < minx ? coord[1] : minx

        maxy = coord[2] > maxy ? coord[2] : maxy
        miny = coord[2] < miny ? coord[2] : miny
    end

    return AABB(Vec{2,Float64}((minx,miny)), Vec{2,Float64}((maxx-minx, maxy-miny)))
end

struct ContactData{S,M,pdim,sdim}
    node::S
    surface::M
    a::NTuple{pdim,Vec{sdim,Float64}}
    n::Vec{sdim,Float64}
    g::Float64
    ξ::Vec{pdim,Float64}
    δξ
end

function collect_active_set(data::Vector{ContactData})
    set = Set{Int}()
    for d in data
        dof = first(d.node.dofs)
        push!(set, dof)
    end
    return set
end

mutable struct Node2SurfaceContact{dim,CS<:ContactSurface}
	
    segments::Vector{CS}
    nodes::Vector{ContactNode{dim}}

    bounding_boxes::Vector{AABB{dim,Float64}}

    nlbox::Vector{Int} #UNUSED, number of buckets a entity occupies
    node_bucketid::Vector{Int}  #bucket id for each slave node
    nnodes_in_bucket::Vector{Int} #number of entities in each bucket
    bucketid_to_nodeid::Vector{Int} #list of entities sorted by bucket id
    bucketid_to_nodeid_offset::Vector{Int} #pointer that identifies first entity in bucketid_to_nodeid

    nbuckets::Int
    bucket_size::Float64

    possible_contacts::Vector{Tuple{Int,Int}}
end

function Node2SurfaceContact(slave_nodes::Vector{CN}, master_segments::Vector{CS}) where {CN,CS}
    nmasters = length(master_segments)
    dim = getdim(CN)
    bounding_boxes = Vector{AABB{dim,Float64}}(undef, nmasters)

    return Node2SurfaceContact{dim,CS}(master_segments, slave_nodes, bounding_boxes,
                            zeros(Int, nmasters), Int[], Int[], Int[], Int[], 
                            0, 0.0,
                            Tuple{Int,Int}[])
end

function global_search!(contact::Node2SurfaceContact{2}, x::Vector{Float64})

    (; segments, bounding_boxes)    = contact
    (; node_bucketid, nnodes_in_bucket, bucketid_to_nodeid, bucketid_to_nodeid_offset, nbuckets, bucket_size) = contact

    dim = 2

    minx = Inf
    maxx = -Inf
    miny = Inf
    maxy = -Inf
    bucket_size = Inf
    #Get maximum and minimum coordinates
    for (i, master) in enumerate(contact.segments)
        aabb = getAABB(master, x)
        bounding_boxes[i] = aabb

        minx = min(minx, aabb.corner[1])
        miny = min(miny, aabb.corner[2])
        
        max_coords = aabb.corner + aabb.sidelength
        maxx = max(maxx, max_coords[1])
        maxy = max(maxy, max_coords[2])

        #bucketsize = is bases on the dimension of the smallest entity 
        max_dimension = max(aabb.sidelength[1], aabb.sidelength[2])
        bucket_size = min(bucket_size, max_dimension) 
    end

    nbucketsx = trunc(Int, (maxx-minx)/bucket_size)
    nbucketsy = trunc(Int, (maxy-miny)/bucket_size)
    nbuckets = nbucketsy*nbucketsx

    resize!(nnodes_in_bucket, nbuckets)
    fill!(nnodes_in_bucket,  0)
    resize!(node_bucketid, length(contact.nodes))
    fill!(node_bucketid, -1)
    resize!(bucketid_to_nodeid_offset, nbuckets)

    _bucket_id(x, min, max, n) = trunc(Int, n*(x - min)/(max-min)) + 1

    for (i, node) in enumerate(contact.nodes)

        node_coord = node.X + x[Vec(node.dofs)]
        ix = _bucket_id(node_coord[1], minx, maxx, nbucketsx)# trunc(Int, (node_coord[1] - minx)/bucket_size) + 1
        (ix > nbucketsx) && continue
        (ix < 1) && continue
        
        iy = _bucket_id(node_coord[2], miny, maxy, nbucketsy)#trunc(Int, (node_coord[2] - miny)/bucket_size) + 1
        (iy > nbucketsy) && continue
        (iy < 1) && continue
        ib = (iy-1)*nbucketsx + ix
        nnodes_in_bucket[ib] += 1
        node_bucketid[i] = ib
    end

    bucketid_to_nodeid_offset[1] = 1
    for j in 2:nbuckets
        bucketid_to_nodeid_offset[j] = bucketid_to_nodeid_offset[j-1] + nnodes_in_bucket[j-1]
    end
    resize!(bucketid_to_nodeid, sum(nnodes_in_bucket))

    fill!(nnodes_in_bucket, 0)
    for i in 1:length(contact.nodes)
        ib = node_bucketid[i] #boxid
        ib == -1 && continue
        bucketid_to_nodeid[nnodes_in_bucket[ib] + bucketid_to_nodeid_offset[ib]] = i
        nnodes_in_bucket[ib] += 1
    end

    contact.nbuckets = nbuckets

    empty!(contact.possible_contacts)
    for (i, master) in enumerate(segments)
        aabb = bounding_boxes[i]
        min_coords = aabb.corner
        max_coords = aabb.corner + aabb.sidelength

        ibox_min = _bucket_id(min_coords[1], minx, maxx, nbucketsx)# min(nbucketsx, trunc(Int, (min_coords[1]-minx)/bucket_size)+1)
        ibox_min = clamp(ibox_min, 1, nbucketsx)
        ibox_max = _bucket_id(max_coords[1], minx, maxx, nbucketsx)#min(nbucketsx, trunc(Int, (max_coords[1]-minx)/bucket_size)+1)
        ibox_max = clamp(ibox_max, 1, nbucketsx)

        jbox_min = _bucket_id(min_coords[2], miny, maxy, nbucketsy)
        jbox_min = clamp(jbox_min, 1, nbucketsy)
        jbox_max = _bucket_id(max_coords[2], miny, maxy, nbucketsy)
        jbox_max = clamp(jbox_max, 1, nbucketsy)

        for ix in ibox_min:ibox_max
            for iy in jbox_min:jbox_max
                ib = (iy-1)*nbucketsx + ix

                pointer = bucketid_to_nodeid_offset[ib]
                nnodes_in_bucket[ib] == 0 && continue

                for j in 1:nnodes_in_bucket[ib]
                    node_id = bucketid_to_nodeid[pointer + j-1]
                    #if i in node_segements[node_id]
                    #    continue
                    #end
                    push!(contact.possible_contacts, (i, node_id))
                end

            end
        end

    end
end


function global_search!(contact::Node2SurfaceContact{3,T}, x) where {T}

    (; masters, nodes, bounding_boxes, node_segements)     = contact
    (; node_bucketid, nnode_bucketid, nnodes_in_bucket, bucketid_to_nodeid, bucketid_to_nodeid_offset, nbuckets, bucket_size) = contact
    (; possible_contacts)              = contact

    dim = 3
    nmasters = length(contact.masters)

    minx = Inf
    maxx = -Inf
    miny = Inf
    maxy = -Inf
    minz = Inf
    maxz = -Inf
    bucket_size = Inf

    #Get maximum and minimum coordinates
    for (i, master) in enumerate(contact.masters)

        aabb = getAABB(master, x)
        bounding_boxes[i] = aabb
        minx = minx < aabb.cornerpos[1] ? minx : aabb.cornerpos[1]
        miny = miny < aabb.cornerpos[2] ? miny : aabb.cornerpos[2]
        minz = minz < aabb.cornerpos[3] ? minz : aabb.cornerpos[3]
        #minz = minz < aabb.cornerpos[3] ? minz : aabb.cornerpos[3]

        max_coords = aabb.cornerpos + aabb.sidelength
        maxx = maxx > max_coords[1] ? maxx : max_coords[1]
        maxy = maxy > max_coords[2] ? maxy : max_coords[2]
        maxz = maxz > max_coords[3] ? maxz : max_coords[3]
        #maxz = maxz > max_coords[3] ? maxz : max_coords[3]

        #bucketsize = is bases on the dimension of the smallest entity 
        max_dimension = max(aabb.sidelength...)
        bucket_size = min(bucket_size, max_dimension)
    end

    nbucketsx = trunc(Int, (maxx-minx)/bucket_size) + 1
    nbucketsy = trunc(Int, (maxy-miny)/bucket_size) + 1
    nbucketsz = trunc(Int, (maxz-minz)/bucket_size) + 1
    nbuckets = nbucketsy*nbucketsx*nbucketsz

    resize!(nnodes_in_bucket, nbuckets)
    fill!(nnodes_in_bucket,0)
    fill!(nnode_bucketid,0)
    resize!(node_bucketid, length(contact.nodes))
    
    resize!(bucketid_to_nodeid_offset, nbuckets)

    for (i, node) in enumerate(contact.nodes)

        node_coord = x[Vec(node.dofs)]

        ix = trunc(Int, (node_coord[1] - minx)/bucket_size) + 1
        iy = trunc(Int, (node_coord[2] - miny)/bucket_size) + 1
        iz = trunc(Int, (node_coord[3] - minz)/bucket_size) + 1
        ib = (iz-1)*nbucketsy*nbucketsx + (iy-1)*nbucketsx + ix
        nnodes_in_bucket[ib] +=1
        node_bucketid[i] = ib
    end
    bucketid_to_nodeid_offset[1] = 1
    for j in 2:nbuckets
        bucketid_to_nodeid_offset[j] = bucketid_to_nodeid_offset[j-1] + nnodes_in_bucket[j-1]
    end
    resize!(bucketid_to_nodeid, sum(nnodes_in_bucket)+1)

    fill!(nnodes_in_bucket, 0)
    for i in 1:length(contact.slavenodes)
        ib = node_bucketid[i] #boxid
        bucketid_to_nodeid[nnodes_in_bucket[ib] + bucketid_to_nodeid_offset[ib]] = i
        nnodes_in_bucket[ib] += 1
    end

    contact.nbuckets = nbuckets

    if maxx > 1000
        @show maxx, nbuckets
    end
    if minx < -1000
        @show minx, nbuckets
    end

    #ib = 1
    #@show bucketid_to_nodeid[ (0:nnodes_in_bucket[ib]) .+ bucketid_to_nodeid_offset[ib]]
    #masterids in bucket ib
    #master_ids = bucketid_to_nodeid[ (0:nnodes_in_bucket[ib]) .+ npoint[ib] ]
    empty!(contact.possible_contacts)
    for (i, master) in enumerate(masters)
        aabb = bounding_boxes[i]
        min_coords = aabb.cornerpos
        max_coords = aabb.cornerpos + aabb.sidelength

        ibox_min = min(nbucketsx, trunc(Int, (min_coords[1]-minx)/bucket_size)+1)
        ibox_max = min(nbucketsx, trunc(Int, (max_coords[1]-minx)/bucket_size)+1)

        jbox_min = min(nbucketsy, trunc(Int, (min_coords[2]-miny)/bucket_size)+1)
        jbox_max = min(nbucketsy, trunc(Int, (max_coords[2]-miny)/bucket_size)+1)

        kbox_min = min(nbucketsz, trunc(Int, (min_coords[3]-minz)/bucket_size)+1)
        kbox_max = min(nbucketsz, trunc(Int, (max_coords[3]-minz)/bucket_size)+1)

        for ix in ibox_min:ibox_max
            for iy in jbox_min:jbox_max
                for iz in kbox_min:kbox_max
                    ib = (iz-1)*nbucketsy*nbucketsx + (iy-1)*nbucketsx + ix

                    pointer = bucketid_to_nodeid_offset[ib]
                    nnodes_in_bucket[ib] == 0 ? continue : nothing #nothing to contact with

                    for j in 1:nnodes_in_bucket[ib]
                        node_id = bucketid_to_nodeid[pointer + j-1]
                        #if i in node_segements[node_id]
                        #    continue
                        #end
                        push!(contact.possible_contacts, (i, node_id))
                    end
                end

            end
        end
    end
end

function local_search!(contact::Node2SurfaceContact, x)
    contacts = ContactData[]
    ncontacts = 0
    nodes_in_contact = Dict{Int,Int}()
    for contact_pair in contact.possible_contacts
        segmentidx, nodeidx = contact_pair
        segment = contact.segments[segmentidx]
        node = contact.nodes[nodeidx]

        iscontact, co = local_contact_search(node, segment, x)
        if iscontact == true
            if haskey(nodes_in_contact, nodeidx)
                idx = nodes_in_contact[nodeidx]
                data = contacts[idx]
                if data.g < co.g #if new contact has smaller penatration
                    contacts[idx] = co
                end
            else
                ncontacts += 1
                nodes_in_contact[nodeidx] = ncontacts
                push!(contacts, co)
            end
        end
    end
    return contacts
end

function search_contact!(contact::Node2SurfaceContact, x)
    global_search!(contact, x)
    @show length(contact.possible_contacts)
    local_search!(contact, x)
end





function Ferrite.function_value(ip::Ferrite.Interpolation{pdim}, ξ::Vec{pdim}, coords::Vector{Vec{sdim,T}}) where {pdim,sdim,T}
    nb = getnbasefunctions(ip)
    @assert length(coords) == getnbasefunctions(ip)
    x = zero(Vec{sdim,T})
    for i in 1:nb
        N = Ferrite.value(ip, i, ξ)
        x += N*coords[i]
    end
    return x
end

function function_derivatives(ip::Ferrite.Interpolation{pdim}, ξ::Vec{pdim}, coords::Vector{Vec{sdim,T}}) where {pdim,sdim,T}
    nb = getnbasefunctions(ip)
    @assert length(coords) == getnbasefunctions(ip)
    a = zeros(Vec{sdim,T}, pdim)
    for i in 1:nb
        dN = Tensors.gradient( (ξ) -> Ferrite.value(ip, i, ξ), ξ)
        for j in 1:pdim
            a[j] += dN[j]*coords[i]
        end
    end
    return a
end

function gap_residual(ip::Interpolation, xˢ::Vec{2,Float64}, coords::Vector{T}, ξ::Vec{1,T2}) where {T,T2}
    nb = getnbasefunctions(ip)
    @assert length(coords) == getnbasefunctions(ip)
    xᵐ = zero(Vec{2,T})
    a1 = zero(Vec{2,T})
    for i in 1:nb
        dN, N = Tensors.gradient( (ξ) -> Ferrite.value(ip, i, ξ), ξ, :all)
        xᵐ += N*coords[i]
        a1 += dN[1]*coords[i]
    end

    n = Tensors.cross(a1)
    n /= norm(n)
    
    return (xˢ - xᵐ) ⋅ n, xᵐ, a1
end

function _asdf()
    nb = getnbasefunctions(ip)
    @assert length(coords) == getnbasefunctions(ip)
    a = zero(Tensor{2,pdim,T})
    e = basevecs(Vec{pdim,Float64})
    c = 0
    for i in 1:nb
        dN = Tensors.gradient( (ξ) -> Ferrite.value(ip, i, ξ), ξ)
        for j in 1:pdim
            c += 1
            a += (e[j] ⊗ dN[j])*u[c]
        end
    end
    return a
end

function min_dist(xs::Vec{sdim}, faceip::Interpolation, coords::Vector{Vec{sdim,T1}}, ξ::Vec{pdim,T2}) where {pdim,sdim,T1,T2}
    T = promote_type(T1, T2)

    xm = zero(Vec{sdim,T})
    dxdξ = zeros(Vec{sdim,T}, pdim)

    N = Ferrite.value(faceip, ξ)
    dN = Ferrite.derivative(faceip, ξ)
    for i in 1:getnbasefunctions(faceip)
        xm += N[i] * coords[i]
        for j in 1:pdim
            dxdξ[j] += dN[i][j] * coords[i]
        end
    end

    r = Vec{pdim,T}( i -> (xs - xm)⋅dxdξ[i])

    return r
end

function compute_normal(a1::Vec{3}, a2::Vec{3})
    n = Tensors.cross(a1,a2)
    n /= norm(n)
end

function compute_normal(a::Vec{2})
    n = Vec((a[2], -a[1]))
    n /= norm(n)
end

function local_contact_search(node::ContactNode{sdim}, surface::ContactSurface, a::Vector{Float64}) where sdim
    pdim = sdim-1

    TOL = 1e-10
    error = TOL + 1.0
    nitr = 0
    
    ndofs = length(surface.dofs) + length(node.dofs)

    #Node coord
    us = Vec{sdim,Float64}(i -> a[node.dofs[i]])
    xs = node.X + us

    #Surface coords
    ae = a[collect(surface.dofs)]
    disps = reinterpret(Vec{sdim,Float64}, ae)
    coords = surface.X .+ disps

    ξ = zero(Vec{sdim-1,Float64})
    ae = a[collect(surface.dofs)]
    local drdξ
    while error > TOL
        drdξ, r = gradient(ξ -> min_dist(xs, surface.ip, coords, ξ), ξ, :all)
        dξ = -drdξ\r
        ξ += dξ
        error = norm(r)
        nitr += 1
        if nitr ≥ 10
            break
        end
    end

    function Rξ(a::Vector{T}) where T # ξ is constant
        #Node
        _us = Vec{sdim,T}(i -> a[node.dofs[i]])
        _xs = node.X + _us
    
        #Surface 
        _ae = a[collect(surface.dofs)]
        _disps = reinterpret(Vec{sdim,T}, _ae)
        _coords = surface.X .+ _disps

        #Function
        return min_dist(_xs, surface.ip, _coords, ξ)
    end
    
    dRda = ForwardDiff.jacobian(Rξ, a)
    dξda = zeros(Vec{pdim,Float64}, ndofs)
    for i in 1:ndofs
        _drda = Vec{pdim,Float64}(j -> dRda[j,i])
        dξda[i] = -inv(drdξ)⋅_drda
    end

    xm = function_value(surface.ip, ξ, coords)
    surface_tangents = function_derivatives(surface.ip, ξ, coords)

    n = compute_normal(surface_tangents...)
    L = sqrt(norm(n)) #characheristic length?

    g = (xs - xm)⋅n

    data = ContactData(node, surface, Tuple(surface_tangents), n, g, ξ, dξda)

    iscontact = false
    if all( -1.0 .< ξ .< 1.0 )
        if g <= L/1000
            iscontact = true
        end
    end

    return iscontact, data
end

#=
xs = Vec{2,Float64}((0.5,0.01))
node = ContactNode((1,2), xs)

faceip = Lagrange{1,RefCube,1}()
Xm = (Vec{2,Float64}((0.0, 0.0)), Vec{2,Float64}((1.0, 0.0)))
surface = ContactSurface{2,typeof(faceip),4,2}(faceip, (2,3,4,5), Xm)

a = zeros(Float64, 6)
local_contact_search(node, surface, a)
=#