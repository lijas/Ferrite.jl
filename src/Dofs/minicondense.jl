
grid = generate_grid(Quadrilateral, (2,2))
dh = DofHandler(grid)
add!(dh, :u, 1)
close!(dh)

celldofs1 = celldofs(dh, 1)
celldofs2 = celldofs(dh, 2)
totaldofs = vcat(celldofs1, celldofs2)

I = Int[]
J = Int[]
for i in eachindex(totaldofs)
    for j in eachindex(totaldofs)
        dofi = totaldofs[i]
        dofj = totaldofs[j]
        push!(I, dofi)
        push!(J, dofj)
    end
end

K = create_sparsity_pattern(dh)#sparse(ones(ndofs(dh),ndofs(dh)))
fill!(K.nzval, 1.0)

_ndofs = ndofs(dh)
K2 = Ferrite.spzeros!!(Float64, I, J, _ndofs, _ndofs)
fill!(K2.nzval, 1)

K .+= K2
fill!(K.nzval, 5.0)


@show K[totaldofs,totaldofs]
a = start_assemble(K)
n = length(totaldofs)
ke = ones(n,n)
assemble!(a, totaldofs, ke)
#minike, minidofs = mini_condense(ke,totaldofs)

ld = length(totaldofs)
sorteddofs = Int[]
resize!(sorteddofs, ld)
copyto!(sorteddofs, totaldofs)
permutation = similar(sorteddofs)
Ferrite.sortperm2!(a.sorteddofs, permutation)


function mini_condense(ke, totaldofs::Vector{Int})

    mapp = Dict{Int,Int}()
    minidofs = Int[]
    cnt = 0
    for i in eachindex(totaldofs)
        get!(mapp, totaldofs[i]) do 
            push!(minidofs, totaldofs[i])
            cnt+=1
        end
    end

    minike = zeros(Float64, cnt, cnt)
    for i in eachindex(totaldofs)
        ii = mapp[totaldofs[i]]
        for j in eachindex(totaldofs)
            jj = mapp[totaldofs[j]]
            minike[ii,jj] += ke[i,j]
        end
    end

    return minike, minidofs
end