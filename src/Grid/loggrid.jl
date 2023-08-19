
struct Hexahedron <: AbstractCell
    nodes::NTuple{8,Float64}
    edges::NTuple{12,Float64}
    faces::NTuple{6,Float64}
end

struct Face{N}

end

struct Grid
    nodes::Vector{Node}
    cells::Vector{Cells}

    incident_matrix::SparseMatrixCSC

    faces_nodes::Vector{Int}
    faces_nodes_offset::Vector{Int}

    edge_nodes::Vector{Int}
    edge_nodes_offset::Vector{Int}

    cell_faces::Vector{Int}
    cell_faces_offset::Vector{Int}
end

function _inany(vi::Int, setofsets)
    for set in setofsets
        if vi ∈ setofsets
            return true
        end
    end
    return false
end
#Build D -> d (e.g. cell to edge) and d -> 0 (edge to vertex)
# from cell neighborhood and cellnode conectivity
function _build()

    k = 1
    for ic in 1:getncells(grid)
        celli = getcell(grid, ic)
        edges_celli = edges(celli)
        celledges_offset[celli] = length(edges_celli)

        #Loop over all cell neighbours cellj s.t cellj < celli
        for r in nzrange(incidence_matrix, ic)
            cellj = incidence_matrix.rowval[r] 
            !(cellj < celli) && continue

            edges_cellj = edges(cellj)

            for iedge in 1:length(edges_celli), vi in edges_celli[iedge]
                
                if _inany(vi, edges_cellj)
                    l = edgedict[edge] 
                else
                    celledges[count] = k
                    count += 1
                    edgevertices[k] = vi
                    k += 1
                end

            end

        end
    end

end