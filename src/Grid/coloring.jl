# Greedy algorithm for coloring a grid such that no two cells with the same node
# have the same color
function create_coloring(g::Grid, cellset::Vector{Int} = collect(1:getncells(g)))
    # Contains the elements that each node contain
    cell_containing_node = Dict{Int, Set{Int}}()
    for cellid in cellset
        cell = g.cells[cellid]
        for v in cell.nodes
            if !haskey(cell_containing_node, v)
                cell_containing_node[v] = Set{Int}()
            end
            push!(cell_containing_node[v], cellid)
        end
    end

    I, J, V = Int[], Int[], Bool[]
    for (node, cells) in cell_containing_node
        for cell1 in cells # All these cells have a neighboring node
            for cell2 in cells
                if cell1 != cell2
                    push!(I, cell1)
                    push!(J, cell2)
                    push!(V, true)
                end
            end
        end
    end

    incidence_matrix = sparse(I, J, V)
    # cell -> color of cell
    cell_colors = Dict{Int, Int}()
    # color -> list of cells
    final_colors = Vector{Int}[]
    occupied_colors = Set{Int}()
    # Zero represents no color set yet
    for cellid in cellset#1:length(g.cells)
        cell_colors[cellid] = 0
    end
    total_colors = 0
    for (ic, cellid) in enumerate(cellset)#1:length(g.cells)
        empty!(occupied_colors)
        # loop over neighbors
        for r in nzrange(incidence_matrix, ic)#cellid)
            cell_neighbour = incidence_matrix.rowval[r]
            color = cell_colors[cell_neighbour]
            if color != 0
                push!(occupied_colors, color)
            end
        end

        # occupied colors now contains all the colors we are not allowed to use
        free_color = 0
        for attempt_color in 1:total_colors
            if attempt_color ∉ occupied_colors
                free_color = attempt_color
                break
            end
        end

        if free_color == 0 # no free color found, need to bump max colors
            total_colors += 1
            free_color = total_colors
            push!(final_colors, Int[])
        end

        cell_colors[cellid] = free_color
        push!(final_colors[free_color], cellid)
    end

    return cell_colors, final_colors
end

function vtk_cell_data_colors(vtkfile, grid, cell_colors)
    color_vector = zeros(getncells(grid))
    for (i, cells_color) in enumerate(cell_colors)
        for cell in cells_color
            color_vector[cell] = i
        end
    end
    vtk_cell_data(vtkfile, color_vector, "coloring")
end
