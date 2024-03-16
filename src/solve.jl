# Try to solve a position to identity (batch version)
function solve(model::Model, cubes::AbstractVector{Cube}, settings::Settings=Settings(); max_distance::Integer=MAX_MOVES, no_exploration::Bool=false)
    num_playouts = settings.num_playouts

    all_seqs = [FaceTurn[] for _ in cubes]

    curr_trees = [MCTSTree(c, model, settings) for c in cubes]
    curr_seqs = copy(all_seqs)

    for d in 1:max_distance+1
        # Filter out all terminated tree
        terminated = findall(t -> is_terminating(t[]), curr_trees)
        deleteat!(curr_trees, terminated)
        deleteat!(curr_seqs, terminated)

        # Nothing to continue, break
        isempty(curr_trees) && break

        # Max distance reached, break
        if d == max_distance + 1
            # Push a dummy move to distinguish unsolved cases to solved cases
            for seq in curr_seqs
                push!(seq, FaceTurn(1))
            end
            break
        end

        # Step the trees (in batch)
        for _ in 1:num_playouts
            step!(curr_trees, root_noise=!no_exploration)
        end

        # Push new moves to the sequences
        for i in eachindex(curr_trees)
            tree = curr_trees[i]
            # Get the best actions
            best = best_action(tree)
            # Add to the sequence
            push!(curr_seqs[i], best)
            # Replace current tree with the subtree
            curr_trees[i] = MCTSTree(tree[][best], tree)
        end
    end

    # Filter out all unsolved cases
    return [length(seq) > max_distance ? nothing : seq for seq in all_seqs]
end

solve(model::Model, cube::Cube, settings::Settings=Settings()) = solve(model, [cube], settings)[1]

# Evaluate the strength of a model based on the number of positions it can solve
function solve_rate(model::Model, cubes::AbstractVector{Cube}, settings::Settings=Settings(); max_distance::Integer=MAX_MOVES, no_exploration::Bool=false)
    solved = count(!isnothing, solve(model, cubes, settings; max_distance, no_exploration))
    return solved / length(cubes)
end

