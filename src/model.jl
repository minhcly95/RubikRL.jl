const MAX_MOVES = 30

# Model to evaluate each Cube for distance and policy
struct Model
    inner
    device
end

# Create a default model
function Model(n_blocks::Integer=4, n_channels::Integer=64; device=cpu)
    trunk = Chain((residual_block(n_channels => n_channels) for _ in 1:n_blocks)...)
    inner = Chain(
        RubikConv(N_INPUT_CHANNELS => n_channels),
        trunk,
        Parallel(tuple,
            make_dist_head(n_channels),
            make_policy_head(n_channels)
        )
    ) |> device
    return Model(inner, device)
end

# Components
residual_block((in, out)::Pair, conv1::Bool=false) = Parallel(+,
    conv1 ? Conv((1,), in => out) : identity,
    Chain(
        BatchNorm(in),
        relu,
        RubikConv(in => out),
        BatchNorm(out),
        relu,
        RubikConv(out => out)
    )
)

make_dist_head(in) = Chain(
    residual_block(in => MAX_MOVES, true),
    GlobalMeanPool(),
    Flux.flatten
)

make_policy_head(in) = Chain(
    residual_block(in => N_FACETURNS, true),
    GlobalMeanPool(),
    Flux.flatten
)

# Evaluation routine
function (model::Model)(cubes::AbstractVector{Cube})
    # Extract the features
    x = features(cubes) |> model.device
    # Run inference
    d, p = model.inner(x) |> cpu
    # Postprocess
    dist = (1:MAX_MOVES)' * softmax(d)     # Average distance in 1:MAX_MOVES
    policy = softmax(p)
    # Override the case where cube is identity
    for (i, c) in enumerate(cubes)
        if isone(c)
            dist[i] = 0
        end
    end
    return dist, policy
end

function (model::Model)(cube::Cube)
    d, p = model([cube])
    return d[1], reshape(p, :)
end

# Try to solve a position to identity (batch version)
function solve(model::Model, cubes::AbstractVector{Cube}, settings::Settings=Settings(); max_distance::Integer=MAX_MOVES)
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
            step!(curr_trees)
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
function solve_rate(model::Model, cubes::AbstractVector{Cube}, settings::Settings=Settings(); max_distance::Integer=MAX_MOVES)
    solved = count(!isnothing, solve(model, cubes, settings; max_distance))
    return solved / length(cubes)
end

