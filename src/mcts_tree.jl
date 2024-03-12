# Represent the overall search tree
struct MCTSTree{M}
    root::MCTSNode
    model::M
    settings::Settings
end

function MCTSTree(cube::Cube, model, settings::Settings = Settings())
    dist, policy = model(cube)
    root = MCTSNode(cube, 0, dist, policy)
    return MCTSTree(root, model, settings)
end

# Construct a tree (or subtree) using another tree as template
MCTSTree(root::MCTSNode, template::MCTSTree) = MCTSTree(root, template.model, template.settings)

# Indexing
Base.getindex(tree::MCTSTree) = tree.root
Base.getindex(tree::MCTSTree, action) = tree[][action]

# Get real distance (distance of node + depth of node)
distance(tree::MCTSTree, node::MCTSNode) = node.distance + node.depth - tree.root.depth

# Utility map: transform distance of position -> node utility
# Positions further away from identity (high distance) -> lower utility
utility_map(dist) = -dist

# Utility statistics
playouts(tree::MCTSTree) = playouts(tree.root)

function sum_utility_and_playouts(tree::MCTSTree, node::MCTSNode)
    self_util = utility_map(distance(tree, node))

    sum_util = self_util * node.self_playouts
    playouts = node.self_playouts

    for c in children(node)
        s, p = sum_utility_and_playouts(tree, c)
        sum_util += s
        playouts += p
    end
    return sum_util, playouts
end

function avg_utility_and_playouts(tree::MCTSTree, node::MCTSNode)
    s, p = sum_utility_and_playouts(tree, node)
    return s / p, p
end

# Get the next node 1 level down
function next_node(tree::MCTSTree, node::MCTSNode, add_noise::Bool=false)
    default_utility = avg_utility_and_playouts(tree, node)[1]
    all_utility = fill(default_utility, N_FACETURNS)
    all_playouts = fill(0, N_FACETURNS)

    for (i, c) in enumerate(node.children)
        if !isnothing(c)
            all_utility[i], all_playouts[i] = avg_utility_and_playouts(tree, c)
        end
    end

    # Add optional Dirichlet noise
    policy = node.policy
    if add_noise
        noise = rand(Dirichlet(N_FACETURNS, tree.settings.policy_noise_param))
        noise_weight = tree.settings.policy_noise_weight
        policy = (1 - noise_weight) * policy + noise_weight * noise
    end

    puct = all_utility .+ tree.settings.puct_weight .* policy .* sqrt(sum(all_playouts)) ./ (1 .+ all_playouts)

    _, action = findmax(puct)
    return node[action], FaceTurn(action)
end

# Descend to the bottom, return the leaf node and the sequence of actions
function descend(tree::MCTSTree; max_depth::Int=1000, root_noise::Bool=true)
    node = tree.root
    actions = FaceTurn[]

    for _ in 0:max_depth
        if is_terminating(node)
            # Identity found
            return node, actions
        end

        # We only add noise at the root node
        is_root = (tree.root == node)
        next, action = next_node(tree, node, is_root && root_noise)
        push!(actions, action)

        if isnothing(next)
            # Reached the bottom of the tree
            return node, actions
        else
            # Existing node, continue descending
            node = next
        end
    end

    error("max depth reached")
end

# Make 1 playout: descend until reached bottom, then create a new node
# We do it in batch to be more efficient
function step!(trees::AbstractVector{<:MCTSTree}; model=trees[1].model, kwargs...)
    # Get all cubes that need to be evaluated
    evals = []
    for tree in trees
        node, actions = descend(tree; kwargs...)

        if is_terminating(node)
            # Identity found. Terminate the playout by increase its self-playouts
            node.self_playouts += 1
        else
            # Reached the bottom of the tree. Add the next position to the evals list
            last_act = actions[end]
            new_pos = node.position * last_act
            push!(evals, (new_pos, node, last_act))
        end
    end
    
    isempty(evals) && return

    # Evaluate in batch
    dist, policy = model(first.(evals))

    # Add the children to the parents
    for (i, (new_pos, node, last_act)) in enumerate(evals)
        child = MCTSNode(new_pos, node, dist[i], policy[:,i])
        node[last_act] = child
    end
end

step!(tree::MCTSTree; kwargs...) = step!([tree]; kwargs...)

# Get the best action (most number of playouts overall)
function best_action(tree::MCTSTree)
    _, action = findmax(playouts, tree[].children)
    return FaceTurn(action)
end

# Pretty-print
Base.show(io::IO, tree::MCTSTree) = print(io, typeof(tree), " with $(playouts(tree)) playouts")

