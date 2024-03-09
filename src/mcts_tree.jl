# Represent the overall search tree
struct MCTSTree{M}
    root::MCTSNode
    model::M
    settings::MCTSSettings
end

MCTSTree(root::MCTSNode, model; kwargs...) = MCTSTree(root, model, MCTSSettings(; kwargs...))

function MCTSTree(cube::Cube, model; kwargs...)
    dist, policy = model(cube)
    root = MCTSNode(cube, 0, dist, policy)
    return MCTSTree(root, model; kwargs...)
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

# Descend to the next node
function descend(tree::MCTSTree, node::MCTSNode, add_noise::Bool=false)
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

# Make 1 playout: descend until reached bottom, then create a new node
function step!(tree::MCTSTree; max_depth::Int=1000, root_noise::Bool=true)
    node = tree.root
    actions = FaceTurn[]

    for _ in 0:max_depth
        if is_terminating(node)
            # Identity found. Terminate the playout by increase its self-playouts
            node.self_playouts += 1
            return node, actions
        end

        # We only add noise at the root node
        is_root = (tree.root == node)
        next, action = descend(tree, node, is_root && root_noise)
        push!(actions, action)

        if isnothing(next)
            # Reached the bottom of the tree. Create a new node
            new_pos = node.position * action
            dist, policy = tree.model(new_pos)
            child = MCTSNode(new_pos, node, dist, policy)
            node[action] = child
            return child, actions
        else
            # Existing node, continue descending
            node = next
        end
    end

    error("max depth reached")
end

# Get the best action (most number of playouts overall)
function best_action(tree::MCTSTree)
    _, action = findmax(playouts, tree[].children)
    return action
end

# Pretty-print
Base.show(io::IO, tree::MCTSTree) = print(io, typeof(tree), " with $(playouts(tree)) playouts")

