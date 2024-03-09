# A node in the search tree
mutable struct MCTSNode
    position::Cube          # Current position
    depth::Int              # The depth of this node
    self_playouts::Int      # The number of *terminated* at this node
    distance::Float64       # Estimated distance to solved
    policy::Vector{Float64} # Probability to choose each move
    children::Vector{Union{Nothing,MCTSNode}}
end

# Other constructors
function MCTSNode(position::Cube, depth::Int, distance, policy)
    return MCTSNode(position, depth, 1, distance, policy, fill(nothing, N_FACETURNS))
end

function MCTSNode(position::Cube, parent::MCTSNode, distance, policy)
    return MCTSNode(position, parent.depth + 1, distance, policy)
end

# Terminating node = identity cube
is_terminating(node::MCTSNode) = isone(node.position)

# Indexing
Base.getindex(node::MCTSNode, action) = node.children[convert(Int, action)]
Base.haskey(node::MCTSNode, action) = !isnothing(node[action])
Base.setindex!(node::MCTSNode, child::MCTSNode, action) = (node.children[convert(Int, action)] = child)

# Get children
children(node) = Iterators.filter(!isnothing, node.children)

# Statistics
function playouts(node::MCTSNode)
    p = node.self_playouts
    for c in node.children
        p += playouts(c)
    end
    return p
end
playouts(::Nothing) = 0

# Pretty-print
function Base.show(io::IO, node::MCTSNode)
    n_children = count(!isnothing, node.children)
    n_playouts = playouts(node)
    print(io, typeof(node), " with {$n_children children, $n_playouts playouts}")
end

