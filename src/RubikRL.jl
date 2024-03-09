module RubikRL

using RubikCore
using Flux
using Distributions, DataStructures, Parameters

import RubikCore: N_FACETURNS

include("features.jl")
export features

include("rubik_conv.jl")
export RubikConv

include("mcts_node.jl")
export MCTSNode, children, playouts

include("mcts_settings.jl")
include("mcts_tree.jl")
export MCTSTree, descend, step!, best_action

end
