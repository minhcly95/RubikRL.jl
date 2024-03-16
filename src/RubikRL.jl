module RubikRL

using RubikCore
using Flux
using Distributions, DataStructures, Parameters, Random, ProgressMeter

import RubikCore: N_FACETURNS, ALL_FACETURNS

include("settings.jl")
export Settings

include("features.jl")
export features

include("rubik_conv.jl")
export RubikConv

include("mcts_node.jl")
export MCTSNode, children, playouts

include("mcts_tree.jl")
export MCTSTree, descend, step!, best_action

include("model.jl")
export Model, evaluate

include("extend.jl")
export extend

include("solve.jl")
export solve, solve_rate

include("training_data.jl")
export TrainingBuffer, populate!, try_advance!

include("train.jl")
export train!

end
