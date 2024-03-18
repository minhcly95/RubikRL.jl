const ENTRY_NUM_ACTIONS = 2

struct TrainingDataEntry
    position::Cube
    distance::Int
    actions::NTuple{ENTRY_NUM_ACTIONS,FaceTurn}
end

abstract type AbstractDataSource end

# Populate the data source with new data
function populate!(::AbstractDataSource) end

# Routine to run after each epoch
# For example, the data source can replace the current data-generating model
# or increase the complexity of the generated data.
function post_epoch!(::AbstractDataSource, ::Model) end

