mutable struct RandomSequenceDataSource <: AbstractDataSource
    buffer::CircularBuffer{TrainingDataEntry}
    complexity::Int
    settings::Settings
end

RandomSequenceDataSource(settings::Settings=Settings()) = RandomSequenceDataSource(
    CircularBuffer{TrainingDataEntry}(settings.buffer_capacity),
    settings.complexity_randomseq,
    settings
)

# Statistics
Base.length(ds::RandomSequenceDataSource) = length(ds.buffer)

# Sample from buffer
Base.rand(rng::AbstractRNG, ds::Random.SamplerTrivial{RandomSequenceDataSource}) = rand(rng, ds[].buffer)

# Populate the buffer (in batch)
function populate!(ds::RandomSequenceDataSource)
    settings = ds.settings
    populate_size = settings.populate_size

    # Create generating sequences of moves of length = complexity
    gen_seqs = rand(CanonSequence(ds.complexity), populate_size)

    # Create cubes from generating sequences
    cubes = Cube.(gen_seqs)

    num_new_entries = 0

    for (c, gseq) in zip(cubes, gen_seqs)
        # Skip trivial cases
        isone(c) && continue

        # Add new entry for each position in the sequence
        num_new_entries += push_seq!(ds.buffer, c, gseq')
        if settings.augment_inv
            num_new_entries += push_seq!(ds.buffer, c', gseq)
        end
    end

    # Return the proportion of positions that the model can solve better
    return num_new_entries, (;)
end

# Post-epoch routine: test the solve rate against the each complexity in 1:20.
function post_epoch!(ds::RandomSequenceDataSource, new_model::Model)
    testmode!(new_model)

    # Create cubes from sequences of moves of length = complexity
    cubes = Cube[]
    for complexity in 1:20
        append!(cubes, Cube.(rand(CanonSequence(complexity), 10)))
    end

    # Solve the cubes (without exploration features)
    time = @elapsed begin
        seqs = solve(new_model, cubes, ds.settings; no_exploration=true)
    end

    # Count the number of solved cubes per complexity
    counts = zeros(Int, 20)
    for (complexity, cseqs) in zip(1:20, Iterators.partition(seqs, 10))
        counts[complexity] = count(!isnothing, cseqs)
    end

    # Report
    count_str = join(["$(counts[i])$(_subscript_string(i))" for i in 1:20], " ")
    @info "Post-epoch test" counts = count_str time
end

# Pretty-print
Base.show(io::IO, ds::RandomSequenceDataSource) = print(io, typeof(ds), " with {length = $(length(ds)), capacity = $(ds.buffer.capacity), complexity = $(ds.complexity)}")

