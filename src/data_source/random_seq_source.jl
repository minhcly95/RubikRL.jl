mutable struct RandomSequenceDataSource <: AbstractDataSource
    buffer::CircularBuffer{TrainingDataEntry}
    complexity::Int
    settings::Settings
end

RandomSequenceDataSource(settings::Settings=Settings()) = RandomSequenceDataSource(
    CircularBuffer{TrainingDataEntry}(settings.buffer_capacity),
    settings.complexity_start,
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

# Post-epoch routine: 
# Advance test: test the solve rate against the next complexity.
# If the model can solve more than success_rate, increase the complexity.
function post_epoch!(ds::RandomSequenceDataSource, new_model::Model)
    settings = ds.settings
    sample_size = settings.test_size
    success_rate = settings.test_success_rate
    complexity = ds.complexity

    testmode!(new_model)

    # Create cubes from sequences of moves of length = complexity
    cubes = Cube.(rand(CanonSequence(ds.complexity), sample_size))

    # Test the solve rate (without exploration features)
    time = @elapsed begin
        rate = solve_rate(new_model, cubes, settings; no_exploration=true)
    end

    # Increase the complexity if rate >= success_rate
    if rate >= success_rate
        @info "Test PASSED" rate complexity time
        ds.complexity += settings.complexity_step
    else
        @info "Test failed" rate complexity time
    end
end

# Pretty-print
Base.show(io::IO, ds::RandomSequenceDataSource) = print(io, typeof(ds), " with {length = $(length(ds)), capacity = $(ds.buffer.capacity), complexity = $(ds.complexity)}")

