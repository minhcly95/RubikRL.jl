mutable struct InlineModelDataSource <: AbstractDataSource
    buffer::CircularBuffer{TrainingDataEntry}
    complexity::Int
    model::Model
    settings::Settings
end

InlineModelDataSource(model::Model, settings::Settings=Settings()) = InlineModelDataSource(
    CircularBuffer{TrainingDataEntry}(settings.buffer_capacity),
    settings.complexity_start,
    copy(model),
    settings
)

# Statistics
Base.length(ds::InlineModelDataSource) = length(ds.buffer)

# Sample from buffer
Base.rand(rng::AbstractRNG, ds::Random.SamplerTrivial{InlineModelDataSource}) = rand(rng, ds[].buffer)

# Populate the buffer (in batch)
function populate!(ds::InlineModelDataSource)
    settings = ds.settings
    populate_size = settings.populate_size

    testmode!(ds.model)

    # Create generating sequences of moves of length = complexity
    gen_seqs = rand(CanonSequence(ds.complexity), populate_size)
    # Create cubes from generating sequences
    cubes = Cube.(gen_seqs)
    # Solve the cubes in batch (max distance = complexity - 1 since we only want better sequences)
    seqs = solve(ds.model, cubes, settings, max_distance=ds.complexity - 1)

    solved = count(!isnothing, seqs)
    num_new_entries = 0

    for (c, gseq, seq) in zip(cubes, gen_seqs, seqs)
        # Skip trivial cases
        isone(c) && continue

        # Choose the shorter sequence between gseq' and seq
        fseq = isnothing(seq) ? gseq' : seq

        # Add new entry for each position in the sequence
        num_new_entries += push_seq!(ds.buffer, c, gseq')
        if settings.augment_inv
            num_new_entries += push_seq!(ds.buffer, c', gseq)
        end
    end

    # Return the proportion of positions that the model can solve better
    return num_new_entries, (;
        better_rate=solved / populate_size
    )
end

# Post-epoch routine: 
# Advance test: test the solve rate against the next complexity.
# If the model can solve more than success_rate, increase the complexity.
function post_epoch!(ds::InlineModelDataSource, new_model::Model)
    settings = ds.settings
    sample_size = settings.test_size
    success_rate = settings.test_success_rate
    complexity = ds.complexity
    old_model = ds.model

    testmode!(old_model)
    testmode!(new_model)

    # Create cubes from sequences of moves of length = complexity
    cubes = Cube.(rand(CanonSequence(ds.complexity), sample_size))

    # Test the solve rate (without exploration features)
    time = @elapsed begin
        old_rate = solve_rate(old_model, cubes, settings; no_exploration=true)
        new_rate = solve_rate(new_model, cubes, settings; no_exploration=true)
    end

    # Replace old_model with new_model if new_rate >= old_rate
    if new_rate >= old_rate
        ds.model = copy(new_model)
        @info "Replace old model" old_rate new_rate complexity time
    else
        @info "Sustain old model" old_rate new_rate complexity time
    end

    # Increase the complexity if rate >= success_rate
    rate = max(old_rate, new_rate)
    if rate >= success_rate
        @info "Test PASSED"
        ds.complexity += settings.complexity_step
    end
end

# Pretty-print
Base.show(io::IO, ds::InlineModelDataSource) = print(io, typeof(ds), " with {length = $(length(ds)), capacity = $(ds.buffer.capacity), complexity = $(ds.complexity)}")


