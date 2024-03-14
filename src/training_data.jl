struct TrainingDataEntry
    position::Cube
    distance::Int
    action::FaceTurn
end

mutable struct TrainingBuffer
    buffer::CircularBuffer{TrainingDataEntry}
    complexity::Int
    model::Model
end

TrainingBuffer(model::Model; max_capacity=1000000) = TrainingBuffer(CircularBuffer{TrainingDataEntry}(max_capacity), 2, model)

# Statistics
Base.length(buffer::TrainingBuffer) = length(buffer.buffer)

# Sample from buffer
Base.rand(rng::AbstractRNG, buffer::Random.SamplerTrivial{TrainingBuffer}) = rand(rng, buffer[].buffer)

# Populate the buffer (in batch)
function populate!(buffer::TrainingBuffer, populate_size::Integer, settings::Settings=Settings())
    # Create generating sequences of moves of length = complexity
    gen_seqs = [rand(FaceTurn, buffer.complexity) for _ in 1:populate_size]
    # Create cubes from generating sequences
    cubes = Cube.(gen_seqs)
    # Solve the cubes in batch (max distance = complexity + 5)
    seqs = solve(buffer.model, cubes, settings, max_distance = buffer.complexity+5)

    solved_better = 0
    for (c, gseq, seq) in zip(cubes, gen_seqs, seqs)
        # Skip trivial cases
        isone(c) && continue
        # Choose the shorter sequence between gseq' and seq (prioritize gseq because it's new data)
        fseq = gseq'
        if !isnothing(seq) && (length(seq) < length(fseq))
            fseq = seq
            solved_better += 1
        end
        # Add new entry for each position in the sequence
        for (i, s) in enumerate(fseq)
            entry = TrainingDataEntry(c, length(fseq) - i + 1, s)
            push!(buffer.buffer, entry)
            c *= s
        end
    end

    # Return the proportion of positions that the model can solve better
    return solved_better / populate_size
end

populate!(buffer::TrainingBuffer, settings::Settings=Settings()) = populate!(buffer, settings.populate_size, settings)

# Advance test: test the solve rate against the next complexity.
# If the model can solve more than 80%, increase the complexity.
function try_advance!(buffer::TrainingBuffer, settings::Settings=Settings())
    sample_size = settings.advance_test_size
    success_rate = settings.advance_test_success_rate
    complexity = buffer.complexity

    # Create cubes from sequences of moves of length = complexity
    cubes = [Cube(rand(FaceTurn, complexity)) for _ in 1:sample_size]
    # Test the solve rate
    time = @elapsed begin
        rate = solve_rate(buffer.model, cubes, settings; max_distance = complexity+5)
    end
    # Increase the complexity if rate >= 80%
    if rate >= success_rate
        @info "Advance test PASSED" rate complexity time
        buffer.complexity += 1
        return true
    else
        @info "Advance test failed" rate complexity time
        return false
    end
end

# Pretty-print
Base.show(io::IO, buffer::TrainingBuffer) = print(io, "TrainingBuffer with {length = $(length(buffer)), complexity = $(buffer.complexity)}")

