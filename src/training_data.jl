const ENTRY_NUM_ACTIONS = 2

struct TrainingDataEntry
    position::Cube
    distance::Int
    actions::NTuple{ENTRY_NUM_ACTIONS,FaceTurn}
end

mutable struct TrainingBuffer
    buffer::CircularBuffer{TrainingDataEntry}
    complexity::Int
    model::Model
end

TrainingBuffer(model::Model, settings::Settings=Settings()) = TrainingBuffer(
    CircularBuffer{TrainingDataEntry}(settings.buffer_capacity),
    settings.complexity_start,
    model
)

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
    seqs = solve(buffer.model, cubes, settings, max_distance=buffer.complexity + 5)

    solved_better = 0
    solved = count(!isnothing, seqs)
    num_new_entries = 0

    for (c, gseq, seq) in zip(cubes, gen_seqs, seqs)
        # Skip trivial cases
        isone(c) && continue
        # Choose the shorter sequence between gseq' and seq (prioritize gseq because it's new data)
        fseq = gseq'
        if !isnothing(seq) && (length(seq) < length(fseq))
            fseq = seq
            solved_better += 1
        end

        # Backup for later
        d = c'
        iseq = fseq'
        n = length(fseq)

        # Add new entry for each position in the sequence
        append!(fseq, rand(FaceTurn, ENTRY_NUM_ACTIONS - 1))  # Pad the sequence with random moves
        for i in 1:n
            entry = TrainingDataEntry(c, n - i + 1, Tuple(fseq[i+j-1] for j in 1:ENTRY_NUM_ACTIONS))
            push!(buffer.buffer, entry)
            c *= fseq[i]
            num_new_entries += 1
        end
        @assert isone(c)    # Sanity check

        # Also add the inverse position and sequence if enabled (data augmentation by inverse)
        if settings.augment_inv
            append!(iseq, rand(FaceTurn, ENTRY_NUM_ACTIONS - 1))  # Pad the sequence with random moves
            for i in 1:n
                entry = TrainingDataEntry(d, n - i + 1, Tuple(iseq[i+j-1] for j in 1:ENTRY_NUM_ACTIONS))
                push!(buffer.buffer, entry)
                d *= iseq[i]
                num_new_entries += 1
            end
            @assert isone(d)    # Sanity check
        end
    end

    # Return the proportion of positions that the model can solve better
    return num_new_entries, solved / populate_size, solved_better / populate_size
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
    # Test the solve rate (without exploration features)
    time = @elapsed begin
        rate = solve_rate(buffer.model, cubes, settings; max_distance=complexity + 5, no_exploration=true)
    end
    # Increase the complexity if rate >= 80%
    if rate >= success_rate
        @info "Advance test PASSED" rate complexity time
        buffer.complexity += settings.complexity_step
        return true
    else
        @info "Advance test failed" rate complexity time
        return false
    end
end

# Pretty-print
Base.show(io::IO, buffer::TrainingBuffer) = print(io, "TrainingBuffer with {length = $(length(buffer)), complexity = $(buffer.complexity)}")

