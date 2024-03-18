function push_seq!(buffer, cube, seq)
    n = length(seq)
    seq = vcat(seq, rand(FaceTurn, ENTRY_NUM_ACTIONS - 1))      # Pad the sequence with random moves
    for i in 1:n
        entry = TrainingDataEntry(cube, n - i + 1, Tuple(seq[i+j-1] for j in 1:ENTRY_NUM_ACTIONS))
        push!(buffer, entry)
        cube *= seq[i]
    end
    @assert isone(cube)     # Sanity check
    return n
end

