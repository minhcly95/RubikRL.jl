function train!(model::Model, buffer::TrainingBuffer, settings::Settings)
    device = model.device
    loss_distance_weight = settings.loss_distance_weight
    loss_policy_weight = settings.loss_policy_weight

    opt_chain = OptimiserChain(
        WeightDecay(settings.weight_decay),
        Momentum(settings.learning_rate, settings.momentum_decay)
    )
    opt_state = Flux.setup(opt_chain, model.inner)
    solved_better = 0.

    for epoch in 1:settings.num_epochs
        @info "EPOCH $epoch started" complexity = buffer.complexity

        loss = 0.0

        prog = Progress(settings.steps_per_epoch; showspeed=true)
        time = @elapsed for step in 1:settings.steps_per_epoch
            GC.gc(false)

            # Populate once every several steps
            if step % settings.steps_per_populate == 1
                solved_better = populate!(buffer, settings)
            end

            data = rand(buffer, settings.batch_size)
            x = features([entry.position for entry in data]) |> device
            d = Flux.onehotbatch([entry.distance for entry in data], 1:MAX_MOVES) |> device
            p = Flux.onehotbatch([entry.action for entry in data], ALL_FACETURNS) |> device

            l, gs = Flux.withgradient(model.inner) do m
                d̂, p̂ = m(x)
                l_d = loss_distance_weight * Flux.logitcrossentropy(d̂, d)
                l_p = loss_policy_weight * Flux.logitcrossentropy(p̂, p)
                return l_d + l_p
            end

            Flux.update!(opt_state, model.inner, gs[1])
            loss += l / settings.steps_per_epoch

            next!(prog, showvalues = [(:buffer_length, length(buffer)), (:solved_better, solved_better)])
        end

        finish!(prog)

        @info "EPOCH $epoch ended" loss time

        try_advance!(buffer, settings)
    end
end

train!(model::Model, buffer::TrainingBuffer; kwargs...) = train!(model, buffer, Settings(; kwargs...))
