function train!(model::Model, buffer::TrainingBuffer, settings::Settings)
    device = model.device
    loss_distance_weight = settings.loss_distance_weight
    loss_policy_weight = settings.loss_policy_weight

    opt_chain = OptimiserChain(
        WeightDecay(settings.weight_decay),
        Momentum(settings.learning_rate, settings.momentum_decay)
    )
    opt_state = Flux.setup(opt_chain, model.inner)
    solve_rate, better_rate = 0.0, 0.0

    trainmode!(model.inner)

    for epoch in 1:settings.num_epochs
        @info "EPOCH $epoch started" complexity = buffer.complexity

        loss = 0.0

        prog = Progress(settings.steps_per_epoch; showspeed=true)
        time = @elapsed for step in 1:settings.steps_per_epoch
            # Populate once every several steps
            if step % settings.steps_per_populate == 1
                testmode!(model.inner)
                solve_rate, better_rate = populate!(buffer, settings)
                trainmode!(model.inner)
            end

            # Preprocess data
            data = rand(buffer, settings.batch_size)
            symm = rand(Symm, settings.batch_size)      # Augment by symmetry

            cubes = [s' * e.position * s for (e, s) in zip(data, symm)]
            distances = [e.distance for e in data]
            actions = [s(e.action) for (e, s) in zip(data, symm)]

            x = features(cubes) |> device
            d = Flux.onehotbatch(distances, 1:MAX_MOVES) |> device
            p = Flux.onehotbatch(actions, ALL_FACETURNS) |> device

            # Calculate gradients and update
            l, gs = Flux.withgradient(model.inner) do m
                d̂, p̂ = m(x)
                l_d = loss_distance_weight * Flux.logitcrossentropy(d̂, d)
                l_p = loss_policy_weight * Flux.logitcrossentropy(p̂, p)
                return l_d + l_p
            end

            Flux.update!(opt_state, model.inner, gs[1])
            loss += l / settings.steps_per_epoch

            next!(prog, showvalues=[
                (:buffer_length, length(buffer)),
                (:solve_rate, solve_rate),
                (:better_rate, better_rate)
            ])
        end

        finish!(prog)

        @info "EPOCH $epoch ended" loss time

        testmode!(model.inner)
        try_advance!(buffer, settings)
        trainmode!(model.inner)
    end
end

train!(model::Model, buffer::TrainingBuffer; kwargs...) = train!(model, buffer, Settings(; kwargs...))
