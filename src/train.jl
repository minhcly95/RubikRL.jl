function train!(model::Model, buffer::TrainingBuffer, settings::Settings)
    device = model.device
    steps_per_epoch = settings.steps_per_epoch
    loss_distance_weight = settings.loss_distance_weight
    loss_policy_weight = settings.loss_policy_weight
    loss_next_policy_weight = settings.loss_next_policy_weight

    opt_chain = OptimiserChain(
        WeightDecay(settings.weight_decay),
        Momentum(settings.learning_rate, settings.momentum_decay)
    )
    opt_state = Flux.setup(opt_chain, model)

    solve_rate, better_rate = 0.0, 0.0
    data_generated = 0

    trainmode!(model)

    for epoch in 1:settings.num_epochs
        @info "EPOCH $epoch started" complexity = buffer.complexity

        loss, loss_d, loss_p1, loss_p2 = 0.0, 0.0, 0.0, 0.0

        prog = Progress(steps_per_epoch; showspeed=true)
        time = @elapsed for step in 1:steps_per_epoch
            # Populate once every several steps
            if step % settings.steps_per_populate == 1
                testmode!(model)
                num_new_entries, solve_rate, better_rate = populate!(buffer, settings)
                data_generated += num_new_entries
                trainmode!(model)
            end

            # Preprocess data
            data = rand(buffer, settings.batch_size)
            symm = settings.augment_symm ?      # Augment data with random symmetry
                   rand(Symm, settings.batch_size) :
                   fill(Symm(), settings.batch_size)

            cubes = [s' * e.position * s for (e, s) in zip(data, symm)]
            distances = [e.distance for e in data]
            next_actions = [s(e.actions[1]) for (e, s) in zip(data, symm)]
            after_next_actions = [s(e.actions[2]) for (e, s) in zip(data, symm)]

            x = features(cubes) |> device
            d = Flux.onehotbatch(distances, 1:MAX_MOVES) |> device
            p1 = Flux.onehotbatch(next_actions, ALL_FACETURNS) |> device
            p2 = Flux.onehotbatch(after_next_actions, ALL_FACETURNS) |> device

            # Calculate gradients and update
            local l_d, l_p1, l_p2
            l, gs = Flux.withgradient(model) do m
                dd, pp1, pp2 = m(x)
                l_d = loss_distance_weight * Flux.logitcrossentropy(dd, d)
                l_p1 = loss_policy_weight * Flux.logitcrossentropy(pp1, p1)
                l_p2 = loss_next_policy_weight * Flux.logitcrossentropy(pp2, p2)
                return l_d + l_p1 + l_p2
            end

            Flux.update!(opt_state, model, gs[1])
            loss += l
            loss_d += l_d
            loss_p1 += l_p1
            loss_p2 += l_p2

            # Progress meter
            next!(prog, showvalues=[
                (:data_generated, data_generated),
                (:solve_rate, solve_rate),
                (:better_rate, better_rate),
                (:(loss, d, p1, p2), round.((loss, loss_d, loss_p1, loss_p2) ./ step, digits=3))
            ])
        end

        finish!(prog)

        loss /= steps_per_epoch
        loss_d /= steps_per_epoch
        loss_p1 /= steps_per_epoch
        loss_p2 /= steps_per_epoch

        @info "EPOCH $epoch ended" loss loss_d loss_p1 loss_p2 time

        # Advance test
        testmode!(model)
        try_advance!(buffer, settings)
        trainmode!(model)
    end
end

train!(model::Model, buffer::TrainingBuffer; kwargs...) = train!(model, buffer, Settings(; kwargs...))
