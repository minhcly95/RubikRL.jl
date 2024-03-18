function train!(model::Model, data_source::AbstractDataSource, settings::Settings)
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

    data_generated = 0
    aux_info = (;)

    for epoch in 1:settings.num_epochs
        @info "EPOCH $epoch started" complexity = data_source.complexity

        loss, loss_d, loss_p1, loss_p2 = 0.0, 0.0, 0.0, 0.0

        prog = Progress(steps_per_epoch; showspeed=true)
        time = @elapsed for step in 1:steps_per_epoch
            # Populate once every several steps
            if step % settings.steps_per_populate == 1
                num_new_entries, aux_info = populate!(data_source)
                data_generated += num_new_entries
            end

            trainmode!(model)
            GC.gc(false)

            # Preprocess data
            data = rand(data_source, settings.batch_size)

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
                (:(loss, d, p1, p2), round.((loss, loss_d, loss_p1, loss_p2) ./ step, digits=3)),
                (:data_generated, data_generated),
                collect(zip(keys(aux_info), aux_info))...,
            ])
        end

        finish!(prog)

        loss /= steps_per_epoch
        loss_d /= steps_per_epoch
        loss_p1 /= steps_per_epoch
        loss_p2 /= steps_per_epoch

        @info "EPOCH $epoch ended" loss loss_d loss_p1 loss_p2 time

        # Post-epoch routine
        post_epoch!(data_source, model)
    end

    testmode!(model)
end

train!(model::Model, buffer::InlineModelDataSource; kwargs...) = train!(model, buffer, Settings(; kwargs...))
