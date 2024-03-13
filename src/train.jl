function train!(model::Model, buffer::TrainingBuffer, settings::Settings)
    device = model.device
    loss_distance_weight = settings.loss_distance_weight
    loss_policy_weight = settings.loss_policy_weight

    opt_state = Flux.setup(Flux.Adam(settings.learning_rate), model.inner)

    ptime = @elapsed begin
        count = populate!(buffer, settings)
    end
    @info "Populate pretrain" count length = length(buffer) time = ptime

    for epoch in 1:settings.num_epochs
        @info "EPOCH $epoch started" complexity = buffer.complexity

        loss = 0.0

        prog = Progress(settings.steps_per_epoch; showspeed=true)
        time = @elapsed for step in 1:settings.steps_per_epoch
            GC.gc(false)

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

            # Populate after some steps
            if step % settings.steps_per_populate == 0
                ptime = @elapsed begin
                    count = populate!(buffer, settings)
                end
                @debug "Populate step $step" count length = length(buffer) time = ptime
            end

            next!(prog, showvalues = [(:buffer_length, length(buffer))])
        end

        finish!(prog)

        @info "EPOCH $epoch ended" loss time

        try_advance!(buffer, settings)
    end
end

train!(model::Model, buffer::TrainingBuffer; kwargs...) = train!(model, buffer, Settings(; kwargs...))
