const MAX_MOVES = 30

# Model to evaluate each Cube for distance and policy
struct Model
    inner
    device
end

Flux.@functor Model

# Create a default model
function Model(n_blocks::Integer=4, n_channels::Integer=64; device=cpu)
    trunk = Chain((residual_block(n_channels => n_channels) for _ in 1:n_blocks)...)
    inner = Chain(
        RubikConv(N_INPUT_CHANNELS => n_channels),
        trunk,
        Parallel(tuple,
            make_dist_head(n_channels),
            make_policy_head(n_channels)
        )
    ) |> device
    return Model(inner, device)
end

# Components
residual_block((in, out)::Pair, conv1::Bool=false) = Parallel(+,
    conv1 ? Conv((1,), in => out) : identity,
    Chain(
        BatchNorm(in),
        relu,
        RubikConv(in => out),
        BatchNorm(out),
        relu,
        RubikConv(out => out)
    )
)

output_head((in, out)::Pair, hidden) = Chain(
    BatchNorm(in),
    relu,
    Conv((1,), in => hidden),
    BatchNorm(hidden),
    relu,
    Conv((1,), hidden => out),
    GlobalMeanPool(),
    Flux.flatten
)

make_dist_head(in, hidden=96) = output_head(in => MAX_MOVES, hidden)

make_policy_head(in, hidden=96) = output_head(in => 2N_FACETURNS, hidden)

trunk(model::Model) = model.inner[2]
num_blocks(model::Model) = length(trunk(model))
num_channels(model::Model) = size(first(model.inner).weight)[end]

# Evaluation routine
function (model::Model)(x)
    d, p = model.inner(x)
    p1 = view(p, 1:N_FACETURNS, :)
    p2 = view(p, N_FACETURNS+1:2N_FACETURNS, :)
    return d, p1, p2
end

function evaluate(model::Model, cubes::AbstractVector{Cube})
    # Extract the features
    x = features(cubes) |> model.device
    # Run inference
    d, p, _ = model(x) |> cpu
    # Postprocess
    dist = (1:MAX_MOVES)' * softmax(d)     # Average distance in 1:MAX_MOVES
    policy = softmax(p)
    # Override the case where cube is identity
    for (i, c) in enumerate(cubes)
        if isone(c)
            dist[i] = 0
        end
    end
    return dist, policy
end

function evaluate(model::Model, cube::Cube)
    d, p = evaluate(model, [cube])
    return d[1], reshape(p, :)
end

Flux.cpu(model::Model) = Model(model.inner |> cpu, cpu)
Flux.gpu(model::Model) = Model(model.inner |> gpu, gpu)

Base.copy(model::Model) = deepcopy(model)

