# Extend to larger network
function extend(model::Model, n_blocks::Integer, n_channels::Integer)
    b, c = num_blocks(model), num_channels(model)
    @assert b <= n_blocks && c <= n_channels
    @debug "Extending ($b, $c) => ($n_blocks, $n_channels)"

    new_model = Model(n_blocks, n_channels, device = model.device)

    # Copy parameters from the old model to the new model
    _extend!(new_model.inner, model.inner)

    return new_model
end

# Extend recursively
_extend!(new_elem, old_elem) = nothing  # Fallback = doing nothing

function _extend!(new_chain::Chain, old_chain::Chain)
    for (new, old) in zip(new_chain, old_chain)
        _extend!(new, old)
    end
end

function _extend!(new_par::Parallel, old_par::Parallel)
    for (new, old) in zip(new_par.layers, old_par.layers)
        _extend!(new, old)
    end
end

function _extend!(new_bn::BatchNorm, old_bn::BatchNorm)
    _extend_vec!(new_bn.β, old_bn.β)
    _extend_vec!(new_bn.γ, old_bn.γ)
    _extend_vec!(new_bn.μ, old_bn.μ)
    _extend_vec!(new_bn.σ², old_bn.σ²)
end
_extend_vec!(new_vec, old_vec) = copy!(view(new_vec, 1:length(old_vec)), old_vec)

function _extend!(new_rc::RubikConv, old_rc::RubikConv)
    _, in, out = size(old_rc.weight)
    copy!(view(new_rc.weight, :, 1:in, 1:out), old_rc.weight)
end

function _extend!(new_conv::Conv, old_conv::Conv)
    _, in, out = size(old_conv.weight)
    copy!(view(new_conv.weight, :, 1:in, 1:out), old_conv.weight)
    _extend_vec!(new_conv.bias, old_conv.bias)
end

