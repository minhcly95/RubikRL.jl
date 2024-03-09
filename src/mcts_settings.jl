@with_kw mutable struct MCTSSettings
    puct_weight::Float64 = 5.0
    policy_noise_weight::Float64 = 0.25
    policy_noise_param::Float64 = 0.03
end

