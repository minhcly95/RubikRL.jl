@with_kw mutable struct Settings
    # General parameters
    num_epochs::Int = 10
    batch_size::Int = 256
    steps_per_epoch::Int = 1000
    num_playouts::Int = 100
    # Learning
    learning_rate::Float64 = 0.01
    momentum_decay::Float64 = 0.9
    weight_decay::Float64 = 0.001
    # Advance test
    advance_test_size::Int = 200
    advance_test_success_rate::Float64 = 0.8
    # Data population
    steps_per_populate::Int = 100
    populate_size::Int = 200
    # Data buffer
    buffer_capacity::Int = 200000
    complexity_start::Int = 3
    complexity_step::Int = 3
    # Augmentation
    augment_symm::Bool = true
    augment_inv::Bool = true
    # Search tree
    puct_weight::Float64 = 5.0
    policy_noise_weight::Float64 = 0.25
    policy_noise_param::Float64 = 0.03
    # Loss function
    loss_distance_weight::Float64 = 1.5
    loss_policy_weight::Float64 = 1.0
end

