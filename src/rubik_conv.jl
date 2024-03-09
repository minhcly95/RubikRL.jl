"""
A Rubik convolution layer is a specilization of graph convolution on Rubik channels.
A Rubik convolution kernel has 9 weights per channel:
1. Center from self
2. Center from edge
3. Center from corner
4. Edge from self
5. Edge from center
6. Edge from edge
7. Corner from self
8. Corner from center
9. Corner from corner

For example, a center neuron in an output channel will receive inputs from:
1. Itself (across all input channels) multiplied by the 1st weight,
2. All 4 of its adjacent edges, multiplied by the 2nd weight,
3. All 4 of its adjacent corners, multiplied by the 3rd weight.
Other weights (4th to 9th) are used for the edge and corner neurons.
"""
struct RubikConv{W}
    weight::W
end

function RubikConv((in, out)::Pair{<:Integer,<:Integer}; init=Flux.glorot_uniform)
    return RubikConv(init(9, in, out))
end

Flux.@layer RubikConv

Base.show(io::IO, a::RubikConv) = print(io, "RubikConv($(size(a.weight,2)) => $(size(a.weight,3)))")

# We convert the kernel and the input to matrix form to take advantage of BLAS functionalities
const CENTER_KERNEL = [1, 2, 2, 2, 2, 3, 3, 3, 3]   # Center from self, 4x from adj. edges, 4x from adj. corners
const EDGE_KERNEL = [4, 5, 6]                       # Edge from self, from adj. center, from other edge cell
const CORNER_KERNEL = [7, 8, 9, 9]                  # Corner from self, from adj. center, 2x from other corner cells

# Windows to reshape input tensor.
# Each row corresponds to an output feature.
# Each column corresponds to an entry in CENTER_KERNEL, EDGE_KERNEL, or CORNER_KERNEL.
# For example, the first column in CENTER_WINDOW is the center cell itself,
# the next 4 columns are its 4 adj. edges, and the last 4 columns are its 4 adj. corners.
const CENTER_WINDOW = [
    1 7 8 9 10 31 32 33 34;
    2 11 12 13 14 35 36 37 38;
    3 15 16 17 18 39 40 41 42;
    4 19 20 21 22 43 44 45 46;
    5 23 24 25 26 47 48 49 50;
    6 27 28 29 30 51 52 53 54
]
# Edge from self, center, other edge cell
const EDGE_WINDOW = [
    7 1 23;
    8 1 27;
    9 1 15;
    10 1 11;
    11 2 10;
    12 2 29;
    13 2 16;
    14 2 19;
    15 3 9;
    16 3 13;
    17 3 24;
    18 3 21;
    19 4 14;
    20 4 30;
    21 4 18;
    22 4 26;
    23 5 7;
    24 5 17;
    25 5 28;
    26 5 22;
    27 6 8;
    28 6 25;
    29 6 12;
    30 6 20
]
# Corner from self, center, other 2 corner cells
const CORNER_WINDOW = [
    31 1 48 51;
    32 1 40 47;
    33 1 35 52;
    34 1 36 39;
    35 2 33 52;
    36 2 34 39;
    37 2 43 54;
    38 2 41 44;
    39 3 34 36;
    40 3 32 47;
    41 3 38 44;
    42 3 46 49;
    43 4 37 54;
    44 4 38 41;
    45 4 50 53;
    46 4 42 49;
    47 5 32 40;
    48 5 31 51;
    49 5 42 46;
    50 5 45 53;
    51 6 31 48;
    52 6 33 35;
    53 6 45 50;
    54 6 37 43
]

# Convert the kernel to matrix form
function kernel_to_col(a::RubikConv)
    kcenter = view(a.weight, CENTER_KERNEL, :, :)
    kedge = view(a.weight, EDGE_KERNEL, :, :)
    kcorner = view(a.weight, CORNER_KERNEL, :, :)

    az = copy(Flux.flatten(kcenter))
    ae = copy(Flux.flatten(kedge))
    ac = copy(Flux.flatten(kcorner))

    return az, ae, ac
end

# Convert the input to matrix form
function input_to_col(x)
    win_center = view(x, CENTER_WINDOW, :, :)
    win_edge = view(x, EDGE_WINDOW, :, :)
    win_corner = view(x, CORNER_WINDOW, :, :)

    xz = copy(reshape(win_center, N_CENTER_FEATURES, :, size(x)[end]))
    xe = copy(reshape(win_edge, N_EDGE_FEATURES, :, size(x)[end]))
    xc = copy(reshape(win_corner, N_CORNER_FEATURES, :, size(x)[end]))

    return xz, xe, xc
end

# Apply the convolution in matrix form
function (a::RubikConv)(x)
    az, ae, ac = kernel_to_col(a)
    xz, xe, xc = input_to_col(x)
    return cat(xz ⊠ az, xe ⊠ ae, xc ⊠ ac, dims=1)
end

