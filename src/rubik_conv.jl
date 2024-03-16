"""
A Rubik convolution layer is a specilization of graph convolution on Rubik channels.
A Rubik convolution kernel has 25 weights per channel:
1. Center from self
2. Center from edge
3. Center from corner
4. Edge from self
5. Edge from center
6. Edge from other cell
7. Edge from U edge
8. Edge from U2 edge
9. Edge from U' edge
10. Edge from F edge
11. Edge from F2 edge
12. Edge from F' edge
13. Corner from self
14. Corner from center
15. Corner from CW cell
16. Corner from CCW cell
17. Edge from U edge
18. Edge from U2 edge
19. Edge from U' edge
20. Edge from F edge
21. Edge from F2 edge
22. Edge from F' edge
23. Edge from R edge
24. Edge from R2 edge
25. Edge from R' edge

For example, a center neuron in an output channel will receive inputs from:
1. Itself (across all input channels) multiplied by the 1st weight,
2. All 4 of its adjacent edges, multiplied by the 2nd weight,
3. All 4 of its adjacent corners, multiplied by the 3rd weight.
Other weights (4th to 25th) are used for the edge and corner neurons.
"""
struct RubikConv{W}
    weight::W
end

function RubikConv((in, out)::Pair{<:Integer,<:Integer}; init=Flux.glorot_uniform)
    return RubikConv(init(25, in, out))
end

Flux.@layer RubikConv

Base.show(io::IO, a::RubikConv) = print(io, "RubikConv($(size(a.weight,2)) => $(size(a.weight,3)))")

# We convert the kernel and the input to matrix form to take advantage of BLAS functionalities
const CENTER_KERNEL = [1, 2, 2, 2, 2, 3, 3, 3, 3]   # Center from self, 4x from adj. edges, 4x from adj. corners
const EDGE_KERNEL = 4:12                            # Edge from self, from adj. center, from other edge cell, 6x from U/F orbits
const CORNER_KERNEL = 13:25                         # Corner from self, from adj. center, 2x from other corner cells, 6x from U/F/R orbits

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
# Edge from self, center, other edge cell, U/F orbits
const EDGE_WINDOW = [
    7 1 23 9 10 8 28 22 17;
    8 1 27 7 9 10 12 20 25;
    9 1 15 10 8 7 24 21 13;
    10 1 11 8 7 9 16 19 29;
    11 2 10 13 14 12 27 23 15;
    12 2 29 11 13 14 20 25 8;
    13 2 16 14 12 11 9 24 21;
    14 2 19 12 11 13 18 26 30;
    15 3 9 17 18 16 11 27 23;
    16 3 13 15 17 18 19 29 10;
    17 3 24 18 16 15 7 28 22;
    18 3 21 16 15 17 26 30 14;
    19 4 14 21 22 20 29 10 16;
    20 4 30 19 21 22 25 8 12;
    21 4 18 22 20 19 13 9 24;
    22 4 26 20 19 21 17 7 28;
    23 5 7 25 26 24 15 11 27;
    24 5 17 23 25 26 21 13 9;
    25 5 28 26 24 23 8 12 20;
    26 5 22 24 23 25 30 14 18;
    27 6 8 29 30 28 23 15 11;
    28 6 25 27 29 30 22 17 7;
    29 6 12 30 28 27 10 16 19;
    30 6 20 28 27 29 14 18 26
]
# Corner from self, center, other 2 corner cells, U/F/R orbits
const CORNER_WINDOW = [
    31 1 51 48 32 34 33 53 46 40 35 43 50;
    32 1 47 40 34 33 31 49 44 36 51 45 42;
    33 1 35 52 31 32 34 37 45 48 39 44 54;
    34 1 39 36 33 31 32 41 43 52 47 46 38;
    35 2 52 33 36 38 37 51 47 39 43 50 31;
    36 2 34 39 38 37 35 32 49 44 52 48 40;
    37 2 43 54 35 36 38 45 48 33 41 49 53;
    38 2 41 44 37 35 36 42 50 54 34 47 46;
    39 3 36 34 40 42 41 35 51 47 44 54 33;
    40 3 32 47 42 41 39 31 53 46 36 52 48;
    41 3 44 38 39 40 42 43 52 34 49 53 37;
    42 3 49 46 41 39 40 50 54 38 32 51 45;
    43 4 54 37 44 46 45 52 34 41 50 31 35;
    44 4 38 41 46 45 43 36 32 49 54 33 39;
    45 4 50 53 43 44 46 48 33 37 42 32 51;
    46 4 42 49 45 43 44 40 31 53 38 34 47;
    47 5 40 32 48 50 49 39 35 51 46 38 34;
    48 5 31 51 50 49 47 33 37 45 40 36 52;
    49 5 46 42 47 48 50 44 36 32 53 37 41;
    50 5 53 45 49 47 48 54 38 42 31 35 43;
    51 6 48 31 52 54 53 47 39 35 45 42 32;
    52 6 33 35 54 53 51 34 41 43 48 40 36;
    53 6 45 50 51 52 54 46 40 31 37 41 49;
    54 6 37 43 53 51 52 38 42 50 33 39 44
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

