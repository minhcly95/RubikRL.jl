"""
A Rubik channel is a 54-dimensional vector, each corresponds to 1 cell.
In particular,
- The first 6 elements (`1:6`) represent the centers,
- The next 24 elements (`7:30`) represent the edges (12 edges * 2 sides per edge),
- The last 24 elements (`31:54`) represent the corners (8 corners * 3 sides per corner).

The cells are numbered as follows:
```
         31  7 32                    
          8  1  9                    
         33 10 34                    
51 27 52 35 11 36 39 15 40 47 23 48  
28  6 29 12  2 13 16  3 17 24  5 25  
53 30 54 37 14 38 41 18 42 49 26 50  
         43 19 44                    
         20  4 21                    
         45 22 46                    
```
"""

const N_CENTER_FEATURES = 6
const N_EDGE_FEATURES = 24
const N_CORNER_FEATURES = 24
const N_FEATURES = N_CENTER_FEATURES + N_EDGE_FEATURES + N_CORNER_FEATURES

const NET_TO_CENTERS = [5, 14, 23, 32, 41, 50]
const NET_TO_EDGES = [2, 4, 6, 8, 11, 13, 15, 17, 20, 22, 24, 26, 29, 31, 33, 35, 38, 40, 42, 44, 47, 49, 51, 53]
const NET_TO_CORNERS = [1, 3, 7, 9, 10, 12, 16, 18, 19, 21, 25, 27, 28, 30, 34, 36, 37, 39, 43, 45, 46, 48, 52, 54]
const NET_TO_FEATURES = vcat(NET_TO_CENTERS, NET_TO_EDGES, NET_TO_CORNERS)

function features(cube::Cube)
    flat_net = collect(Iterators.flatten(RubikCore.net(cube)))
    return Float32.(Flux.onehotbatch(flat_net[NET_TO_FEATURES], RubikCore.ALL_FACES)')
end

features(cubes::AbstractVector{Cube}) = stack(features(cube) for cube in cubes)

