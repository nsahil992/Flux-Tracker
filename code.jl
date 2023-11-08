using Statistics
using Flux.Tracker:

inputs = [0.2, -0.3, 0.5, 1, -0.9]
outputs = [-0.2, 0.3, -0.5, -1, 0.9]
weight = 0.4

predict(inputs, weight) = inputs .* weight
loss(inputs, outputs, weight) = mean((outputs - (inputs .* weight)) .^2)

