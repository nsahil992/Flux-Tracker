using Statistics
using Flux.Tracker:

inputs = [0.2, -0.3, 0.5, 1, -0.9]
outputs = [-0.2, 0.3, -0.5, -1, 0.9]
weight = 0.4

predict(inputs, weight) = inputs .* weight
loss(inputs, outputs, weight) = mean((outputs - (inputs .* weight)) .^2)

dloss(inputs, outputs, weight) = Tracker.data(Tracker .gradient(loss, inputs, outputs, weight)[3])

for i in 1:100
    println("Current prediction: $(predict(inputs, weight))")
    println("Current loss: $(loss(inputs, outputs, weight))")
    println("Current weight: $(weight)")
