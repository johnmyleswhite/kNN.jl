module TestRegress
	using Base.Test
	using kNN
	using StatsBase

	srand(1)
	n = 1_000
	x = 10 * randn(n)
	y = sin(x) + 0.5 * randn(n)

	fit = kernelregression(x, y)
	grid = minimum(x):0.1:maximum(x)
	predictions = predict(fit, grid)

	fit = kernelregression(x, y, kernel = :gaussian)
	grid = minimum(x):0.1:maximum(x)
	predictions = predict(fit, grid)

	fit = kernelregression(x, y, kernel = :gaussian, bandwidth = 1.0)
	grid = minimum(x):0.1:maximum(x)
	predictions = predict(fit, grid)
end
