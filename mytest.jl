### A Pluto.jl notebook ###
# v0.19.36

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 595c1654-b3b4-11ee-3d06-e3e426e3fbf9
begin 
	using Pkg; 
	Pkg.activate("Project.toml");
	Pkg.add("PlutoUI")
	Pkg.add("Optim"); 
	Pkg.status();
	using PlutoUI; 
end

# ╔═╡ 2345ae2f-aac8-4783-af52-0fe3d3372191
md"""
# Define test centroids dataset

Choose resolution: $(@bind resolution Slider(1:1:10, default=1))
"""

# ╔═╡ f0fc562a-41e5-4c06-ae6f-b7da65d1ca0b
begin 
	using Dierckx
	using Plots
	using LinearAlgebra

	
	# function -- start
	# sigma calculates the distance
	σ(point1,point2)=abs(sqrt((point1[1]-point2[1])^2+(point1[1]-point2[1])^2+(point1[1]-point2[1])^2))
	
	function closest(points::Matrix{Float64}, 
		spline::LinearAlgebra.Adjoint{Float64, Matrix{Float64}},
		plot::Plots.Plot{Plots.GRBackend})
		closest = []
		for point in range(1, size(points, 1))
			distances = [σ(point,spline[a,:]) for a in range(1, size(spline, 1))]
			dist_id = sortperm(distances)
			ordered = spline[dist_id, :]
			proxima = ordered[1, :]
			# print(points[point,:])
			println(size(ordered))
			println(size(proxima))
			
			plot!(
				[proxima[1], points[point,1]], 
				[proxima[2], points[point,2]], 
				[proxima[3], points[point,3]], 
			)
			

		end
	end
	# functions -- end
	
	rayo=[-10.2548981 10.37426088 2.56011651
	    -10.11793877 8.0069942 2.34776499
	    -9.40531934 5.96339742 1.53314975
	    -8.24462178 4.05571606 0.87486114
	    -6.64535405 2.42014212 0.3832646
	    -4.70369491 1.14270792 0.08551336
	    -2.44332978 0.28911826 -0.04121438
	     0.07714017 0.013431069 -0.05332121
		-6.645352335405 2.42014212 0.3832646
	    -4.70369491 1.11270792 0.08551336
	    -2.44932978 4.28911826 -0.045121438
	     1.07714017 0.81360069 -0.059632121
	]
	
	# sort rayo -- start
	sorted_indeses = sortperm(rayo[:, 1])
	rayo = rayo[sorted_indeses, :]
	# sort rayo -- end
	N = size(rayo, 1)
	spl = ParametricSpline(1:N, rayo') 
	# new points -- start
	points = [
		-2.5 10.37426088 0.2
	    -7.5 5 1
	    -2 2 0.53314975
	]
	# new points -- end
	
	# 
	tfine = LinRange(1, N, resolution*N)  # interpolates more points
	xyz = evaluate(spl,tfine)'
	println(typeof(xyz))
	myplot = plot(xyz[:, 1], xyz[:, 2], xyz[:, 3], label="spline")
	closest(points, xyz, myplot)
	
	scatter!(rayo[:, 1], rayo[:, 2], rayo[:, 3], label="centroid data")
	scatter!(points[:, 1], points[:, 2], points[:, 3], label="new points")
end

# ╔═╡ ebd5c24b-4afe-44f1-ac7c-36ca8dd99b71


# ╔═╡ 650773fb-7291-4781-a636-7a5dde431988
md"""
The ' operator is a way of transposing in julia.
```
a = LinRange(0, 10, 20)
println(size(a))
println(size(a'))
```
- a is of dimension (20, )
- a' is of dimension (1, 20) -> ((1), (2), (3))
"""

# ╔═╡ 5c23b1c8-560f-427e-ac50-340c09b6e96c
md"""
Evaluate the 1-d spline spl at points given in x, which can be a 1-d array or scalar. If a 1-d array, the values must be monotonically increasing.
"""

# ╔═╡ 59b08d49-02c3-44e0-9ff0-f983d90e0735
md"""
Here is the initial code I used
```julia
begin 
	using Dierckx
	
	N = size(rayo,1) # get size of rayo matrix
	spl = ParametricSpline(1:N, rayo, k=2) # gets length of the spline
	
	tfine = LinRange(1, N, 10*N)  # interpolates with 10x more points
	
	xyz = evaluate(spl,tfine)'
end
```
"""

# ╔═╡ a75a1b0b-0036-4c95-a3b6-483b08dff89f
begin 
	
end

# ╔═╡ e52e6a48-0e83-4915-9606-a12e25fe7eee
md"""
We want to estimate z from x and y data. The previous examples does not work because the evaluate function only accepts a 1D array as input for predict function. Here I am going to try the Spline 2D function.
"""

# ╔═╡ 2e61ddfe-4dfd-4471-a9eb-d0f193bf58a9
# ╠═╡ disabled = true
#=╠═╡

begin 
	using Dierckx
	# Assuming you have already defined rayo[:, 1], rayo[:, 2], and rayo[:, 3]
	
	# Extracting x, y, z
	x = rayo[:, 1]
	y = rayo[:, 2]
	z = rayo[:, 3]
	
	# Get the permutation to sort x
	sorted_indices = sortperm(x)
	
	# Apply the permutation to x, y, and z
	x_sorted = x[sorted_indices]
	y_sorted = y[sorted_indices]
	z_sorted = z[sorted_indices]
	# adapting kx and ky
	
	# Initialize Spline2D
	myspline = Spline2D(x_sorted, y_sorted, z_sorted, kx=2, ky=2, s=1)
	# evaluating data
	output = evaluate(myspline, x_sorted, y_sorted)
end

  ╠═╡ =#

# ╔═╡ 0cd1836f-ea6b-4f6f-80fa-984fd35e7edd
# ╠═╡ disabled = true
#=╠═╡
md"""
$(@bind deriv_x Slider(-20:0.1:10, default=5)) 
$(@bind deriv_y Slider(-20:0.1:10, default=-5))
"""
  ╠═╡ =#

# ╔═╡ 62994db1-6c09-4753-9d7b-a1625b88b72b
#=╠═╡
begin 
	my_derivative = derivative(myspline, deriv_x, deriv_y)
end
  ╠═╡ =#

# ╔═╡ e42c65c0-d08a-4e9a-9981-8bd63efd76e3
md"""
# Creating new data 
Now that we have defined our model we are going to use the spline to find spline(x, y)-> y for a new dataset
"""

# ╔═╡ 9afbf18b-2079-4fe4-b48a-fc6a2b7ccd89


# ╔═╡ df553e8f-7456-4acc-89e2-d780e5413b29
md"""
$(@bind max_x Slider(-10:1:10, default=5)) 
$(@bind min_x Slider(-10:1:10, default=-5))

$(min_x) $(max_x) 

"""

# ╔═╡ aa241f2b-93f8-4313-a503-45fe92e11f48
#=╠═╡
begin 
	x_new = LinRange(minimum(x), maximum(x), 250)
	y_new = LinRange(minimum(x), maximum(x), 250)
	z_predicted = evaluate(myspline, x_new, y_new)
end
  ╠═╡ =#

# ╔═╡ 59865bff-ca3f-4212-9e8e-9e9ec0de0e57
md"""
Plotting data
"""

# ╔═╡ 73ae4684-4bbb-44b9-b7de-c3fb0f69abe0
# ╠═╡ disabled = true
#=╠═╡
begin 
	using Plots
	scatter(x_sorted, y_sorted, z_sorted, label="original")
	plot!(x_sorted, y_sorted, output, label="predicted")
	scatter!([deriv_x],[deriv_y], [my_derivative])
end
  ╠═╡ =#

# ╔═╡ Cell order:
# ╠═595c1654-b3b4-11ee-3d06-e3e426e3fbf9
# ╟─2345ae2f-aac8-4783-af52-0fe3d3372191
# ╠═ebd5c24b-4afe-44f1-ac7c-36ca8dd99b71
# ╠═f0fc562a-41e5-4c06-ae6f-b7da65d1ca0b
# ╟─650773fb-7291-4781-a636-7a5dde431988
# ╟─5c23b1c8-560f-427e-ac50-340c09b6e96c
# ╟─59b08d49-02c3-44e0-9ff0-f983d90e0735
# ╠═a75a1b0b-0036-4c95-a3b6-483b08dff89f
# ╟─e52e6a48-0e83-4915-9606-a12e25fe7eee
# ╠═2e61ddfe-4dfd-4471-a9eb-d0f193bf58a9
# ╠═0cd1836f-ea6b-4f6f-80fa-984fd35e7edd
# ╟─62994db1-6c09-4753-9d7b-a1625b88b72b
# ╠═e42c65c0-d08a-4e9a-9981-8bd63efd76e3
# ╟─9afbf18b-2079-4fe4-b48a-fc6a2b7ccd89
# ╠═df553e8f-7456-4acc-89e2-d780e5413b29
# ╟─aa241f2b-93f8-4313-a503-45fe92e11f48
# ╟─59865bff-ca3f-4212-9e8e-9e9ec0de0e57
# ╠═73ae4684-4bbb-44b9-b7de-c3fb0f69abe0
