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
	# Pkg.add("PlutoUI")
	# Pkg.add("Optim");
	# Pkg.add("PlyIO"); 
	# Pkg.add("Clustering"); 
	# Pkg.status();
	# Pkg.add("GMT");
	# Pkg.add("Tables");
	# Pkg.add("Interpolations"); 
	# Pkg.add("Query");
	# Pkg.add("AbstractPlotting");
	# Pkg.add("Images")
	using Tables; 
	using PlutoUI; 
	using Clustering; 
	using PlyIO;
	using GMT; 
	using CSV; 
	print()
end

# ╔═╡ 77c4951e-ab8b-4470-8856-6e334c8b88d7
using Images

# ╔═╡ ebd5c24b-4afe-44f1-ac7c-36ca8dd99b71
begin
	using Dierckx
	using Plots
	using LinearAlgebra

	
	# function -- start
	# sigma calculates the distance
	σ(point1,point2)=abs(
		sqrt(
			(point1[1]-point2[1])^2+(point1[2]-point2[2])^2+(point1[3]-point2[3])^2
		)
	)
	
	function closest(points::Matrix{Float64}, 
		spline::LinearAlgebra.Adjoint{Float64, Matrix{Float64}},
		plot::Plots.Plot{Plots.GRBackend}
	)
		
		closest = [] # list of closest points
		for point in eachrow(points)
			# 1D f64 list of distances
			distances = [σ(point,spl) for spl in eachrow(spline)]
			# 1D integer list of indexes
			dist_id = sortperm(distances)
			# we use dist id to create a new list sorted list distance point
			ordered = spline[dist_id, :]
			# we keep the first point
			proxima = ordered[1, :]
			# plotting
			Plots.plot!(
				[proxima[1], point[1]], 
				[proxima[2], point[2]], 
				[proxima[3], point[3]], 
			)
		end
	end
	# multiple dispatch
	function closest(
		point, 
		spline::LinearAlgebra.Adjoint{Float64, Matrix{Float64}},
	)
		# 1D f64 list of distances
		distances = [σ(point,spl) for spl in eachrow(spline)]
		# 1D integer list of indexes
		dist_id = sortperm(distances)
		# we use dist id to create a new list sorted list distance point
		ordered = spline[dist_id, :]
		# we keep the first point
		return ordered[1, :]
end
	
end

# ╔═╡ da75109a-939a-4367-8894-bb4f6e16bfe0
md"""
General structure:
1. Project presentation
2. Classification and Machine Learning methods
3. Spline method
4. Elevation raster
5. Conclusion
"""

# ╔═╡ ded4bc1a-4b2a-4245-940f-52c0e4860983
html"""
<iframe src="https://www.youtube.com/embed/watch?v=ZbuKgo3JJGY&list=TLGG2GyTBScYxnMxODAzMjAyNA;autoplay=1&mute=1" width="560" height="315" frameborder="0" allowfullscreen></iframe>
"""

# ╔═╡ 4a7984cd-82d4-4165-8004-0ce9c86418a9
begin
	lidar_image = load("images/lidar.png")
	beauvais_gallery = load("images/beauvais_gallery.png")
	simple_models = load("images/simple_models.png")
	class1 = load("images/class_1.png")
	class2 = load("images/class_2.png")
	class3 = load("images/class_3.png")
	random_forest = load("images/random_forest.png")
	random_forest_result = load("images/random_forest_2.png")
	spline_method = load("images/spline_method.png")
	sorted_img = load("images/sorted.png")
	unsorted_img = load("images/unsorted.png")
	unsorted_img = imresize(unsorted_img, ratio=0.8)
	unsortedexpl = load("images/unsorted_expl.png")
	classification_img = load("images/classification_img.png")
	mars = load("images/mars.png")
	paris = load("images/paris.jpg")
	geolab = load("images/geolab.jpg")
	depth_map = load("images/depth_map.png")
	orga = load("images/orga.png")
	myfloor = load("images/floor.png")
	mypoint_cloud = load("images/point_cloud.png")
	mycluster = load("images/cluster.png")
	small_lidar = load("images/small_lidar.png")
	print()
end

# ╔═╡ 6f10bdd7-1538-48eb-8768-de2b9bb6e3e7
md"""
# 1. Project


- Initiated by Benoit Proudon **Enseignant-Chercheur, Géologie structurale et Tectonique synsédimentaire** in collaboration with the Géolab at UniLasalle.
- 2 week project

> How to automate the creation of underground gallery plans based on LiDAR point clouds

## 1.2. Example
- Rising temperatures increase the need for cooling in Towns
- Paris a vast web of underground galleries that could theoretically be used for temperature control if used for aerothermy
$paris

"""

# ╔═╡ fc852bd0-3eab-41d7-aaaa-6462222c1491
md"""
## 1.3. Géolab
$geolab
Innovation center at UniLasalle Beauvais that lends it's experience to aid innovation and learning. 

### 1.3.1. Areas of expertise
- 3D modelling from mountain ranges to micrometric cristals
- Numerical modelling
- 3D printing and scanning
- Bio-inspired designs
- Virtual reality
"""

# ╔═╡ dd8db273-82a1-408b-902b-b70f11dd9246
md"""
## 1.4. Team and organisation

|Team Member|Profession|Role|
|---|---|---|
|Adrien Leroux Engineer|Apprentice at EDF|Manager|
|Gaspard Serres|Apprentice at Esri|GIS expert|
|Matthieu Paulin|Student (FISE)|Developer|
|Léane Peyrot|Student (FISE)|Developer|
|George Husband|Apprentice at Ginger CEBTP|Lead Developer|

$orga

- Short meetings every 1~2 days
- Small, motivated team -> flexible management style
- Regular feedback with manager to stay on course
"""

# ╔═╡ b78594cb-a297-4477-9854-1a2457b0cf5c
md"""
## 1.5. LIDAR
$lidar_image

$$v=\frac{d}{t} \rightarrow d = t \times v \rightarrow d=\frac{t \times v}{2}$$
- Creates a point cloud

$mypoint_cloud
"""

# ╔═╡ 1967e07f-0a03-4078-a909-61022ac64c4a
md"""
## 1.6. Dataset
### 1.6.1. Complex datasets
$beauvais_gallery
### 1.6.2. Simple models
$simple_models
"""

# ╔═╡ 8aaf3917-9598-42a3-9aab-4648d5450f56
md"""
|Parameter|Slider|
|---|----|
|Rotate axis 1| $(@bind α2 Slider(1:1:360, default=1))|
|Rotate axis 2| $(@bind β2 Slider(1:1:360, default=1))|
|Marker size| $(@bind sizemarker Slider(1:1:5, default=1))|
"""

# ╔═╡ cd6d8e98-0545-4968-bbe2-9015f9b5a9a3
md"""
$(@bind path Select(
	["cavernes/caverne3.ply" => "cave 1", 
	"cavernes/caverne2.ply" => "cave 2",
	"cavernes/caverne4.ply" => "cave 3",
	"cavernes/complex.ply" => "cave 4"
]))
"""

# ╔═╡ 5c39bc20-715e-41bf-a7c6-d66774fd92b3
begin 
	# TODO: allow the user to choose a filepath
	# open ply files
	# path = "/home/throgg/Documents/Lasalle/cavernes/caverne3.ply"
	ply = load_ply(path)
	# hcat is really cool
	ply_points = hcat(ply["vertex"]["x"], ply["vertex"]["y"], ply["vertex"]["z"])
	print()
end

# ╔═╡ 2af27561-472c-41db-9a41-d7b6d578d8bd
begin
	# seperated plotting from opening data, maybe it will be faster
	Plots.scatter(ply_points[:, 1], ply_points[:, 2], ply_points[:, 3], 
		markersize = sizemarker,
		camera = (α2, β2),
		markeralpha=0.5,
		aspect_ratio=:equal,
		markerstrokewidth=0.1,
		size=(1000,1000)
	)
end

# ╔═╡ ad077917-afa9-4d43-9439-aae4a58b015f
md"""
# 2. Classification & Machine Learning methods
> Goal: Roof & floor separation
## 2.1. K mean clustering
$class1
$class2
$class3
## 2.2.  Random forest classification
$random_forest
- Independant decision trees
- 100 decision trees used in our model
- Reduces overfitting and Bias
$random_forest_result
"""

# ╔═╡ 81050872-ad40-45d6-aa3b-582c40a52135
md"""
# 3. Spline Method
1. Cluster
2. Extract centroids from clusters
3. 3D interpolation between centroid
4. Compare each point from point cloud to it's projection on interpolated line

$spline_method
"""

# ╔═╡ 2345ae2f-aac8-4783-af52-0fe3d3372191
md"""

## Example


Choose resolution: $(@bind resolution Slider(1:1:20, default=1))


Rotate axis 1: $(@bind α Slider(1:1:360, default=1))


Rotate axis 2: $(@bind β Slider(1:1:360, default=1))


Here is the [link](https://discourse.julialang.org/t/interpolate-a-3d-curve/56401/4) that inspired this method
"""

# ╔═╡ f0fc562a-41e5-4c06-ae6f-b7da65d1ca0b
begin 
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
	xyz = Dierckx.evaluate(spl,tfine)'
	println(typeof(xyz))
	myplot = Plots.plot(xyz[:, 1], 
		xyz[:, 2], 
		xyz[:, 3], 
		label="spline",
		xlabel = "x", 
		ylabel = "y", 
		zlabel = "z",
		camera = (α, β)
	)
	# find the closest point on 
	closest(points, xyz, myplot)
	
	Plots.scatter!(rayo[:, 1], rayo[:, 2], rayo[:, 3], label="centroid data")
	Plots.scatter!(points[:, 1], points[:, 2], points[:, 3], label="new points")
end

# ╔═╡ 85f10f93-4a25-4059-8254-f5b196fce1b8
md"""
## 3.1. Clustering and sorting
Here we group the points into clusters

### 3.1.1. Clustering
- Cluster using kmeans
- Calculate the centroid for each cluster

$(mycluster)
"""

# ╔═╡ 6355ca08-fde2-4894-b16e-61022972b1eb
md"""
### 3.1.2. Sorting
|sorted|unsorted|
|---|---|
|$sorted_img|$unsorted_img|
"""

# ╔═╡ 551fa7ef-8a1c-476a-b219-293b7138938e
function sort_by(points, order)
	# sorts using value on axis
    if order == "x"
		return points[sortperm(points[:, 1]), :]
    elseif order == "y"
        return points[sortperm(points[:, 2]), :]
    elseif order == "z"
        return points[sortperm(points[:, 3]), :]
    else
        println("Invalid order, defaulting to x")
        return points[sortperm(points[:, 1]), :]
    end
end
# todo: add path sorting algorithm

# ╔═╡ 0892033f-7409-455a-ab4c-b9d96be5bfe4
md"""
|Parameter|Value|
|---|---|
|Number of clusters|$(@bind numclusters Slider(5:1:20, default=10))|
|Sort axis| $(@bind sortaxis Select(["x", "y", "z"]))|
|Display point cloud ? | $(@bind disppointcloud CheckBox(default=false))|
|Display point not ordered ? | $(@bind notordered CheckBox(default=true))|
|Rotate axis 1| $(@bind αα Slider(1:1:360, default=20))|
|Rotate axis 2| $(@bind ββ Slider(1:1:360, default=20))|
|xscale| $(@bind xscale Slider(1:1:100, default=1))|
|yscale| $(@bind yscale Slider(1:1:100, default=1))|
|zscale| $(@bind zscale Slider(1:1:100, default=1))|
"""

# ╔═╡ 39bc4e6d-9d9d-496d-87f7-2f94871fa293
begin 
	# cluster X into 20 clusters using K-means
	R = Clustering.kmeans(ply_points', numclusters)
	@assert nclusters(R) == numclusters # verify the number of clusters

	centers = R.centers' # transposing it
	sorted = sort_by(centers, sortaxis)
	print()
end

# ╔═╡ 9236ff1d-86d7-49ae-8de9-4e9ffc7aa050
begin 
	Plots.scatter(centers[:, 1], centers[:, 2], centers[:, 3],label="centroids",
		camera = (αα, ββ),
		#xlims=(minimum(ply_points[:, 1]), maximum(ply_points[:, 1])).*xscale,
		#ylims=(minimum(ply_points[:, 2]), maximum(ply_points[:, 2])).*yscale,
		#zlims= (minimum(ply_points[:, 3]), maximum(ply_points[:, 3])).*zscale,
		c="red"
	)
	if notordered
		Plots.plot!(centers[:, 1], centers[:, 2], centers[:, 3],label="not ordered")
	end
	# ask the user of the he want to display the point cloud
	if disppointcloud
			Plots.scatter!(ply_points[:, 1], ply_points[:, 2], ply_points[:, 3], 
				markersize = 1,
				markeralpha=0.1,
				label="point cloud"
		)
	end
	Plots.plot!(sorted[:, 1], sorted[:, 2], sorted[:, 3],label="sorted")
end

# ╔═╡ 4cd85be7-f72c-4604-b8f6-58b1e213674b
md"""
## 3.2 Interpolation

> A spline is a type of piecewise polynomial function. In mathematics, splines are often used in a type of interpolation known as spline interpolation.(All the science, 2023)


|Parameter|Slider|
|---|----|
|Rotate axis 1| $(@bind α3 Slider(1:1:360, default=1))|
|Rotate axis 2| $(@bind β3 Slider(1:1:360, default=1))|
|Choose resolution| $(@bind resolution3 Slider(1:1:20, default=5))|


"""

# ╔═╡ bf045d7e-8cc9-47d5-85fc-1b4581c1d835
begin
	# sort rayo -- start
	sorted_indeses2 = sortperm(sorted[:, 1])
	sorted_indeses2 = sorted[sorted_indeses2, :]
	# sort rayo -- end
	N2 = size(sorted, 1)
	# size of list on A axis, the list we fit the spline to
	spl2 = ParametricSpline(1:N2, sorted') # fi
	# 
	tfine2 = LinRange(1, N2, resolution3*N2)  # interpolates more points
	xyz2 = Dierckx.evaluate(spl2,tfine2)'
	println(typeof(xyz2))
	myplot2 = Plots.plot(xyz2[:, 1], 
		xyz2[:, 2], 
		xyz2[:, 3], 
		label="spline",
		xlabel = "x", 
		ylabel = "y", 
		zlabel = "z",
		camera = (α3, β3)
	)
	# find the closest point on 
	closest(points, xyz2, myplot2)
	
	Plots.scatter!(sorted[:, 1], sorted[:, 2], sorted[:, 3], label="centroid data")
	Plots.scatter!(points[:, 1], points[:, 2], points[:, 3], label="new points")
end

# ╔═╡ 14fdecd5-68d5-4ab1-83ad-eb8f649dade1
md"""
## 3.3. Classifying points

$classification_img
"""

# ╔═╡ 3d499a7c-f56e-4a97-9b8a-bf3718dfa50d
function classify_points(
	line,
	pointcloud,
)
	classification = []
	for point in eachrow(pointcloud)
		if closest(point, line)[3] <= point[3] # 
			append!(classification, ["roof"])
		else
			append!(classification, ["floor"])
		end
	end
	return classification
end

# ╔═╡ 62fb9c5a-229f-4579-88f8-60602ef0d320
classified = classify_points(xyz2, ply_points)

# ╔═╡ 0c3dbc83-bdce-4ab2-8a5d-1c50960237d7
md"""

|Parameter|Slider|
|---|----|
|Rotate axis 1| $(@bind α4 Slider(1:1:360, default=1))|
|Rotate axis 2| $(@bind β4 Slider(1:1:360, default=1))|
|Display roof ? | $(@bind disproof4 CheckBox(default=true))|
|Display floor ? | $(@bind dispfloor4 CheckBox(default=true))|
"""

# ╔═╡ 85979053-8fcb-4310-b300-9a2f60c70d24
begin 
	roof = Vector{Float64}()
	a = hcat(ply_points, classified)
	sol = a[(a[:,4] .== "floor") ,1:3]
	roof = a[(a[:,4] .== "roof") ,1:3]

	max=maximum(ply_points);min=minimum(ply_points);
	
	thisplot = Plots.plot(
		camera = (α4, β4),normalize=true,
		#xlims=(minimum(a[:, 1]), maximum(a[:, 1])),
		#ylims=(minimum(a[:, 2]), maximum(a[:, 2])),
		#zlims= (minimum(a[:, 3]), maximum(a[:, 3])).*5,
		aspect_ratio=:equal,
		size=(1000,1000)
	)
	
	if dispfloor4==true
		Plots.scatter!(sol[:, 1], sol[:, 2], sol[:, 3],
			markersize=2,
			markerstrokewidth=0.2, 
			alpha=0.3,
			label="sol",
			color="red",
			aspect_ratio=:equal,
		)
	end
	if disproof4==true
		Plots.scatter!(roof[:, 1],
			roof[:, 2],
			roof[:, 3],
			markersize=2,
			markerstrokewidth=0,
			alpha=0.3,
			label="toit",
			color="green",
			aspect_ratio=:equal
		)
	end
	# this makes sure that is is shown
	thisplot
end

# ╔═╡ 0b82de54-d79e-4a10-9cdc-943d32b2c517
md"""
# 4. Elevation raster
Terrain elevation is represented with pixel colour
$mars
Example of elevation layer from Mars (source JPL)

## 4.1. Elevation raster process
$myfloor
$depth_map

"""

# ╔═╡ 8a377759-81a5-4ef7-a49a-f52f36af5d76
function minmax(a, b, x, minx, maxx)
	z = a + ((x-minx)*(b-a))/(maxx-minx)
	return ceil(z)
end

# ╔═╡ f3694af5-0e2d-47e8-987e-94d3f7737b41
function aspect(original_height, original_width, resized_width)::Int64
	resized_height = (original_height/original_width) * resized_width
	return floor(resized_height)
end

# ╔═╡ c3950a87-5ac4-480f-ad90-e28bf8dfec4b
md"""
Choose resolution $(@bind taille Select(["25", "50", "75", "100", "125", "150", "175", "200"]))
"""

# ╔═╡ 6da65f2a-97b2-4795-ad04-fd3aa2fd72b0
begin 
	using Query; 
	using DataFrames; 
	
	# getting extent of floor variable
	maxₓ = maximum(sol[:, 1])
	minₓ = minimum(sol[:, 1])
	maxy= maximum(sol[:, 2])
	miny = minimum(sol[:, 2])
	
	grid_width = parse(Int32, taille)
	grid_height = aspect((maxy-miny), (maxₓ-minₓ), grid_width)

	# normalize each point
	normalized_x = [minmax(0, grid_width, x, minₓ, maxₓ) for x in sol[:, 1]]
	normalized_y = [minmax(0, grid_height, x, miny, maxy) for x in sol[:, 2]]

	normalized = DataFrame(:x=>normalized_x, :y=>normalized_y, :z=>sol[:, 3])

	# define an empty grid of zeros
	grid = zeros(Float64, grid_width, grid_height)
	# Iterate through normalized DataFrame and populate the grid
	for x in 1:grid_width
		for y in 1:grid_height
			mypoints = filter(row -> row.x == x && row.y == y, normalized)
			if isempty(mypoints) == false
				grid[x, y] = mean(mypoints.z)
			else
				grid[x, y] = NaN
			end
		end
	end
end

# ╔═╡ 675dd1ae-83dc-4c7d-911a-e4b2dd87ce4d
Plots.heatmap(grid, aspect_ratio=:equal)

# ╔═╡ 0400ffa3-cb50-42fa-bc60-099dac6f1771
md"""
# Conclusion
- Depth map creation
- Open source code

---

- Extract walls
- Depth map of roof and walls
- Create user interface
- Optimizations
- Topology map creation

$small_lidar
"""

# ╔═╡ 5f15e1ba-146d-409d-92fa-c512c685d1b3
md"""
# Litterature

All the science. 2023. What is a spline ?
Link : https://www.allthescience.org/what-is-a-spline.htm

BRGM. 2021. Modélisation 3D des cavités souterrainesLink : https://www.youtube.com/watch?v=hDvbweCH2OA

International Business Machines Corporation. 2023. Qu'est-ce que l'algorithme de forêt aléatoire (random forest) ?
Link : https://www.ibm.com/fr-fr/topics/random-forest

KHAN M. et al. 2021. Automated prediction of Good Dictionary EXamples (GDEX): a comprehensive experiment with distant supervision, machine learning, and word embedding-based deep learning techniques. Complexity, vol. 2021, p. 1-18.

Open3D, 2023. Open3D 0.18.0 documentation in open3D. Site d’Open3D. Disponible sur < https://www.open3d.org/docs/release/index.html>
"""

# ╔═╡ 76fcab9a-d277-44e2-ab92-8a3dc7b602f7
md"""
# Annexes
"""

# ╔═╡ 67aae635-359c-4770-9247-96b43baf9f70
md"""
## TODO make a contour plot
This can be used to make maps using the elvation data of the tunnels
"""

# ╔═╡ c07ae83c-edfb-4b32-8f4e-37b7937068ac
md"""
## Finding shortest path
This is an alternate way of finding a path through the points

Here is the initial code in python: 
```python
centroids_copy = centroids
start_point = 6
start_point = centroids_copy[start_point]
mylist = []
import math
while len(centroids_copy) > 1:
  sorted_numbers = np.array(sorted(centroids_copy, key=lambda point:
      math.sqrt(pow((start_point[0]-point[0]), 2) +
                pow((start_point[1]-point[1]), 2) +
                pow((start_point[2]-point[2]), 2)), reverse = False))

  mylist.append(sorted_numbers[0])
  start_point = sorted_numbers[0]
  centroids_copy = [point for point in centroids_copy if not np.array_equal(point, start_point)]
mylist.append(centroids_copy[0])


my_list = np.array(mylist)
fig, axs = plt.subplots(subplot_kw={'projection': '3d'})
axs.plot(my_list[:, 0], my_list[:, 1], my_list[:, 2], c='black', marker='x', label='Centroids')

for i, (x, y, z) in enumerate(my_list):
    axs.text(x, y, z, str(i), color='red')

mylist = np.array(mylist)

axs.set_xlim(centroids[:, 0].min(), centroids[:, 0].max())  # Set x-axis limits
axs.set_ylim(points['y'].min()*5, points['y'].max()*5)  # Set y-axis limits
axs.set_zlim(0, 25)  # Set z-axis limits
```
"""

# ╔═╡ 4a67239b-44db-4fe2-8bec-243c117ba379
# TODO: Translate to julia

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
# ╠═╡ disabled = true
#=╠═╡
md"""
# Creating new data 
Now that we have defined our model we are going to use the spline to find spline(x, y)-> y for a new dataset
"""
  ╠═╡ =#

# ╔═╡ 9afbf18b-2079-4fe4-b48a-fc6a2b7ccd89


# ╔═╡ df553e8f-7456-4acc-89e2-d780e5413b29
# ╠═╡ disabled = true
#=╠═╡
md"""
$(@bind max_x Slider(-10:1:10, default=5)) 
$(@bind min_x Slider(-10:1:10, default=-5))

$(min_x) $(max_x) 

"""
  ╠═╡ =#

# ╔═╡ aa241f2b-93f8-4313-a503-45fe92e11f48
# ╠═╡ disabled = true
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
# ╟─595c1654-b3b4-11ee-3d06-e3e426e3fbf9
# ╟─da75109a-939a-4367-8894-bb4f6e16bfe0
# ╟─6f10bdd7-1538-48eb-8768-de2b9bb6e3e7
# ╟─fc852bd0-3eab-41d7-aaaa-6462222c1491
# ╟─ded4bc1a-4b2a-4245-940f-52c0e4860983
# ╟─dd8db273-82a1-408b-902b-b70f11dd9246
# ╟─77c4951e-ab8b-4470-8856-6e334c8b88d7
# ╟─4a7984cd-82d4-4165-8004-0ce9c86418a9
# ╟─b78594cb-a297-4477-9854-1a2457b0cf5c
# ╟─1967e07f-0a03-4078-a909-61022ac64c4a
# ╟─8aaf3917-9598-42a3-9aab-4648d5450f56
# ╟─cd6d8e98-0545-4968-bbe2-9015f9b5a9a3
# ╟─5c39bc20-715e-41bf-a7c6-d66774fd92b3
# ╟─2af27561-472c-41db-9a41-d7b6d578d8bd
# ╟─ad077917-afa9-4d43-9439-aae4a58b015f
# ╟─81050872-ad40-45d6-aa3b-582c40a52135
# ╟─2345ae2f-aac8-4783-af52-0fe3d3372191
# ╟─ebd5c24b-4afe-44f1-ac7c-36ca8dd99b71
# ╟─f0fc562a-41e5-4c06-ae6f-b7da65d1ca0b
# ╠═85f10f93-4a25-4059-8254-f5b196fce1b8
# ╟─6355ca08-fde2-4894-b16e-61022972b1eb
# ╟─551fa7ef-8a1c-476a-b219-293b7138938e
# ╟─39bc4e6d-9d9d-496d-87f7-2f94871fa293
# ╟─9236ff1d-86d7-49ae-8de9-4e9ffc7aa050
# ╟─0892033f-7409-455a-ab4c-b9d96be5bfe4
# ╟─4cd85be7-f72c-4604-b8f6-58b1e213674b
# ╟─bf045d7e-8cc9-47d5-85fc-1b4581c1d835
# ╟─14fdecd5-68d5-4ab1-83ad-eb8f649dade1
# ╟─3d499a7c-f56e-4a97-9b8a-bf3718dfa50d
# ╟─62fb9c5a-229f-4579-88f8-60602ef0d320
# ╟─85979053-8fcb-4310-b300-9a2f60c70d24
# ╟─0c3dbc83-bdce-4ab2-8a5d-1c50960237d7
# ╟─0b82de54-d79e-4a10-9cdc-943d32b2c517
# ╟─8a377759-81a5-4ef7-a49a-f52f36af5d76
# ╟─f3694af5-0e2d-47e8-987e-94d3f7737b41
# ╠═6da65f2a-97b2-4795-ad04-fd3aa2fd72b0
# ╟─675dd1ae-83dc-4c7d-911a-e4b2dd87ce4d
# ╟─c3950a87-5ac4-480f-ad90-e28bf8dfec4b
# ╟─0400ffa3-cb50-42fa-bc60-099dac6f1771
# ╟─5f15e1ba-146d-409d-92fa-c512c685d1b3
# ╟─76fcab9a-d277-44e2-ab92-8a3dc7b602f7
# ╟─67aae635-359c-4770-9247-96b43baf9f70
# ╟─c07ae83c-edfb-4b32-8f4e-37b7937068ac
# ╠═4a67239b-44db-4fe2-8bec-243c117ba379
# ╟─650773fb-7291-4781-a636-7a5dde431988
# ╟─5c23b1c8-560f-427e-ac50-340c09b6e96c
# ╟─59b08d49-02c3-44e0-9ff0-f983d90e0735
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
