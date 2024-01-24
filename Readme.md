#unilasalle 
# Intro
- The goal of this project is to assist in creation of 2D maps from [[Lidar]] in caves
- End of 5th year project 
## Goals
- Spitting the gallery tunnel into roof, walls and floor
# Splitting gallery
To split the tunnel we tried with a machine learning model in [[Python]] with weights on x, y and z variables. This did not work very well when the galleries had a slope. 
Later we used [[Blender Scripting]] to create some mock up tunnels and tried to make a linéaire from them. 
## Making the linéaire 
### Opening ply files in [[Python]]
Whe start by opening the ply files we made in [[Blender]]: 
```python
import open3d as o3d
import polars as pl
import numpy as np
import os
import matplotlib.pyplot as plt
  
prima_via = r"/content/drive/MyDrive/Lasalle/S9/mine/cavernes/caverne2.ply"
secunda_via = r"/content/drive/MyDrive/Lasalle/S9/mine/cavernes/caverne3.ply"
tertia_via = r"/content/drive/MyDrive/Lasalle/S9/mine/cavernes/caverne4.ply"
  
# hic, tres ordinatas facimus
ordinata_prima = np.array(o3d.io.read_point_cloud(os.path.join(prima_via)).points)
ordinata_secunda = np.array(o3d.io.read_point_cloud(os.path.join(secunda_via)).points)
ordinata_tertia = np.array(o3d.io.read_point_cloud(os.path.join(tertia_via)).points)
# hic, tres nubes punctorum facimus
nubes_prima = pl.DataFrame({"x": ordinata_prima[:, 0], "y": ordinata_prima[:, 1], "z": ordinata_prima[:, 2]})
nubes_secunda = pl.DataFrame({"x": ordinata_secunda[:, 0], "y": ordinata_secunda[:, 1], "z": ordinata_secunda[:, 2]})
nubes_tertia = pl.DataFrame({"x": ordinata_tertia[:, 0], "y": ordinata_tertia[:, 1], "z": ordinata_tertia[:, 2]}
# Plotting 
# nubes punctum duco
fig, axs = plt.subplots(figsize = (20, 20), subplot_kw={'projection': '3d'})
axs.scatter(nubes_prima["x"], nubes_prima["y"], nubes_prima["z"], c=nubes_prima["z"])
max_mea = max([nubes_prima["x"].max(), nubes_prima["y"].max(), nubes_prima["z"].max()])
min_mea = min([nubes_prima["x"].min(), nubes_prima["y"].min(), nubes_prima["z"].min()])
axs.set_xlim(min_mea, max_mea) # Set x-axis limits
axs.set_ylim(min_mea, max_mea) # Set y-axis limits
axs.set_zlim(min_mea, max_mea) # Set z-axis limits
# nubes secunda duco
fig, axs = plt.subplots(figsize = (20, 20), subplot_kw={'projection': '3d'})
axs.scatter(nubes_secunda["x"], nubes_secunda["y"], nubes_secunda["z"], c=nubes_secunda["z"])
max_mea = max([nubes_secunda["x"].max(), nubes_secunda["y"].max(), nubes_secunda["z"].max()])
min_mea = min([nubes_secunda["x"].min(), nubes_secunda["y"].min(), nubes_secunda["z"].min()])
axs.set_xlim(min_mea, max_mea) # Set x-axis limits
axs.set_ylim(min_mea, max_mea) # Set y-axis limits
axs.set_zlim(min_mea, max_mea) # Set z-axis limits
# nubes tertia duco
fig, axs = plt.subplots(figsize = (20, 20), subplot_kw={'projection': '3d'})
axs.scatter(nubes_tertia["x"], nubes_tertia["y"], nubes_tertia["z"], c=nubes_tertia["z"])
max_mea = max([nubes_tertia["x"].max(), nubes_tertia["y"].max(), nubes_tertia["z"].max()])
min_mea = min([nubes_tertia["x"].min(), nubes_tertia["y"].min(), nubes_tertia["z"].min()])
axs.set_xlim(min_mea, max_mea) # Set x-axis limits
axs.set_ylim(min_mea, max_mea) # Set y-axis limits
axs.set_zlim(min_mea, max_mea) # Set z-axis limits
```
For more details see [[Point Clouds Python]].
Gave us results like so: 

|  |  |  |
| ---- | ---- | ---- |
| ![[Pasted image 20240109120043.png\|200]] | ![[Pasted image 20240109120114.png\|200]] | ![[Pasted image 20240109120139.png\|200]]
### Clustering and centroid extraction
The easiest way we found to find a spine for the point cloud, was to cluster the point clouds and find the centroid of each cluster. Connecting each centroid in the right order could then be used to create a spine for the geometry. 
```python
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Select the columns containing the 3D points
points = nubes_prima[['x', 'y', 'z']]
points_df = pd.DataFrame(points, columns=['x', 'y', 'z'])
# Specify the number of clusters (you can adjust this)
num_clusters = 8
# Perform K-means clustering
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(points)
cluster_labels = kmeans.labels_
centroids = kmeans.cluster_centers_

```
The results of the code above can be plotted: 
```python
# Visualize the clusters in a 3D plot
fig, axs = plt.subplots(ncols=2, figsize =(20, 20), subplot_kw={'projection': '3d'})
# Plot each point colored by its cluster label
for cluster_label in range(num_clusters):
	cluster_points = points_df[cluster_labels == cluster_label] # extracting clusters with a specific label
	axs[0].scatter(cluster_points['x'], cluster_points['y'], cluster_points['z'], label=f'Cluster {cluster_label}')
# Plot centroids
	axs[0].scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], c='black', marker='x', s=100, label='Centroids')

# Plot centroids
axs[1].plot(centroids[:, 0], centroids[:, 1], centroids[:, 2], c='black', marker='x', label='Centroids')
  
axs[0].set_xlabel('X axis')
axs[0].set_ylabel('Y axis')
axs[0].set_zlabel('Z axis')
# setting dimensions
axs[0].set_xlim(0, 50) # Set x-axis limits
axs[0].set_ylim(0, 25) # Set y-axis limits
axs[0].set_zlim(0, 25) # Set z-axis limits
axs[1].set_xlim(0, 50) # Set x-axis limits
axs[1].set_ylim(0, 25) # Set y-axis limits
axs[1].set_zlim(0, 25) # Set z-axis limits

for i, (x, y, z) in enumerate(centroids):
	axs[1].text(x, y, z, str(i), color='red')
plt.title('K-means Clustering of 3D Points')
plt.show()
```
And yield the following result: 

| Clusters | Centroids |
| --- | --- |
| ![[Pasted image 20240109121028.png]] | ![[Pasted image 20240109121038.png]] |
- We can see that the default index of the centroids to not allow the construction of a logical path. The next step is to determine the correct path. 
### Path creation
This part is on how to create a logical path connectiong all the centroid points together in a logical fashion: 
```python
centroids_copy = centroids
start_point = centroids_copy[6] # <- manual start point definition 
mylist = []
while len(centroids_copy) > 1:
	sorted_numbers = np.array(sorted(centroids_copy, key=lambda point:
		math.sqrt(pow((start_point[0]-point[0]), 2) +
		pow((start_point[1]-point[1]), 2) +
		pow((start_point[2]-point[2]), 2)), reverse = False))

	mylist.append(sorted_numbers[0])
	start_point = sorted_numbers[0]
	centroids_copy = [point for point in centroids_copy if not np.array_equal(point, start_point)]

# add the last remaining point list
mylist.append(centroids_copy[0])
# convert to numy arrya 
my_list = np.array(mylist)
```
> In the code above the start point needs to be manually defined. This can be done using the indeces displayed on the previous centroid plots. A possible upgrade would be to use a path finding algorithm as what was done with the ant algo ([link](https://github.com/Dunckleosteus/Ants/tree/main/src))

The ordered list can the be plotted using [[Matplotlib]] with the following code: 
```python
fig, axs = plt.subplots(subplot_kw={'projection': '3d'})
axs.plot(my_list[:, 0], my_list[:, 1], my_list[:, 2], c='black', marker='x', label='Centroids')
# plotting the index of each point 
for i, (x, y, z) in enumerate(my_list):
	axs.text(x, y, z, str(i), color='red')
```

![[Pasted image 20240109121734.png]]
Here is what the ordered python list looks like: 
```
[array([-10.2548981 , 10.37426088, 2.56011651]), array([-10.11793877, 8.0069942 , 2.34776499]), array([-9.40531934, 5.96339742, 1.53314975]), array([-8.24462178, 4.05571606, 0.87486114]), array([-6.64535405, 2.42014212, 0.3832646 ]), array([-4.70369491, 1.14270792, 0.08551336]), array([-2.44332978, 0.28911826, -0.04121438]), array([ 0.07714017, 0.01360069, -0.05332121])]
```
## Fitting
For this project I tried 2 main methods of fitting a [[3D Functions]] to my points. 
### Line fitting
- representing the tunnel with mathematical formula ([[Curve Fitting]])
- lets us create a plance from the tunnel spine and horizontal and vertical axis. 
![[Geoquarry 2024-01-10 11.41.41.excalidraw]]
We want to define a function or a set of functions $f$ from which it's possible to calculate $x, y, z$ using combinations of $x, y, z$. For example: 
- $f(x, y)\rightarrow{z}$ will allow us to detect if a point A $\in$ the floor or the roof by checking if $f(point.x,point.y)\rightarrow{z}>point.z$
### Curve fitting
[[3D Functions]] have two inputs and one output value, a little like [[Scalar Fields]]. I found the best solution to be to create a 3D surface and fit it to the point could generated using the point cloud. See examples in [[3D Graph]] for more details. 
We start with the usual imports, note that **mylist** variable is simply the ordered list create above. I'm not sure if it's really useful in this case.
```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit
# create x, y and z list from data 
x = mylist[:,0]
y = mylist[:,1]
z = mylist[:,2]
```
To constrain the curve we can add extra constraints by appending offset start points to the dataset. This is not always the best option and the axis of the offset will depend on the geometry and orientation of the gallery. 
```python
# creating horizontal offsets -- start
x2 = x + np.array([10])
y2 = y + np.array([10])
x3 = x + np.array([-10])
y3 = y + np.array([-10])
x = np.append(np.append(x, x), x)
y = np.append(np.append(y, y2), y3)
z = np.append(np.append(z, z), z)
# transposing data
data = np.array([x, y, z]).T
```
Here we define the functions we are going to add weight to. Only func is used in the code but it illustrates a higher order function that may fit the data better. ==Trying different types of functions may be a good idea. ==
```python
def func2(xy, a, b, c, d, e, f, g, h, i, j):
	x, y = xy
	return a + b * x + c * y + d * x**2 + e * y**2 + f * x * y + g * x**3 + h * y**3 + i * x**2 * y + j * x * y**2

def func(xy, a, b, c, d, e, f):
	x, y = xy
	return a + b * x + c * y + d * x**2 + e * y**2 + f * x * y
```
Here we perform the curve fitting on our modified centroid points data: 
```python
# Perform curve fitting
popt, pcov = curve_fit(func, (x, y), z)
# Print optimized parameters
print(popt)
```
Now we add the points from our point cloud so as to be able to project them on the surface created by the centroid dataset. 
```python
# test points
input_data = points_df[["x", "y", "z"]].values
input_x = input_data[:, 0]
input_y = input_data[:, 1]
input_z = input_data[:, 2]
# predicting using fitted function
pred_z = func((input_x, input_y), *popt)
```
Now that we have a fitted function we can proceed to the classification of the points based on if they are above or below their projection on the function's surface. ($f(point.x,point.y)\rightarrow{} z > point.z$ ??)
```python
# classifying the points into ground or roof lists
humus = [] # ground list
tectum = [] # roof list

for (point, predicted) in zip(input_data, pred_z):
	if point[-1] <= predicted:
		humus.append(point)
	else:
		tectum.append(point)
# converting array to numpy array
humus = np.array(humus)
tectum = np.array(tectum)
```
I removed the plotting of the function surface, see [[Curve Fitting]] for an example. Here are the plots: 
```python
ax.scatter(humus[:, 0], humus[:, 1], humus[:, -1], label="humus")
ax.scatter(tectum[:, 0], tectum[:, 1], tectum[:, -1], c = 'red', label="tectum")
# setting labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
# setting limits
axs.set_xlim(centroids[:, 0].min(), centroids[:, 0].max()) # Set x-axis limits
axs.set_ylim(points['y'].min()*5, points['y'].max()*5) # Set y-axis limits
axs.set_zlim(0, 25) # Set z-axis limits
plt.show()
```

| Photo | Explanation |
| ---- | ---- |
| ![[Pasted image 20240118163456.png\|400]] | The problem with this method is that the curve struggles to fit to the shape of the tunnel. The centroids are offset on the y axis to force it to be horizontal. For each new geometry we need to define an axis. It could probably improved by increasing the order of the function.    |
## Using [[Julia]]
The difference between this method and the old one is that I'm using a line instead of a surface.

Basically we want to find closest point a the curve for the points new points that represent point from our lidar point cloud. For the sake of simplicity we say that for all point in point point cloud we have a function such as:

> f(x, y, z) -> (x', y', z')

Where:
- x, y, z are the cartesian coordinates of our point in the point cloud
- (x', y', z') are the cartesian coordinates of the closest point on our curve
Because the centroids should be in the center of the tunnel we can affirm that:
- if z < z' then the point belongs to the floor of the gallery
- if z > z' then the point belongs to the roof of the gallery

We start by loading our blender datasets: 
```julia
begin 
	# TODO: allow the user to choose a filepath
	# open ply files
	# path = "/home/throgg/Documents/Lasalle/cavernes/caverne3.ply"
	ply = load_ply(path)
	# hcat is really cool
	ply_points = hcat(ply["vertex"]["x"], ply["vertex"]["y"], ply["vertex"]["z"])
	print()
end
```
The points can then be plotted using this code, the alpha and beta values are simply variables that the user can change using [[Pluto]]IU, they can be replaced using numbers: 
```julia
begin
	# seperated plotting from opening data, maybe it will be faster
	scatter(ply_points[:, 1], ply_points[:, 2], ply_points[:, 3], 
		markersize = sizemarker,
		camera = (α2, β2),
		markeralpha=0.5
	)
end
```
It yields the following result: ![[Pasted image 20240118165036.png]]
Then we cluster the points and extract the centroids using the following code: 
```julia
begin 
	# cluster X into 20 clusters using K-means
	R = kmeans(ply_points', numclusters)
	@assert nclusters(R) == numclusters # verify the number of clusters
	centers = R.centers' # transposing it
	sorted = sort_by(centers, sortaxis)
	print() # print just stops the cell from printing
end
```
Here is the sorted function, it will sort values based on an axis. It's simpler than the python implementation done earlier. The sorting is done using the **sortperm** function: 
```julia
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
```
The result can then be plotted:

|  |  |
| ---- | ---- |
| ![[Pasted image 20240118165752.png\|300]] | ![[Pasted image 20240118165951.png]] |
| As we can see the centroids are in the center of the tunnel.  | Without the the point clouds we can see 2 plots, the green one is ordered after going through the **sort_by** function. The orange one is the order before sorting. |

### Sorting the points
![[Geoquarry 2024-01-23 15.04.26.excalidraw]]
Now we need to sort the points in the point cloud. In the current example we want to divide them up into points belonging to floor and to the roof of the gallery. The idea is simply to compare the z value of the closest point on the curve with the initial point. We start by creating a new list called classified: 
```julia
classified = classify_points(xyz2, ply_points)
```
This list is a 1D matrix containint either the "roof" or "floor" string that is generated using the following function: 
```julia
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
```
closest gets the nearest point in the line dataset. The function then compares the z value of the 2 points to see which is above the other. This allows them to be classified. 

After, we create the a vector that is a horizontal concatenation of ply_points and classified. The matrix looks like this: 
> ((x, y, z, classification), (xn, yn, zn, classification_n))
Filtering the lists by this third column was pretty hard. After that all that remains to do is to plot it. 
```julia
begin 
	roof = Vector{Float64}()
	a = hcat(ply_points, classified)
	sol = a[(a[:,4] .== "floor") ,1:3]
	roof = a[(a[:,4] .== "roof") ,1:3]

	thisplot = plot(
		camera = (α4, β4)
	)
	
	if dispfloor4==true
		scatter!(sol[:, 1], sol[:, 2], sol[:, 3],
			markersize=2,
			alpha=0.3,
			label="sol",
			color="red"
		)
	end
	if disproof4==true
		scatter!(roof[:, 1],
			roof[:, 2],
			roof[:, 3],
			markersize=2,
			alpha=0.3,
			label="toit",
			color="green"
		)
	end
	# this makes sure that is is shown
	thisplot
end
```
This yields the following result: 

| Cave 1 | Cave 2 | Cave 3 |
| ---- | ---- | ---- |
| ![[Pasted image 20240118185736.png\|300]] | ![[Pasted image 20240118185816.png\|300]]| ![[Pasted image 20240118185852.png\|300]]
We notice that: 
- It adapts to all the case studies
- The order of the centroid list does not seem to have any effect on the quality of the results, maybe it can work on more complex shapes
# Making a depth map
The final step is to make a depth map, like an mnt ([[Plotting Mnt]]) from the points making up the floor. ![[Geoquarry 2024-01-22 13.44.29.excalidraw]] The overall process is illustrated in the image above, note that when there are no values in a given cell it is equal to NaN.
Here is how it translates in [[Julia]] code: 

First we start by rescaling the point cloud from our sol dataset to fit the grid.
```julia
begin 
	using DataFrames; 
	
	# getting extent of floor variable
	maxₓ = maximum(sol[:, 1])
	minₓ = minimum(sol[:, 1])
	maxy= maximum(sol[:, 2])
	miny = minimum(sol[:, 2])
```
### Not strecth image
To not strech the image, the user only chooses grid width. 
```julia
	grid_width = 50
	grid_height = aspect((maxy-miny), (maxₓ-minₓ), grid_width)

	# normalize each point
	normalized_x = [minmax(0, grid_width, x, minₓ, maxₓ) for x in sol[:, 1]]
	normalized_y = [minmax(0, grid_height, x, miny, maxy) for x in sol[:, 2]]

	normalized = DataFrame(:x=>normalized_x, :y=>normalized_y, :z=>sol[:, 3])
```
Grid height is calculated using a function: 
```julia
function aspect(original_height, original_width, resized_width)::Int64
	resized_height = (original_height/original_width) * resized_width
	return floor(resized_height)
end
```
The values of x and y are then normalized over this interval and we make a [[Julia DataFrame]]. This eases helps with filtering.
### Filling the grid 
Now we need to find the average value of the points in each cell. We do this with a for loop. 
```julia
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
```
### Plotting the result
```julia
Plots.heatmap(grid, aspect_ratio=:equal)
```
![[Pasted image 20240122141011.png]]