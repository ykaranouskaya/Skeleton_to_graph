# Skeleton_to_graph
A script that allows to convert 3D skeleton into a graph

The file skel_to_graph.py computes the reduced graph and allows to get rid of artificial loops. It seems to work quite good for the skeletons, that contain just the branches and the vertices. However, our structures also creates the "bubbles" in the skeleton (the image is attached) and in this case the graph are quite complicated to draw them and to check the result. I add the graphs (reduced and without loops) for the small volume (small npy array) that are quite good. Unfortunately, for large volumes the adjacency matrix is calculated really slowly, and the problem can once again raise from these "bubbles" in the initial skeleton.
