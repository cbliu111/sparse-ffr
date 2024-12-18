### Pipline
1. label cell images using labelme
2. `json_to_contour.py`, turn json file into contours
3. `contour_to_adata.py`, collect contours and stored into anndata file ended with `.h5ad`
4. `compute_gw_with_coupling.py`, compute pairwise Gromov-Wasserstein distances between cells
5. `embedding.py`, compute nearest neighbors and embed cells into low dimensional space
6. `clustering.py`, identify macro-states
7. `gaussian_mixture.py`, represents the macro-state density using Gaussian mixture model
8. `latent_time.py`, compute latent time through Waddington optimal transport and diffusion pseudotime algorithm
9. `vector_field.py`, compute flows using latent time and transition matrix, compute scores using density gradient, vector field is obtained afterward.
10. `simulate.py`, simulate the Langevin trajectory ensemble based on the learned vector field
11. `landscape.py`, plot the landscape as the logarithm of the steady-state distribution
12. `mode_shape.py` and `mode_distribution.py`, are used to calculate the cluster cells and obtain the shape modes following vampire [1]
13. `least_action_path.py`, identify the location of basins in the potential landscape, calculate the basin boundaries, plot the shape mode and shape distribution corresponding to each basin, calculate the least action path across basins
14. `react_coord.py`, identify the reaction coordinates through the least action path, plot small, middle and large noise landscape and potential along reaction coordinates
15. `thermo_relation.py`, calculate the thermodynamical relationship inferred from the vector field. 





[1] Phillip, J. M., Han, K.-S., Chen, W.-C., Wirtz, D. & Wu, P.-H. A robust unsupervised machine-learning method to quantify the morphological heterogeneity of cells and nuclei. Nat Protoc 16, 754–774 (2021).
