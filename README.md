# Cryobiology_team1

The project is aimed at automating the analysis of 2D cell cultures by processing microimages.

The system provides a full cycle of processing brightfield and fluorescent data, which includes:
* **Classification:** automatic determination of morphological cell types.
* **Clustering:** automatically binds vesicles to nuclei.
* **Segmentation:** isolation of intracellular structures (nuclei, vesicles)
* **Metric analysis:** calculation of quantitative characteristics (area and linear dimensions).

### Cell shape - classification of cell shapes
This script is designed to analyze cell segmentation masks (in .npy format). It calculates the morphological metrics of each cell and classifies them by shape, then visualizes the result by superimposing colored contours on the original image.

### Clustering - wavefront vesicle clustering algorithm
This script implements a wavefront algorithm for binding vesicles to cell nuclei based on spatial proximity and pre-computed segmentation masks.
