# mCGCNN

mCGCNN is a dual-stream crystal graph convolutional neural network for the efficient prediction of magnetic properties of crystalline materials. It augments the full structural graph with a dedicated magnetic subgraph. The magnetic stream performs angle-aware message passing over magnetic centers using metal-ligand-metal exchange-path descriptors motivated by Goodenough-Kanamori-Anderson physics, while layer-wise cross-coupling transfers structural and ligand-field information from the full crystal graph. A separate magnetic-sublattice pooling operation prevents the magnetic interaction from being diluted by nonmagnetic atoms. The incorporation of exchange geometry directly into graph architectures provides a physically grounded route to predictive models of magnetic materials.


## Dual graph representation in the mCGCNN architecture
<img src="images/mcgcnn-graph.png" alt="Dual graph schematic" width="550">

---

## Architecture of mCGCNN
<img src="images/mcgcnn-arch.png" alt="mcgcnn architecture schematic" width="550">

