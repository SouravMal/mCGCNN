# mCGCNN

mCGCNN is a dual-stream crystal graph convolutional neural network for the efficient prediction of magnetic properties of crystalline materials. It augments the full structural graph with a dedicated magnetic subgraph. The magnetic stream performs angle-aware message passing over magnetic centers using metal-ligand-metal exchange-path descriptors motivated by Goodenough-Kanamori-Anderson physics, while layer-wise cross-coupling transfers structural and ligand-field information from the full crystal graph. A separate magnetic-sublattice pooling operation prevents the magnetic interaction from being diluted by nonmagnetic atoms. The incorporation of exchange geometry directly into graph architectures provides a physically grounded route to predictive models of magnetic materials.


## Dual graph representation in the mCGCNN architecture
<img src="images/mcgcnn-graph.png" alt="Dual graph schematic" width="800">


## Architecture of mCGCNN
<img src="images/mcgcnn-arch.png" alt="mcgcnn architecture schematic" width="800">


## Generate the Dual Graphs (Crystal Graph + Magnetic Subgraph)

Before training or inference, the crystal structures must be converted into dual graph representations consisting of **crystal graph** and its corresponding **magnetic subgraph**.

### Dataset Organization

Organize your dataset as follows:

```text
my_dataset/
├── dataset.csv
├── atom_init.json
├── magnetic_atom_init.json
├── material_1.cif
├── material_2.cif
├── material_3.cif
└── ...
```
where
* `dataset.csv` contains the material IDs and target properties.
* `atom_init.json` contains the elemental feature vectors as used in CGCNN.
* `magnetic_atom_init.json` contains the magnetic elemental feature vectors.
* `*.cif` files contain the crystal structures.  

The first column of `dataset.csv` must contain the material IDs, which **must exactly match** the corresponding CIF filenames (without the `.cif` extension).

For example,

| **material_id** | **tot_mom_mub** |
| :--- | :--- |
| mp-100 | 5.82 |
| mp-101 | 1.37 |
| mp-102 | 0.00 |

where **tot_mom_mub** denotes the DFT total magnetic moment per unit cell in $\mu_B$.

### Generate the Graphs

Run
```bash
python preprocess.py my_dataset --target tot_mom_mub 
```
By default, the generated graph files are written to
my_dataset/
├── processed_graphs/
│   ├── mp-100.pt
│   ├── mp-101.pt
│   ├── mp-102.pt
│   └── ...


## How to genearte the dual graph (crystal graph + corresponding magnetic subgraph)?

We have to run the following command:

```bash
python preprocess.py data/sample --target tot_mom_mub --workers 8
```


## License

This project is licensed under the **MIT License**.

See the [LICENSE](LICENSE) file for details.


## Citation

Please consider citing our work if you find it helpful:

```bibtex
@misc{mal2026mcgcnndualstreamcrystalgraph,
      title={mCGCNN: A Dual-Stream Crystal Graph Convolutional Neural Network for the Efficient Prediction of Magnetic Properties of Crystalline Materials}, 
      author={Sourav Mal and Satadeep Bhattacharjee},
      year={2026},
      eprint={2606.28458},
      archivePrefix={arXiv},
      primaryClass={cond-mat.mtrl-sci},
      url={https://arxiv.org/abs/2606.28458}, 
}
```


