# mCGCNN

mCGCNN is a dual-stream crystal graph convolutional neural network for the efficient prediction of magnetic properties of crystalline materials. It augments the full structural graph with a dedicated magnetic subgraph. The magnetic stream performs angle-aware message passing over magnetic centers using metal-ligand-metal exchange-path descriptors motivated by Goodenough-Kanamori-Anderson physics, while layer-wise cross-coupling transfers structural and ligand-field information from the full crystal graph. A separate magnetic-sublattice pooling operation prevents the magnetic interaction from being diluted by nonmagnetic atoms. The incorporation of exchange geometry directly into graph architectures provides a physically grounded route to predictive models of magnetic materials.


## Dual graph representation in the mCGCNN architecture
<img src="images/mcgcnn-graph.png" alt="Dual graph schematic" width="800">

---

## Architecture of mCGCNN
<img src="images/mcgcnn-arch.png" alt="mcgcnn architecture schematic" width="800">

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


