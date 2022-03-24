# Geometry-Aware Supertagging with Heterogeneous Dynamic Convolutions

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/geometry-aware-supertagging-with/ccg-supertagging-on-ccgbank)](https://paperswithcode.com/sota/ccg-supertagging-on-ccgbank?p=geometry-aware-supertagging-with)

This is our code for the paper [Geometry-Aware Supertagging with Heterogeneous Dynamic Convolutions](https://arxiv.org/abs/2203.12235).

## Citing
While unpublished, you can cite the arxiv preprint:
```latex
@misc{https://doi.org/10.48550/arxiv.2203.12235,
  doi = {10.48550/ARXIV.2203.12235},  
  url = {https://arxiv.org/abs/2203.12235},
  author = {Kogkalidis, Konstantinos and Moortgat, Michael},
  keywords = {Computation and Language (cs.CL), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Geometry-Aware Supertagging with Heterogeneous Dynamic Convolutions},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}

```

## About
The model presents a new approach to constructive supertagging, based on an explicit graph representation that accounts
for both the _intra_-tree interactions between a single supertag and the _inter_-tree interactions between (partially decoded) supertag
sequences.
To account for the disparity between the various modalities in the graph 
(i.e. entential word order, subword contextualized vectors, tree-sequence order and intra-tree edges) we adopt a 
heterogeneous formulation.
Decoding is performed in parallel over trees, each temporal step associated with an increased tree depth.
Statefulness is achieved by representing each partially decoded tree with a single state-tracking vector, which is 
updated twice at each step: once with feedback from its own tree's last decoded fringe, and once with feedback
from surrounding trees.
The result is a highly parallel yet partially auto-regressive architecture with input-scaling memory complexity and 
near-constant decoding time that achieves new state-of-the-art scores on four datasets while retaining the ability to 
predict rare supertags reliably.

## Results
Averages of 6 repetitions compared to recent results on the respective datasets (crawled 23/03/2022).

<table>
    <tr>
        <td><center><b>Model</b></center></td>
        <td><center><b>Accuracy</b></center></td>
        <td><center><b>Frequent</b> (100+)</center></td>
        <td><center><b>Uncommon</b> (10-99)</center></td>
        <td><center><b>Rare</b> (1-9)</center></td>
        <td><center><b>OOV</b></center></td>
    </tr>
    <tr>
        <td colspan="6"></td>
    </tr>
    <tr>
        <td colspan="6"><center><b><i>CCGbank</i></b></center></td>
    </tr>
    <tr>
        <td><a href="https://github.com/cuhksz-nlp/NeST-CCG">Attentive Convolutions</a> </td>
        <td>96.25</td>
        <td>96.64</td>
        <td>71.04</td>
        <td>n/a</td>
        <td>n/a</td>
    </tr>
    <tr>
        <td><i>Ours</i></td>
        <td><b>96.29</b></td>
        <td>96.61</td>
        <td>72.06</td>
        <td>34.45</td>
        <td>4.55</td>
    </tr>
    <tr>
        <td colspan="6"></td>
    </tr>
    <tr>
        <td colspan="6"><center><b><i>CCGrebank</i></b></center></td>
    </tr>
    <tr>
        <td><a href="https://github.com/jakpra/treeconstructive-supertagging">Recursive Tree Addressing</a></td>
        <td>94.70</td>
        <td>95.11</td>
        <td>68.86</td>
        <td>36.76</td>
        <td>4.94</td>
    </tr>
    <tr>
        <td><i>Ours</i></td>
        <td><b>95.07</b></td>
        <td>95.45</td>
        <td>71.06</td>
        <td>34.45</td>
        <td>4.55</td>
    </tr>
    <tr>
        <td colspan="6"></td>
    </tr>
    <tr>
        <td colspan="6"><center><b><i>French TLGbank</i></b></center></td>
    </tr>
    <tr>
        <td><a href="https://richardmoot.github.io/Slides/WoLLIC2019.pdf">ELMO & LSTM</a></td>
        <td>93.20</td>
        <td>95.10</td>
        <td>75.19</td>
        <td>25.85</td>
        <td>n/a</td>
    </tr>
    <tr>
        <td><i>Ours</i></td>
        <td><b>95.92</b></td>
        <td>96.40</td>
        <td>81.48</td>
        <td>55.37</td>
        <td>7.25</td>
    </tr>
    <tr>
        <td colspan="6"></td>
    </tr>
    <tr>
        <td colspan="6"><center><b><i>Æthel</i></b></center></td>
    </tr>
    <tr>
        <td><a href="https://github.com/konstantinosKokos/neural-proof-nets">Symbol-Sequential Transformer</a></td>
        <td>83.67</td>
        <td>84.55</td>
        <td>64.70</td>
        <td>50.58</td>
        <td>24.55</td>
    </tr>
    <tr>
        <td><i>Ours</i></td>
        <td><b>93.67</b></td>
        <td>94.83</td>
        <td>73.45</td>
        <td>53.83</td>
        <td>15.79</td>
</table>

## Project Structure
`dyngraphst.neural` contains the model architecture , and `dyngraphst.data` contains the data preprocessing code; see 
the READMEs of the respective directories for more details.
While anticipating the next stable release of [PyTorchGeometric](https://pytorch-geometric.readthedocs.io/en/latest/),
the code will require two distinct python environments: python3.9 with `pytorch 1.10.2` and 
`torch_geometric 2.0.3` for training/inference, and python3.10 for data processing and evaluation.
The two will be coallesced at a later stage.

## How-to
Detailed instructions coming soon. 

## Contact & Support
If you have any questions or comments or would like a grammar/language specific pretrained model,
feel free to [get in touch](k.kogkalidis@uu.nl).