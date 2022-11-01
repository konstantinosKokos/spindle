# Geometry-Aware Supertagging meets Neural Proof Nets

This repository combines
* https://github.com/konstantinosKokos/dynamic-graph-supertagging/
and
* https://github.com/konstantinosKokos/neural-proof-nets/

into a single package for parsing Dutch into syntactic proofs of multiplicative intuitionuistic linear logic with 
dependency modalities, and the equivalent semantic λ-terms.

---

## How-To
Pending better packaging, installation involves the following steps:

> ### 1. Create a local clone of this repository
>   ```
>   git clone git@github.com:konstantinosKokos/dynamic-proof-nets.git
>  ``` 
> ### 2. Create a local clone of lassy-tlg-extraction
>   ```
>   git clone git@github.com:konstantinosKokos/lassy-tlg-extraction.git
>   ```
>   If you're having compatibility issues, you can revert to the last tested commit:
>   ```
>   git checkout -b safe e9ddab53d8013eb9271628ae1e01a241949702be
>   ```
> ### 3. Link the two 
>   Walk into the `dynamic-proof-nets` directory and make a symlink to the LassyExtraction module.
>   ```
>   cd dynamic-proof-nets
>   ln -s ../lassy-tlg-extraction/LassyExtraction ./LassyExtraction
>   ```
> ### 4. Prepare your environment
>   * #### Setup and source a **fresh** python 3.10 environment, for instance using conda. 
>     ```
>      conda create -n [VENV_NAME] python=3.10
>      conda activate [VENV_NAME]
>      ```
>   * #### Install PyTorch 1.11 and opt_einsum
>     ```
>     conda install pytorch==1.11.0 -c pytorch
>     conda install opt_einsum -c conda-forge
>     ```
>   * #### Install Transformers
>     ```
>     pip install transformers
>     ```
>   * #### Finally, install PyTorch Geometric.
>    If you're lucky, this should work:
>    ```
>     conda install pyg -c pyg
>    ```
>    Chances are it won't.
>    If it doesn't, refer to the [installation instructions](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).
>    The usual way out would look something like:
>    ```
>     pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.11.0+cpu.html
>    ```
>    Where you'd need to substitute for your own system's cuda version if you have one.
> ### 5. Download pretrained model weights
>   These can be found [here](https://surfdrive.surf.nl/files/index.php/s/tJ2Htq6NhgTmtvP).
>  Extract and place them in the `data` directory.

You're good to go!
Parse your first sentences as follows:
```python
from inference import InferenceWrapper
inferer = InferenceWrapper(weight_path='./data/model_weights.pt',
                           atom_map_path='./data/atom_map.tsv',
                           config_path='./data/bert_config.json', 
                           device='cuda')  # replace with 'cpu' if no GPU accelaration
analyses = inferer.analyze(['Dit is een vorbeeldzin'])
```

If you want to inspect examples outside the console, you can compile proofs into TeX using the extraction code:
```python
from LassyExtraction.utils.tex import compile_tex, sample_to_tex
compile_tex(sample_to_tex(sample=...,                   # an Analysis object
                          show_intermediate_terms=...,  # bool
                          show_words_at_leaves=...,     # bool
                          show_sentence=...,            # bool
                          show_final_term=...,))        # bool
```

---

## Citing
Please cite the following two papers if you use (parts of) this repository for your own work.

If you use the package as an end-to-end parser, also cite this repository.

> ### Geometry-Aware Supertagging with Heterogeneous Dynamic Convolutions
> ```latex
> @misc{https://doi.org/10.48550/arxiv.2203.12235,
>   doi = {10.48550/ARXIV.2203.12235},  
>  url = {https://arxiv.org/abs/2203.12235},
>  author = {Kogkalidis, Konstantinos and Moortgat, Michael},
>  keywords = {Computation and Language (cs.CL), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
>  title = {Geometry-Aware Supertagging with Heterogeneous Dynamic Convolutions},
>  publisher = {arXiv},
>  year = {2022},
>  copyright = {Creative Commons Attribution 4.0 International}
> }
> ```
> ### Neural Proof Nets
> ```latex
> @inproceedings{kogkalidis-etal-2020-neural,
>    title = "Neural Proof Nets",
>    author = "Kogkalidis, Konstantinos  and
>      Moortgat, Michael  and
>      Moot, Richard",
>    booktitle = "Proceedings of the 24th Conference on Computational Natural Language Learning",
>    month = nov,
>    year = "2020",
>    address = "Online",
>    publisher = "Association for Computational Linguistics",
>    url = "https://aclanthology.org/2020.conll-1.3",
>    doi = "10.18653/v1/2020.conll-1.3",
>    pages = "26--40",
> }
> ```
> ### ÆTHEL
> ```latex
> @inproceedings{kogkalidis2020aethel,
>  title={{\AE}THEL: Automatically Extracted Typelogical Derivations for Dutch},
>  author={Kogkalidis, Konstantinos and Moortgat, Michael and Moot, Richard},
>  booktitle={Proceedings of the 12th Language Resources and Evaluation Conference},
>  pages={5257--5266},
>  year={2020}
> }
> ```
> ### This repository
> ```latex
> @software{dynamic-proof-nets,
>   title = {{Dynamic Proof Nets}}
>   author = {Kogkalidis, Konstantinos},
> }
> ```

---

## Contact & Support
If you have any questions or comments or encounter any difficulties, please feel free to [get in touch](k.kogkalidis@uu.nl).
