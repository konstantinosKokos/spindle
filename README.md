# :spider: :spider_web: spind<sup>2</sup>λe
> **S**pindle **P**arses **In**to **D**ependency-**D**ecorated **λ** **E**xpressions 

Use to parse Dutch sentences into deep syntactic parses expressed as proofs/terms of multiplicative intuitionuistic linear logic with 
dependency modalities.

---

## :computer: At Home

Try out the online version at the link below (wip):

>https://parseport.hum.uu.nl/spindle

---

## :wrench: How-To

Pending better packaging, installation involves the following steps:

> ### 1. Create a local clone of this repository
>   ```
>   git clone git@github.com:konstantinosKokos/spindle.git
>  ```
> ### 2. Prepare your environment
>   * #### Setup and source a **fresh** python 3.10 environment, for instance using conda. 
>     ```
>      conda create -n [VENV_NAME] python=3.10
>      conda activate [VENV_NAME]
>      ```
>   * #### Install [aethel](https://github.com//konstantinosKokos/aethel)
>     ``` 
>       pip install git+https://github.com/konstantinosKokos/aethel@stable
>     ```
>   * #### Install PyTorch 1.11 and opt_einsum
>     ```
>     conda install pytorch==1.11.0 -c pytorch
>     conda install opt_einsum -c conda-forge
>     ```
>   * #### Install Transformers
>     ```
>     pip install transformers==4.20.1
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
> ### 3. Download pretrained model weights
>   These can be found [here](https://www.dropbox.com/scl/fi/osdraor5zzcf43l2q001z/model_weights.zip?rlkey=e1bcfbji93ug4wcmx9zrlfqo0&dl=0).
>  Extract and place them in the `data` directory.

You're good to go!
Parse your first sentences as follows:
```python
from inference import InferenceWrapper
inferer = InferenceWrapper(weight_path='./data/model_weights.pt',
                           atom_map_path='./data/atom_map.tsv',
                           config_path='./data/bert_config.json', 
                           device='cuda')  # replace with 'cpu' if no GPU accelaration
analyses = inferer.analyze(['Dit is een voοrbeeldzin'])
```

If you want to inspect examples outside the console, you can compile proofs into TeX using the extraction code:
```python
from aethel.utils.tex import compile_tex, sample_to_tex
compile_tex(sample_to_tex(sample=...,                   # an Analysis object
                          show_intermediate_terms=...,  # bool
                          show_words_at_leaves=...,     # bool
                          show_sentence=...,            # bool
                          show_final_term=...,))        # bool
```

---

## :notebook: Citing
Please cite the following [paper](https://arxiv.org/abs/2302.12050) if you use spindle:

```bibtex
@inproceedings{spindle,
    title = "{SPINDLE}: Spinning Raw Text into Lambda Terms with Graph Attention",
    author = {Kogkalidis, Konstantinos  and
		Moortgat, Michael and
		Moot, Richard},
	booktitle = "Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics: System Demonstrations",
	month = may,
	year = "2023",
 	address = "Dubrovnik, Croatia",
    publisher = "Association for Computational Linguistics",
```

---

## :question: Contact & Support
If you have any questions or comments or encounter any difficulties, please feel free to [get in touch](k.kogkalidis@uu.nl),
or [open an issue](https://github.com/konstantinosKokos/spindle/issues/new/choose).
