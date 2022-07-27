### neural
Requires: pytorch 1.10.2, torch-geometric 2.0.3.

* `batching` contains utilities for converting vectorized samples into a single batch.
* `embedding` implements an invertible embedder.
* `encoder` wraps a BERT-model with a pooling and a normalization layer.
* `loss` contains the loss function for the neural model (redundant, likely to be removed).
* `model` wraps components into a single decoderlayer (front-end functions will be implemented there).
* `train` contains the training loop for the neural model.
* `tree_decoding` implements each message passing component as a standalone class.
* `utils` contains utility functions and modules for the neural model.
