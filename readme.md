# Support Vecotor Machine pixel classifier for radargrams
**Author:** Jordy Dal Corso, [RSLab](https://rslab-tech.disi.unitn.it), University of Trento, Italy. For comments and corrections write to jordy.dalcorso@unitn.it

With this small codebase we replicate the work of [Ilisei and Bruzzone, 2015](https://ieeexplore.ieee.org/document/7001584). With these scripts we do not aim at computational speed nor perfect matching of the results. We focus on a correct replication of the computation of hand-crafted feature maps for the SVM and few snippets to implement the machine learning pipeline.

Readers should refer to:
* `create_features.py` for the creation and saving of handcrafted features
* `svm_classification.py` for the fitting a testing of the SVM pixel classifier
* `visualise_features.py` to save images of the features generated
Readers should pay attention to the parser arguments, in particular to the setting of correct file and folder paths.

A small report in .pdf format is provided, comprising a minimal analysis on kernel performance and feature importance.