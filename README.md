
# Methods

- The prepare_dataset() method of the [a_preprocess_text.py](a_preprocess_text.py) file prepares the original dataset for simulation by preprocessing the text and generating additional features. Note: the actions carried out in this method will depend on the characteristics of the experimental data set; by default, it is optimized for the Yelp data set. 

- The incremental_features() method of the [b_incremental_features.py](b_incremental_features.py) file will compute the accumulated feature values to start the simulation.

- The balance_and_merge_datasets() method of the [c_downsampling_dataset.py](c_downsampling_dataset.py) file balances the classes and creates one experimental file for execution in sequential mode and twenty experimental files for parallel evaluation.

- The run_paralell_experiment() and run_one_experiment() methods in the [d_ML_streaming.py](d_ML_streaming.py) file execute the model training by applying the data drift techniques described in the article.

- The experiment_run() method of the [d_ML_streaming.py](d_ML_streaming.py) file runs a complete test using the parameters configured in the config_exp_params dictionary.

- The drift_detection() method of the [machine_learning_utils.py](utils%2Fmachine_learning_utils.py) file contains the core code for drift detection; it is parametrizable in terms of threshold values that limit the number of samples in the sliding windows and the maximum p_value value.

# Citation
If you use this code, please cite the following reference.
```text
@article{arriba2024,
  title={Online detection and infographic explanation of spam reviews with data drift adaptation},
  author={de Arriba-Pérez, Francisco and García-Méndez, Silvia and Leal, Fátima and Malheiro, Benedita and Burguillo, Juan C.},
  journal={Under review},
  year={2024}
}
```
