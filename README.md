# ReLU Code Space: A Basis for Rating Network Quality Besides Accuracy

Provided code can be used for the purpose of reproducibility of the Autoencoder results described in the paper

## How to use the code:

To reproduce results shown in the paper for Autoencoder trained on MNIST data with 2 classes only run `main.py`.
Results will be stored in folder `experiments`:
 - chkpnt_logs - saving training checkpoint
 - info_logs - saving processed code space information
 - plot_logs - saving resultsing plots
 - train_logs - saving training logs
 - weight_logs - saving weights and biases during training procedure
 
 ## How can I modify training?
 If you would like to modify some of the training parameters and see it's affect on the code space you can:
  - (option 1) modify following parametrs in `config\config_AutoEncoder_3l_lr0001_bs_256.py` 
  - (option 2) or create new config file Ex.: `config\config_AutoEncoder_3l_lr01_bs_16.py`
  
 In case of (option 2) please change the names of `experiment_id`, `filepath`, and `plotpath` in `main.py` according to your new config file. 
  
 ### Parameters whcih can be modified in config file:
  - `config.image_size` - image size
  - `config.train_data_file` - dataset for training
  - `config.valid_data_file` - dataset for validation
  - `config.test_data_file` - dataset for test
  - `config.lr` - learning rate
  - `config.batch_size` - batch size
  - `config.num_epochs` - number of epochs
  - `config.num_classes` - number of classes in dataset 
  
 ### Parameters whcih can be modified in `main.py` file:
  - `epoch_list` - for which epochs hamming distance and plot are computed 
  - `num_classes` - should be aligned with number of classes in the config file 
  - `num_epochs` - should be aligned with number of epochs in the config file 
  - `mode_list` - should be computation done for training, validation or both
  
 ## What do functions in the `main.py` do? 
- `training_procedure` - trains given arhitecture architecture 
- `uniqueness_count` - computations done over the code space for every epoch for training and validation, need to be computed once for fully trained network
- `plot_umap`-computes hamming distance and produces UMAP plots for given epoch_list, can be called many time, for every existing epoch as soon as job of  `uniqueness_count` function is finished 
