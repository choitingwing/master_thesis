import datasets

project_name = "nuNet"
run_version = "runModel"
dataset_name = "SouthPole"

# Dataset setup
# Call Dataset(dataset_name, em, noise) with
#     dataset_name:
#         ALVAREZ (only had + noise) / ARZ
#     em: (True means em+had, False means had)
#         True / False (default)
#     noise:
#         True (default) / False
dataset_name = "ARZ"
dataset_em = False
dataset_noise = True

dataset = datasets.Dataset(dataset_name, dataset_em, dataset_noise)

# Paths
datapath = dataset.datapath
data_filename = dataset.data_filename
label_filename = dataset.label_filename

# numbers
n_files = dataset.n_files
n_files_val = dataset.n_files_val
test_file_ids = dataset.test_file_ids

#running stuff

epochs = 25
norm = 1e-6

train_files = 33
val_files = 5
test_files = 3
train_data_points = 3290000   
val_data_points = 490000     
test_data_points = 299995  

# train_files = 2
# val_files = 1
# test_files = 1
# train_data_points = 190000   
# val_data_points = 49000    
# test_data_points = 20000 

sample_numbers = 10000

learning_rate = 0.00005
es_patience = 8
es_min_delta = 0.0001 # Old value: es_min_delta = 0.0001
batchSize = 64

# Directories
plots_dir = "plots"
