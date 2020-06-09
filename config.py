DATA = "data/"
DATA_POINT_CLOUD = DATA + "modelnet40_ply_hdf5_2048"
TRAINED_MODEL = "trained_models/"

NUM_POINTS = 2048
NUM_CLASSES = 40


PARAMS = {
    "batch_size": 32,
    "max_epoch": 250, 
    "learning_rate": 0.001,
    "momentum": 0.9,
    "optimizer": "Adam", 
    "decay_step": 200000,
    "decay_rate": 0.7
}
