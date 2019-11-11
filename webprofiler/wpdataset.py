# Import
import json
import numpy as np
import os

# Class 'WPDataset'
class WPDataset:
    """ A WebProfiler dataset which can generate train, validation, and test sets from a preprocessed profile log. """

    # Constants
    TRAIN_RATIO = 0.7
    VALID_TEST_RATIO = 0.5 # out of (1 - TRAIN_RATIO); 0.5 means half and half
    META_FILE_PREFIX = "metadata_"

    def __init__(self, root_dir="./", log_id=None, random_seed=42, use_embedding=False):
        """ (WPDataset, str, str, int, bool) -> NoneType
        Create a new dataset with a profile log ID.
        """
        # Initialize attributes
        self.root_dir = root_dir
        if log_id == None:
            log_id = input("Log ID: ")
        self.log_id = log_id
        self.random_seed = random_seed
        self.use_embedding = use_embedding
        
        # Read metadata and dictionary files with the profile log ID
        print("Read metadata and dictionary files...", end = ' ')
        META_PATH = os.path.join(self.root_dir, self.META_FILE_PREFIX + self.log_id + ".txt")
        self.dict_metadata = json.load(open(META_PATH, 'r', encoding='UTF8'))
        DICT_PATH = os.path.join(self.root_dir,  self.dict_metadata["dictionary"])
        self.dict_event_code = json.load(open(DICT_PATH, 'r', encoding='UTF8'))
        print("done!")

        # Read input, input length, and output files
        print("Read input and output files...", end = ' ')
        INPUT_PATH = os.path.join(self.root_dir,  self.dict_metadata["input"])
        INPUT_LEN_PATH = os.path.join(self.root_dir,  self.dict_metadata["input_len"])
        OUTPUT_PATH = os.path.join(self.root_dir,  self.dict_metadata["output"])
        self.arr_X = np.loadtxt(INPUT_PATH, dtype=np.uint16, delimiter=',', encoding='UTF8').reshape(-1, self.dict_metadata["n_steps"], 1)
        self.arr_Xlen = np.loadtxt(INPUT_LEN_PATH, dtype=np.uint16, delimiter=',', encoding='UTF8')
        self.arr_y = np.loadtxt(OUTPUT_PATH, dtype=np.uint16, delimiter=',', encoding='UTF8')
        print("done!")

        # Re-generate input data, if Web embedding enabled; otherwise, normalize input data with the maximum event ID
        self.arr_X_raw = self.arr_X
        if self.use_embedding == True: # embedding
            EMBED_PATH = os.path.join(self.root_dir,  self.dict_metadata["embedding"])
            self.embeddings = np.load(EMBED_PATH)

            arr_X_embedded = list()
            for sample in self.arr_X_raw:
                new_sample = list()
                for event in sample:
                    new_sample.append(self.embeddings[event].reshape(-1))
                arr_X_embedded.append(new_sample)
            self.arr_X = np.array(arr_X_embedded)
        else: # Integerization with division by max(event ID)
            max_X = float(max(self.dict_event_code.values()))
            self.arr_X = (self.arr_X / max_X).astype(np.float32)
        
        # Generate an output dictionary
        self.dict_output = dict()
        output_count = 0
        for key in self.dict_metadata["label_list"]:
            self.dict_output[np.uint16(key)] = output_count
            output_count = output_count + 1
        arr_coded = list()
        for y in self.arr_y:
            arr_coded.append(self.dict_output[y])
        self.arr_y_coded = np.array(arr_coded)
    
    def load_data(self):
        """ (WPDataset, str) -> (numpy.ndarray, numpy.ndarray), (numpy.ndarray, numpy.ndarray), (numpy.ndarray, numpy.ndarray)
        Split the dataset into train/validation/test sets and return them.
        """
        # Initialize the random seed
        np.random.seed(self.random_seed)

        # Split the dataset into train/validation/test sets
        data_size = self.dict_metadata["sample_num"]
        shuffled_indices = np.random.permutation(data_size)
        train_set_size = int(data_size * self.TRAIN_RATIO)
        train_indices = shuffled_indices[:train_set_size]
        valid_test_indices = shuffled_indices[train_set_size:]
        valid_set_size = int(len(valid_test_indices) * self.VALID_TEST_RATIO)
        valid_indices = valid_test_indices[:valid_set_size]
        test_indices = valid_test_indices[valid_set_size:]

        # Return
        return (self.arr_X[train_indices], self.arr_Xlen[train_indices], self.arr_y_coded[train_indices]), (self.arr_X[valid_indices], self.arr_Xlen[valid_indices], self.arr_y_coded[valid_indices]), (self.arr_X[test_indices], self.arr_Xlen[test_indices], self.arr_y_coded[test_indices])
    
    def load_raw_data(self):
        """ (WPDataset, str) -> (numpy.ndarray, numpy.ndarray), (numpy.ndarray, numpy.ndarray), (numpy.ndarray, numpy.ndarray)
        Split the dataset into train/validation/test sets and return them without any input normalization or Web embedding.
        """
        # Initialize the random seed
        np.random.seed(self.random_seed)

        # Split the dataset into train/validation/test sets
        data_size = self.dict_metadata["sample_num"]
        shuffled_indices = np.random.permutation(data_size)
        train_set_size = int(data_size * self.TRAIN_RATIO)
        train_indices = shuffled_indices[:train_set_size]
        valid_test_indices = shuffled_indices[train_set_size:]
        valid_set_size = int(len(valid_test_indices) * self.VALID_TEST_RATIO)
        valid_indices = valid_test_indices[:valid_set_size]
        test_indices = valid_test_indices[valid_set_size:]

        # Return
        return (self.arr_X_raw[train_indices], self.arr_Xlen[train_indices], self.arr_y_coded[train_indices]), (self.arr_X_raw[valid_indices], self.arr_Xlen[valid_indices], self.arr_y_coded[valid_indices]), (self.arr_X_raw[test_indices], self.arr_Xlen[test_indices], self.arr_y_coded[test_indices])
