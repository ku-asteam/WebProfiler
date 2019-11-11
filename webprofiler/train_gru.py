# Import
import getopt, sys
import numpy as np
import os
import tensorflow as tf
from wpdataset import WPDataset

# Process arguments
opts, args = getopt.getopt(sys.argv[1:], "", ["train_model=", "log_time_tag=", "dir=", "log_id=", "random_seed=", "use_embedding=", "n_neurons=", "n_layers=", "learning_rate=", "n_epochs=", "batch_size=", "act_func=", "use_peepholes=", "opt_algo=", "top_k_thresh="])
for opt, arg in opts:
    if opt == "--train_model":
        train_model = arg
    elif opt == "--log_time_tag":
        log_time_tag = arg
    elif opt == "--dir":
        log_dir = arg
    elif opt == "--log_id":
        log_id = arg
    elif opt == "--random_seed":
        random_seed = int(arg)
    elif opt == "--use_embedding":
        if arg == "True":
            use_embedding = True
        elif arg == "False":
            use_embedding = False
    elif opt == "--n_neurons":
        n_neurons = int(arg) # equal for all layers/cells
    elif opt == "--n_layers":
        n_layers = int(arg)
    elif opt == "--learning_rate":
        learning_rate = float(arg)
    elif opt == "--n_epochs":
        n_epochs = int(arg)
    elif opt == "--batch_size":
        batch_size = int(arg)
    elif opt == "--act_func":
        act_func = arg
    elif opt == "--use_peepholes":
        if arg == "True":
            _ = True
        elif arg == "False":
            _ = False
    elif opt == "--opt_algo":
        opt_algo = arg
    elif opt == "--top_k_thresh":
        top_k_thresh = int(arg)
    else:
        pass

# Reset all pseudo numbers
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
def reset_graph(seed):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)
reset_graph(random_seed)

# Load preprocessed profile log data and make train/validation/test sets
wp_dataset = WPDataset(log_dir, log_id, random_seed, use_embedding)
(X_train, Xlen_train, y_train), (X_valid, Xlen_valid, y_valid), (X_test, Xlen_test, y_test) = wp_dataset.load_data()
Xlen_train = Xlen_train.astype(np.int32)
Xlen_valid = Xlen_valid.astype(np.int32)
Xlen_test = Xlen_test.astype(np.int32)
y_train = y_train.astype(np.int32)
y_valid = y_valid.astype(np.int32)
y_test = y_test.astype(np.int32)

# Set variables for deep neural networks
n_steps = X_train.shape[1]
n_inputs = X_train.shape[2]
n_outputs = wp_dataset.dict_metadata["n_labels"]

print("\n* Variables for training " + train_model)
print(" - len(X_train)|len(X_valid)|len(X_test): {0:d}|{1:d}|{2:d}".format(len(X_train), len(X_valid), len(X_test)))
print(" - n_steps (# of time steps): {0:d}".format(n_steps))
print(" - n_inputs (dimension of inputs): {0:d}".format(n_inputs))
print(" - n_neurons (# of neurons at each cell): {0:d}".format(n_neurons))
print(" - n_layers (# of cells): {0:d}".format(n_layers))
print(" - n_outputs (dimension of outputs): {0:d}".format(n_outputs))
print(" - learning_rate (learning rate for optimizer): {0:f}".format(learning_rate))
print(" - n_epochs (# of epochs to iterate): {0:d}".format(n_epochs))
print(" - batch_size (# of samples in each batch): {0:d}".format(batch_size))

# Set functions for batch construction
def shuffle_batch(X, y, Xlen, batch_size):
	rnd_idx = np.random.permutation(len(X))
	n_batches = len(X) // batch_size # quotient
	for batch_idx in np.array_split(rnd_idx, n_batches):
		X_batch, y_batch, Xlen_batch = X[batch_idx], y[batch_idx], Xlen[batch_idx]
		yield X_batch, y_batch, Xlen_batch


# Define placeholders for input and output variables
X = tf.placeholder(tf.float32, [None, n_steps, n_inputs], name="X")
y = tf.placeholder(tf.int32, [None], name="y")
seq_length = tf.placeholder(tf.int32, [None])
TP = tf.placeholder_with_default(0, shape=(), name="TP")
FP = tf.placeholder_with_default(0, shape=(), name="FP")
FN = tf.placeholder_with_default(0, shape=(), name="FN")
TN = tf.placeholder_with_default(0, shape=(), name="TN")

# Construct neural networks from input layer to output layer
if act_func == "sigmoid":
    activation = tf.keras.activations.sigmoid
elif act_func == "tanh":
    activation = tf.keras.activations.tanh
elif act_func == "relu":
    activation = tf.keras.activations.relu
elif act_func == "elu":
    activation = tf.keras.activations.elu
else: # default is tanh
    act_func = "tanh"
    activation = tf.keras.activations.tanh
print(" - activation: {0:s}".format(act_func))
print(" - initializer: {0:s}".format("He")) # for ReLU and its variants
with tf.variable_scope(train_model, initializer=tf.variance_scaling_initializer()):
    gru_cells = [tf.contrib.rnn.GRUCell(num_units=n_neurons, activation=activation)
				for layer in range(n_layers)]
    multi_layer_cell = tf.contrib.rnn.MultiRNNCell(gru_cells)
    outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32, sequence_length=seq_length)
    states_concat = tf.concat(axis=1, values=states)
    logits = tf.layers.dense(states_concat, n_outputs)

# Construct graph nodes for loss
with tf.name_scope("loss"):
	xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
	loss = tf.reduce_mean(xentropy)

# Construct graph nodes for training and optimizer
with tf.name_scope("train"):
    if opt_algo == "GD":
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    elif opt_algo == "Momentum":
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
    elif opt_algo == "Nesterov":
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9, use_nesterov=True)
    elif opt_algo == "RMSProp":
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, momentum=0.9, decay=0.9, epsilon=1e-10)
    elif opt_algo == "Adam":
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    else: # default is Nesterov
        opt_algo = "Nesterov"
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9, use_nesterov=True)
    print(" - optimizer: {0:s}".format(opt_algo))
    training_op = optimizer.minimize(loss)

# Construct graph nodes for evaluation
print(" - TOP_K_THRESH (top K threshold for accuracy): {0:d}".format(top_k_thresh))
with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, top_k_thresh)
    #accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    n_corr = tf.keras.backend.sum(tf.dtypes.cast(correct, dtype=tf.int32))
    n_wrong = tf.shape(y)[0] - n_corr

    # If correct, (k) TPs and (n_outputs - k) TNs are counted, respectively
    TP = top_k_thresh * n_corr
    # If not correct, (k) FPs, (1) FN, and (n_outputs - k - 1) FNs are counted, respectively
    FP = top_k_thresh * n_wrong
    FN = 1 * n_wrong
    TN = (n_outputs - top_k_thresh) * n_corr + (n_outputs - top_k_thresh - 1) * n_wrong

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f_measure = 2 * precision * recall / (precision + recall)

# Construct graph nodes for initializer and saver
init = tf.global_variables_initializer()
saver = tf.train.Saver()


# Run training as a session
print("\n* Training")
BOOL_DEVICE_LOG = False # Set the flag True to check device placement
DIR_PATH = os.path.join(log_dir, "result", log_time_tag)
if not os.path.isdir(DIR_PATH):
    os.mkdir(DIR_PATH)
RESULT_PATH = os.path.join(DIR_PATH, str(random_seed) + "_" + train_model + "_train-valid-accuracy.txt")
SAVE_PATH = os.path.join(log_dir, "checkpoint", log_time_tag, str(random_seed) + "_my_" + train_model + "_model_final.ckpt")
f = open(RESULT_PATH, 'w', encoding='UTF8')
with tf.Session(config=tf.ConfigProto(log_device_placement=BOOL_DEVICE_LOG)) as sess:
    init.run()
    for epoch in range(n_epochs):
        for X_batch, y_batch, Xlen_batch in shuffle_batch(X_train, y_train, Xlen_train, batch_size):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch, seq_length: Xlen_batch})
        #acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch, seq_length: Xlen_batch})
        #acc_valid = accuracy.eval(feed_dict={X: X_valid, y: y_valid, seq_length: Xlen_valid})
        #print("Epoch {0:d}: Batch Accuracy: {1:f}, Validation Accuracy: {2:f}".format(epoch, acc_batch, acc_valid))
        
        # Print out precision, recall, and f-measure for the last batch set and the validation set, respectively
        batch_feed = {X: X_batch, y: y_batch, seq_length: Xlen_batch}
        valid_feed = {X: X_valid, y: y_valid, seq_length: Xlen_valid}
        prec_batch, reca_batch, f1_batch = sess.run([precision, recall, f_measure], feed_dict=batch_feed)
        prec_valid, reca_valid, f1_valid = sess.run([precision, recall, f_measure], feed_dict=valid_feed)        
        
        data = "{0:d} {1:f} {2:f} {3:f} {4:f} {5:f} {6:f}\n".format(epoch, prec_batch, reca_batch, f1_batch, prec_valid, reca_valid, f1_valid)
        f.write(data)
        print("Epoch {0:d}: Batch Precision/Recall/F-measure: {1:f}/{2:f}/{3:f}, Validation Precision/Recall/F-measure: {4:f}/{5:f}/{6:f}".format(epoch, prec_batch, reca_batch, f1_batch, prec_valid, reca_valid, f1_valid))

    # Save the final model    
    path_saved = saver.save(sess, SAVE_PATH)
f.close()
print("   >>> The training and validation set precision/recall/f-measure values as epochs are stored in {0:s}".format(RESULT_PATH))
print("   >>> The checkpoint for the trained model is stored in {0:s}".format(SAVE_PATH))


# Run testing as a session
print("\n* Testing")
RESULT_PATH = os.path.join(DIR_PATH, str(random_seed) + "_test-accuracy.txt")
f = open(RESULT_PATH, 'a', encoding='UTF8')
with tf.Session() as sess:
    saver.restore(sess, SAVE_PATH)
    #acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test, seq_length: Xlen_test})
    test_feed = {X: X_test, y: y_test, seq_length: Xlen_test}
    prec_test, reca_test, f1_test = sess.run([precision, recall, f_measure], feed_dict=test_feed)
    data = "[{0:s}] Test Precision/Recall/F-measure: {1:f}/{2:f}/{3:f}\n".format(train_model, prec_test, reca_test, f1_test)
    f.write(data)
    print(data, end=' ')
f.close()
print("   >>> The testing set precision/recall/f-measure values is stored in {0:s}\n".format(RESULT_PATH))
