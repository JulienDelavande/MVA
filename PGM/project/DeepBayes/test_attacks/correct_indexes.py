from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf
import os
import pickle
from tensorflow.python.platform import flags
import sys

sys.path.append('../utils/')
sys.path.append('../cleverhans/')
from cleverhans.utils import set_log_level
from model_eval import model_eval

import keras.backend
sys.path.append('load/')
from load_classifier import load_classifier

FLAGS = flags.FLAGS

def extract_correct_indices(data_name, model_name, batch_size=128):
    """
    Extract indices of correctly classified images.
    :param data_name: Name of the dataset (e.g., 'mnist', 'cifar10').
    :param model_name: Name of the model to evaluate.
    :param batch_size: Batch size for evaluation.
    :return: List of indices of correctly classified images.
    """
    # Set TF random seed for reproducibility
    tf.set_random_seed(1234)

    # Create TF session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    print("Created TensorFlow session.")
    #set_log_level(logging.DEBUG)

    # Load dataset
    if data_name == 'mnist':
        from cleverhans.utils_mnist import data_mnist
        X_train, Y_train, X_test, Y_test = data_mnist(train_start=0, train_end=60000,
                                                      test_start=0, test_end=10000)
    elif data_name in ['cifar10', 'plane_frog']:
        from import_data_cifar10 import load_data_cifar10
        labels = None
        if data_name == 'plane_frog':
            labels = [0, 6]
        datapath = '../cifar_data/'
        X_train, X_test, Y_train, Y_test = load_data_cifar10(datapath, labels=labels)
    else:
        raise ValueError("Unsupported dataset: {}".format(data_name))

    img_rows, img_cols, channels = X_test[0].shape
    nb_classes = Y_test.shape[1]

    # Define placeholders
    x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols, channels))
    y = tf.placeholder(tf.float32, shape=(None, nb_classes))

    # Load model
    model = load_classifier(sess, model_name, data_name)

    # Ensure correct learning phase for evaluation
    if 'bnn' not in model_name:
        keras.backend.set_learning_phase(0)
    else:
        keras.backend.set_learning_phase(1)

    # Model predictions
    preds = model.predict(x, softmax=False)

    # Evaluate accuracy and retrieve predictions
    eval_params = {'batch_size': batch_size}
    accuracy, y_pred_clean = model_eval(sess, x, y, preds, X_test, Y_test,
                                        args=eval_params, return_pred=True)
    print('Test accuracy on legitimate test examples: {:.2f}%'.format(accuracy * 100))

    # Extract indices of correctly classified images
    correct_prediction = (np.argmax(Y_test, axis=1) == np.argmax(y_pred_clean, axis=1))
    correct_indices = np.where(correct_prediction)[0]
    print('Number of correctly classified images: {}/{}'.format(len(correct_indices), len(X_test)))

    # Save indices
    output_dir = 'correct_indices'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, f'{data_name}_{model_name}_correct_indices.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump(correct_indices, f)
    print(f"Correct indices saved to {output_file}")

    return correct_indices


if __name__ == '__main__':
    # Define command-line flags
    flags.DEFINE_string('data_name', 'mnist', 'Dataset name (e.g., mnist, cifar10)')
    flags.DEFINE_string('model_name', 'bayes_K10_A', 'Model name to evaluate')
    flags.DEFINE_integer('batch_size', 128, 'Batch size for evaluation')

    # Parse flags and execute
    args = FLAGS
    extract_correct_indices(data_name=args.data_name, model_name=args.model_name, batch_size=args.batch_size)
