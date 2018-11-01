from __future__ import absolute_import

from pyspark.mllib.linalg import Vector as MLLibVector, Matrix as MLLibMatrix
from pyspark.ml.linalg import Vector as MLVector, Matrix as MLMatrix
from pyspark.mllib.regression import LabeledPoint
import numpy as np

from ..mllib.adapter import to_vector, from_vector, from_matrix
from six.moves import zip


def to_simple_rdd(sc, features, labels):
    """Convert numpy arrays of features and labels into
    an RDD of pairs.

    :param sc: Spark context
    :param features: numpy array with features
    :param labels: numpy array with labels
    :return: Spark RDD with feature-label pairs
    """
    pairs = [(x, y) for x, y in zip(features, labels)]
    return sc.parallelize(pairs)


def to_labeled_point(sc, features, labels, categorical=False):
    """Convert numpy arrays of features and labels into
    a LabeledPoint RDD for MLlib and ML integration.

    :param sc: Spark context
    :param features: numpy array with features
    :param labels: numpy array with labels
    :param categorical: boolean, whether labels are already one-hot encoded or not
    :return: LabeledPoint RDD with features and labels
    """
    labeled_points = []
    for x, y in zip(features, labels):
        if categorical:
            lp = LabeledPoint(np.argmax(y), to_vector(x))
        else:
            lp = LabeledPoint(y, to_vector(x))
        labeled_points.append(lp)
    return sc.parallelize(labeled_points)


def from_labeled_point(rdd, categorical=False, nb_classes=None):
    """Convert a LabeledPoint RDD back to a pair of numpy arrays

    :param rdd: LabeledPoint RDD
    :param categorical: boolean, if labels should be one-hot encode when returned
    :param nb_classes: optional int, indicating the number of class labels
    :return: pair of numpy arrays, features and labels
    """
    features = np.asarray(rdd.map(lambda lp: from_vector(lp.features)).collect())
    labels = np.asarray(rdd.map(lambda lp: lp.label).collect(), dtype='int32')
    if categorical:
        if not nb_classes:
            nb_classes = np.max(labels) + 1
        temp = np.zeros((len(labels), nb_classes))
        for i, label in enumerate(labels):
            temp[i, label] = 1.
        labels = temp
    return features, labels


def encode_label(label, nb_classes):
    """One-hot encoding of a single label

    :param label: class label (int or double without floating point digits)
    :param nb_classes: int, number of total classes
    :return: one-hot encoded vector
    """
    encoded = np.zeros(nb_classes)
    encoded[int(label)] = 1.
    return encoded


def lp_to_simple_rdd(lp_rdd, categorical=False, nb_classes=None):
    """Convert a LabeledPoint RDD into an RDD of feature-label pairs

    :param lp_rdd: LabeledPoint RDD of features and labels
    :param categorical: boolean, if labels should be one-hot encode when returned
    :param nb_classes: int, number of total classes
    :return: Spark RDD with feature-label pairs
    """
    if categorical:
        if not nb_classes:
            labels = np.asarray(lp_rdd.map(lambda lp: lp.label).collect(), dtype='int32')
            nb_classes = np.max(labels) + 1
        rdd = lp_rdd.map(lambda lp: (from_vector(lp.features), encode_label(lp.label, nb_classes)))
    else:
        rdd = lp_rdd.map(lambda lp: (from_vector(lp.features), lp.label))
    return rdd

def row_to_simple_rdd(row_rdd, categorical=False, nb_classes=None):
    """Convert a Row RDD into an RDD of feature-label pairs

    :param row_rdd: RDD of sql.Row with columns features and labels
    :param categorical: boolean, if labels should be one-hot encode when returned
    :param nb_classes: int, number of total classes
    :return: Spark RDD with feature-label pairs
    """
    if categorical:
        if not nb_classes:
            labels = np.asarray(row_rdd.map(lambda row: float(row.label)).collect(), dtype='int32')
            nb_classes = np.max(labels) + 1
        rdd = row_rdd.map(lambda row: (from_vector(row.features), encode_label(float(row.label), nb_classes)))
    else:
        first_row = row_rdd.first()
        if isinstance(first_row.label, MLLibVector) or isinstance(first_row.label, MLVector):
            label_fn = from_vector
        elif isinstance(first_row.label, MLLibMatrix) or isinstance(first_row.label, MLMatrix):
            label_fn = from_matrix
        else:
            label_fn = float

        rdd = row_rdd.map(lambda row: (from_vector(row.features), label_fn(row.label)))
    return rdd
