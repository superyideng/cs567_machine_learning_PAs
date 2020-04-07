import numpy as np
from knn import KNN

############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################


# TODO: implement F1 score
def f1_score(real_labels, predicted_labels):
    """
    Information on F1 score - https://en.wikipedia.org/wiki/F1_score
    :param real_labels: List[int]
    :param predicted_labels: List[int]
    :return: float
    """

    product_sum = np.dot(real_labels, predicted_labels)
    real_sum = np.sum(real_labels)
    predicted_sum = np.sum(predicted_labels)

    f1 = 2 * product_sum / (real_sum + predicted_sum)
    return f1


class Distances:
    @staticmethod
    # TODO
    def minkowski_distance(point1, point2):
        """
        Minkowski distance is the generalized version of Euclidean Distance
        It is also know as L-p norm (where p>=1) that you have studied in class
        For our assignment we need to take p=3
        Information on Minkowski distance - https://en.wikipedia.org/wiki/Minkowski_distance
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """

        vector1 = np.array(point1)
        vector2 = np.array(point2)

        return np.linalg.norm(vector1-vector2, 3)

    @staticmethod
    # TODO
    def euclidean_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """

        vector1 = np.array(point1)
        vector2 = np.array(point2)
        distance = np.sqrt(np.sum(np.square(vector1 - vector2)))

        return distance

    @staticmethod
    # TODO
    def inner_product_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """

        distance = np.dot(point1, point2)

        return distance

    @staticmethod
    # TODO
    def cosine_similarity_distance(point1, point2):
        """
       :param point1: List[float]
       :param point2: List[float]
       :return: float
       """
        if np.linalg.norm(point1) == 0 or np.linalg.norm(point2) == 0:
            return 0

        cos_sim = np.dot(point1, point2) / (np.linalg.norm(point1) * np.linalg.norm(point2))

        distance = 1 - cos_sim
        return distance

    @staticmethod
    # TODO
    def gaussian_kernel_distance(point1, point2):
        """
       :param point1: List[float]
       :param point2: List[float]
       :return: float
       """

        euclidean_square = np.square(Distances.euclidean_distance(point1, point2))

        distance = -np.exp(- euclidean_square / 2)
        return distance


class HyperparameterTuner:
    def __init__(self):
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None

    # TODO: find parameters with the best f1 score on validation dataset
    def tuning_without_scaling(self, distance_funcs, x_train, y_train, x_val, y_val):
        """
        In this part, you should try different distance function you implemented in part 1.1, and find the best k.
        Use k range from 1 to 30 and increment by 2. Use f1-score to compare different models.

        :param distance_funcs: dictionary of distance functions you must use to calculate the distance.
            Make sure you loop over all distance functions for each data point and each k value.
            You can refer to test.py file to see the format in which these functions will be
            passed by the grading script
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val:  List[List[int]] Validation data set will be used on your KNN predict function to produce
            predicted labels and tune k and distance function.
        :param y_val: List[int] validation labels

        Find(tune) best k, distance_function and model (an instance of KNN) and assign to self.best_k,
        self.best_distance_function and self.best_model respectively.
        NOTE: self.best_scaler will be None

        NOTE: When there is a tie, choose model based on the following priorities:
        Then check distance function  [euclidean > minkowski > gaussian > inner_prod > cosine_dist]
        If they have same distance function, choose model which has a less k.
        """

        max_f1 = -1

        distance_sequence = ['euclidean', 'minkowski', 'gaussian', 'inner_prod', 'cosine_dist']

        for dist in distance_sequence:
            for k in range(1, 30, 2):
                current_knn = KNN(k, distance_funcs[dist])
                current_knn.train(x_train, y_train)
                predicted_labels = current_knn.predict(x_val)
                current_f1 = f1_score(y_val, predicted_labels)
                if current_f1 > max_f1:
                    max_f1 = current_f1
                    self.best_k = k
                    self.best_distance_function = dist
                    self.best_model = current_knn

    # TODO: find parameters with the best f1 score on validation dataset, with normalized data
    def tuning_with_scaling(self, distance_funcs, scaling_classes, x_train, y_train, x_val, y_val):
        """
        This part is similar to Part 1.3 except that before passing your training and validation data to KNN model to
        tune k and distance function, you need to create the normalized data using these two scalers to transform your
        data, both training and validation. Again, we will use f1-score to compare different models.
        Here we have 3 hyperparameters i.e. k, distance_function and scaler.

        :param distance_funcs: dictionary of distance funtions you use to calculate the distance. Make sure you
            loop over all distance function for each data point and each k value.
            You can refer to test.py file to see the format in which these functions will be
            passed by the grading script
        :param scaling_classes: dictionary of scalers you will use to normalized your data.
        Refer to test.py file to check the format.
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val: List[List[int]] validation data set you will use on your KNN predict function to produce predicted
            labels and tune your k, distance function and scaler.
        :param y_val: List[int] validation labels

        Find(tune) best k, distance_funtion, scaler and model (an instance of KNN) and assign to self.best_k,
        self.best_distance_function, self.best_scaler and self.best_model respectively

        NOTE: When there is a tie, choose model based on the following priorities:
        For normalization, [min_max_scale > normalize];
        Then check distance function  [euclidean > minkowski > gaussian > inner_prod > cosine_dist]
        If they have same distance function, choose model which has a less k.
        """

        max_f1 = -1

        distance_sequence = ['euclidean', 'minkowski', 'gaussian', 'inner_prod', 'cosine_dist']
        scaling_sequence = ['min_max_scale', 'normalize']

        for scaler in scaling_sequence:
            scale_instance = scaling_classes[scaler]()
            scaled_x_train = scale_instance(x_train)
            scaled_x_val = scale_instance(x_val)
            for dist in distance_sequence:
                for k in range(1, 30, 2):
                    current_knn = KNN(k, distance_funcs[dist])
                    current_knn.train(scaled_x_train, y_train)
                    predicted_labels = current_knn.predict(scaled_x_val)
                    current_f1 = f1_score(y_val, predicted_labels)
                    if current_f1 > max_f1:
                        max_f1 = current_f1
                        self.best_k = k
                        self.best_distance_function = dist
                        self.best_model = current_knn
                        self.best_scaler = scaler


class NormalizationScaler:
    def __init__(self):
        pass

    # TODO: normalize data
    def __call__(self, features):
        """
        Normalize features for every sample

        Example
        features = [[3, 4], [1, -1], [0, 0]]
        return [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """

        scaled_features = [0 * len(features[0])] * len(features)
        for i in range(len(features)):
            current_feature = np.array(features[i])
            length = np.linalg.norm(current_feature)
            if length != 0:
                scaled_features[i] = (np.array(features[i]) / length).tolist()
            else:
                scaled_features[i] = features[i]

        return scaled_features


class MinMaxScaler:
    """
    Please follow this link to know more about min max scaling
    https://en.wikipedia.org/wiki/Feature_scaling
    You should keep some states inside the object.
    You can assume that the parameter of the first __call__
    will be the training set.

    Hints:
        1. Use a variable to check for first __call__ and only compute
            and store min/max in that case.

    Note:
        1. You may assume the parameters are valid when __call__
            is being called the first time (you can find min and max).

    Example:
        train_features = [[0, 10], [2, 0]]
        test_features = [[20, 1]]

        scaler1 = MinMaxScale()
        train_features_scaled = scaler1(train_features)
        # train_features_scaled should be equal to [[0, 1], [1, 0]]

        test_features_scaled = scaler1(test_features)
        # test_features_scaled should be equal to [[10, 0.1]]

        new_scaler = MinMaxScale() # creating a new scaler
        _ = new_scaler([[1, 1], [0, 0]]) # new trainfeatures
        test_features_scaled = new_scaler(test_features)
        # now test_features_scaled should be [[20, 1]]

    """

    def __init__(self):
        self.never_been_called = True
        self.max = None
        self.min = None

    def __call__(self, features):
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        if self.never_been_called:
            self.max = features[0][:]
            self.min = features[0][:]

            for i in range(len(features)):
                for j in range(len(features[0])):
                    self.max[j] = np.maximum(self.max[j], features[i][j])
                    self.min[j] = np.minimum(self.min[j], features[i][j])

            self.never_been_called = False

        arr_min = np.array(self.min)
        arr_max = np.array(self.max)
        arr_length = arr_max - arr_min
        for i in range(len(arr_length)):
            if arr_length[i] == 0:
                arr_min[i] = 0
                arr_length[i] = 1
        scaled_features = ((np.array(features) - arr_min) / arr_length).tolist()

        return scaled_features
