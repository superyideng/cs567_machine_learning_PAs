import numpy as np


def get_k_means_plus_plus_center_indices(n, n_cluster, x, generator=np.random):
    '''

    :param n: number of samples in the data
    :param n_cluster: the number of cluster centers required
    :param x: data-  numpy array of points
    :param generator: random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.


    :return: the center points array of length n_clusters with each entry being index to a sample
             which is chosen as centroid.
    '''
    # TODO:
    # implement the Kmeans++ algorithm of how to choose the centers according to the lecture and notebook
    # Choose 1st center randomly and use Euclidean distance to calculate other centers.

    first_ind = generator.randint(0, n)
    # to store the index of center points
    centers = [first_ind]
    for _ in range(n_cluster - 1):
        cur_distsquare = np.array([])
        # size: 1 * cur_num_centers
        center_points = x[centers]
        for point in x:
            cur_min_dist = np.min(np.sum(np.square(center_points - point), axis=1))
            cur_distsquare = np.append(cur_distsquare, cur_min_dist)
        next_center_ind = np.argmax(cur_distsquare)
        # what if there is more than one
        centers.append(next_center_ind)

    # DO NOT CHANGE CODE BELOW THIS LINE

    print("[+] returning center for [{}, {}] points: {}".format(n, len(x), centers))
    return centers


def get_lloyd_k_means(n, n_cluster, x, generator):
    return generator.choice(n, size=n_cluster)


class KMeans():

    '''
        Class KMeans:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
            generator - random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.
    '''
    def __init__(self, n_cluster, max_iter=100, e=0.0001, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator
        # self.centers = []
        # self.quality = 0

    def fit(self, x, centroid_func=get_lloyd_k_means):

        '''
            Finds n_cluster in the data x
            params:
                x - N X D numpy array
                centroid_func - To specify which algorithm we are using to compute the centers(Lloyd(regular) or Kmeans++)
            returns:
                A tuple
                (centroids a n_cluster X D numpy array,
                 y a length (N,) numpy array where cell i is the ith sample's assigned cluster,
                 number_of_updates a Int)
            Note: Number of iterations is the number of time you update the assignment
        '''
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"

        # N = # data points, D = len of every point
        N, D = x.shape

        # init n_cluster center indexes
        self.centers = centroid_func(len(x), self.n_cluster, x, self.generator)

        # TODO:
        # - comment/remove the exception.
        # - Initialize means by picking self.n_cluster from N data points
        # - Update means and membership until convergence or until you have made self.max_iter updates.
        # - return (means, membership, number_of_updates)

        # DONOT CHANGE CODE ABOVE THIS LINE

        centroids = x[self.centers]
        Q_pre = -1
        y = np.zeros(N, dtype=int)
        num_of_updates = 0

        while num_of_updates < self.max_iter:
            norm2 = np.sum(((x - np.expand_dims(centroids, axis=1)) ** 2), axis=2)
            y = np.argmin(norm2, axis=0)
            
            l = []
            for k in range(self.n_cluster):
                l.append([np.sum((x[y == k] - centroids[k]) ** 2)])
            Q_cur = np.sum(l) / N

            num_of_updates += 1

            if (abs(Q_cur - Q_pre)) <= self.e and Q_pre >= 0:
                break

            Q_pre = Q_cur

            muk_update = np.zeros((self.n_cluster, D))
            for n_cluster in range(self.n_cluster):
                muk_update[n_cluster] = np.sum(x[y == n_cluster], axis=0) / len(x[y == n_cluster])
            if np.isnan(muk_update).any():
                index = np.where(np.isnan(centroids))
                muk_update[index] = centroids[index]
            centroids = muk_update

        # DO NOT CHANGE CODE BELOW THIS LINE
        return centroids, y, num_of_updates


class KMeansClassifier():

    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
            generator - random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.
    '''

    def __init__(self, n_cluster, max_iter=100, e=1e-6, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator

    def fit(self, x, y, centroid_func=get_lloyd_k_means):
        '''
            Train the classifier
            params:
                x - N X D size  numpy array
                y - (N,) size numpy array of labels
                centroid_func - To specify which algorithm we are using to compute the centers(Lloyd(regular) or Kmeans++)

            returns:
                None
            Stores following attributes:
                self.centroids : centroids obtained by kmeans clustering (n_cluster X D numpy array)
                self.centroid_labels : labels of each centroid obtained by
                    majority voting (N,) numpy array)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        self.generator.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the classifier
        # - assign means to centroids
        # - assign labels to centroid_labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        kmeans = KMeans(self.n_cluster, self.max_iter, self.e)
        centroids, membership, num_of_updates = kmeans.fit(x, centroid_func)
        centroid_labels = []

        for i in range(len(centroids)):
            label_idx = np.where(membership == i)[0]
            labels = y[label_idx]
            bincnt = np.bincount(np.array(labels))
            centroid_labels.append(np.argmax(bincnt))

        centroid_labels = np.array(centroid_labels)
        
        # DONOT CHANGE CODE BELOW THIS LINE

        self.centroid_labels = centroid_labels
        self.centroids = centroids

        assert self.centroid_labels.shape == (
            self.n_cluster,), 'centroid_labels should be a numpy array of shape ({},)'.format(self.n_cluster)

        assert self.centroids.shape == (
            self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(self.n_cluster, D)

    def predict(self, x):
        '''
            Predict function
            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        self.generator.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the prediction algorithm
        # - return labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        labels = []
        for i in range(len(x)):
            point = x[i]
            dist = np.argmin(np.sum(np.square(self.centroids - point), axis=1))
            labels.append(self.centroid_labels[dist])
        
        # DO NOT CHANGE CODE BELOW THIS LINE
        return np.array(labels)
        

def transform_image(image, code_vectors):
    '''
        Quantize image using the code_vectors

        Return new image from the image by replacing each RGB value in image with nearest code vectors (nearest in euclidean distance sense)

        returns:
            numpy array of shape image.shape
    '''

    assert image.shape[2] == 3 and len(image.shape) == 3, \
        'Image should be a 3-D array with size (?,?,3)'

    assert code_vectors.shape[1] == 3 and len(code_vectors.shape) == 2, \
        'code_vectors should be a 2-D array with size (?,3)'

    # TODO
    # - comment/remove the exception
    # - implement the function

    # DONOT CHANGE CODE ABOVE THIS LINE

    row = len(image)
    col = len(image[0])
    cur_row = row * col
    reshaped_img = image.reshape(cur_row, 3)

    min_inds = [np.argmin(np.sum(np.square(code_vectors - point), axis=1)) for point in reshaped_img]
    min_points = code_vectors[min_inds]

    new_im = min_points.reshape(row, col, 3)

    # DONOT CHANGE CODE BELOW THIS LINE
    return new_im

