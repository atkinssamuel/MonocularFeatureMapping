# ROB521 Lab 3
# Mackenzie Clark, Najah Hassan, Samuel Atkins

import os
import glob

import numpy as np
import cv2
import scipy.io as sio
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy.linalg as la
from sympy import Plane, Point3D

class FeatureProcessor:
    def __init__(self, data_folder, n_features=500, median_filt_multiplier=1.0):
        # Initiate ORB detector and the brute force matcher
        self.n_features = n_features
        self.median_filt_multiplier = median_filt_multiplier
        self.orb = cv2.ORB_create(nfeatures=n_features)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.data_folder = data_folder
        self.num_images = len(glob.glob(data_folder + '*.jpeg'))
        self.feature_match_locs = []  # [img_i, feat_i, [x, y] of match ((-1, -1) if no match)]

        # store the features found in the first image here. you may find it useful to store both the raw
        # keypoints in kp, and the numpy (u, v) pairs (keypoint.pt) in kp_np
        self.features = dict(kp=[], kp_np=[], des=[])
        self.first_matches = True
        return

    def get_image(self, id):
        # Load image and convert to grayscale
        # print(os.path.join(self.data_folder, 'camera_image_{}.jpeg'.format(id)))
        img = cv2.imread(os.path.join(self.data_folder, 'camera_image_{}.jpeg'.format(id)))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray

    def get_features(self, id):
        """ Get the keypoints and the descriptors for features for the image with index id."""
        # harris corner detector
        img = self.get_image(id)
        # img = np.float32(img)
        # dst = cv2.cornerHarris(img,2,3,0.04)
        # img[dst>0.01*dst.max()]=[0,0,255]

        # ORB feature detector
        orb = cv2.ORB_create()
        # find the keypoints with ORB
        kp = orb.detect(img, None)
        # print(kp[0].pt)
        # compute the descriptors with ORB
        kp, des = orb.compute(img, kp)
        # img2 = cv2.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
        # plt.imshow(img2), plt.show()

        return kp, des

    def get_matches(self):
        """ Get all of the locations of features matches for each image to the features found in the
        first image. Output should be a numpy array of shape (num_images, num_features_first_image, 2), where
        the actual data is the locations of each feature in each image."""
        self.feature_match_locs = -1 * np.ones((self.num_images, self.n_features, 2))

        kp_0, des_0 = self.get_features(0)  # detect features on first image

        self.features['kp'] = kp_0
        self.features['des'] = des_0

        for f_num, kp in enumerate(kp_0):
            self.features['kp_np'].append(kp.pt)
            self.feature_match_locs[0, f_num, 0] = kp.pt[0]
            self.feature_match_locs[0, f_num, 1] = kp.pt[1]

        # print(self.num_images-1)
        for img in range(1, self.num_images):
            # print(img)
            kp, des = self.get_features(img)
            matches = self.bf.match(des, des_0)

            # matches = sorted(matches, key=lambda x: x.distance)
            avg = sum(point.distance for point in matches) / len(matches)

            # lowes ratio test to get the good matches
            for match in matches:  # not sure if there's a better way to filter this
                if match.distance < 1.2 * avg:
                    train_f_num = match.trainIdx
                    query_f_num = match.queryIdx
                    # print(train_f_num, query_f_num)
                    self.feature_match_locs[img, train_f_num, 0] = kp[query_f_num].pt[0]
                    self.feature_match_locs[img, train_f_num, 1] = kp[query_f_num].pt[1]

                    # includes the first frame feature locations
        return self.feature_match_locs

    def ransac(self, triangulated_points):
        """ Feature rejection function
        :param triangulated_points: np array, (num_features, 3) all triangulated points
        :return bestinliers: (N, ), indices of the inliers, where N is the total number of inliers
        """
        # we only expect to have the 5-10% outliers, so we'll set the minimum inliers
        # to be 92% of the total number of features.
        num_features = triangulated_points.shape[0]

        # tunable parameters
        mininliers = .92 * num_features
        alpha = 0.042

        maxinliers = 0
        sympy_points = np.ndarray((num_features, 3))
        for j in range(num_features):
            sympy_points[j] = Point3D(triangulated_points[j, 0], triangulated_points[j, 1], triangulated_points[j, 2])

        # make an array of possible indices that we will randomly draw from
        for i in range(1000):
            # select 3 random points from triangulated points to compute a plane
            rand_pts = np.random.choice(triangulated_points.shape[0], 3)
            normal, d = self.compute_plane(triangulated_points[rand_pts, :])

            # make the points into a plane object
            plane = Plane(Point3D(triangulated_points[rand_pts[0], :]), normal_vector=normal)

            # compute the reprojection error for all the points
            reproj = np.ndarray((num_features, ))
            for j in range(num_features):
                reproj[j] = plane.distance(Point3D(sympy_points[j, :]))
            # print(reproj)
            inliers = reproj < alpha
            ninliers = np.sum(inliers)

            # update the max values if applicable
            if ninliers > maxinliers:
                maxinliers = ninliers
                bestinliers = inliers
                # check exit condition
                if maxinliers > mininliers:
                    print('required', i+1, 'RANSAC attempts to remove outliers.')
                    break

        print('number of inliers: ' + str(maxinliers))

        # returns the inlier indices, not the actual values
        return bestinliers

    def compute_plane(self, points):
        """ Compute the equation of the plane that best fits the set of points provided
        If given 3 points, then the problem will have an exact solution.
        :param points: np array, Nx3 array of points (x, y, z)
        :return normal, d: normal is (3,) np array of [a, b, c], and d is the constant
            in the equation of the plane. """
        # fit a plane to the feature point cloud for illustration
        # see https://math.stackexchange.com/questions/99299/best-fitting-plane-given-a-set-of-points
        centroid = points.mean(axis=0)
        points_minus_cent = points - centroid  # subtract centroid
        u, _, _ = np.linalg.svd(points_minus_cent.T)
        normal = u.T[2]

        # normal from svd, point is centroid, get d = -(ax + by + cz)
        d = -centroid.dot(normal)

        # print('The equation is {0}x + {1}y + {2}z = {3}'.format(normal[0], normal[1], normal[2], d))
        return normal, d

def triangulate(feature_data, tf, inv_K):
    """ For (u, v) image locations of a single feature for every image, as well as the corresponding
    robot poses as T matrices for every image, calculate an estimated xyz position of the feature.

    You're free to use whatever method you like, but we recommend a solution based on least squares, similar
    to section 7.1 of Szeliski's book "Computer Vision: Algorithms and Applications".
    :param feature_data: np array (num_images, 2) - (x, y) coordinates for each feature in each frame
    :param tf: np array (num_images, 4, 4) - homogeneous transformations for each frame
    :param inv_K: np array (3, 3) - inverse of the camera intrinsics matrix
    :returns pt: np array (3, ) - point in 3D space """

    tot_features = feature_data.shape[0]  # total number of features
    sum1 = np.zeros((3, 3), dtype=float)
    sum2 = np.zeros((3, 1), dtype=float)

    for j in range(0, tot_features):
        x, y = feature_data[j][0], feature_data[j][1]

        if x < 0 or y < 0:
            # print("Invalid")
            continue

        r = np.array([x, y, 1]).reshape((3, 1))
        C = tf[j, 0:3, 0:3]
        vj = C @ inv_K @ r
        vj /= np.linalg.norm(vj)

        cent_j = (tf[j][:3, 3]).reshape((3, 1))
        # print(cent_j.shape)

        sum1 += np.identity(3) - (vj @ vj.T)
        sum2 += (np.identity(3) - (vj @ vj.T)) @ cent_j

    point = np.dot(la.inv(sum1), sum2)
    return point.squeeze()


def main():
    min_feature_views = 20  # minimum number of images a feature must be seen in to be considered useful
    K = np.array([[530.4669406576809, 0.0, 320.5],  # K from sim
                  [0.0, 530.4669406576809, 240.5],
                  [0.0, 0.0, 1.0]])
    inv_K = np.linalg.inv(K)  # will be useful for triangulating feature locations

    # load in data, get consistent feature locations
    data_folder = os.path.join(os.getcwd(), 'l3_mapping_data/')
    f_processor = FeatureProcessor(data_folder)
    # get_matches includes lowes ratio test for extracting good matches.
    feature_locations = f_processor.get_matches()  # output shape should be (num_images, num_features, 2)
    # print('shape of matches: ' + str(good_feature_locations.shape))
    num_landmarks = feature_locations.shape[1]
    # print('num landmarks: ' + str(num_landmarks))
    num_frames = feature_locations.shape[0]

    validFeat = np.zeros(num_landmarks)

    for frame in range(0, num_frames):
        for landmark in range(0, num_landmarks):
            if feature_locations[frame, landmark, 0] > -1 and feature_locations[frame, landmark, 1] > -1:
                validFeat[landmark] += 1

    # create a boolean array for ever frame that meets our min feature criteria
    final_landmarks = validFeat > min_feature_views

    # num_landmarks = np.sum(valid_features)
    good_feature_locations = feature_locations[:, final_landmarks, :]
    num_landmarks = np.sum(final_landmarks)
    print('landmarks after filtering features: ', num_landmarks)

    pc = np.zeros((num_landmarks, 3))

    # create point cloud map of features
    tf = sio.loadmat("l3_mapping_data/tf.mat")['tf']
    tf_fixed = np.linalg.inv(tf[0, :, :]).dot(tf).transpose((1, 0, 2))  # (num_images, 4, 4)
    # print(tf_fixed)
    # print('one good landmark has the shape: ' + str(good_feature_locations[:, 0, :].shape))

    for i in range(num_landmarks):
        pc[i] = triangulate(good_feature_locations[:, i, :], tf_fixed, inv_K)

    # perform RANSAC on the point cloud based on the equation of a plane
    ransac_ind = f_processor.ransac(pc)
    # update good_feature_locations from the points that were returned after ransac
    pc = pc[ransac_ind, :]
    good_feature_locations = good_feature_locations[:, ransac_ind, :]
    num_landmarks = pc.shape[0]
    print('landmarks after RANSAC: ', num_landmarks)

    # ------- PLOTTING TOOLS ------------------------------------------------------------------------------
    # you don't need to modify anything below here unless you change the variable names, or want to modify
    # the plots somehow.

    # get point cloud of trajectory for comparison
    traj_pc = tf_fixed[:, :3, 3]

    # view point clouds with matplotlib
    # set colors based on y positions of point cloud
    max_y = pc[:, 1].max()
    min_y = pc[:, 1].min()
    colors = np.ones((num_landmarks, 4))
    colors[:, :3] = .5
    colors[:, 1] = (pc[:, 1] - min_y) / (max_y - min_y)
    pc_fig = plt.figure()
    ax = pc_fig.add_subplot(111, projection='3d')
    ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], marker='^', color=colors, label='features')

    ax.scatter(traj_pc[:, 0], traj_pc[:, 1], traj_pc[:, 2], marker='o', label='robot poses')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(elev=-30, azim=-88)
    ax.legend()

    # fit a plane to the feature point cloud for illustration
    # see https://math.stackexchange.com/questions/99299/best-fitting-plane-given-a-set-of-points
    centroid = pc.mean(axis=0)
    pc_minus_cent = pc - centroid  # subtract centroid
    u, s, vh = np.linalg.svd(pc_minus_cent.T)
    normal = u.T[2]

    # plot the plane
    # plane is ax + by + cz + d = 0, so z = (-ax - by - d) / c, normal is [a, b, c]
    # normal from svd, point is centroid, get d = -(ax + by + cz)
    d = -centroid.dot(normal)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    X, Y = np.meshgrid(np.linspace(xlim[0], xlim[1], 10),
                       np.linspace(ylim[0], ylim[1], 10))
    Z = (-normal[0] * X - normal[1] * Y - d) * 1. / normal[2]
    ax.plot_wireframe(X, Y, Z, color='k')

    # view all final good features matched on first image (to compare to point cloud)
    feat_fig = plt.figure()
    ax = feat_fig.add_subplot(111)
    ax.imshow(f_processor.get_image(0), cmap='gray')
    ax.scatter(good_feature_locations[0, :, 0], good_feature_locations[0, :, 1], marker='^', color=colors)

    plt.show()

    pc_fig.savefig('point_clouds.png', bbox_inches='tight')
    feat_fig.savefig('feat_fig.png', bbox_inches='tight')

if __name__ == '__main__':
    main()
