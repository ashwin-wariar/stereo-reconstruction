import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from mpl_toolkits.mplot3d import Axes3D


def compute_F(pts1, pts2):
    inliers_best = 0
    F_best = np.zeros((3,3))

    for i in range(10000):
        A = np.zeros((8,9))
        eight_points = random.sample(range(pts1.shape[0]), 8)
        for j, l in enumerate(eight_points):
            A[j,0] = pts1[l,0] * pts2[l,0]
            A[j,1] = pts1[l,1] * pts2[l,0]
            A[j,2] = pts2[l,0]
            A[j,3] = pts1[l,0] * pts2[l,1]
            A[j,4] = pts1[l,1] * pts2[l,1]
            A[j,5] = pts2[l,1]
            A[j,6] = pts1[l,0]
            A[j,7] = pts1[l,1]
            A[j,8] = 1
        # Compute F    
        u, d, v = np.linalg.svd(A)
        F = v.T[:,-1].reshape(3,3)
        # SVD Cleanup
        u, d, v = np.linalg.svd(F)
        d[-1] = 0
        F = u @ np.diag(d) @ v
  
        # RANSAC implementation
        inliers = 0
        pad = np.ones((pts1.shape[0],1))
        pt1 = np.hstack((pts1, pad))
        pt2 = np.hstack((pts2, pad))
        for k in range(pts1.shape[0]):
            u1 = F.T @ pt2[k,:]
            u2 = F @ pt1[k,:]
            d1 = (pt1[k,:].T @ F.T @ pt2[k,:]) / np.sqrt(u1[0]**2 + u1[1]**2)
            d2 = (pt2[k,:].T @ F @ pt1[k,:]) / np.sqrt(u2[0]**2 + u2[1]**2) 
            if abs(d1) < 1 and abs(d2) < 1:
                inliers += 1
            
        if inliers > inliers_best:
            inliers_best = inliers
            F_best = F

    F = F_best 
    return F

def triangulation(P1, P2, pts1, pts2):
    pts3D = np.zeros((pts1.shape[0], 3))

    for i in range(pts1.shape[0]):
        p1 = np.append(pts1[i,:], [1])
        p2 = np.append(pts2[i,:], [1])
        # Compute u and v through the skew-symmetric matrix
        u = [[0, -p1[2], p1[1]],
            [p1[2], 0, -p1[0]],
            [-p1[1], p1[0], 0]]
        v = [[0, -p2[2], p2[1]],
            [p2[2], 0, -p2[0]],
            [-p2[1], p2[0], 0]]
        
        ux = u @ P1
        vx = v @ P2
        # Make A 4x4
        A = np.vstack((ux[:-1], vx[:-1]))
        _, _, v = np.linalg.svd(A)
        x = v[-1]
        # Remove the last point through normalization
        x /= x[-1]
        pts3D[i] = x[:-1].reshape((1,3))
    return pts3D


def disambiguate_pose(Rs, Cs, pts3Ds):
    max_inliers = -math.inf
    for i in range(len(Rs)):
        r = Rs[i][2].reshape((3,1))

        disc = pts3Ds[i] - Cs[i].T
        chirality = disc.dot(r)

        inliers = np.sum(chirality > 0)

        if inliers > max_inliers:
            R = Rs[i]
            C = Cs[i]
            pts3D = pts3Ds[i]
            max_inliers = inliers        

    return R, C, pts3D


def compute_rectification(K, R, C):
    R_x = (C / np.linalg.norm(C)).reshape((1,3))

    R_z_tilde = np.array([0,0,1]).reshape((1,3))
    R_z_1 = R_z_tilde.dot(R_x.T)
    R_z_2 = R_z_tilde - R_z_1.dot(R_x)
    R_z = R_z_2 / np.linalg.norm(R_z_2)

    R_y = np.cross(R_z, R_x)
    
    R_rect = np.vstack((R_x, R_y, R_z))

    H1 = K @ R_rect @ np.linalg.inv(K)
    H2 = K @ R_rect @ R.T @ np.linalg.inv(K)
    return H1, H2


def dense_match(img1, img2, descriptors1, descriptors2):
    h, w = img1.shape
    disparity = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            # Check to see if the pixel is black
            if not img2[i][j]:
                continue
            # Find the difference throughout all the rows
            d = descriptors1[i][j:] - descriptors2[i][j]
            # Minimize the difference by taking the norm and using argmin
            distance = np.linalg.norm(d, axis=1)
            disparity[i][j] = np.argmin(distance)
    # Remove the negative pixel values
    disparity = np.clip(disparity, a_min=0, a_max=math.inf)

    return disparity


# PROVIDED functions
def compute_camera_pose(F, K):
    E = K.T @ F @ K
    R_1, R_2, t = cv2.decomposeEssentialMat(E)
    # 4 cases
    R1, t1 = R_1, t
    R2, t2 = R_1, -t
    R3, t3 = R_2, t
    R4, t4 = R_2, -t

    Rs = [R1, R2, R3, R4]
    ts = [t1, t2, t3, t4]
    Cs = []
    for i in range(4):
        Cs.append(-Rs[i].T @ ts[i])
    return Rs, Cs


def visualize_img_pair(img1, img2):
    img = np.hstack((img1, img2))
    if img1.ndim == 3:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img, cmap='gray')
    plt.axis('off')
    fig = plt.gcf()
    fig.set_size_inches(16, 9)
    plt.show()


def visualize_find_match(img1, img2, pts1, pts2):
    assert pts1.shape == pts2.shape, 'x1 and x2 should have same shape!'
    img_h = img1.shape[0]
    scale_factor1 = img_h/img1.shape[0]
    scale_factor2 = img_h/img2.shape[0]
    img1_resized = cv2.resize(img1, None, fx=scale_factor1, fy=scale_factor1)
    img2_resized = cv2.resize(img2, None, fx=scale_factor2, fy=scale_factor2)
    pts1 = pts1 * scale_factor1
    pts2 = pts2 * scale_factor2
    pts2[:, 0] += img1_resized.shape[1]
    img = np.hstack((img1_resized, img2_resized))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    for i in range(pts1.shape[0]):
        plt.plot([pts1[i, 0], pts2[i, 0]], [pts1[i, 1], pts2[i, 1]], 'b.-', linewidth=1, markersize=10)
    plt.axis('off')
    fig = plt.gcf()
    fig.set_size_inches(16, 9)
    fig.tight_layout()
    plt.show()


def visualize_epipolar_lines(F, pts1, pts2, img1, img2):
    assert pts1.shape == pts2.shape, 'x1 and x2 should have same shape!'
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    ax1.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    ax2.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))

    for i in range(pts1.shape[0]):
        x1, y1 = int(pts1[i][0] + 0.5), int(pts1[i][1] + 0.5)
        ax1.scatter(x1, y1, s=20)
        p1, p2 = find_epipolar_line_end_points(img2, F, (x1, y1))
        ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], linewidth=1)

    for i in range(pts2.shape[0]):
        x2, y2 = int(pts2[i][0] + 0.5), int(pts2[i][1] + 0.5)
        ax2.scatter(x2, y2, s=20)
        p1, p2 = find_epipolar_line_end_points(img1, F.T, (x2, y2))
        ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], linewidth=1)

    ax1.axis('off')
    ax2.axis('off')
    fig = plt.gcf()
    fig.set_size_inches(16, 9)
    fig.tight_layout()
    plt.show()


def find_epipolar_line_end_points(img, F, p):
    img_width = img.shape[1]
    el = (F @ np.array([[p[0], p[1], 1]]).T).flatten()
    p1, p2 = (0, int(-el[2] / el[1])), (img.shape[1], int((-img_width * el[0] - el[2]) / el[1]))
    _, p1, p2 = cv2.clipLine((0, 0, img.shape[1], img.shape[0]), p1, p2)
    return p1, p2


def visualize_camera_poses(Rs, Cs):
    assert(len(Rs) == len(Cs) == 4)
    fig = plt.figure()
    R1, C1 = np.eye(3), np.zeros((3, 1))
    for i in range(4):
        R2, C2 = Rs[i], Cs[i]
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        draw_camera(ax, R1, C1)
        draw_camera(ax, R2, C2)
        set_axes_equal(ax)
        ax.set_xlabel('x axis')
        ax.set_ylabel('y axis')
        ax.set_zlabel('z axis')
        ax.view_init(azim=-85, elev=0)
        ax.title.set_text('Configuration {}'.format(i))
    fig.set_size_inches(8, 8)
    fig.tight_layout()
    plt.show()


def visualize_camera_poses_with_pts(Rs, Cs, pts3Ds):
    assert len(Rs) == len(Cs) == len(pts3Ds)
    fig = plt.figure()
    R1, C1 = np.eye(3), np.zeros((3, 1))
    for i in range(len(Rs)):
        R2, C2, pts3D = Rs[i], Cs[i], pts3Ds[i]
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        draw_camera(ax, R1, C1, 5)
        draw_camera(ax, R2, C2, 5)
        ax.plot(pts3D[:, 0], pts3D[:, 1], pts3D[:, 2], 'b.')
        set_axes_equal(ax)
        ax.set_xlabel('x axis')
        ax.set_ylabel('y axis')
        ax.set_zlabel('z axis')
        ax.view_init(azim=-85, elev=0)
        ax.title.set_text('Configuration {}'.format(i))
    fig.set_size_inches(8, 8)
    fig.tight_layout()
    plt.show()


def draw_camera(ax, R, C, scale=0.2):
    axis_end_points = C + scale * R.T  # (3, 3)
    vertices = C + scale * R.T @ np.array([[1, 1, 1], [-1, 1, 1], [-1, -1, 1], [1, -1, 1]]).T  # (3, 4)
    vertices_ = np.hstack((vertices, vertices[:, :1]))  # (3, 5)
    C = C.flatten()

    # draw coordinate system of camera
    ax.plot([C[0], axis_end_points[0, 0]], [C[1], axis_end_points[1, 0]], [C[2], axis_end_points[2, 0]], 'r-')
    ax.plot([C[0], axis_end_points[0, 1]], [C[1], axis_end_points[1, 1]], [C[2], axis_end_points[2, 1]], 'g-')
    ax.plot([C[0], axis_end_points[0, 2]], [C[1], axis_end_points[1, 2]], [C[2], axis_end_points[2, 2]], 'b-')

    # draw square window and lines connecting it to camera center
    ax.plot(vertices_[0, :], vertices_[1, :], vertices_[2, :], 'k-')
    ax.plot([C[0], vertices[0, 0]], [C[1], vertices[1, 0]], [C[2], vertices[2, 0]], 'k-')
    ax.plot([C[0], vertices[0, 1]], [C[1], vertices[1, 1]], [C[2], vertices[2, 1]], 'k-')
    ax.plot([C[0], vertices[0, 2]], [C[1], vertices[1, 2]], [C[2], vertices[2, 2]], 'k-')
    ax.plot([C[0], vertices[0, 3]], [C[1], vertices[1, 3]], [C[2], vertices[2, 3]], 'k-')


def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range, x_middle = abs(x_limits[1] - x_limits[0]), np.mean(x_limits)
    y_range, y_middle = abs(y_limits[1] - y_limits[0]), np.mean(y_limits)
    z_range, z_middle = abs(z_limits[1] - z_limits[0]), np.mean(z_limits)

    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def visualize_disparity_map(disparity):
    disparity[disparity > 150] = 150
    plt.imshow(disparity, cmap='jet')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    # read in left and right images as RGB images
    img_left = cv2.imread('./left.bmp', 1)
    img_right = cv2.imread('./right.bmp', 1)
    #visualize_img_pair(img_left, img_right)

    # Step 0: get correspondences between image pair
    data = np.load('./correspondence.npz')
    pts1, pts2 = data['pts1'], data['pts2']
    #visualize_find_match(img_left, img_right, pts1, pts2)

    # Step 1: compute fundamental matrix and recover four sets of camera poses
    F = compute_F(pts1, pts2)
    #visualize_epipolar_lines(F, pts1, pts2, img_left, img_right)

    K = np.array([[350, 0, 960/2], [0, 350, 540/2], [0, 0, 1]])
    Rs, Cs = compute_camera_pose(F, K)
    #visualize_camera_poses(Rs, Cs)

    # Step 2: triangulation
    pts3Ds = []
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    for i in range(len(Rs)):
        P2 = K @ np.hstack((Rs[i], -Rs[i] @ Cs[i]))
        pts3D = triangulation(P1, P2, pts1, pts2)
        pts3Ds.append(pts3D)
    visualize_camera_poses_with_pts(Rs, Cs, pts3Ds)

    # Step 3: disambiguate camera poses
    R, C, pts3D = disambiguate_pose(Rs, Cs, pts3Ds)

    # Step 4: rectification
    H1, H2 = compute_rectification(K, R, C)
    img_left_w = cv2.warpPerspective(img_left, H1, (img_left.shape[1], img_left.shape[0]))
    img_right_w = cv2.warpPerspective(img_right, H2, (img_right.shape[1], img_right.shape[0]))
    visualize_img_pair(img_left_w, img_right_w)

    # Step 5: generate disparity map
    img_left_w = cv2.resize(img_left_w, (int(img_left_w.shape[1] / 2), int(img_left_w.shape[0] / 2)))  # resize image for speed
    img_right_w = cv2.resize(img_right_w, (int(img_right_w.shape[1] / 2), int(img_right_w.shape[0] / 2)))
    img_left_w = cv2.cvtColor(img_left_w, cv2.COLOR_BGR2GRAY)  # convert to gray scale
    img_right_w = cv2.cvtColor(img_right_w, cv2.COLOR_BGR2GRAY)
    data = np.load('./dsift_descriptor.npz')
    desp1, desp2 = data['descriptors1'], data['descriptors2']
    disparity = dense_match(img_left_w, img_right_w, desp1, desp2)
    #visualize_disparity_map(disparity)
