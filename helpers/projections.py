import cv2
import open3d as o3d
import numpy as np


def project_points_3d_to_points_2d(points_3d, projection_matrix):
    """
    Apply the perspective projection
    Args:
        points_3d:     3D points_3d in camera coordinate [npoints, 3]
        projection_matrix:   Projection matrix [3, 4]
    """
    num_pts = points_3d.shape[1]

    # Change to homogenous coordinate
    points_3d = np.vstack((points_3d, np.ones((1, num_pts))))
    points_2d = projection_matrix @ points_3d
    points_2d[:2, :] /= points_2d[2, :]
    return points_2d.T

def project_points_2d_to_points_3d(points_2d, inverse_projection_matrix):
    # todo
    pass

def create_filter(points_2d, points_3d, image_width, image_height):
    """
    Filter lidar points to be within image FOV
    :param points_2d:
    :param points_3d:
    :param image_width:
    :param image_height:
    :return:
    """
    ordered_indexes = np.where((points_2d[:, 0] >= -0.5) & (points_2d[:, 0] < image_width - 0.5) &
                               (points_2d[:, 1] >= -0.5) & (points_2d[:, 1] < image_height - 0.5) &
                               (points_3d[:, 0] > 0)
                               )[0]

    return ordered_indexes

def pointcloud_array_to_pcd(self, array):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(array[..., 0:3])
    return pcd

# =========================================================
# Visualizations
# =========================================================


def render_lidar_on_image(points_3d, base_img, calib):
    image_height, image_width, _ = base_img.shape

    # projection matrix (project from velo2cam2)
    proj_velo2cam2 = get_projection_matrix(calib)

    # apply projection
    points_2d = project_points_3d_to_points_2d(points_3d.T, proj_velo2cam2)

    ordered_indexes = create_filter(points_2d, points_3d, image_width = image_width, image_height = image_height)

    # Filter out 2d points
    filtered_points_2d = points_2d[ordered_indexes]

    image = np.full((image_height, image_width, 1), dtype = np.float32, fill_value = 0.0)

    for point_2d in filtered_points_2d:
        u = int(round((point_2d[0])))
        v = int(round((point_2d[1])))
        z = point_2d[2]
        image[v, u] = z

    cv2.imwrite("/root/ffabi_shared_folder/MSC/UnsupervisedDepth/test_complete.png", image)
    return image


def to_bgra(depth):
    assert depth.shape[-1] == 1
    depth = depth[:, :, 0]
    jet_img = cm.jet(1. - depth / np.amax(depth))[..., :3]
    jet_img *= 255 * np.stack(((depth != 0), (depth != 0), (depth != 0)), axis = 2)
    return jet_img


def merge_colored_rgb(depth_in_meters, base_image):
    colored_depth = to_bgra(depth_in_meters)
    image = base_image * (colored_depth == 0) + colored_depth * (colored_depth != 0)
    cv2.imwrite("/root/ffabi_shared_folder/MSC/UnsupervisedDepth/test_colored.png", image)
