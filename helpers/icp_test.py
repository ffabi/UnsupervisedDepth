import open3d as o3d
import numpy as np

from kitti_helpers.KittiLoader import KittiLoader
from global_variables import *


def icp_point_to_point(source, target, max_distance = 1.0):
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(source)

    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target)

    result = o3d.registration.registration_icp(
        source = source_pcd,
        target = target_pcd,
        max_correspondence_distance = max_distance,
        # init = init_translation,
        estimation_method = o3d.registration.TransformationEstimationPointToPoint())

    return result


def icp_demo(add_noise = True):
    kitti_loader = KittiLoader(date = test_date, drive = test_drive)

    source = kitti_loader.load_velodyne_pointcloud_array(test_frame_index)[:, :3]
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(source)
    colors = np.empty(shape = source.shape)
    colors[:, 0] = 0.99
    source_pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(os.path.join(pointcloud_save_path, "source.pcd"), source_pcd)

    # init_translation = kitti_loader.calib.T_velo_imu
    init_translation = np.eye(4)

    target = init_translation @ np.hstack((source, np.ones((np.asarray(source).shape[0], 1)))).T
    target = target.T
    target = target[:, :3]

    if add_noise:
        # set 10 centimeters for standard deviation (xyz) # maybe make it uniform for distance!
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.special_ortho_group.html
        noise = np.random.normal(0., .01, target.shape)
        target += noise

    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target)
    colors[:, 0] = 0.0
    colors[:, 1] = 0.99
    target_pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(os.path.join(pointcloud_save_path, "noisy_target.pcd"), target_pcd)

    print("init_translation = ")
    print(init_translation)

    result = o3d.registration.registration_icp(
        source = source_pcd,
        target = target_pcd,
        max_correspondence_distance = 10,  # todo check: meters?
        # init = init_translation,
        estimation_method = o3d.registration.TransformationEstimationPointToPoint())

    print("icp translation = ")
    print(result.transformation)

    # threshold: 1 meter
    evaluation_results = o3d.registration.evaluate_registration(source_pcd, target_pcd, 1, result.transformation)

    print(evaluation_results)
    print()


if __name__ == '__main__':
    icp_demo(True)
