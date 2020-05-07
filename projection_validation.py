import time

from kitti_helpers.KittiLoader import KittiLoader

from global_variables import *

from helpers.data_helper import *

from helpers.projections import *

import open3d as o3d

from global_variables import *


def compare_own_depth_to_kitti_depth(add_noise = False):
    kitti_depth = load_kitti_depth(
        "/root/ffabi_shared_folder/datasets/_original_datasets/kitti_all/kitti_depth/val/2011_09_26_drive_0005_sync/proj_depth/velodyne_raw/image_02/" + test_image_name + ".png")
    kitti_points_2d = convert_2d_image_to_2d_points(kitti_depth)

    # set Z value to zero for this experiment
    kitti_points_2d[:, 2] = 0

    kitti_loader = KittiLoader(test_date, test_drive)
    points_3d, rgb_image, calibration, projection_matrix = kitti_loader.get_all_data(test_frame_index)

    # remove intensity
    points_3d = points_3d[:, :3]

    if add_noise:
        noise = np.random.normal(0., .01, points_3d.shape)
        points_3d += noise

    points_2d = project_points_3d_to_points_2d(points_3d.T, projection_matrix)

    image_height, image_width = kitti_depth.shape
    ordered_indexes = create_filter(points_2d, points_3d, image_width = image_width, image_height = image_height)
    # Filter out 2d points
    points_2d = points_2d[ordered_indexes]
    points_2d[:, 0] = np.round(points_2d[:, 0])
    points_2d[:, 1] = np.round(points_2d[:, 1])

    # set Z value to zero for this experiment
    points_2d[:, 2] = 0

    # icp

    points_2d_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_2d))

    kitti_points_2d_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(kitti_points_2d))

    result = o3d.registration.registration_icp(
        source = points_2d_pcd,
        target = kitti_points_2d_pcd,
        max_correspondence_distance = 2,
        # init = init_translation,
        estimation_method = o3d.registration.TransformationEstimationPointToPoint())

    print("icp result:")
    print(result.transformation)

    print(len(kitti_points_2d), len(points_2d))
    # threshold: 1 meter
    evaluation_results = o3d.registration.evaluate_registration(points_2d_pcd, kitti_points_2d_pcd, 10,
                                                                result.transformation)

    print(evaluation_results)
    print()


def compare_own_noisy_depth_to_own_depth(add_noise = True):
    kitti_loader = KittiLoader(test_date, test_drive)
    source_points_3d, rgb_image, calibration, projection_matrix = kitti_loader.get_all_data(test_frame_index)
    target_points_3d = source_points_3d.copy()

    # remove intensity
    source_points_3d = source_points_3d[:, :3]
    target_points_3d = target_points_3d[:, :3]

    if add_noise:
        noise = np.random.normal(0., 0.1, source_points_3d.shape)
        source_points_3d += noise

    source_points_2d = project_points_3d_to_points_2d(source_points_3d.T, projection_matrix)
    target_points_2d = project_points_3d_to_points_2d(target_points_3d.T, projection_matrix)

    image_height, image_width, _ = rgb_image.shape

    # Filter out 2d points
    ordered_indexes = create_filter(source_points_2d, source_points_3d, image_width = image_width,
                                    image_height = image_height)
    source_points_2d = source_points_2d[ordered_indexes]
    source_points_2d[:, 0] = np.round(source_points_2d[:, 0])
    source_points_2d[:, 1] = np.round(source_points_2d[:, 1])
    source_points_2d[:, 2] = 0

    # Filter out 2d points
    ordered_indexes = create_filter(target_points_2d, target_points_3d, image_width = image_width,
                                    image_height = image_height)
    target_points_2d = target_points_2d[ordered_indexes]
    target_points_2d[:, 0] = np.round(target_points_2d[:, 0])
    target_points_2d[:, 1] = np.round(target_points_2d[:, 1])
    target_points_2d[:, 2] = 0

    target_points_3d = target_points_3d[ordered_indexes]

    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target_points_3d)
    colors = np.empty(shape = target_points_3d.shape)
    colors[:, 0] = 0.0
    colors[:, 1] = 0.0
    colors[:, 2] = 0.99
    target_pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(os.path.join(pointcloud_save_path, "filtered.pcd"), target_pcd)

    # icp

    source_points_2d_pcd = o3d.geometry.PointCloud()
    source_points_2d_pcd.points = o3d.utility.Vector3dVector(source_points_2d)

    target_points_2d_pcd = o3d.geometry.PointCloud()
    target_points_2d_pcd.points = o3d.utility.Vector3dVector(target_points_2d)

    result = o3d.registration.registration_icp(
        source = source_points_2d_pcd,
        target = target_points_2d_pcd,
        max_correspondence_distance = 2,
        # init = init_translation,
        estimation_method = o3d.registration.TransformationEstimationPointToPoint())

    print("icp result:")
    print(result.transformation)

    print(len(target_points_2d), len(source_points_2d))
    # threshold: 1 meter
    evaluation_results = o3d.registration.evaluate_registration(source_points_2d_pcd, target_points_2d_pcd, 10,
                                                                result.transformation)

    print(evaluation_results)
    print()


def filter_and_color_pointcloud(add_noise = False):
    kitti_loader = KittiLoader(test_date, test_drive)
    points_3d, rgb_image, calibration, projection_matrix = kitti_loader.get_all_data(test_frame_index)

    # remove intensity
    points_3d = points_3d[:, :3]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.append(points_3d, [[0, 0, 0], [-0.27, 0, 0]], axis = 0))
    pcd.paint_uniform_color([0, 1, 0])
    o3d.io.write_point_cloud(os.path.join(pointcloud_save_path, "original.pcd"), pcd)

    if add_noise:
        noise = np.random.normal(0., .01, points_3d.shape)
        points_3d += noise

    points_2d = project_points_3d_to_points_2d(points_3d.T, projection_matrix)

    image_height, image_width, _ = rgb_image.shape

    # Filter out points based on both projection
    ordered_indexes = create_filter(points_2d, points_3d, image_width = image_width,
                                    image_height = image_height)

    points_2d = points_2d[ordered_indexes]
    points_2d[:, 0] = np.round(points_2d[:, 0])
    points_2d[:, 1] = np.round(points_2d[:, 1])
    points_3d = points_3d[ordered_indexes]

    pcd.points = o3d.utility.Vector3dVector(points_3d)
    pcd.paint_uniform_color([0, 1, 0])
    o3d.io.write_point_cloud(os.path.join(pointcloud_save_path, "filtered.pcd"), pcd)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)

    colors = []
    for point_2d in points_2d:
        colors.append(rgb_image[int(point_2d[1]), int(point_2d[0])] / 255.)

    colors = np.asarray(colors)
    colors = colors[:, ::-1]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(os.path.join(pointcloud_save_path, "rgb_colored.pcd"), pcd)


def synthesize_next_rgb_image(dilation = 1):
    kitti_loader = KittiLoader(test_date, test_drive)
    source_points_3d, source_rgb_image, _, projection_matrix = kitti_loader.get_all_data(test_frame_index)
    source_points_3d = np.append(source_points_3d, [[1, 0, 0, 0]], axis = 0)
    _, next_rgb_image, _, _ = kitti_loader.get_all_data(test_frame_index + dilation)

    source_T_w_imu = kitti_loader.dataset.oxts[test_frame_index].T_w_imu
    next_T_w_imu = kitti_loader.dataset.oxts[test_frame_index + dilation].T_w_imu

    # remove intensity
    source_points_3d = source_points_3d[:, :3]

    next_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(source_points_3d))
    o3d.io.write_point_cloud(os.path.join(pointcloud_save_path, "source_points_3d.pcd"), next_pcd)

    source_points_3d_homogeneous = np.hstack((source_points_3d, np.ones((np.asarray(source_points_3d).shape[0], 1))))
    next_points_3d = source_points_3d_homogeneous @ np.linalg.inv(source_T_w_imu) @ next_T_w_imu

    # remove homogeneous coordinate 1
    next_points_3d = next_points_3d[:, :3]

    next_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(next_points_3d))
    o3d.io.write_point_cloud(os.path.join(pointcloud_save_path, "next_points_3d.pcd"), next_pcd)

    source_points_2d = project_points_3d_to_points_2d(source_points_3d.T, projection_matrix)
    next_points_2d = project_points_3d_to_points_2d(next_points_3d.T, projection_matrix)

    image_height, image_width, _ = source_rgb_image.shape

    source_ordered_indexes = create_filter(source_points_2d, source_points_3d, image_width = image_width,
                                           image_height = image_height)
    next_ordered_indexes = create_filter(next_points_2d, next_points_3d, image_width = image_width,
                                         image_height = image_height)
    ordered_indexes = list(set(source_ordered_indexes) & set(next_ordered_indexes))

    source_points_2d = source_points_2d[ordered_indexes]
    source_points_2d[:, 0] = np.round(source_points_2d[:, 0])
    source_points_2d[:, 1] = np.round(source_points_2d[:, 1])

    next_points_2d = next_points_2d[ordered_indexes]
    next_points_2d[:, 0] = np.round(next_points_2d[:, 0])
    next_points_2d[:, 1] = np.round(next_points_2d[:, 1])

    next_points_3d = next_points_3d[ordered_indexes]

    next_pcd.points = o3d.utility.Vector3dVector(next_points_3d)
    o3d.io.write_point_cloud(os.path.join(pointcloud_save_path, "next_points_3d_filtered.pcd"), next_pcd)

    next_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(next_points_3d))

    colors = []
    for point_2d in source_points_2d:
        colors.append(source_rgb_image[int(point_2d[1]), int(point_2d[0])])

    synthesized_image = np.zeros((image_height, image_width, 3), dtype = np.float32)
    for i in range(len(next_points_2d)):
        point = next_points_2d[i]
        synthesized_image[int(point[1]), int(point[0])] = colors[i]
    cv2.imwrite(os.path.join(synthesized_image_save_path, "synthesized_image_" + str(dilation) + ".png"),
                synthesized_image)

    colors = np.asarray(colors)
    colors = colors[:, ::-1]  # bgr to rgb
    next_pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(os.path.join(pointcloud_save_path, "next_points_3d_rgb_colored.pcd"), next_pcd)


def compare_noisy_pointcloud_to_pointcloud(add_noise = True):
    kitti_loader = KittiLoader(date = test_date, drive = test_drive)

    source = kitti_loader.load_velodyne_pointcloud_array(test_frame_index)[:, :3]
    source_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(source))
    o3d.io.write_point_cloud(os.path.join(pointcloud_save_path, "source.pcd"), source_pcd)

    init_transformation = kitti_loader.calib.T_velo_imu
    # init_transformation = np.eye(4)

    target = init_transformation @ np.hstack((source, np.ones((np.asarray(source).shape[0], 1)))).T
    target = target.T
    target = target[:, :3]

    if add_noise:
        noise = np.random.normal(0., .01, target.shape)
        target += noise

    target_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target))
    o3d.io.write_point_cloud(os.path.join(pointcloud_save_path, "noisy_target.pcd"), target_pcd)

    print("init_transformation = ")
    print(init_transformation)

    result = o3d.registration.registration_icp(
        source = source_pcd,
        target = target_pcd,
        max_correspondence_distance = 10,  # todo check: meters?
        init = init_transformation,
        estimation_method = o3d.registration.TransformationEstimationPointToPoint())

    print("icp translation = ")
    print(result.transformation)

    # threshold: 1 meter
    evaluation_results = o3d.registration.evaluate_registration(source_pcd, target_pcd, 1, result.transformation)

    print(evaluation_results)
    print()


def imu_validation():
    kitti_loader = KittiLoader(date = test_date, drive = test_drive)

    frame_offset = 70

    T_w_imu_0 = kitti_loader.dataset.oxts[test_frame_index].T_w_imu
    T_w_imu_1 = kitti_loader.dataset.oxts[test_frame_index + frame_offset].T_w_imu

    target = kitti_loader.load_velodyne_pointcloud_array(test_frame_index)[:, :3]
    source = kitti_loader.load_velodyne_pointcloud_array(test_frame_index + frame_offset)[:, :3]
    # source[:, 2] += 1.0

    target_pcd = o3d.geometry.PointCloud()
    colors = np.empty(shape = target.shape)
    colors[:, 0] = 0.99
    colors[:, 1] = 0.0
    colors[:, 2] = 0.0
    target_pcd.colors = o3d.utility.Vector3dVector(colors)

    source_pcd = o3d.geometry.PointCloud()
    colors = np.empty(shape = source.shape)
    colors[:, 0] = 0.0
    colors[:, 1] = 0.99
    colors[:, 2] = 0.0
    source_pcd.colors = o3d.utility.Vector3dVector(colors)

    target = T_w_imu_0 @ np.hstack((target, np.ones((np.asarray(target).shape[0], 1)))).T
    target = target.T
    target = target[:, :3]

    source = T_w_imu_0 @ np.hstack((source, np.ones((np.asarray(source).shape[0], 1)))).T
    source = source.T
    source = source[:, :3]

    target_pcd.points = o3d.utility.Vector3dVector(target)
    source_pcd.points = o3d.utility.Vector3dVector(source)

    o3d.io.write_point_cloud(os.path.join(pointcloud_save_path, "frame_target_initial.pcd"), target_pcd)
    o3d.io.write_point_cloud(os.path.join(pointcloud_save_path, "frame_source_initial.pcd"), source_pcd)

    evaluation_results = o3d.registration.evaluate_registration(source_pcd, target_pcd, 5, np.eye(4))
    print("initial error without IMU nor ICP:", evaluation_results.inlier_rmse)
    print()

    target = kitti_loader.load_velodyne_pointcloud_array(test_frame_index)[:, :3]
    source = kitti_loader.load_velodyne_pointcloud_array(test_frame_index + frame_offset)[:, :3]

    target = T_w_imu_0 @ np.hstack((target, np.ones((np.asarray(target).shape[0], 1)))).T
    target = target.T
    target = target[:, :3]

    source = T_w_imu_1 @ np.hstack((source, np.ones((np.asarray(source).shape[0], 1)))).T
    source = source.T
    source = source[:, :3]

    target_pcd.points = o3d.utility.Vector3dVector(target)
    source_pcd.points = o3d.utility.Vector3dVector(source)

    evaluation_results = o3d.registration.evaluate_registration(source_pcd, target_pcd, 1, np.eye(4))
    print("error after IMU correction:", evaluation_results.inlier_rmse)
    print()
    o3d.io.write_point_cloud(os.path.join(pointcloud_save_path, "frame_target_imu.pcd"), target_pcd)
    o3d.io.write_point_cloud(os.path.join(pointcloud_save_path, "frame_source_imu.pcd"), source_pcd)

    # point-to-point ICP
    result = o3d.registration.registration_icp(
        source = source_pcd,
        target = target_pcd,
        max_correspondence_distance = 1,
        init = np.eye(4),
        estimation_method = o3d.registration.TransformationEstimationPointToPoint())

    source = result.transformation @ np.hstack((source, np.ones((np.asarray(source).shape[0], 1)))).T
    source = source.T
    source = source[:, :3]
    source_pcd.points = o3d.utility.Vector3dVector(source)

    evaluation_results = o3d.registration.evaluate_registration(source_pcd, target_pcd, 1, np.eye(4))
    print("error after point-to-point ICP:", evaluation_results.inlier_rmse)
    print()
    o3d.io.write_point_cloud(os.path.join(pointcloud_save_path, "frame_target_point.pcd"), target_pcd)
    o3d.io.write_point_cloud(os.path.join(pointcloud_save_path, "frame_source_point.pcd"), source_pcd)

    # point-to-plane ICP
    source_pcd.estimate_normals(search_param = o3d.geometry.KDTreeSearchParamHybrid(radius = 0.15, max_nn = 50))
    target_pcd.estimate_normals(search_param = o3d.geometry.KDTreeSearchParamHybrid(radius = 0.15, max_nn = 50))
    result = o3d.registration.registration_icp(
        source = source_pcd,
        target = target_pcd,
        max_correspondence_distance = 1,
        init = np.eye(4),
        estimation_method = o3d.registration.TransformationEstimationPointToPlane())

    source = result.transformation @ np.hstack((source, np.ones((np.asarray(source).shape[0], 1)))).T
    source = source.T
    source = source[:, :3]
    source_pcd.points = o3d.utility.Vector3dVector(source)

    evaluation_results = o3d.registration.evaluate_registration(source_pcd, target_pcd, 1, np.eye(4))
    print("error after point-to-plane ICP:", evaluation_results.inlier_rmse)
    print()

    source_pcd.normals = o3d.utility.Vector3dVector([])
    o3d.io.write_point_cloud(os.path.join(pointcloud_save_path, "frame_target_plane.pcd"), target_pcd)

    target_pcd.normals = o3d.utility.Vector3dVector([])
    o3d.io.write_point_cloud(os.path.join(pointcloud_save_path, "frame_source_plane.pcd"), source_pcd)

    print()


def accumulate():
    kitti_loader = KittiLoader(date = test_date, drive = test_drive)
    size = len(kitti_loader.dataset.velo_files)
    print(size)
    T_w_imu = kitti_loader.dataset.oxts[0].T_w_imu
    points = kitti_loader.load_velodyne_pointcloud_array(0)[:, :3]
    points = T_w_imu @ np.hstack((points, np.ones((np.asarray(points).shape[0], 1)))).T
    points = points.T
    points = points[:, :3]

    # todo np array bug!!!
    accumulated_points = points

    accumulated_pointcloud = o3d.geometry.PointCloud()
    accumulated_pointcloud.points = o3d.utility.Vector3dVector(accumulated_points)

    tmp_pointcloud = o3d.geometry.PointCloud()
    transform = np.eye(4)
    for index in range(1, size, 1):
        if index == size:
            break
        print(index)
        T_w_imu = kitti_loader.dataset.oxts[index].T_w_imu
        points = kitti_loader.load_velodyne_pointcloud_array(index)[:, :3]
        points = T_w_imu @ np.hstack((points, np.ones((np.asarray(points).shape[0], 1)))).T
        points = points.T
        points = points[:, :3]

        tmp_pointcloud.points = o3d.utility.Vector3dVector(points)

        result = o3d.registration.registration_icp(
            source = tmp_pointcloud,
            target = accumulated_pointcloud,
            max_correspondence_distance = 1,
            init = transform,
            estimation_method = o3d.registration.TransformationEstimationPointToPoint())

        transform = result.transformation
        points = transform @ np.hstack((points, np.ones((np.asarray(points).shape[0], 1)))).T
        points = points.T
        points = points[:, :3]

        accumulated_points = np.concatenate((accumulated_points, points))

        accumulated_pointcloud.points = o3d.utility.Vector3dVector(accumulated_points)

    print("Number of accumulated points:", accumulated_points.shape)
    colors = np.empty(shape = (len(accumulated_points), 3))
    colors[:, 0] = 0.99
    colors[:, 1] = 0.0
    colors[:, 2] = 0.0
    accumulated_pointcloud.colors = o3d.utility.Vector3dVector(colors)

    o3d.io.write_point_cloud(os.path.join(pointcloud_save_path, "accumulated.pcd"), accumulated_pointcloud)

    print()


if __name__ == '__main__':
    # filter_and_color_pointcloud()
    # synthesize_next_rgb_image(1)
    # compare_own_depth_to_kitti_depth()
    compare_own_noisy_depth_to_own_depth()
    # compare_noisy_pointcloud_to_pointcloud()
    # imu_validation()
    # accumulate()
    print()
