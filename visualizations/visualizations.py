# with the help of https://github.com/intel-isl/Open3D/blob/master/examples/Python/Advanced/load_save_viewpoint.py
# location: https://g.page/berthold-apotheke?share

# advertising pillar at
"""
{
	"class_name" : "ViewTrajectory",
	"interval" : 29,
	"is_loop" : false,
	"trajectory" :
	[
		{
			"boundingbox_max" : [ 103.63133239746094, 51.669380187988281, 4.1334571838378906 ],
			"boundingbox_min" : [ -58.307891845703125, -132.62893676757813, -14.400491714477539 ],
			"field_of_view" : 59.999999999999993,
			"front" : [ 0.018973145918580625, -0.019439299310908174, 0.99963099860711269 ],
			"lookat" : [ 44.169410236514629, -32.691518989036041, -5.9244249593080172 ],
			"up" : [ 0.80701176464346913, -0.58992741062627274, -0.026789212732277495 ],
			"zoom" : 0.054556141492597077
		}
	],
	"version_major" : 1,
	"version_minor" : 0
}
"""
# simple
"""
{
	"class_name" : "ViewTrajectory",
	"interval" : 29,
	"is_loop" : false,
	"trajectory" : 
	[
		{
			"boundingbox_max" : [ 79.689888000488281, 50.037471771240234, 2.9135265350341797 ],
			"boundingbox_min" : [ 5.544562816619873, -39.130084991455078, -2.0257730484008789 ],
			"field_of_view" : 60.0,
			"front" : [ -0.99806364301781392, -0.038824296606872533, 0.048596692058115648 ],
			"lookat" : [ 42.617225408554077, 5.4536933898925781, 0.44387674331665039 ],
			"up" : [ 0.048459653127398854, 0.0044635752809101415, 0.99881516734302944 ],
			"zoom" : 0.39783333333333359
		}
	],
	"version_major" : 1,
	"version_minor" : 0
}
"""

import os
from global_variables import *
import numpy as np
import pykitti
import open3d as o3d


def simple():
    pointcloud = o3d.io.read_point_cloud(os.path.join(pointcloud_load_path, "rgb_colored.pcd"))
    # pointcloud = o3d.io.read_point_cloud(os.path.join(pointcloud_load_path, "original.pcd"))
    # pointcloud = o3d.io.read_point_cloud(os.path.join(pointcloud_load_path, "filtered.pcd"))
    # pointcloud = o3d.io.read_point_cloud(os.path.join(pointcloud_load_path, "filtered.pcd"))

    o3d.visualization.draw_geometries_with_editing([pointcloud])


def demo_orig_filter_rgb():
    pointcloud = o3d.io.read_point_cloud(os.path.join(pointcloud_load_path, "original.pcd"))
    custom_draw_geometry(pointcloud)
    # pointcloud = o3d.io.read_point_cloud(os.path.join(pointcloud_load_path, "filtered.pcd"))
    # custom_draw_geometry(pointcloud)
    # pointcloud = o3d.io.read_point_cloud(os.path.join(pointcloud_load_path, "original.pcd"))
    # print(pointcloud.points[0])
    # custom_draw_geometry(pointcloud)
    # pointcloud = o3d.io.read_point_cloud(os.path.join(pointcloud_load_path, "transformed_original.pcd"))
    # print(pointcloud.points[0])
    # custom_draw_geometry(pointcloud)
    # pointcloud = o3d.io.read_point_cloud(os.path.join(pointcloud_load_path, "rgb_colored.pcd"))
    # custom_draw_geometry(pointcloud)


def custom_draw_geometry(pcd):
    # The following code achieves the same effect as:
    # o3d.visualization.draw_geometries([pcd])
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    ctr = vis.get_view_control()
    param = o3d.io.read_pinhole_camera_parameters("new_view.json")
    cam = vis.get_view_control().convert_to_pinhole_camera_parameters()
    cam.extrinsic = param.extrinsic
    cam.intrinsic = param.intrinsic
    vis.add_geometry(pcd)
    ctr.convert_from_pinhole_camera_parameters(cam)
    vis.run()

    param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters("new_view.json", param)

    vis.destroy_window()


def pillar_demo():
    # demo_names = ["initial", "imu", "point", "plane"]
    demo_names = ["imu", "point", "plane"]

    for name in demo_names:
        frame_target = o3d.io.read_point_cloud(os.path.join(pointcloud_load_path, "frame_target_" + name + ".pcd"))
        frame_source = o3d.io.read_point_cloud(os.path.join(pointcloud_load_path, "frame_source_" + name + ".pcd"))

        custom_draw_geometry(frame_source + frame_target)


if __name__ == '__main__':
    # pillar_demo()
    # frame = o3d.io.read_point_cloud(os.path.join(pointcloud_load_path, "accumulated.pcd"))
    # custom_draw_geometry(frame)

    simple()
    # demo_orig_filter_rgb()
