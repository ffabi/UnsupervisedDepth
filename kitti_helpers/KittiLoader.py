import pykitti
import open3d as o3d
from global_variables import *
import cv2
import numpy as np


class KittiLoader:

    def __init__(self, date, drive) -> None:
        super().__init__()

        self.date = date
        self.drive = drive

        self.dataset = pykitti.raw(dataset_base_path, date, drive)
        self.calib = self.dataset.calib

    def load_velodyne_pointcloud_array(self, frame):
        return self.dataset.get_velo(frame)

    def load_rgb_image(self, frame, camera_id = "02"):
        image_path = os.path.join(dataset_base_path, self.date, "2011_09_26_drive_" + self.drive + "_sync",
                                  "image_" + camera_id, "data", str(frame).zfill(10) + ".png")
        # print(image_path)
        assert os.path.exists(image_path), f"Not found: {image_path}"
        return cv2.imread(image_path)

    def get_all_data(self, frame):
        velo = self.load_velodyne_pointcloud_array(frame)
        image = self.load_rgb_image(frame)
        proj_mat = self.get_velo_to_cam2_projection_matrix()
        return velo, image, self.calib, proj_mat

    def get_velo_to_cam2_projection_matrix(self):
        P_velo2cam_ref = self.calib.T_cam0_velo_unrect
        R_ref2rect = self.calib.R_rect_00
        P_rect2cam2 = self.calib.P_rect_20  # .reshape((3, 4))
        projection_matrix = P_rect2cam2 @ R_ref2rect @ P_velo2cam_ref
        return projection_matrix
