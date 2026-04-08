import open3d as o3d
import numpy as np
# import pcl  # Uncomment if python-pcl is successfully built in your environment

class PointCloudProcessor:
    def __init__(self, pcd_path):
        self.pcd = o3d.io.read_point_cloud(pcd_path)
        
    def downsample_and_remove_noise(self):
        """Voxel downsampling and statistical outlier removal"""
        voxel_down_pcd = self.pcd.voxel_down_sample(voxel_size=0.05)
        cl, ind = voxel_down_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        clean_pcd = voxel_down_pcd.select_by_index(ind)
        return clean_pcd