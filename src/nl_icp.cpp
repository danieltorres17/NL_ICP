#include <iostream>

#include <Eigen/Dense>

#include <ceres/ceres.h>

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/visualization/common/common.h>
#include <pcl/visualization/pcl_visualizer.h>

#include "PointToPointResidual.h"

int main(int argc, char** argv) {
  const std::string cloud_path = "../data/region_growing_tutorial.pcd";
  // load point cloud
  pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud(new pcl::PointCloud<pcl::PointXYZ>());
  if (pcl::io::loadPCDFile(cloud_path, *target_cloud) < 0) {
    std::cout << "Unable to load target point cloud\n";
  }

  pcl::PointXYZ pt_min, pt_max;
  pcl::getMinMax3D(*target_cloud, pt_min, pt_max);
  std::cout << "Min: " << pt_min << "\n";
  std::cout << "Max: " << pt_max << "\n";

  std::cout << "Before downsampling: " << target_cloud->size() << "\n";
  pcl::VoxelGrid<pcl::PointXYZ> grid;
  grid.setInputCloud(target_cloud);
  grid.setLeafSize(0.1, 0.1, 0.1);
  grid.filter(*target_cloud);
  std::cout << "After downsampling: " << target_cloud->size() << "\n";

  // normal estimation
  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
  ne.setInputCloud(target_cloud);
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
  ne.setSearchMethod(tree);
  pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
  ne.setRadiusSearch(0.05);
  ne.compute(*cloud_normals);
  std::cout << "Number of normals estimated: " << cloud_normals->size() << "\n";

  // apply transformation to the cloud to generate a test set
  const double theta = M_PI / 10;
  Eigen::Affine3d transform = Eigen::Affine3d::Identity();
  transform.translation() << 0.1, 0.0, -0.02;
  transform.rotate(Eigen::AngleAxisd(theta, Eigen::Vector3d::UnitZ()));

  // create new point cloud to load transform cloud into
  pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud(new pcl::PointCloud<pcl::PointXYZ>());
  pcl::transformPointCloud(*target_cloud, *source_cloud, transform);

  // visualize results
  pcl::visualization::PCLVisualizer viewer("Transform Viewer");
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> target_cloud_color_handler(target_cloud,
                                                                                             255, 255, 255);
  viewer.addPointCloud(target_cloud, target_cloud_color_handler, "Target Cloud");
  viewer.addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(target_cloud, cloud_normals);

  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> source_cloud_color_handler(source_cloud,
                                                                                             230, 20, 20);
  viewer.addPointCloud(source_cloud, source_cloud_color_handler, "Source Cloud");

  viewer.addCoordinateSystem(1.0, "cloud", 0);
  viewer.setBackgroundColor(0.05, 0.05, 0.05, 0);  // background color
  viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "Target Cloud");
  viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "Source Cloud");

  Eigen::Vector3d rand = 0.1 * Eigen::Vector3d::Random();
  std::cout << "rand: " << rand << "\n";

  // set up ICP problem
  ceres::Problem problem;
  double params[6] = {0.01, 0.0, M_PI / 6, 0.05, 0.0, 0.1};
  for (int ii = 0; ii < target_cloud->size(); ii++) {
    Eigen::Vector3d target_pt(target_cloud->at(ii).x, target_cloud->at(ii).y, target_cloud->at(ii).z);
    Eigen::Vector3d source_pt(source_cloud->at(ii).x, source_cloud->at(ii).y, source_cloud->at(ii).z);

    ceres::CostFunction* cost_function = Pt2PtResidual::Create(target_pt, source_pt);

    problem.AddResidualBlock(cost_function, nullptr, params);
  }
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.minimizer_progress_to_stdout = true;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << "Summary:\n" << summary.FullReport() << "\n";
  std::cout << "Params final: " << params[0] << ", " << params[1] << ", " << params[2] << ", " << params[3]
            << ", " << params[4] << ", " << params[5] << "\n";

  Eigen::Affine3f T_final =
      pcl::getTransformation(params[3], params[4], params[5], params[0], params[1], params[2]);
  Eigen::Affine3f T_final_inv = T_final.inverse();
  float roll, pitch, yaw;
  pcl::getEulerAngles(T_final_inv, roll, pitch, yaw);
  std::cout << "x: " << T_final_inv.translation().x() << ", y: " << T_final_inv.translation().y()
            << ", z: " << T_final_inv.translation().z() << "\n";
  std::cout << "roll: " << roll << ", pitch: " << pitch << ", yaw: " << yaw << "\n";

  // create new point cloud to load transform cloud into
  pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>());
  pcl::transformPointCloud(*source_cloud, *transformed_cloud, T_final);
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> transformed_cloud_color_handler(
      transformed_cloud, 0, 255, 255);
  viewer.addPointCloud(transformed_cloud, transformed_cloud_color_handler, "Transformed Cloud");
  viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2,
                                          "Transformed Cloud");

  while (!viewer.wasStopped()) {
    viewer.spinOnce(100);
  }

  return 0;
}
