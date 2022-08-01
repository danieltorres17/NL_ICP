#ifndef POINT_TO_PLANE_RESIDUAL_H_
#define POINT_TO_PLANE_RESIDUAL_H_

#include <ceres/ceres.h>

class Pt2PlResidual {
public:
  Pt2PlResidual(const Eigen::Vector3d& target_pt_normal, const Eigen::Vector3d& target_pt,
                const Eigen::Vector3d& source_pt);
  // operator overload used by ceres
  template <typename T>
  bool operator()(const T* const params, T* residuals) const;
  // to create new residual block
  static ceres::CostFunction* Create(const Eigen::Vector3d& target_pt_normal,
                                     const Eigen::Vector3d& target_pt, const Eigen::Vector3d& source_pt);

private:
  const Eigen::Vector3d target_pt_normal_;
  const Eigen::Vector3d target_pt_;
  const Eigen::Vector3d source_pt_;
};

#endif
