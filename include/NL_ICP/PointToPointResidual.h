#ifndef POINT_TO_POINT_RESIDUAL_H_
#define POINT_TO_POINT_RESIDUAL_H_

#include <ceres/ceres.h>

class Pt2PtResidual {
public:
  Pt2PtResidual(const Eigen::Vector3d& target_pt, const Eigen::Vector3d& source_pt);
  // operator overload used by ceres solver
  template <typename T>
  bool operator()(const T* const params, T* residuals) const;
  // to create new residual block
  static ceres::CostFunction* Create(const Eigen::Vector3d& target_pt, const Eigen::Vector3d& source_pt);

private:
  const Eigen::Vector3d target_pt_;
  const Eigen::Vector3d source_pt_;
};

#endif
