#include "NL_ICP/PointToPointResidual.h"

#include <ceres/rotation.h>
#include <ceres/autodiff_cost_function.h>

Pt2PtResidual::Pt2PtResidual(const Eigen::Vector3d& target_pt, const Eigen::Vector3d& source_pt)
  : target_pt_(target_pt), source_pt_(source_pt) {}

template <typename T>
bool Pt2PtResidual::operator()(const T* const params, T* residuals) const {
  // rotate source point by estimated rotation
  T pt[3] = {T(source_pt_[0]), T(source_pt_[1]), T(source_pt_[2])};
  ceres::AngleAxisRotatePoint(params, pt, pt);

  // apply translation to source point
  pt[0] += params[3];
  pt[1] += params[4];
  pt[2] += params[5];

  // update residuals
  residuals[0] = pt[0] - T(target_pt_[0]);
  residuals[1] = pt[1] - T(target_pt_[1]);
  residuals[2] = pt[2] - T(target_pt_[2]);

  return true;
}

ceres::CostFunction* Pt2PtResidual::Create(const Eigen::Vector3d& target_pt, const Eigen::Vector3d& source_pt) {
  return (new ceres::AutoDiffCostFunction<Pt2PtResidual, 3, 6> (new Pt2PtResidual(target_pt, source_pt)));
}
