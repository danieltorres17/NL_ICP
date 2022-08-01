#include "NL_ICP/PointToPlaneResidual.h"

#include <ceres/rotation.h>
#include <ceres/autodiff_cost_function.h>

Pt2PlResidual::Pt2PlResidual(const Eigen::Vector3d& target_pt_normal, const Eigen::Vector3d& target_pt,
                             const Eigen::Vector3d& source_pt)
  : target_pt_normal_(target_pt_normal), target_pt_(target_pt), source_pt_(source_pt) {}

template <typename T>
bool Pt2PlResidual::operator()(const T* const params, T* residuals) const {
  // rotate source point by estimated rotation
  T pt[3] = {T(source_pt_[0]), T(source_pt_[1]), T(source_pt_[2])};
  ceres::AngleAxisRotatePoint(params, pt, pt);

  // apply translation to source point
  pt[0] += params[3];
  pt[1] += params[4];
  pt[2] += params[5];

  // update residual
  T res0 = pt[0] - T(target_pt_[0]);
  T res1 = pt[1] - T(target_pt_[1]);
  T res2 = pt[2] - T(target_pt_[2]);
  T res[3] = {res0, res1, res2};

  // dot residual with the target point's normal vector
  T normal[3] = {T(target_pt_normal_[0]), T(target_pt_normal_[1]), T(target_pt_normal_[2])};
  residuals[0] = ceres::DotProduct(res, normal);

  return true;
}

ceres::CostFunction* Pt2PlResidual::Create(const Eigen::Vector3d& target_pt_normal,
                                           const Eigen::Vector3d& target_pt,
                                           const Eigen::Vector3d& source_pt) {
  return new ceres::AutoDiffCostFunction<Pt2PlResidual, 1, 6>(
      new Pt2PlResidual(target_pt_normal, target_pt, source_pt));
}
