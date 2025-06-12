#pragma once

#include <Eigen/Dense>

namespace common {

using Eigen::Matrix3d;
using Eigen::Vector3d;
using std::vector;

Matrix3d strain_to_stress(const Matrix3d& strain, double young, double poisson);

Matrix3d stress_to_strain(const Matrix3d& stress, double young, double poisson);

std::tuple<Matrix3d, Matrix3d> calc_deviatoric_hydrostatic_parcel(
    Matrix3d& matriz);

Matrix3d calc_n(Matrix3d& Sigma);

Matrix3d deviatoric_to_tensor(Matrix3d& deviatoric_parcel);

vector<double> create_hist(
    vector<std::tuple<double, double, unsigned>> intervals);

Vector3d calc_valores_principais(Matrix3d& matriz);

Vector3d sigma_princ_to_epsilon_princ(Vector3d& sigma_princ, double young,
                                      double poisson);

}  // namespace common
