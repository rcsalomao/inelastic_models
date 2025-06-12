#include <Eigen/Dense>
#include <ranges>

namespace common {

namespace vw = std::ranges::views;
using Eigen::Matrix3d;
using Eigen::Vector3d;
using std::vector;

Matrix3d strain_to_stress(const Matrix3d& strain, double young,
                          double poisson) {
    Matrix3d stress = Matrix3d::Zero();
    double a = (young / (1 + poisson)) * ((1 - poisson) / (1 - 2 * poisson));
    double b = poisson * young / ((1 + poisson) * (1 - 2 * poisson));
    double G = young / (2 * (1 + poisson));
    double c = 2 * G;
    stress(0, 0) = a * strain(0, 0) + b * (strain(1, 1) + strain(2, 2));
    stress(1, 1) = a * strain(1, 1) + b * (strain(0, 0) + strain(2, 2));
    stress(2, 2) = a * strain(2, 2) + b * (strain(1, 1) + strain(0, 0));
    stress(0, 1) = stress(1, 0) = c * strain(0, 1);
    stress(0, 2) = stress(2, 0) = c * strain(0, 2);
    stress(2, 1) = stress(1, 2) = c * strain(2, 1);
    return stress;
}

Matrix3d stress_to_strain(const Matrix3d& stress, double young,
                          double poisson) {
    Matrix3d strain = Matrix3d::Zero();
    double a = 1 / young;
    double b = -poisson / young;
    double G = young / (2 * (1 + poisson));
    double c = 1 / (2 * G);
    strain(0, 0) = a * stress(0, 0) + b * (stress(1, 1) + stress(2, 2));
    strain(1, 1) = a * stress(1, 1) + b * (stress(0, 0) + stress(2, 2));
    strain(2, 2) = a * stress(2, 2) + b * (stress(1, 1) + stress(0, 0));
    strain(0, 1) = strain(1, 0) = c * stress(0, 1);
    strain(0, 2) = strain(2, 0) = c * stress(0, 2);
    strain(2, 1) = strain(1, 2) = c * stress(2, 1);
    return strain;
}

std::tuple<Matrix3d, Matrix3d> calc_deviatoric_hydrostatic_parcel(
    Matrix3d& matriz) {
    Matrix3d hydrostatic = matriz.trace() / 3 * Matrix3d::Identity();
    return {matriz - hydrostatic, hydrostatic};
}

Matrix3d calc_n(Matrix3d& Sigma) { return Sigma / Sigma.norm(); }

Matrix3d transform_matrix{
    {2.0 / 3, -1.0 / 3, -1.0 / 3},
    {-1.0 / 3, 2.0 / 3, -1.0 / 3},
    {-1.0 / 3, -1.0 / 3, 2.0 / 3},
};

Matrix3d deviatoric_to_tensor(Matrix3d& deviatoric_parcel) {
    Matrix3d out = deviatoric_parcel;
    Vector3d deviatoric_parcel_diag = deviatoric_parcel.diagonal();
    Vector3d out_diag = transform_matrix.ldlt().solve(deviatoric_parcel_diag);
    for (long i = 0; i < out_diag.size(); i++) {
        out(i, i) = out_diag(i);
    }
    return out;
}

vector<double> create_hist(
    vector<std::tuple<double, double, unsigned>> intervals) {
    vector<double> vo;
    for (auto [a, b, n] : intervals) {
        double dx = (b - a) / n;
        for (unsigned i{0}; i < n + 1; i++) {
            vo.push_back(a + i * dx);
        }
    }
    return vo;
}

Vector3d calc_valores_principais(Matrix3d& matriz) {
    return matriz.eigenvalues().real();
}

Vector3d sigma_princ_to_epsilon_princ(Vector3d& sigma_princ, double young,
                                      double poisson) {
    double sum_sigma_princ = sigma_princ.sum();
    Vector3d out;
    for (auto [i, s] : vw::enumerate(sigma_princ)) {
        out(i) = ((1.0 + poisson) * s - poisson * sum_sigma_princ) / young;
    }
    return out;
}

}  // namespace common
