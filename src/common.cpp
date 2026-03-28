#include <gel.hpp>
#include <vector>

namespace common {

using std::vector;

void strain_to_stress(double const* strain, double young, double poisson,
                      double* stress) {
    double a = (young / (1 + poisson)) * ((1 - poisson) / (1 - 2 * poisson));
    double b = poisson * young / ((1 + poisson) * (1 - 2 * poisson));
    double G = young / (2 * (1 + poisson));
    double c = 2 * G;
    stress[0] = a * strain[0] + b * (strain[4] + strain[8]);
    stress[4] = a * strain[4] + b * (strain[0] + strain[8]);
    stress[8] = a * strain[8] + b * (strain[4] + strain[0]);
    stress[1] = stress[3] = c * strain[1];
    stress[2] = stress[6] = c * strain[2];
    stress[7] = stress[5] = c * strain[7];
}

void calc_deviatoric_parcel(double const* matriz, double* deviatoric) {
    gel::cpyarr(matriz, 9, deviatoric);
    double hydrostatic_parcel = gel::Aii3x3(matriz) / 3.0;
    deviatoric[0] -= hydrostatic_parcel;
    deviatoric[4] -= hydrostatic_parcel;
    deviatoric[8] -= hydrostatic_parcel;
}

void calc_n(double const* Sigma, double* n) {
    gel::Aij3x3_mul_v(Sigma, 1.0 / gel::Aij3x3norm(Sigma), n);
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

void calc_valores_principais(double const* matriz, double* valores_principais) {
    double E_vec[9];
    gel::Aij3x3eigen_sym(matriz, valores_principais, E_vec);
}

void sigma_princ_to_epsilon_princ(double const* sigma_princ, double young,
                                  double poisson, double* epsilon_princ) {
    double sum_sigma_princ = sigma_princ[0] + sigma_princ[1] + sigma_princ[2];
    for (size_t i{0}; i < 3; i++) {
        epsilon_princ[i] =
            ((1.0 + poisson) * sigma_princ[i] - poisson * sum_sigma_princ) /
            young;
    }
}

}  // namespace common
