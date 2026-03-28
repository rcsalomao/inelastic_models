#pragma once

#include <vector>

namespace common {

using std::vector;

struct MaterialProperties {
    double young;
    double poisson;
};

void strain_to_stress(double const* strain, double young, double poisson,
                      double* stress);

void calc_deviatoric_parcel(double const* matriz, double* deviatoric);

void calc_n(double const* Sigma, double* n);

vector<double> create_hist(
    vector<std::tuple<double, double, unsigned>> intervals);

void calc_valores_principais(double const* matriz, double* valores_principais);

void sigma_princ_to_epsilon_princ(double const* sigma_princ, double young,
                                  double poisson, double* epsilon_princ);

}  // namespace common
