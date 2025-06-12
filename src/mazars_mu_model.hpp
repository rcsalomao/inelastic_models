#pragma once

#include <vector>

#include "common.hpp"

namespace mazars_mu_model {

struct ModelParams {
    double epsilon_d0t;
    double epsilon_d0c;
    double At;
    double Bt;
    double Ac;
    double Bc;
    double k;

    ModelParams(double epsilon_d0t, double epsilon_d0c, double At, double Bt,
                double Ac, double Bc, double k)
        : epsilon_d0t{epsilon_d0t},
          epsilon_d0c{epsilon_d0c},
          At{At},
          Bt{Bt},
          Ac{Ac},
          Bc{Bc},
          k{k} {};
};

std::vector<std::tuple<double, double>> run(common::MaterialProperties& mat_p,
                                            ModelParams& mod_p,
                                            std::vector<double> hist);
}  // namespace mazars_mu_model
