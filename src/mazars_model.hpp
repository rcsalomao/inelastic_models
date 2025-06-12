#pragma once

#include <vector>

#include "common.hpp"

namespace mazars_model {

struct ModelParams {
    double epsilon_d0;
    double At;
    double Bt;
    double Ac;
    double Bc;

    ModelParams(double epsilon_d0, double At, double Bt, double Ac, double Bc)
        : epsilon_d0{epsilon_d0}, At{At}, Bt{Bt}, Ac{Ac}, Bc{Bc} {};
};

std::vector<std::tuple<double, double>> run(common::MaterialProperties& mat_p,
                                            ModelParams& mod_p,
                                            std::vector<double> hist);
}  // namespace mazars_model
