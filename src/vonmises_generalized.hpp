#pragma once

#include <vector>

#include "common.hpp"

namespace vonmises_generalized {

struct ModelParams {
    double sigma_y;
    double Hiso;
    double Hkin;
    double beta;
    double delta;

    ModelParams(double sigma_y, double Hiso, double Hkin, double beta,
                double delta)
        : sigma_y{sigma_y}, Hiso{Hiso}, Hkin{Hkin}, beta{beta}, delta{delta} {};
};

std::vector<std::tuple<double, double>> run(common::MaterialProperties& mat_p,
                                            ModelParams& mod_p,
                                            std::vector<double> hist);
}  // namespace vonmises_generalized
