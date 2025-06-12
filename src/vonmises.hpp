#pragma once

#include <vector>

#include "common.hpp"

namespace vonmises {

struct ModelParams {
    double sigma_y;
    double Hiso;
    double Hkin;

    ModelParams(double sigma_y, double Hiso, double Hkin)
        : sigma_y{sigma_y}, Hiso{Hiso}, Hkin{Hkin} {};
};

std::vector<std::tuple<double, double>> run(common::MaterialProperties& mat_p,
                                            ModelParams& mod_p,
                                            std::vector<double> hist);
}  // namespace vonmises
