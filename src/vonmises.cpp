#include <Eigen/Dense>
#include <algorithm>
#include <print>
#include <ranges>
#include <vector>

#include "common.hpp"

namespace vonmises {

namespace rg = std::ranges;
namespace vw = std::ranges::views;
using Eigen::Matrix3d;
using Eigen::Vector3d;
using std::println;
using std::vector;

vector<std::tuple<double, double>> run(vector<double> hist) {
    // material params
    double poisson = 0.0;
    double young = 100;

    // vonmises params
    double sigma_y = 0.4 * sqrt(2.0 / 3.0);
    double Hiso = 20;
    double Hkin = 0;

    // initial values
    Matrix3d backstress_0 = Matrix3d::Zero();
    Matrix3d epsilon_p_0 = Matrix3d::Zero();
    double gamma_0 = 0.0;

    vector<double> strains{0.0};
    vector<double> stresses{0.0};
    vector<std::tuple<double, double>> data;

    for (double epsilon_11 : hist) {
        Matrix3d epsilon_1{
            {epsilon_11, 0, 0},
            {0, -poisson * epsilon_11, 0},
            {0, 0, -poisson * epsilon_11},
        };
        Matrix3d sigma_TR =
            common::strain_to_stress(epsilon_1 - epsilon_p_0, young, poisson);
        auto [s_TR, hyd_s_TR] =
            common::calc_deviatoric_hydrostatic_parcel(sigma_TR);
        Matrix3d backstress_TR = backstress_0;
        Matrix3d Sigma_TR = s_TR - backstress_TR;
        double gamma_TR = gamma_0;
        auto [e_p_0, hyd_e_p_0] =
            common::calc_deviatoric_hydrostatic_parcel(epsilon_p_0);
        Matrix3d e_p_TR = e_p_0;

        double f = Sigma_TR.norm() - (sigma_y + Hiso * gamma_TR);

        if (f <= 0.0) {
            Matrix3d e_p_1 = e_p_TR;
            Matrix3d epsilon_p_1 = common::deviatoric_to_tensor(e_p_1);
            Matrix3d s_1 = s_TR;
            Matrix3d sigma_1 = common::deviatoric_to_tensor(s_1);
            Matrix3d backstress_1 = backstress_TR;
            double gamma_1 = gamma_TR;

            backstress_0 = backstress_1;
            epsilon_p_0 = epsilon_p_1;
            gamma_0 = gamma_1;
            stresses.push_back(sigma_1(0, 0));
        } else {
            double G = young / (2 * (1 + poisson));
            double H = Hiso + Hkin;
            double lambda_1 = f / (2 * G + H);
            Matrix3d n = common::calc_n(Sigma_TR);
            Matrix3d e_p_1 = e_p_TR + lambda_1 * n;
            Matrix3d epsilon_p_1 = common::deviatoric_to_tensor(e_p_1);
            Matrix3d s_1 = s_TR - 2 * G * lambda_1 * n;
            Matrix3d sigma_1 = common::deviatoric_to_tensor(s_1);
            Matrix3d backstress_1 = backstress_TR + Hkin * lambda_1 * n;
            double gamma_1 = gamma_TR + lambda_1;

            epsilon_p_0 = epsilon_p_1;
            backstress_0 = backstress_1;
            gamma_0 = gamma_1;
            stresses.push_back(sigma_1(0, 0));
        }

        strains.push_back(epsilon_1(0, 0));
    }

    for (auto [a, b] : vw::zip(strains, stresses)) {
        data.push_back({a, b});
    }
    return data;
}

}  // namespace vonmises
