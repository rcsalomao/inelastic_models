#include "vonmises_generalized.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <print>
#include <ranges>
#include <vector>

#include "common.hpp"

namespace vonmises_generalized {

namespace rg = std::ranges;
namespace vw = std::ranges::views;
using Eigen::Matrix3d;
using Eigen::Vector3d;
using std::vector;

double calc_lambda(double f, double H, double G, Matrix3d& Sigma_TR,
                   Matrix3d& Sigma_0, double delta, double beta) {
    double G1 = G + H / 2;
    double A1 = f;
    double A2 = Sigma_TR.norm() - Sigma_0.norm();
    double A3 = delta - 2.0 * G;
    double A4 = (delta + H) * beta;
    double c1 = 2.0 * G1 * A3;
    double c2 = A4 - A1 * A3 + 2.0 * G1 * A2;
    double c3 = -A1 * A2;

    vector<double> vl{-(c2 - sqrt(c2 * c2 - 4.0 * c1 * c3)) / (2.0 * c1),
                      -(c2 + sqrt(c2 * c2 - 4.0 * c1 * c3)) / (2.0 * c1)};

    auto lambdas = vector<double>{
        std::from_range, vl | vw::filter([](auto l) { return l >= 0; })};
    if (lambdas.empty()) {
        std::println("[ERROR] Couldn't find suitable lambdas.");
        exit(2);
    }

    double lambda_1 = *rg::min_element(lambdas);

    return lambda_1;
}

vector<std::tuple<double, double>> run(common::MaterialProperties& mat_p,
                                       vonmises_generalized::ModelParams& mod_p,
                                       vector<double> hist) {
    // material params
    double young = mat_p.young;
    double poisson = mat_p.poisson;

    // vonmises params
    double sigma_y = mod_p.sigma_y;
    double Hiso = mod_p.Hiso;
    double Hkin = mod_p.Hkin;
    double beta = mod_p.beta;
    double delta = mod_p.delta;

    // initial values
    Matrix3d s_0 = Matrix3d::Zero();
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

            s_0 = s_1;
            backstress_0 = backstress_1;
            epsilon_p_0 = epsilon_p_1;
            gamma_0 = gamma_1;
            stresses.push_back(sigma_1(0, 0));
        } else {
            double G = young / (2 * (1 + poisson));
            double H = Hiso + Hkin;
            Matrix3d Sigma_0 = s_0 - backstress_0;
            double lambda_1 =
                calc_lambda(f, H, G, Sigma_TR, Sigma_0, delta, beta);
            Matrix3d n = common::calc_n(Sigma_TR);
            Matrix3d e_p_1 = e_p_TR + lambda_1 * n;
            Matrix3d epsilon_p_1 = common::deviatoric_to_tensor(e_p_1);
            Matrix3d s_1 = s_TR - 2 * G * lambda_1 * n;
            Matrix3d sigma_1 = common::deviatoric_to_tensor(s_1);
            Matrix3d backstress_1 = backstress_TR + Hkin * lambda_1 * n;
            double gamma_1 = gamma_TR + lambda_1;

            s_0 = s_1;
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

}  // namespace vonmises_generalized
