#include "mazars_mu_model.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <iostream>
#include <ranges>
#include <vector>

#include "common.hpp"

namespace mazars_mu_model {

namespace rg = std::ranges;
namespace vw = std::ranges::views;
using Eigen::Matrix3d;
using Eigen::Vector3d;
using std::vector;

Vector3d calc_valores_positivos(Vector3d& valores_principais) {
    return valores_principais.unaryExpr(
        [](double v) { return v >= 0.0 ? v : 0.0; });
}

double calc_r(Vector3d& sigma_pos, Vector3d& sigma_princ) {
    double a{0};
    rg::for_each(sigma_princ, [&a](double v) { a += abs(v); });
    if (a == 0.0) {
        return 1.0;
    } else {
        return sigma_pos.sum() / a;
    }
}

double calc_Ie(Vector3d& epsilon_princ) { return epsilon_princ.sum(); }

double calc_Je(Vector3d& epsilon_princ) {
    return 0.5 * (std::pow(epsilon_princ(0) - epsilon_princ(1), 2) +
                  std::pow(epsilon_princ(1) - epsilon_princ(2), 2) +
                  std::pow(epsilon_princ(2) - epsilon_princ(0), 2));
}

double calc_Et(double poisson, double Ie, double Je) {
    return Ie / (2 * (1 - 2 * poisson)) + sqrt(Je) / (2 * (1 + poisson));
}

double calc_Ec(double poisson, double Ie, double Je) {
    return Ie / (5 * (1 - 2 * poisson)) + 6 * sqrt(Je) / (5 * (1 + poisson));
}

double calc_A(double At, double Ac, double r, double k) {
    return At * (2 * (r * r) * (1 - 2 * k) - r * (1 - 4 * k)) +
           Ac * (2 * (r * r) - 3 * r + 1);
}

double calc_B(double Bt, double Bc, double r) {
    double a = (r * r) - 2 * r + 2;
    return Bt * std::pow(r, a) + Bc * (1 - std::pow(r, a));
}

double calc_D(double Y, double Y0, double A, double B) {
    return 1 - Y0 * (1 - A) / Y - A * std::pow(std::numbers::e, -B * (Y - Y0));
}

std::tuple<double, double, double> calc_D_Yt_Yc(Matrix3d& sigma, double young,
                                                double poisson, double Yt,
                                                double Yc, ModelParams& mod_p) {
    auto sigma_princ = common::calc_valores_principais(sigma);
    auto sigma_pos = calc_valores_positivos(sigma_princ);
    double r = calc_r(sigma_pos, sigma_princ);
    double Y0 = r * mod_p.epsilon_d0t + (1 - r) * mod_p.epsilon_d0c;

    auto epsilon_princ =
        common::sigma_princ_to_epsilon_princ(sigma_princ, young, poisson);
    double Ie = calc_Ie(epsilon_princ);
    double Je = calc_Je(epsilon_princ);
    double Et = calc_Et(poisson, Ie, Je);
    double Ec = calc_Ec(poisson, Ie, Je);
    Yt = std::max(Et, Yt);
    Yc = std::max(Ec, Yc);
    double Y = r * Yt + (1 - r) * Yc;

    double A = calc_A(mod_p.At, mod_p.Ac, r, mod_p.k);
    double B = calc_B(mod_p.Bt, mod_p.Bc, r);

    double D = calc_D(Y, Y0, A, B);

    return {D, Yt, Yc};
}

vector<std::tuple<double, double>> run(common::MaterialProperties& mat_p,
                                       ModelParams& mod_p,
                                       vector<double> hist) {
    // material params
    double young0 = mat_p.young;
    double poisson = mat_p.poisson;

    // initial values
    double Yt_0 = mod_p.epsilon_d0t;
    double Yc_0 = mod_p.epsilon_d0c;
    double dano_0 = 0;

    vector<double> strains{0.0};
    vector<double> stresses{0.0};
    vector<std::tuple<double, double>> data;

    for (double epsilon_11 : hist) {
        Matrix3d epsilon_1{
            {epsilon_11, 0, 0},
            {0, -poisson * epsilon_11, 0},
            {0, 0, -poisson * epsilon_11},
        };
        double young = (1 - dano_0) * young0;
        Matrix3d sigma_TR = common::strain_to_stress(epsilon_1, young, poisson);
        auto [dano_1, Yt_1, Yc_1] =
            calc_D_Yt_Yc(sigma_TR, young, poisson, Yt_0, Yc_0, mod_p);
        young = (1 - dano_1) * young0;
        Matrix3d sigma_1 = common::strain_to_stress(epsilon_1, young, poisson);

        Yt_0 = Yt_1;
        Yc_0 = Yc_1;
        dano_0 = dano_1;

        strains.push_back(epsilon_1(0, 0));
        stresses.push_back(sigma_1(0, 0));
    }

    for (auto [a, b] : vw::zip(strains, stresses)) {
        data.push_back({a, b});
    }
    return data;
}
}  // namespace mazars_mu_model
