cmake_minimum_required(VERSION 4.0.0)

project(
    inelastic_models
    DESCRIPTION "This is a project description."
    LANGUAGES CXX
)

add_executable(
    inelastic_models
)

target_sources(
    inelastic_models
    PRIVATE
    ./example/inelastic_models.cpp
    ./src/common.cpp
    ./src/vonmises.cpp
    ./src/vonmises_generalized.cpp
    ./src/mazars_model.cpp
    ./src/mazars_mu_model.cpp
    ./lib/gel/gel.cpp
)

target_compile_features(
    inelastic_models
    PRIVATE
    cxx_std_23
)

target_include_directories(
    inelastic_models
    PRIVATE
    ~/.local/include
    ./src
    ./lib/gel
)

target_compile_options(
    inelastic_models
    PRIVATE
    -Wall
    -Wextra
    -Og
    -march=native
    -fdiagnostics-color=always
    # -fsanitize=address,undefined
)

target_link_directories(
    inelastic_models
    PRIVATE
    ~/.local/lib
)

target_link_libraries(
    inelastic_models
    PRIVATE
    -lboost_iostreams
    -lboost_system
    -lboost_filesystem
)

target_link_options(
    inelastic_models
    PRIVATE
    # -fsanitize=address,undefined
)
