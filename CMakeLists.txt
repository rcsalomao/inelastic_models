cmake_minimum_required(VERSION 4.0.0)

project(
    inelastic_models
    DESCRIPTION "This is a project description."
    LANGUAGES CXX
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 23)

add_executable(
    inelastic_models
    src/inelastic_models.cpp
    src/common.cpp
    src/vonmises.cpp
    src/vonmises_generalized.cpp
    src/mazars_model.cpp
    src/mazars_mu_model.cpp
)

target_include_directories(
    inelastic_models
    PRIVATE
    ~/.local/include
)

target_compile_options(
    inelastic_models
    PRIVATE
    -fdiagnostics-color=always
    -Wall
    -Wextra
    -Og
    -march=native
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
)
