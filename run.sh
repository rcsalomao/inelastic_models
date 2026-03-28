export LD_LIBRARY_PATH="$HOME"/.local/lib/:"$GCC_DIR"/lib64/:"$LD_LIBRARY_PATH"

cmake -S . -B ./build -DCMAKE_CXX_COMPILER="$GCC_DIR"/bin/g++ -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build ./build -j "$(nproc)"

./build/inelastic_models
