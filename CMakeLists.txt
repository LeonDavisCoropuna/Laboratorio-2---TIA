cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(torch_project)

# Estándar de C++
set(CMAKE_CXX_STANDARD 17)

# Ruta de instalación de LibTorch
set(CMAKE_PREFIX_PATH "$ENV{HOME}/libtorch")

# Buscar LibTorch
find_package(Torch REQUIRED)

# Incluir directorio de tus modelos
include_directories(${CMAKE_SOURCE_DIR}/models)

# Ejecutable principal y los headers (aunque no es necesario listarlos aquí)
add_executable(main main.cpp)

# Enlazar Torch al ejecutable
target_link_libraries(main "${TORCH_LIBRARIES}")

# Usar C++17 (redundante si ya usaste set(CMAKE_CXX_STANDARD 17), pero por claridad)
set_property(TARGET main PROPERTY CXX_STANDARD 17)
