cmake_minimum_required(VERSION 3.20)
project(MPCReducedF16 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(MPCJSBSim SHARED
  reduced_predictor.cpp
  reduced_predictor.h
  generated_f16_data.h
)
target_compile_definitions(MPCJSBSim PRIVATE MPC_PREDICTOR_EXPORTS NOMINMAX _USE_MATH_DEFINES)
if(MSVC)
  target_compile_options(MPCJSBSim PRIVATE /O2 /EHsc /W4)
endif()
set_target_properties(MPCJSBSim PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/../../runtime/predictor"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/../../runtime/predictor"
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/../../build/lib"
)
