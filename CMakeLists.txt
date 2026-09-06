cmake_minimum_required(VERSION 3.16)
project(packed_pkpcf CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -maes -DNDEBUG")
find_package(OpenMP)

# Intel HEXL (https://github.com/intel/hexl), built and installed separately:
#   cmake -S hexl -B hexl/build -DCMAKE_INSTALL_PREFIX=<prefix> && cmake --build hexl/build --target install
# Pass -DHEXL_ROOT=<prefix> (default: ~/hexl-install).
if(NOT HEXL_ROOT)
  set(HEXL_ROOT "$ENV{HOME}/hexl-install")
endif()
message(STATUS "Using HEXL from ${HEXL_ROOT}")

add_executable(pcf_bench src/main.cpp)
add_executable(rns_test src/rns_test.cpp)
foreach(t pcf_bench rns_test)
  target_include_directories(${t} PRIVATE ${HEXL_ROOT}/include)
  target_link_libraries(${t} PRIVATE ${HEXL_ROOT}/lib/libhexl.a)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(${t} PRIVATE OpenMP::OpenMP_CXX)
  endif()
endforeach()
