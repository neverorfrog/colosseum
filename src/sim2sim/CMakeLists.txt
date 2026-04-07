# ================== PROJECT SETUP ===================
cmake_minimum_required(VERSION 3.22)

project(sim2sim VERSION 0.1 LANGUAGES CXX)

set(LIBRARY sim2sim)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

file(
    WRITE "${CMAKE_CURRENT_SOURCE_DIR}/.clangd"
    "CompileFlags:
  Compiler: g++
  CompilationDatabase: ${CMAKE_CURRENT_BINARY_DIR}
"
)

add_compile_definitions(PROJECT_ROOT="${CMAKE_SOURCE_DIR}")
# ====================================================

# ================ Dependencies ======================
find_package(Threads REQUIRED)
find_package(booster_robotics_sdk REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(onnxruntime REQUIRED)

# libbooster_robotics_sdk.a was built against glibc >= 2.32 which has
# __libc_single_threaded; stub it out when linking on older glibc.
if(UNIX AND NOT APPLE)
    add_link_options(-Wl,--defsym=__libc_single_threaded=0)
endif()
# ====================================================

# =================== Libraries ====================

# Source files
file(
    GLOB_RECURSE SOURCES
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)
list(FILTER SOURCES EXCLUDE REGEX ".*/main\\.cpp$")
file(
    GLOB_RECURSE HEADERS
    "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp"
)

# Debug message
message(STATUS "Building application library '${LIBRARY}':")
foreach(SOURCE ${ALL_APP_SOURCES})
    file(RELATIVE_PATH REL_SOURCE ${CMAKE_CURRENT_SOURCE_DIR} ${SOURCE})
    message(STATUS "  - ${REL_SOURCE}")
endforeach()

# Library build
add_library(${LIBRARY} SHARED ${SOURCES} ${HEADERS})
target_include_directories(
    ${LIBRARY}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(
    ${LIBRARY}
    PUBLIC
        Threads::Threads
        Eigen3::Eigen
        onnxruntime::onnxruntime
    PRIVATE
        booster::booster_robotics_sdk
)
# ====================================================



# ============================== TESTS ==============================================
find_package(GTest QUIET)

if(GTest_FOUND)
    message(STATUS "GTest found - building tests for ${PROJECT_NAME}")

    # Automatically discover all test source files
    file(GLOB TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_*.cpp")
    file(GLOB TEST_FIXTURES "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures.cpp")

    # Optional: Test common library for shared fixtures (only if fixtures exist)
    if(TEST_FIXTURES)
        add_library(${PROJECT_NAME}_test_common STATIC ${TEST_FIXTURES})
        target_include_directories(
            ${PROJECT_NAME}_test_common
            PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/tests
        )
        target_link_libraries(
            ${PROJECT_NAME}_test_common
            PUBLIC GTest::gtest ${LIBRARY}
        )
        set(TEST_COMMON_LIB ${PROJECT_NAME}_test_common)
    endif()

    # Testable sources: only units that don't pull in booster DDS static init
    file(
        GLOB_RECURSE TESTABLE_SOURCES
        CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/src/OnnxPolicy.cpp"
    )

    # Main test executable
    add_executable(
        ${PROJECT_NAME}_tests
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/main.cpp
        ${TEST_SOURCES}
        ${TESTABLE_SOURCES}
    )

    target_include_directories(
        ${PROJECT_NAME}_tests
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    target_link_libraries(
        ${PROJECT_NAME}_tests
        PRIVATE GTest::gtest Eigen3::Eigen onnxruntime::onnxruntime ${TEST_COMMON_LIB}
    )

    # Enable CTest integration
    enable_testing()
    add_test(NAME ${PROJECT_NAME}_tests COMMAND ${PROJECT_NAME}_tests)

    install(TARGETS ${PROJECT_NAME}_tests RUNTIME DESTINATION bin)
else()
    message(STATUS "GTest not found - skipping tests for ${PROJECT_NAME}")
endif()
# ===================================================================================

# ================== Installation ====================
install(
    TARGETS ${LIBRARY}
    EXPORT ${PROJECT_NAME}Targets
    LIBRARY DESTINATION lib
)

install(DIRECTORY models/ DESTINATION models)

# install(
#     EXPORT ${PROJECT_NAME}Targets
#     FILE ${PROJECT_NAME}Targets.cmake
#     NAMESPACE colosseum::
#     DESTINATION lib/cmake/${PROJECT_NAME}
# )
# ====================================================

# ====================== Main ======================
# Main executable - minimal bootstrap that only initializes SDK and B1LocoClient
add_executable(main ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp)
target_link_libraries(main PRIVATE ${CMAKE_DL_LIBS})
install(TARGETS main RUNTIME DESTINATION bin)

# ====================================================
