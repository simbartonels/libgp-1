#INCLUDE(ExternalProject)

# ----- Download and build gtest -----
#ExternalProject_Add(hayai
#  GIT_REPOSITORY "https://github.com/nickbruun/hayai.git"
##  INSTALL_COMMAND ""
##  UPDATE_COMMAND ""
#  PREFIX "hayai_clone"
#  LOG_DOWNLOAD 1            # Wrap download in script to log output
#  LOG_UPDATE 1              # Wrap update in script to log output
#  LOG_CONFIGURE 1           # Wrap configure in script to log output
#  LOG_BUILD 1               # Wrap build in script to log output
#  LOG_TEST 1                # Wrap test in script to log output
#  LOG_INSTALL 1
#)

# ----- Testing -----
#INCLUDE_DIRECTORIES (${CMAKE_CURRENT_BINARY_DIR}/hayai_clone/src/hayai/src)
#LINK_DIRECTORIES (${CMAKE_CURRENT_BINARY_DIR}/hayai/src/hayai-build)


INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/util)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/naive)


# ----- Add sources -----
INCLUDE("Sources.cmake")

#ADD_EXECUTABLE(gpbenchmark ${LIBGP_BENCHMARKS})
#TARGET_LINK_LIBRARIES(gpbenchmark gp hayai)
#TARGET_LINK_LIBRARIES(gpbenchmark gp)

ADD_EXECUTABLE(gpbenchm_fic gp_fic.cc  ${LIBGP_BENCHMARKS})
TARGET_LINK_LIBRARIES(gpbenchm_fic gp)

ADD_EXECUTABLE(gpbenchm_matrix_mult symmetric_matrix_multiplication.cc  ${LIBGP_BENCHMARKS})
TARGET_LINK_LIBRARIES(gpbenchm_matrix_mult gp)

ADD_EXECUTABLE(gpbenchm_multiscale bf_multi_scale.cc  ${LIBGP_BENCHMARKS})
TARGET_LINK_LIBRARIES(gpbenchm_multiscale gp)

