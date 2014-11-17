# - Try to find SpiralWHT lib
#
# This will define
#
#  SpiralWHT_FOUND - system has SpiralWHT lib with correct version
#  SpiralWHT_INCLUDE_DIR - the SpiralWHT include directory

# Copyright (c) 2006, 2007 Montel Laurent, <montel@kde.org>
# Copyright (c) 2008, 2009 Gael Guennebaud, <g.gael@free.fr>
# Copyright (c) 2009 Benoit Jacob <jacob.benoit.1@gmail.com>
# Redistribution and use is allowed according to the terms of the 2-clause BSD license.

if (SpiralWHT_INCLUDE_DIR)

  # in cache already
  set(SpiralWHT_FOUND True)

else (SpiralWHT_INCLUDE_DIR)
  find_path(SpiralWHT_INCLUDE_DIR NAMES spiral_wht.h
      PATHS
      ${CMAKE_INSTALL_PREFIX}/include
      ${KDE4_INCLUDE_DIR}
      $ENV{HOME}/local/include
      ${PROJECT_SOURCE_DIR}
      PATH_SUFFIXES spiral spiral_wht
  )

  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(SpiralWHT DEFAULT_MSG SpiralWHT_INCLUDE_DIR)

  mark_as_advanced(SpiralWHT_INCLUDE_DIR)

endif(SpiralWHT_INCLUDE_DIR)
