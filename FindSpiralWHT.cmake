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

macro(_spiral_check_installed)
  find_path(SpiralWHT_INSTALLED_DIR NAMES transpose.h
      PATHS
      ${CMAKE_INSTALL_PREFIX}/include
      ${KDE4_INCLUDE_DIR}
      $ENV{HOME}/local/include
      ${PROJECT_SOURCE_DIR}
      PATH_SUFFIXES spiral spiral_wht spiral_Wht
  )
  if(NOT SpiralWHT_INSTALLED_DIR)
    message(STATUS "Spiral WHT found in ${SpiralWHT_INCLUDE_DIR}, "
                   "but it appears it was not properly installed. Did you run ./configure, make and make install in that folder?")
    message(STATUS "If you want mex support you should make sure wht is comiled with -fPIC option!")
  endif(NOT SpiralWHT_INSTALLED_DIR)
endmacro(_spiral_check_installed)

if (SpiralWHT_INCLUDE_DIR)

  # in cache already
  set(SpiralWHT_FOUND True)
  _spiral_check_installed()

else (SpiralWHT_INCLUDE_DIR)
  find_path(SpiralWHT_INCLUDE_DIR NAMES spiral_wht.h
      PATHS
      ${CMAKE_INSTALL_PREFIX}/include
      ${KDE4_INCLUDE_DIR}
      $ENV{HOME}/local/include
      ${PROJECT_SOURCE_DIR}
      PATH_SUFFIXES spiral spiral_wht
  )

  if (SpiralWHT_INCLUDE_DIR)
    _spiral_check_installed()
  endif (SpiralWHT_INCLUDE_DIR)

  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(SpiralWHT DEFAULT_MSG SpiralWHT_INCLUDE_DIR SpiralWHT_INSTALLED_DIR)

  mark_as_advanced(SpiralWHT_INCLUDE_DIR)

endif(SpiralWHT_INCLUDE_DIR)
