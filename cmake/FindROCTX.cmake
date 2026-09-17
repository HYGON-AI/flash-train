find_path(
  ROCTX_INCLUDE_DIR
  NAMES roctx.h
  HINTS "${DTKROOT}" "$ENV{DTKROOT}"
  PATH_SUFFIXES roctracer/include include/roctracer)

find_library(
  ROCTX_LIBRARY
  NAMES roctx64
  HINTS "${DTKROOT}" "$ENV{DTKROOT}"
  PATH_SUFFIXES lib lib64 roctracer/lib)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ROCTX REQUIRED_VARS ROCTX_LIBRARY
                                                      ROCTX_INCLUDE_DIR)

if(ROCTX_FOUND AND NOT TARGET ROCTX::roctx)
  add_library(ROCTX::roctx UNKNOWN IMPORTED)
  set_target_properties(
    ROCTX::roctx
    PROPERTIES IMPORTED_LOCATION "${ROCTX_LIBRARY}"
               INTERFACE_INCLUDE_DIRECTORIES "${ROCTX_INCLUDE_DIR}")
endif()

mark_as_advanced(ROCTX_INCLUDE_DIR ROCTX_LIBRARY)
