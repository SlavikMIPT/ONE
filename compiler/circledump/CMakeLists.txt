if(NOT TARGET mio_circle08)
  message(STATUS "Skip circledump: mio_circle08 not found")
  return()
endif(NOT TARGET mio_circle08)

set(DRIVER "driver/Driver.cpp")

file(GLOB_RECURSE SOURCES "src/*.cpp")

add_executable(circledump ${DRIVER} ${SOURCES})
target_include_directories(circledump PRIVATE include)
target_link_libraries(circledump arser)
target_link_libraries(circledump foder)
target_link_libraries(circledump mio_circle08)
target_link_libraries(circledump mio_circle08_helper)
target_link_libraries(circledump safemain)

install(TARGETS circledump DESTINATION bin)
