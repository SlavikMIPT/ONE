if(NOT BUILD_STYLE_TRANSFER_APP)
  return()
endif(NOT BUILD_STYLE_TRANSFER_APP)

if(NOT BUILD_ONERT)
  return()
endif(NOT BUILD_ONERT)

find_package(JPEG)
if(JPEG_FOUND)
  add_definitions(-DNNFW_ST_APP_JPEG_SUPPORTED)
else(JPEG_FOUND)
  message(WARNING "JPEG not found.")
endif(JPEG_FOUND)

list(APPEND STYLE_TRANSFER_APP_SRCS "src/style_transfer_app.cc")
list(APPEND STYLE_TRANSFER_APP_SRCS "src/args.cc")
list(APPEND STYLE_TRANSFER_APP_SRCS "src/bitmap_helper.cc")
if(JPEG_FOUND)
  list(APPEND STYLE_TRANSFER_APP_SRCS "src/jpeg_helper.cc")
endif(JPEG_FOUND)

nnfw_find_package(Boost REQUIRED program_options)

add_executable(style_transfer_app ${STYLE_TRANSFER_APP_SRCS})
target_include_directories(style_transfer_app PRIVATE src)
target_include_directories(style_transfer_app PRIVATE ${Boost_INCLUDE_DIRS})
if(JPEG_FOUND)
  target_include_directories(style_transfer_app PRIVATE ${JPEG_INCLUDE_DIRS})
endif(JPEG_FOUND)

target_link_libraries(style_transfer_app ${LIB_PTHREAD} dl)
target_link_libraries(style_transfer_app nnfw-dev)
target_link_libraries(style_transfer_app ${Boost_PROGRAM_OPTIONS_LIBRARY})
if(JPEG_FOUND)
  target_link_libraries(style_transfer_app ${JPEG_LIBRARIES})
endif(JPEG_FOUND)

install(TARGETS style_transfer_app DESTINATION bin)
