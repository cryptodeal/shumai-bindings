cmake_minimum_required(VERSION 3.15)
cmake_policy(SET CMP0091 NEW)
cmake_policy(SET CMP0042 NEW)
project(shumai_bindings LANGUAGES CXX C)
set(CMAKE_CXX_STANDARD 19)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(flashlight CONFIG REQUIRED)

# flashlight Bun FFI bindings
add_library(
  flashlight_binding
  SHARED
  ffi_impl/bindings/bindings.cc
  )

# Write lib to the project root
set_target_properties(
  flashlight_binding
  PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
)

target_link_libraries(
  flashlight_binding
  PRIVATE
  flashlight::flashlight
)

# Enable strict handling of warnings
add_definitions("-Wall -Wextra")
add_definitions(-DNAPI_VERSION=6)

include_directories(${CMAKE_JS_INC})
file(GLOB SOURCE_FILES "napi_impl/bindings/*.cc")
add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES} ${CMAKE_JS_SRC})
set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "" SUFFIX ".node")
target_link_libraries(${PROJECT_NAME} ${CMAKE_JS_LIB})
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_JS_INC})
# Include Node-API wrappers
execute_process(COMMAND node -p "require('node-addon-api').include"
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE NODE_ADDON_API_DIR
        )
string(REGEX REPLACE "[\r\n\"]" "" NODE_ADDON_API_DIR ${NODE_ADDON_API_DIR})
target_include_directories(${PROJECT_NAME} PRIVATE ${NODE_ADDON_API_DIR})
target_link_libraries(
  ${PROJECT_NAME}
  PRIVATE 
  flashlight::flashlight
)

