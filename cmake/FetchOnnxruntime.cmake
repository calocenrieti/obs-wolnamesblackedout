# Local ONNX Runtime reference (version 1.24.4 with DirectML, built from source)

set(ONNXRUNTIME_ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}/../onnxruntime")

# Find ONNX Runtime include directory
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime/core/session/onnxruntime_c_api.h
    HINTS ${ONNXRUNTIME_ROOT_DIR}/include
    NO_DEFAULT_PATH)

if(NOT ONNXRUNTIME_INCLUDE_DIR)
    message(FATAL_ERROR "ONNX Runtime include directory not found at ${ONNXRUNTIME_ROOT_DIR}/include")
endif()

# Set correct include path (parent of onnxruntime/)
set(ONNX_RUNTIME_CXX_INCLUDE_DIR "${ONNXRUNTIME_INCLUDE_DIR}")

# Find ONNX Runtime static library and DLL
find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime
    HINTS ${ONNXRUNTIME_ROOT_DIR}/Release
    NO_DEFAULT_PATH)

if(NOT ONNXRUNTIME_LIBRARY)
    message(FATAL_ERROR "onnxruntime.lib not found in ${ONNXRUNTIME_ROOT_DIR}/Release")
endif()

find_file(ONNXRUNTIME_DLL NAMES onnxruntime.dll
    HINTS ${ONNXRUNTIME_ROOT_DIR}/Release
    NO_DEFAULT_PATH)

# Find DirectML DLL (import lib may not exist in local build, use onnxruntime_providers_dml)
find_file(DIRECTML_DLL NAMES DirectML.dll
    HINTS ${ONNXRUNTIME_ROOT_DIR}/Release
    NO_DEFAULT_PATH)

if(NOT DIRECTML_DLL)
    message(FATAL_ERROR "DirectML.dll not found in ${ONNXRUNTIME_ROOT_DIR}/Release")
endif()

# Create ONNX Runtime library target (static import)
# Include multiple directories for full API access
set(ONNX_RUNTIME_SESSION_INCLUDE "${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime/core/session")
set(ONNX_RUNTIME_DML_INCLUDE "${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime/core/providers/dml")
add_library(OnnxRuntime STATIC IMPORTED)
set_target_properties(OnnxRuntime PROPERTIES
    IMPORTED_LOCATION ${ONNXRUNTIME_LIBRARY}
    INTERFACE_INCLUDE_DIRECTORIES "${ONNX_RUNTIME_SESSION_INCLUDE};${ONNX_RUNTIME_DML_INCLUDE}")

# Link with onnxruntime_providers_dml for DirectML support (from local build)
find_library(DIRECTML_PROVIDER_LIB NAMES onnxruntime_providers_dml
    HINTS ${ONNXRUNTIME_ROOT_DIR}/Release
    NO_DEFAULT_PATH)

# Link ONNX Runtime and providers to the project
target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE OnnxRuntime ${DIRECTML_PROVIDER_LIB} d3d12.lib dxgi.lib dxguid.lib Dxcore.lib)

# Install DLLs at build time
if(ONNXRUNTIME_DLL)
    add_custom_command(TARGET ${CMAKE_PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${ONNXRUNTIME_DLL}" $<TARGET_FILE_DIR:${CMAKE_PROJECT_NAME}>)
endif()

if(DIRECTML_DLL)
    add_custom_command(TARGET ${CMAKE_PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DIRECTML_DLL}" $<TARGET_FILE_DIR:${CMAKE_PROJECT_NAME}>)
endif()
