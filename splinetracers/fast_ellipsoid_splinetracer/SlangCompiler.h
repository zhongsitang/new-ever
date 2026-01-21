// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <slang.h>
#include <slang-com-ptr.h>

#include <string>
#include <vector>
#include <stdexcept>

/**
 * SlangCompiler - Runtime Slang shader compilation to PTX
 *
 * This class wraps the Slang C++ API to compile .slang files directly to PTX
 * at runtime, eliminating the need for build-time slangc invocation and
 * embedding PTX as C++ arrays.
 *
 * Usage:
 *   SlangCompiler compiler;
 *   compiler.addSearchPath("/path/to/slang/modules");
 *   std::string ptx = compiler.compileToPTX("/path/to/shader.slang");
 *   // Use ptx with optixModuleCreate()
 */
class SlangCompiler {
public:
    SlangCompiler();
    ~SlangCompiler() = default;

    // Non-copyable
    SlangCompiler(const SlangCompiler&) = delete;
    SlangCompiler& operator=(const SlangCompiler&) = delete;

    /**
     * Add a search path for resolving `import` statements
     */
    void addSearchPath(const std::string& path);

    /**
     * Add NVRTC include path (for CUDA headers like optix.h)
     */
    void addNvrtcIncludePath(const std::string& path);

    /**
     * Compile a .slang file to PTX
     *
     * @param slangFilePath Path to the .slang source file
     * @return PTX code as a string (null-terminated)
     * @throws std::runtime_error on compilation failure
     */
    std::string compileToPTX(const std::string& slangFilePath);

    /**
     * Compile Slang source code (as string) to PTX
     *
     * @param sourceCode The Slang source code
     * @param moduleName Name for the module (used in error messages)
     * @return PTX code as a string (null-terminated)
     * @throws std::runtime_error on compilation failure
     */
    std::string compileSourceToPTX(const std::string& sourceCode,
                                    const std::string& moduleName = "shader");

private:
    Slang::ComPtr<slang::IGlobalSession> globalSession_;
    std::vector<std::string> searchPaths_;
    std::vector<std::string> nvrtcIncludePaths_;

    /**
     * Create a new compilation session with current settings
     */
    Slang::ComPtr<slang::ISession> createSession();

    /**
     * Extract PTX from a compiled program
     */
    std::string extractPTX(slang::IComponentType* linkedProgram);

    /**
     * Format diagnostic messages
     */
    static std::string formatDiagnostics(slang::IBlob* diagnostics);
};

/**
 * Exception thrown on Slang compilation errors
 */
class SlangCompilationError : public std::runtime_error {
public:
    explicit SlangCompilationError(const std::string& message)
        : std::runtime_error(message) {}
};
