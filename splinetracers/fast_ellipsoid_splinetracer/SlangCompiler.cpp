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

#include "SlangCompiler.h"

#include <fstream>
#include <sstream>

SlangCompiler::SlangCompiler() {
    // Create the global Slang session
    SlangResult result = slang_createGlobalSession(SLANG_API_VERSION, globalSession_.writeRef());
    if (SLANG_FAILED(result)) {
        throw SlangCompilationError("Failed to create Slang global session");
    }
}

void SlangCompiler::addSearchPath(const std::string& path) {
    searchPaths_.push_back(path);
}

void SlangCompiler::addNvrtcIncludePath(const std::string& path) {
    nvrtcIncludePaths_.push_back(path);
}

Slang::ComPtr<slang::ISession> SlangCompiler::createSession() {
    // Setup PTX target
    slang::TargetDesc targetDesc = {};
    targetDesc.format = SLANG_PTX;
    targetDesc.profile = globalSession_->findProfile("sm_75");  // Compute capability 7.5+

    // Build search paths array
    std::vector<const char*> searchPathPtrs;
    for (const auto& path : searchPaths_) {
        searchPathPtrs.push_back(path.c_str());
    }

    // Setup session
    slang::SessionDesc sessionDesc = {};
    sessionDesc.targetCount = 1;
    sessionDesc.targets = &targetDesc;
    sessionDesc.searchPaths = searchPathPtrs.empty() ? nullptr : searchPathPtrs.data();
    sessionDesc.searchPathCount = static_cast<SlangInt>(searchPathPtrs.size());

    // Add NVRTC include paths as compiler options
    std::vector<slang::CompilerOptionEntry> compilerOptions;
    std::vector<slang::CompilerOptionValue> nvrtcValues;

    for (const auto& nvrtcPath : nvrtcIncludePaths_) {
        slang::CompilerOptionValue value = {};
        value.kind = slang::CompilerOptionValueKind::String;
        value.stringValue0 = nvrtcPath.c_str();
        nvrtcValues.push_back(value);
    }

    // Note: NVRTC include paths are handled through downstream compiler options
    // For now, we rely on the search paths for Slang modules

    Slang::ComPtr<slang::ISession> session;
    SlangResult result = globalSession_->createSession(sessionDesc, session.writeRef());
    if (SLANG_FAILED(result)) {
        throw SlangCompilationError("Failed to create Slang compilation session");
    }

    return session;
}

std::string SlangCompiler::compileToPTX(const std::string& slangFilePath) {
    // Read the source file
    std::ifstream file(slangFilePath);
    if (!file.is_open()) {
        throw SlangCompilationError("Failed to open file: " + slangFilePath);
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string sourceCode = buffer.str();

    // Extract module name from filename
    size_t lastSlash = slangFilePath.find_last_of("/\\");
    size_t lastDot = slangFilePath.find_last_of('.');
    std::string moduleName = slangFilePath.substr(
        lastSlash == std::string::npos ? 0 : lastSlash + 1,
        lastDot == std::string::npos ? std::string::npos : lastDot - (lastSlash == std::string::npos ? 0 : lastSlash + 1)
    );

    // Add the directory containing the file to search paths temporarily
    if (lastSlash != std::string::npos) {
        std::string dir = slangFilePath.substr(0, lastSlash);
        bool alreadyAdded = false;
        for (const auto& path : searchPaths_) {
            if (path == dir) {
                alreadyAdded = true;
                break;
            }
        }
        if (!alreadyAdded) {
            searchPaths_.push_back(dir);
        }
    }

    return compileSourceToPTX(sourceCode, moduleName);
}

std::string SlangCompiler::compileSourceToPTX(const std::string& sourceCode,
                                              const std::string& moduleName) {
    auto session = createSession();

    Slang::ComPtr<slang::IBlob> diagnosticBlob;

    // Load module from source string
    auto module = session->loadModuleFromSourceString(
        moduleName.c_str(),
        (moduleName + ".slang").c_str(),
        sourceCode.c_str(),
        diagnosticBlob.writeRef()
    );

    if (!module) {
        std::string errorMsg = "Failed to load Slang module: " + moduleName;
        if (diagnosticBlob) {
            errorMsg += "\n" + formatDiagnostics(diagnosticBlob);
        }
        throw SlangCompilationError(errorMsg);
    }

    // Get the number of entry points defined in the module
    SlangInt entryPointCount = module->getDefinedEntryPointCount();

    if (entryPointCount == 0) {
        throw SlangCompilationError("No entry points found in module: " + moduleName);
    }

    // Collect all entry points
    std::vector<Slang::ComPtr<slang::IEntryPoint>> entryPoints;
    std::vector<slang::IComponentType*> components;
    components.push_back(module);

    for (SlangInt i = 0; i < entryPointCount; i++) {
        Slang::ComPtr<slang::IEntryPoint> entryPoint;
        SlangResult result = module->getDefinedEntryPoint(i, entryPoint.writeRef());
        if (SLANG_FAILED(result) || !entryPoint) {
            throw SlangCompilationError("Failed to get entry point " + std::to_string(i));
        }
        entryPoints.push_back(entryPoint);
        components.push_back(entryPoint.get());
    }

    // Create composite component type
    Slang::ComPtr<slang::IComponentType> compositeProgram;
    SlangResult result = session->createCompositeComponentType(
        components.data(),
        static_cast<SlangInt>(components.size()),
        compositeProgram.writeRef(),
        diagnosticBlob.writeRef()
    );

    if (SLANG_FAILED(result) || !compositeProgram) {
        std::string errorMsg = "Failed to create composite component type";
        if (diagnosticBlob) {
            errorMsg += "\n" + formatDiagnostics(diagnosticBlob);
        }
        throw SlangCompilationError(errorMsg);
    }

    // Link the program
    Slang::ComPtr<slang::IComponentType> linkedProgram;
    result = compositeProgram->link(linkedProgram.writeRef(), diagnosticBlob.writeRef());

    if (SLANG_FAILED(result) || !linkedProgram) {
        std::string errorMsg = "Failed to link Slang program";
        if (diagnosticBlob) {
            errorMsg += "\n" + formatDiagnostics(diagnosticBlob);
        }
        throw SlangCompilationError(errorMsg);
    }

    return extractPTX(linkedProgram);
}

std::string SlangCompiler::extractPTX(slang::IComponentType* linkedProgram) {
    Slang::ComPtr<slang::IBlob> codeBlob;
    Slang::ComPtr<slang::IBlob> diagnosticBlob;

    // Get the compiled target code (PTX)
    // targetIndex = 0 since we only have one target
    SlangResult result = linkedProgram->getTargetCode(
        0,  // targetIndex
        codeBlob.writeRef(),
        diagnosticBlob.writeRef()
    );

    if (SLANG_FAILED(result) || !codeBlob) {
        std::string errorMsg = "Failed to get PTX code";
        if (diagnosticBlob) {
            errorMsg += "\n" + formatDiagnostics(diagnosticBlob);
        }
        throw SlangCompilationError(errorMsg);
    }

    // Convert blob to string
    const char* ptxData = static_cast<const char*>(codeBlob->getBufferPointer());
    size_t ptxSize = codeBlob->getBufferSize();

    // Ensure null-termination (PTX should already be null-terminated)
    if (ptxSize > 0 && ptxData[ptxSize - 1] == '\0') {
        return std::string(ptxData, ptxSize - 1);
    }
    return std::string(ptxData, ptxSize);
}

std::string SlangCompiler::formatDiagnostics(slang::IBlob* diagnostics) {
    if (!diagnostics) {
        return "";
    }
    const char* text = static_cast<const char*>(diagnostics->getBufferPointer());
    size_t size = diagnostics->getBufferSize();
    if (size > 0 && text[size - 1] == '\0') {
        return std::string(text, size - 1);
    }
    return std::string(text, size);
}
