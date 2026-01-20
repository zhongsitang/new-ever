#pragma once

#include "Types.h"
#include <memory>
#include <string>

// Forward declarations for Slang/RHI types
namespace slang {
    struct ISession;
    struct IGlobalSession;
}

namespace rhi {
    class IDevice;
    class ICommandQueue;
    class ICommandBuffer;
    class IBuffer;
    class ITexture;
    class IAccelerationStructure;
}

namespace gaussian_rt {

/**
 * @brief Device management class for Slang-RHI
 *
 * Handles initialization and management of the RHI device and Slang session.
 * Uses singleton pattern for global device access.
 */
class Device {
public:
    Device();
    ~Device();

    // Non-copyable
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

    // Movable
    Device(Device&&) noexcept;
    Device& operator=(Device&&) noexcept;

    /**
     * @brief Initialize the device
     * @param cudaDeviceId CUDA device ID to use
     * @param enableValidation Enable RHI validation layers
     * @return Result code
     */
    Result initialize(int cudaDeviceId = 0, bool enableValidation = false);

    /**
     * @brief Shutdown and release all resources
     */
    void shutdown();

    /**
     * @brief Check if device is initialized
     */
    bool isInitialized() const { return m_initialized; }

    /**
     * @brief Get CUDA device ID
     */
    int getCudaDeviceId() const { return m_cudaDeviceId; }

    /**
     * @brief Synchronize all pending operations
     */
    void synchronize();

    //--------------------------------------------------------------------------
    // Buffer creation helpers
    //--------------------------------------------------------------------------

    /**
     * @brief Create a GPU buffer
     * @param size Buffer size in bytes
     * @param initialData Optional initial data to upload
     * @return Device pointer to buffer, or nullptr on failure
     */
    void* createBuffer(size_t size, const void* initialData = nullptr);

    /**
     * @brief Create a GPU buffer with specific usage flags
     */
    void* createBufferWithUsage(size_t size, uint32_t usage, const void* initialData = nullptr);

    /**
     * @brief Free a GPU buffer
     */
    void freeBuffer(void* buffer);

    /**
     * @brief Copy data to GPU buffer
     */
    Result uploadToBuffer(void* buffer, const void* data, size_t size, size_t offset = 0);

    /**
     * @brief Copy data from GPU buffer
     */
    Result downloadFromBuffer(void* buffer, void* data, size_t size, size_t offset = 0);

    //--------------------------------------------------------------------------
    // Slang/RHI access (for internal use)
    //--------------------------------------------------------------------------

    void* getRhiDevice() const { return m_rhiDevice; }
    void* getCommandQueue() const { return m_commandQueue; }
    void* getSlangSession() const { return m_slangSession; }
    void* getSlangGlobalSession() const { return m_slangGlobalSession; }

    // Get CUDA stream for interop
    void* getCudaStream() const { return m_cudaStream; }

private:
    bool m_initialized = false;
    int m_cudaDeviceId = 0;

    // RHI objects (stored as void* to avoid header dependencies)
    void* m_rhiDevice = nullptr;
    void* m_commandQueue = nullptr;
    void* m_slangSession = nullptr;
    void* m_slangGlobalSession = nullptr;

    // CUDA interop
    void* m_cudaStream = nullptr;
    void* m_cudaContext = nullptr;

    // Internal implementation
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

/**
 * @brief Get global device instance
 *
 * Creates a singleton device instance if not already created.
 * Call initializeGlobalDevice() first to initialize.
 */
Device& getGlobalDevice();

/**
 * @brief Initialize the global device
 */
Result initializeGlobalDevice(int cudaDeviceId = 0, bool enableValidation = false);

/**
 * @brief Shutdown the global device
 */
void shutdownGlobalDevice();

} // namespace gaussian_rt
