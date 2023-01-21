#pragma once

#include <filesystem>
#include <memory>
#include <optional>

class Network final {
public:
    Network(Network&&);
    Network& operator=(Network&&);
    Network(const Network&) = delete;
    Network& operator=(const Network&) = delete;
    ~Network();
    
    static std::optional<Network> init(std::filesystem::path cfgfile,
                                       std::filesystem::path weightfile,
                                       const int imw,
                                       const int imh,
                                       const int imch);
    
    float const * const run(unsigned char* data);
    
    
    
private:
    struct Impl;
    explicit Network(Impl&& impl);

    std::unique_ptr<Impl> impl;
    
};
