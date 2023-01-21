#include "network.hpp"

#include "darknet.h"

using namespace std;

using network_ptr = std::unique_ptr<network, void(*)(network*)>;

network_ptr make_null_network_ptr()
{
    return {nullptr, [](network*){}};
}

network_ptr make_network_ptr(network* net)
{
    return {net, [](struct network* net){ free_network(net); }};
}

struct Network::Impl
{
    network_ptr net{make_null_network_ptr()};
    image im;
    image crop;
    int srcw;
    int srch;
    int srcch;
};

std::optional<Network> Network::init(std::filesystem::path cfgfile,
                                   std::filesystem::path weightfile,
                                   const int imw,
                                   const int imh,
                                   const int imch)
{
    auto sCfgFile = std::move(cfgfile).string();
    auto sWeightFile = std::move(weightfile).string();
    auto const network = load_network(sCfgFile.data(), sWeightFile.data(), 0);
    if (auto net = make_network_ptr(network))
    {
        auto impl = Impl{};
        impl.net = move(net);
        impl.im = make_image(imw, imh, imch);
        impl.srcw = imw;
        impl.srch = imh;
        impl.srcch = imch;
        
        return { Network{move(impl)} };
    }
    return {};
}

float const * const Network::run(unsigned char* indata)
{
    int i = 0;
    int j = 0;
    int k = 0;
    
    auto const srch = impl->srch;
    auto const srcch = impl->srcch;
    auto const srcw = impl->srcw;
    network* net = impl->net.get();
    image* crop = &impl->crop;
    for(i = 0; i < srch; ++i)
        {
        for(k= 0; k < srcch; ++k)
            {
            for(j = 0; j < srcw; ++j)
                {
                impl->im.data[k * srcw * srch + i * srcw + j] = indata[i * srcch * srcw + j * srcch + k] / 255.;
                }
            }
        }
    rgbgr_image(impl->im);
    *crop = center_crop_image(impl->im, net->w, net->h);
    float *pred = network_predict(net, crop->data);
    return pred;
}

Network::~Network() = default;

Network::Network(Network&&) = default;
Network& Network::operator=(Network&&) = default;

 Network::Network(Network::Impl &&impl)
: impl{make_unique<Impl>(move(impl))}  {
}
