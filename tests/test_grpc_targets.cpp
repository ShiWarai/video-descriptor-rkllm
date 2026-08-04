#include <iostream>
#include <vector>

#include "grpc/tensor_codec.hpp"

int main()
{
    const auto targets = vlm::grpc_util::parseTargets("MISSING_ENV", "host-a:50051,host-b:50052");
    if (targets.size() != 2 || targets[0] != "host-a:50051" || targets[1] != "host-b:50052") {
        std::cerr << "FAIL: parseTargets default\n";
        return 1;
    }

    const std::vector<std::string> expanded =
        vlm::grpc_util::expandTargets({"127.0.0.1:50051"});
    if (expanded.empty() || expanded[0].find("50051") == std::string::npos) {
        std::cerr << "FAIL: expandTargets localhost\n";
        return 1;
    }

    std::cout << "test_grpc_targets: ok\n";
    return 0;
}
