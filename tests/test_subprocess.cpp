#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "core/subprocess.hpp"

namespace {

void expect(bool cond, const char* msg)
{
    if (!cond) {
        std::cerr << "FAIL: " << msg << '\n';
        std::exit(1);
    }
}

void test_env_values_not_shell_interpreted()
{
    const auto dir = std::filesystem::temp_directory_path() / "vlm_subprocess_test";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);

    const auto script = dir / "echo_env.sh";
    const auto out_file = dir / "out.txt";
    const auto pwned_file = dir / "pwned.txt";
    {
        std::ofstream f(script);
        f << "#!/usr/bin/env bash\n";
        f << "printf '%s' \"$MODELS_DIR\" > \"" << out_file.string() << "\"\n";
    }
    std::filesystem::permissions(
        script, std::filesystem::perms::owner_read | std::filesystem::perms::owner_exec);

    const std::string malicious =
        R"(/tmp/foo"; id > )" + pwned_file.string() + R"(; echo "bar)";

    expect(vlm::runBashScript(script, {{"MODELS_DIR", malicious}}),
           "runBashScript should succeed");

    std::ifstream in(out_file);
    std::string captured;
    std::getline(in, captured, '\0');
    expect(captured == malicious, "env value must be passed literally to the child");

    expect(!std::filesystem::exists(pwned_file), "shell metacharacters must not execute");

    std::filesystem::remove_all(dir);
}

}  // namespace

int main()
{
    test_env_values_not_shell_interpreted();
    std::cout << "test_subprocess: OK\n";
    return 0;
}
