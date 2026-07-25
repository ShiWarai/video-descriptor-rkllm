#include "core/subprocess.hpp"

#include <sys/wait.h>
#include <unistd.h>

#include <cstdlib>
#include <vector>

namespace vlm {

bool runBashScript(const std::filesystem::path& script_path,
                   const std::vector<std::pair<std::string, std::string>>& env_overrides)
{
    std::string script = script_path.string();
    std::vector<char> script_argv(script.begin(), script.end());
    script_argv.push_back('\0');

    const pid_t pid = fork();
    if (pid < 0) {
        return false;
    }
    if (pid == 0) {
        for (const auto& [key, value] : env_overrides) {
            setenv(key.c_str(), value.c_str(), 1);
        }
        execl("/bin/bash", "bash", script_argv.data(), static_cast<char*>(nullptr));
        _exit(127);
    }

    int status = 0;
    if (waitpid(pid, &status, 0) < 0) {
        return false;
    }
    return WIFEXITED(status) && WEXITSTATUS(status) == 0;
}

}  // namespace vlm
