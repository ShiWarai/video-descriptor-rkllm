#include <filesystem>
#include <fstream>
#include <iostream>

#include "api/openai_handlers.hpp"

namespace {

void expect(bool cond, const char* msg)
{
    if (!cond) {
        std::cerr << "FAIL: " << msg << '\n';
        std::exit(1);
    }
}

void test_empty_upload_throws()
{
    const auto dir = std::filesystem::temp_directory_path() / "vlm_upload_test";
    std::filesystem::create_directories(dir);
    bool threw = false;
    try {
        vlm::saveUploadedFile("x.mp4", "", dir);
    } catch (const std::exception&) {
        threw = true;
    }
    expect(threw, "empty content must throw");

    threw = false;
    try {
        vlm::saveUploadedFile("", "abc", dir);
    } catch (const std::exception&) {
        threw = true;
    }
    expect(threw, "empty filename must throw");
    std::filesystem::remove_all(dir);
}

void test_workdir_clear_and_owned_remove()
{
    const auto dir = std::filesystem::temp_directory_path() / "vlm_work_test";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    const auto f1 = dir / "a.mp4";
    {
        std::ofstream(f1) << "data";
    }
    expect(std::filesystem::exists(f1), "file created");
    vlm::clearWorkdir(dir);
    expect(!std::filesystem::exists(f1), "clearWorkdir removes files");
    expect(std::filesystem::exists(dir), "clearWorkdir keeps directory");

    const auto owned = vlm::saveUploadedFile("clip.mp4", "payload", dir);
    expect(std::filesystem::exists(owned), "upload saved");
    vlm::removeWorkFileIfOwned(owned, dir);
    expect(!std::filesystem::exists(owned), "owned work file removed");

    const auto outside = std::filesystem::temp_directory_path() / "vlm_outside_keep.mp4";
    {
        std::ofstream(outside) << "keep";
    }
    vlm::removeWorkFileIfOwned(outside, dir);
    expect(std::filesystem::exists(outside), "outside file not removed");
    std::filesystem::remove(outside);
    std::filesystem::remove_all(dir);
}

}  // namespace

int main()
{
    test_empty_upload_throws();
    test_workdir_clear_and_owned_remove();
    std::cout << "test_workdir: ok\n";
    return 0;
}
