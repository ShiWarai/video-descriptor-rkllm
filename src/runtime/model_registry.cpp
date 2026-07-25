#include "runtime/model_registry.hpp"

#include <algorithm>
#include <cctype>

namespace vlm {

namespace {

bool containsIgnoreCase(std::string_view haystack, std::string_view needle)
{
    if (needle.empty()) {
        return true;
    }
    std::string h(haystack);
    std::string n(needle);
    std::transform(h.begin(), h.end(), h.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    std::transform(n.begin(), n.end(), n.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return h.find(n) != std::string::npos;
}

}  // namespace

void ModelRegistry::setDefaultModelId(std::string id)
{
    default_model_id_ = std::move(id);
}

void ModelRegistry::add(ModelSpec spec)
{
    const std::string key = normalizeKey(spec.id);
    if (index_.contains(key)) {
        models_[index_.at(key)] = std::move(spec);
        return;
    }
    index_[key] = models_.size();
    models_.push_back(std::move(spec));
}

std::string ModelRegistry::normalizeKey(std::string_view key)
{
    std::string out(key);
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return out;
}

const ModelSpec* ModelRegistry::find(std::string_view id) const
{
    const auto it = index_.find(normalizeKey(id));
    if (it == index_.end()) {
        return nullptr;
    }
    return &models_[it->second];
}

std::optional<std::string> ModelRegistry::resolveId(std::string_view requested) const
{
    if (models_.empty()) {
        return std::nullopt;
    }

    if (requested.empty()) {
        if (!default_model_id_.empty()) {
            return default_model_id_;
        }
        return models_.front().id;
    }

    if (const ModelSpec* exact = find(requested)) {
        return exact->id;
    }

    const std::string norm = normalizeKey(requested);

    // Short aliases: "4b", "2b", "0.8b"
    if (norm == "4b" || norm == "qwen3.5-4b") {
        for (const auto& model : models_) {
            if (containsIgnoreCase(model.id, "4b")) {
                return model.id;
            }
        }
    }
    if (norm == "2b" || norm == "qwen3.5-2b") {
        for (const auto& model : models_) {
            if (containsIgnoreCase(model.id, "2b") && !containsIgnoreCase(model.id, "0.8") &&
                !containsIgnoreCase(model.id, "4b")) {
                return model.id;
            }
        }
    }
    if (norm == "0.8b" || norm == "qwen3.5-0.8b") {
        for (const auto& model : models_) {
            if (containsIgnoreCase(model.id, "0.8")) {
                return model.id;
            }
        }
    }

    // Partial match on id
    for (const auto& model : models_) {
        if (containsIgnoreCase(model.id, requested)) {
            return model.id;
        }
    }

    return std::nullopt;
}

}  // namespace vlm
