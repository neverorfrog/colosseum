#pragma once

#include "Policy.h"
#include <functional>
#include <unordered_map>

class TaskRegistry {
public:
    using Factory = std::function<std::unique_ptr<Policy>()>;

    static TaskRegistry& instance();

    void register_task(const std::string& name, Factory factory);
    std::unique_ptr<Policy> create(const std::string& name) const;
    bool has(const std::string& name) const;

    // Nested helper: construct one of these as a static local variable
    // to register a task before main() runs.
    struct Registrar {
        Registrar(const std::string& name, Factory factory) {
            TaskRegistry::instance().register_task(name, std::move(factory));
        }
    };

private:
    TaskRegistry() = default;
    std::unordered_map<std::string, Factory> factories_;
};

// Convenience macro — mirrors REGISTER_MODULE from spqrbooster.
// Place once per task .cpp file (outside any namespace/class).
#define REGISTER_TASK(task_name, ClassName)                                 \
    static TaskRegistry::Registrar __##ClassName##_registrar(               \
        task_name,                                                          \
        []() -> std::unique_ptr<Policy> {                                   \
            return std::make_unique<ClassName>();                           \
        }                                                                   \
    );
