// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <gtest/gtest.h>

#include "flash_train/logging.hpp"

extern char** environ;

namespace ftrain {
namespace {

struct ProbeResult {
    int status;
    std::string error;
    std::string output;
};

std::string readFile(FILE* file) {
    std::rewind(file);
    std::string result;
    char buffer[4096];
    std::size_t n;
    while ((n = std::fread(buffer, 1, sizeof(buffer), file)) != 0) { result.append(buffer, n); }
    return result;
}

ProbeResult runProbe(const char* scenario, const char* level) {
    FILE* error  = std::tmpfile();
    FILE* output = std::tmpfile();
    if (error == nullptr || output == nullptr) { std::abort(); }
    std::vector<std::string> environment;
    for (char** entry = environ; *entry != nullptr; ++entry) {
        if (std::strncmp(*entry, "FTRAIN_LOG_LEVEL=", 17) != 0) { environment.emplace_back(*entry); }
    }
    if (level != nullptr) { environment.emplace_back(std::string{"FTRAIN_LOG_LEVEL="} + level); }
    std::vector<char*> env;
    for (auto& entry : environment) { env.push_back(entry.data()); }
    env.push_back(nullptr);
    char* args[] = {const_cast<char*>(FTRAIN_LOGGING_PROBE_PATH), const_cast<char*>(scenario), nullptr};
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_adddup2(&actions, fileno(error), STDERR_FILENO);
    posix_spawn_file_actions_adddup2(&actions, fileno(output), STDOUT_FILENO);
    pid_t pid{};
    const int spawn_status = posix_spawn(&pid, args[0], &actions, nullptr, args, env.data());
    posix_spawn_file_actions_destroy(&actions);
    int status = -1;
    if (spawn_status == 0) {
        while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
        status = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    }
    ProbeResult result{status, readFile(error), readFile(output)};
    std::fclose(error);
    std::fclose(output);
    return result;
}

std::size_t occurrences(const std::string& text, const std::string& needle) {
    std::size_t count = 0, offset = 0;
    while ((offset = text.find(needle, offset)) != std::string::npos) {
        ++count;
        offset += needle.size();
    }
    return count;
}

// The bounded-record contract shared by every overflow path: one line under
// 4096 bytes, visibly marked.
void expectBoundedMarkedLine(const ProbeResult& result) {
    EXPECT_LE(result.error.size(), 4095u);
    EXPECT_EQ(occurrences(result.error, "\n"), 1);
    EXPECT_NE(result.error.find("...[truncated]\n"), std::string::npos);
}

TEST(LoggingTest, LevelsFilterAndDefaultIsWarning) {
    struct Case {
        const char* level;
        int count;
    };
    for (const Case test : {
             Case{nullptr,     2},
             {"",          2},
             {"  ",        2},
             {"OFF",       0},
             {"ERROR",     1},
             {"WARNING",   2},
             {"INFO",      3},
             {"DEBUG",     4},
             {"TRACE",     5},
             {"  dEbUg\t", 4}
    }) {
        const auto result = runProbe("levels", test.level);
        ASSERT_EQ(result.status, 0);
        EXPECT_EQ(occurrences(result.error, " marker"), test.count) << result.error;
        if (test.count == 0) { EXPECT_TRUE(result.error.empty()); }
    }
}

TEST(LoggingTest, PrefixHasNoModuleAndPlacesLevelBeforeTheMessage) {
    const auto result = runProbe("prefix", "WARNING");
    ASSERT_EQ(result.status, 0);
    ASSERT_EQ(occurrences(result.error, "Prefix format check"), 1);
    const auto program = result.error.find("[ftrain][pid=");
    const auto thread  = result.error.find("[thread=");
    const auto level   = result.error.find("][WARNING] Prefix format check");
    ASSERT_NE(program, std::string::npos);
    ASSERT_NE(thread, std::string::npos);
    ASSERT_NE(level, std::string::npos);
    EXPECT_LT(program, thread);
    EXPECT_LT(thread, level);
    EXPECT_EQ(occurrences(result.error, "\n"), 1);
}

TEST(LoggingTest, InvalidConfigurationWarnsOnceAndKeepsDefault) {
    const auto result = runProbe("init_race", "not-a-level");
    ASSERT_EQ(result.status, 0);
    EXPECT_EQ(occurrences(result.error, "Unknown FTRAIN_LOG_LEVEL; using WARNING"), 1);
    EXPECT_EQ(occurrences(result.error, "Thread "), 400);
}

TEST(LoggingTest, ConfigurationIsReadOnlyOnce) {
    const auto result = runProbe("once", "WARNING");
    ASSERT_EQ(result.status, 0);
    EXPECT_EQ(result.error.find("Changed level marker"), std::string::npos);
    EXPECT_NE(result.error.find("Retained level marker"), std::string::npos);
}

TEST(LoggingTest, GatedEntriesEvaluateNothing) {
    EXPECT_EQ(runProbe("lazy", "WARNING").output, "calls=0\n");
    EXPECT_EQ(runProbe("lazy_custom", "WARNING").output, "calls=0\n");
    EXPECT_EQ(runProbe("lazy", "DEBUG").output, "calls=1\n");
    EXPECT_EQ(runProbe("lazy_custom", "DEBUG").output, "calls=1\n");
}

TEST(LoggingTest, CallbackExceptionsDoNotEscapeAndPreserveErrno) {
    const auto result = runProbe("exceptions", "DEBUG");
    ASSERT_EQ(result.status, 0);
    EXPECT_NE(result.error.find("Logging failure did not escape"), std::string::npos);
}

TEST(LoggingTest, ThreadedRecordsHaveCompletePrefixesAndTerminators) {
    const auto result = runProbe("threads", "WARNING");
    ASSERT_EQ(result.status, 0);
    EXPECT_EQ(occurrences(result.error, "[ftrain][pid="), 400);
    EXPECT_EQ(occurrences(result.error, "[WARNING] Thread "), 400);
    EXPECT_EQ(occurrences(result.error, "[thread="), 400);
    EXPECT_EQ(occurrences(result.error, "completed\n"), 400);
    EXPECT_EQ(occurrences(result.error, "\n"), 400);
}

TEST(LoggingTest, WriteFailurePreservesErrno) {
    const auto result = runProbe("write_failure", "ERROR");
    ASSERT_EQ(result.status, 0);
    EXPECT_EQ(result.output, "errno=preserved\n");
}

TEST(LoggingTest, EscapingPreservesSingleLineAndReplacesInvalidUtf8) {
    const auto result = runProbe("escape", "WARNING");
    ASSERT_EQ(result.status, 0);
    EXPECT_NE(result.error.find("esc [a\\nb\\rc\\td??]"), std::string::npos) << result.error;
    EXPECT_EQ(occurrences(result.error, "\n"), 1);
}

TEST(LoggingTest, LongRecordsAreBoundedAndMarked) {
    const auto result = runProbe("large", "ERROR");
    ASSERT_EQ(result.status, 0);
    expectBoundedMarkedLine(result);
}

TEST(LoggingTest, Utf8TruncationLandsOnCodepointBoundary) {
    const auto result = runProbe("utf8_trunc", "ERROR");
    ASSERT_EQ(result.status, 0);
    const auto marker = result.error.find("...[truncated]");
    ASSERT_NE(marker, std::string::npos);
    // The three bytes before the marker must be one complete U+4E2D codepoint.
    EXPECT_EQ(result.error.compare(marker - 3, 3, "\xe4\xb8\xad"), 0) << result.error.substr(0, 96);
    expectBoundedMarkedLine(result);
}

TEST(LoggingTest, PrintfEdgesAreSafeThroughTheMacro) {
    const auto result = runProbe("printf_edge", "ERROR");
    ASSERT_EQ(result.status, 0);
    EXPECT_NE(result.error.find("lit 100%"), std::string::npos) << result.error;
    EXPECT_NE(result.error.find("wide ["), std::string::npos);
    EXPECT_NE(result.error.find("...[truncated]\n"), std::string::npos);
    EXPECT_NE(result.error.find("ERROR] [format_error]"), std::string::npos) << result.error;
    EXPECT_EQ(occurrences(result.error, "\n"), 3);
}

TEST(LoggingTest, CustomTypesLogTransparentlyAndPreserveErrno) {
    const auto result = runProbe("custom", "WARNING");
    ASSERT_EQ(result.status, 0);
    EXPECT_NE(result.error.find("transparent=Order(id=7, items=3) ok"), std::string::npos) << result.error;
    EXPECT_NE(result.error.find("mix=Order(id=7, items=3) and=Order(id=9, items=9)"), std::string::npos)
        << result.error;
    EXPECT_NE(result.error.find("route=gemm:gfx938#42"), std::string::npos) << result.error;
    EXPECT_NE(result.error.find("at level=WARNING"), std::string::npos) << result.error;
    EXPECT_NE(result.error.find("runtime level=INFO"), std::string::npos) << result.error;
}

TEST(LoggingTest, ComposedWritesConcatenateIntoOneArgument) {
    const auto result = runProbe("compose", "INFO");
    ASSERT_EQ(result.status, 0);
    EXPECT_NE(result.error.find("team: [0] [1] [2]"), std::string::npos) << result.error;
}

TEST(LoggingTest, FullArenaArgumentRendersEmptyAndStaysBounded) {
    const auto result = runProbe("full_arena", "ERROR");
    ASSERT_EQ(result.status, 0);
    expectBoundedMarkedLine(result);
}

}  // namespace
}  // namespace ftrain
