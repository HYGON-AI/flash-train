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

TEST(LoggingTest, LevelsFilterAndDefaultIsWarning) {
    struct Case {
        const char* level;
        int count;
    };
    for (const Case test : {Case{nullptr, 2}, {"", 2}, {"  ", 2}, {"OFF", 0}, {"ERROR", 1}, {"WARNING", 2},
                            {"INFO", 3}, {"DEBUG", 4}, {"  dEbUg\t", 4}}) {
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

TEST(LoggingTest, DisabledArgumentsAreNotEvaluated) {
    EXPECT_EQ(runProbe("lazy", "WARNING").output, "calls=0\n");
    EXPECT_EQ(runProbe("lazy", "DEBUG").output, "calls=1\n");
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

TEST(LoggingTest, BufferEscapesControlsAndReplacesInvalidUtf8) {
    logging::Buffer record;
    record.append("a\nb\rc\t");
    record.append(std::string_view{"\xff\0", 2});
    EXPECT_STREQ(record.data(), "a\\nb\\rc\\t?\\x00");
    record.append("中文");
    EXPECT_NE(std::string{record.data()}.find("中文"), std::string::npos);
}

TEST(LoggingTest, LongUtf8RecordsAreBoundedAndMarked) {
    const auto result = runProbe("large", "ERROR");
    ASSERT_EQ(result.status, 0);
    EXPECT_LE(result.error.size(), logging::Buffer::kCapacity - 1);
    EXPECT_EQ(occurrences(result.error, "\n"), 1);
    EXPECT_NE(result.error.find("...[truncated]\n"), std::string::npos);
    logging::Buffer record;
    for (int i = 0; i < 5000; ++i) { record.append("中"); }
    EXPECT_TRUE(record.full());
    const auto marker_pos = std::string{record.data()}.find("...[truncated]");
    ASSERT_NE(marker_pos, std::string::npos);
    EXPECT_EQ(marker_pos % 3, 0u);
    const auto size = record.size();
    record.append("ignored");
    EXPECT_EQ(record.size(), size);
}

TEST(LoggingTest, PrintfFormattingIsBoundedAndLiteralPercentIsSafe) {
    logging::Buffer record;
    record.format("number=%d text=%s", 7, "100%");
    EXPECT_STREQ(record.data(), "number=7 text=100%");
    logging::Buffer failure;
    failure.format("%lc", static_cast<unsigned int>(0xd800));
    EXPECT_STREQ(failure.data(), "[format_error]");
    record.format("%10000d", 1);
    EXPECT_TRUE(record.full());
    EXPECT_LT(record.size(), logging::Buffer::kCapacity);
}

}  // namespace
}  // namespace ftrain
