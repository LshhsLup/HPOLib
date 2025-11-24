#ifndef __HPOLIB_LOGGER_H__
#define __HPOLIB_LOGGER_H__

#include <atomic>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>

namespace hpolib {

#define LOGD(...)                                                          \
  do {                                                                     \
    if (hpolib::Logger::shouldLog(hpolib::LogLevel::LOG_DEBUG))            \
      hpolib::Logger::log(hpolib::LogLevel::LOG_DEBUG, __FILE__, __LINE__, \
                          __VA_ARGS__);                                    \
  } while (0)
#define LOGI(...)                                                         \
  do {                                                                    \
    if (hpolib::Logger::shouldLog(hpolib::LogLevel::LOG_INFO))            \
      hpolib::Logger::log(hpolib::LogLevel::LOG_INFO, __FILE__, __LINE__, \
                          __VA_ARGS__);                                   \
  } while (0)
#define LOGW(...)                                                         \
  do {                                                                    \
    if (hpolib::Logger::shouldLog(hpolib::LogLevel::LOG_WARN))            \
      hpolib::Logger::log(hpolib::LogLevel::LOG_WARN, __FILE__, __LINE__, \
                          __VA_ARGS__);                                   \
  } while (0)
#define LOGE(...)                                                          \
  do {                                                                     \
    if (hpolib::Logger::shouldLog(hpolib::LogLevel::LOG_ERROR))            \
      hpolib::Logger::log(hpolib::LogLevel::LOG_ERROR, __FILE__, __LINE__, \
                          __VA_ARGS__);                                    \
  } while (0)

enum class LogLevel : int8_t {
  LOG_DEBUG = 0,
  LOG_INFO = 1,
  LOG_WARN = 2,
  LOG_ERROR = 3,
  LOG_NONE = 100
};

class Logger {
 public:
  Logger() = delete;

  static void setMinLevel(LogLevel lvl) noexcept { min_level_.store(lvl); }
  static void enableColor(bool on) noexcept { color_enabled_.store(on); }
  static void setLogFile(const std::string& path) {
    std::lock_guard<std::mutex> lock(io_mutex_);
    if (log_file_.is_open())
      log_file_.close();
    log_file_.open(path, std::ios::out | std::ios::app);
    file_enabled_.store(log_file_.is_open());
  }

  static void log(LogLevel lvl, const char* file, int line, const char* fmt,
                  ...) {
    if (lvl < min_level_.load())
      return;

    char msg_buf[1024];
    va_list args;
    va_start(args, fmt);
    vsnprintf(msg_buf, sizeof(msg_buf), fmt, args);
    va_end(args);

    std::ostringstream oss;
    oss << nowStr() << " [" << threadIdStr() << "] " << levelStr(lvl) << " "
        << basename(file) << ":" << line << " - " << msg_buf;

    std::string out = oss.str();
    if (color_enabled_.load())
      out = colorWrap(lvl, out);

    writeSync(out);
  }

  static bool shouldLog(LogLevel lvl) noexcept {
    return lvl >= min_level_.load();
  }

 private:
  static void writeSync(const std::string& s) {
    std::lock_guard<std::mutex> lock(io_mutex_);
    if (file_enabled_.load() && log_file_.is_open()) {
      log_file_ << s << '\n';
      log_file_.flush();
    } else {
      std::fprintf(stderr, "%s\n", s.c_str());
      std::fflush(stderr);
    }
  }

  static std::string nowStr() {
    using namespace std::chrono;
    auto now = system_clock::now();
    auto t = system_clock::to_time_t(now);
    auto ms = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    char buf[64];
    snprintf(buf, sizeof(buf), "%04d-%02d-%02d %02d:%02d:%02d.%03d",
             tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour,
             tm.tm_min, tm.tm_sec, (int)ms.count());
    return buf;
  }

  static std::string threadIdStr() {
    std::ostringstream oss;
    oss << std::this_thread::get_id();
    return oss.str();
  }

  static const char* levelStr(LogLevel lvl) {
    switch (lvl) {
      case LogLevel::LOG_DEBUG:
        return "DEBUG";
      case LogLevel::LOG_INFO:
        return "INFO ";
      case LogLevel::LOG_WARN:
        return "WARN ";
      case LogLevel::LOG_ERROR:
        return "ERROR";
      default:
        return "UNKWN";
    }
  }

  static std::string basename(const char* path) {
    if (!path)
      return "";
    const char* p = std::strrchr(path, '/');
    if (p)
      return p + 1;
    p = std::strrchr(path, '\\');
    return p ? p + 1 : path;
  }

  static std::string colorWrap(LogLevel lvl, const std::string& s) {
    const char* color = "";
    switch (lvl) {
      case LogLevel::LOG_DEBUG:
        color = "\033[36m";
        break;  // cyan
      case LogLevel::LOG_INFO:
        color = "\033[32m";
        break;  // green
      case LogLevel::LOG_WARN:
        color = "\033[33m";
        break;  // yellow
      case LogLevel::LOG_ERROR:
        color = "\033[31m";
        break;  // red
    }
    return std::string(color) + s + "\033[0m";
  }

 private:
  static inline std::atomic<LogLevel> min_level_{LogLevel::LOG_DEBUG};
  static inline std::atomic<bool> color_enabled_{true};
  static inline std::mutex io_mutex_;
  static inline std::ofstream log_file_;
  static inline std::atomic<bool> file_enabled_{false};
};
}  // namespace hpolib

#endif