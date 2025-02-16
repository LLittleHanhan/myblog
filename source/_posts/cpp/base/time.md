---
title: time
date: 2024-05-25
tags: cpp
---
cpp11 chrono库
<!--more-->
## radio
时间的粒度
```cpp
template <intmax_t N, intmax_t D = 1> class ratio;

ratio<3600, 1>                //hours
ratio<60, 1>                  //minutes
ratio<1, 1>                   //seconds
ratio<1, 1000>                //microseconds
ratio<1, 1000000>             //microseconds
ratio<1, 1000000000>          //nanosecons
```
## duration
描述一段时间
```cpp
template <typename Rep, typename Period = ratio<1> > class duration;
//Period为radio即粒度单位
//Rep数值类型计数

std::chrono::hours hours(1);                // 1小时
std::chrono::minutes minutes(60);           // 60分钟
std::chrono::seconds seconds(3600);         // 3600秒
std::chrono::milliseconds milliseconds(1);  // 1毫秒
std::chrono::microseconds microseconds(1000); // 1000微秒
//以上Rep为int64_t，且不会隐式转换?因为是模板参数？
std::chrono::duration<double,ratio<1,1000>>
```
```cpp
duration.count();
duration_cast<daration>
//存在隐式转换的情况，一个duration通过rep和radio确定，隐式转换什么是可以？算了全用显式
//rep类型相同的情况下，大时间可以变为小时间s->ns
```
## time_point
时间点，怎么确定时间点？要有一个起始位置clock和时间段duration
```cpp
template<class Clock,class Duration = typename Clock::duration> class time_point;
```

## clock
system_clock：系统的时钟，系统的时钟可以修改，甚至可以网络对时，因此使用系统时间计算时间差可能不准。
steady_clock：是固定的时钟，相当于秒表。开始计时后，时间只会增长并且不能修改，适合用于记录程序耗时
high_resolution_clock：和时钟类 steady_clock 是等价的（是它的别名）

## reference
[C++ 使用 chrono 库处理日期和时间](https://www.eet-china.com/mp/a74844.html)

```cpp
//一份计时库的实现
#pragma once

#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <string_view>
#include <vector>

class ScopeProfiler {
   public:
    using ClockType = std::chrono::high_resolution_clock;

    struct Record {
        const char* tag;
        int us;
    };

   private:
    inline thread_local static std::vector<Record> records;

    ClockType::time_point beg;
    ClockType::time_point end;
    const char* tag;

    inline ScopeProfiler(const char* tag, ClockType::time_point beg);
    inline void onDestroy(ClockType::time_point end);

   public:
    ScopeProfiler(const char* tag_)
        : ScopeProfiler(tag_, ClockType::now()) {}
    ~ScopeProfiler() { onDestroy(ClockType::now()); }

    static std::vector<Record> const& getRecords() { return records; }
    inline static void printLog(std::ostream& out = std::cout);
};

ScopeProfiler::ScopeProfiler(const char* tag_, ScopeProfiler::ClockType::time_point beg_)
    : beg(beg_), tag(tag_) {
}

void ScopeProfiler::onDestroy(ScopeProfiler::ClockType::time_point end) {
    auto diff = end - beg;
    int us = std::chrono::duration_cast<std::chrono::microseconds>(diff).count();
    records.push_back({tag, us});
}

void ScopeProfiler::printLog(std::ostream& out) {
    if (records.size() == 0) {
        return;
    }

    struct Statistic {
        int max_us = 0;
        int min_us = 0;
        int total_us = 0;
        int count_rec = 0;
        const char* tag = nullptr;
    };
    std::map<std::string_view, Statistic> stats;
    for (auto const& [tag, us] : records) {
        auto& stat = stats[tag];
        stat.total_us += us;
        stat.max_us = std::max(stat.max_us, us);
        stat.min_us = !stat.count_rec ? us : std::min(stat.min_us, us);
        stat.count_rec++;
        stat.tag = tag;
    }

    struct StatisticCompare {
        using value_type = std::pair<std::string_view, Statistic>;
        bool operator()(value_type const& lhs, value_type const& rhs) const {
            return lhs.second.total_us > rhs.second.total_us;
        }
    };

    std::multiset<std::pair<std::string_view, Statistic>, StatisticCompare> sortstats(stats.begin(), stats.end());

    auto dump = [&out](int val, int w) {
        auto tpwv = 1;
        for (int i = 0; i < w - 1; i++)
            tpwv *= 10;
        if (val > tpwv) {
            if (val / 1000 > tpwv / 10) {
                out << std::setw(w - 1) << val / 1000000 << 'M';
            } else {
                out << std::setw(w - 1) << val / 1000 << 'k';
            }
        } else {
            out << std::setw(w) << val;
        }
    };

    out << "   avg   |   min   |   max   |  total  | cnt | tag\n";
    for (auto const& [tag, stat] : sortstats) {
        dump(stat.total_us / stat.count_rec, 9);
        out << '|';
        dump(stat.min_us, 9);
        out << '|';
        dump(stat.max_us, 9);
        out << '|';
        dump(stat.total_us, 9);
        out << '|';
        dump(stat.count_rec, 5);
        out << '|';
        out << ' ' << tag << '\n';
    }
}

#if defined(__GNUC__) || defined(__clang__)
#define DefScopeProfiler ScopeProfiler _scopeProfiler(__PRETTY_FUNCTION__);
#elif defined(_MSC_VER)
#define DefScopeProfiler ScopeProfiler _scopeProfiler(__FUNCSIG__);
#else
#define DefScopeProfiler ScopeProfiler _scopeProfiler(__func__);
#endif

static void printScopeProfiler(std::ostream& out = std::cout) {
    ScopeProfiler::printLog(out);
}
```