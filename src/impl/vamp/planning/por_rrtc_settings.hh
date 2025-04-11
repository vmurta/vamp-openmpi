#pragma once
#include <atomic>
#include <memory>

namespace vamp::planning
{
    struct Finish_Flag
    {

        std::shared_ptr<std::atomic<int>> f_ptr = std::make_shared<std::atomic<int>>(-1);

        void set(int i)
        {
            *f_ptr = i;
        }

        void reset()
        {
            *f_ptr = -1;
        }

        int get()
        {
            return *f_ptr;
        }
    };



    struct POR_RRTCSettings
    {
        float range = 2.;

        bool dynamic_domain = true;
        float radius = 4.;
        float alpha = 0.0001;
        float min_radius = 1.;

        bool balance = true;
        float tree_ratio = 1.;

        std::size_t max_iterations = 100000;
        std::size_t max_samples = 100000;
        bool start_tree_first = true;

        int thread_id = 0;

        POR_RRTCSettings copy()
        {
            return *this;
        }
    };
}  // namespace vamp::planning
