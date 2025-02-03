#pragma once

#include <memory>

#include <vamp/collision/environment.hh>
#include <vamp/planning/nn.hh>
#include <vamp/planning/plan.hh>
#include <vamp/planning/validate.hh>
#include <vamp/planning/rrtc_settings.hh>
#include <vamp/random/halton.hh>
#include <vamp/random/rng.hh>
#include <vamp/utils.hh>
#include <vamp/vector.hh>
#include <iostream>
#include <omp.h>

namespace vamp::planning
{
    template <typename Robot, std::size_t rake, std::size_t resolution>
    struct RRTC
    {
        using Configuration = typename Robot::Configuration;
        static constexpr auto dimension = Robot::dimension;
        using RNG = typename vamp::rng::RNG<Robot::dimension>;

        inline static auto solve(
            const Configuration &start,
            const Configuration &goal,
            const collision::Environment<FloatVector<rake>> &environment,
            const RRTCSettings &settings,
            typename RNG::Ptr rng) noexcept -> PlanningResult<dimension>
        {
            return solve(start, std::vector<Configuration>{goal}, environment, settings, rng);
        }

        inline static auto solve(
            const Configuration &start,
            const std::vector<Configuration> &goals,
            const collision::Environment<FloatVector<rake>> &environment,
            const RRTCSettings &settings,
            typename RNG::Ptr rng) noexcept -> PlanningResult<dimension>
        {
            //@TODO: start a timer here
            PlanningResult<dimension> result;

            constexpr const std::size_t start_index = 0;
            constexpr const std::size_t goal_index = 1;

            auto start_time = std::chrono::steady_clock::now();

            for (const auto &goal : goals)
            {
                if (validate_motion<Robot, rake, resolution>(start, goal, environment))
                {
                    result.path.emplace_back(start);
                    result.path.emplace_back(goal);
                    result.nanoseconds = vamp::utils::get_elapsed_nanoseconds(start_time);
                    result.iterations = 0;
                    result.size.emplace_back(1);
                    result.size.emplace_back(1);

                    return result;
                }
            }
            // trees
            
            //@TODO: make sure each process has a unique seed
            // NOTE: don't actually do this until we have everything else working for the OK implementation
            // it looks like by default, the seed is the same for all processes
            // (technically I think rng_skip_iterations is not a seed, but for our purposes it is)
            // To make sure that each process isn't just creating the exact same random tree, we want this
            // rng_skip_iterations value to be non overlapping for each process. We need to figure out how to 
            // prorgammatically set this value for each process. I believe it should be 
            // rng_skip_iterations += process_id * max_iterations, but we may want to test this out to make sure
            // auto rng_skip_time = std::chrono::steady_clock::now();



            // std::cout << "Rank " << rank << " rng set in " << vamp::utils::get_elapsed_nanoseconds(rng_skip_time) << std::endl;

            std::size_t free_index = start_index + 1;


            bool found_solution = false;
            std::size_t iter = 0;


            // std::cout << "Starting parallel region" << std::endl;
            #pragma omp parallel private(iter, free_index)
            {
                //#TODO: make this as memory efficient as possible
                // std::cout << "Thread " << omp_get_thread_num() << " started" << std::endl;
                alignas(FloatVectorAlignment) std::array<float, dimension> init_v;
                std::copy_n(vamp::rng::Halton<dimension>::primes.cbegin() + omp_get_thread_num(), dimension, init_v.begin()); // #only works when dimension + num_ranks < 32
                vamp::rng::Halton<dimension> rng(init_v);
                
                std::vector<std::size_t> parents(settings.max_samples);
                std::vector<float> radii(settings.max_samples);

                NN<dimension> start_tree;
                NN<dimension> goal_tree;

                auto buffer = std::shared_ptr<float>(
                    vamp::utils::vector_alloc<float, FloatVectorAlignment, FloatVectorWidth>(
                        settings.max_samples * Configuration::num_scalars_rounded),
                        &free);

                const auto buffer_index = [&buffer](std::size_t index) -> float *
                    { return buffer.get() + index * Configuration::num_scalars_rounded; };


                // add start to tree
                start.to_array(buffer_index(start_index));
                start_tree.insert(NNNode<dimension>{start_index, {buffer_index(start_index)}});
                parents[start_index] = start_index;
                radii[start_index] = std::numeric_limits<float>::max();

                for (const auto &goal : goals)
                {
                    goal.to_array(buffer_index(free_index));
                    goal_tree.insert(NNNode<dimension>{free_index, {buffer_index(free_index)}});
                    parents[free_index] = free_index;
                    radii[free_index] = std::numeric_limits<float>::max();
                    free_index++;
                }

                auto *tree_a = (settings.start_tree_first) ? &goal_tree : &start_tree;
                auto *tree_b = (settings.start_tree_first) ? &start_tree : &goal_tree;
                bool tree_a_is_start = not settings.start_tree_first;

                while (iter++ < settings.max_iterations and free_index < settings.max_samples)
                {
                    // std::cout << "Thread " << omp_get_thread_num() << " iter " << iter << std::endl;
                    //@TODO: check if anything global has an answer 
                    // experiment with frequency of checking for global solution (maybe only do every 10 iters? 100 iters? play around)
                    // THE OK APPROACH:
                    // irecv and isendv the global solution to all procs, have all procs return the global solution
                    // will probably need to split this up into a couple of calls to communicate solution size, so we can make sure the buffer
                    // for receiving the answer will be the right size
                    // THE GOOD APPROACH:
                    // only communicate the existence of a solution, not the actual solution itself. 
                    // have all non solution procs return -1 ( or somehow indicate failure, maybe try /catch?)
                    // have only the proc that found the solution return the solution
                    // this approach will need changes at a higher stack level (have to do make sure solve caller knows what to do with the processes returning no solution)
                    // /////////////////////////////////////////////////////////////////
                    // do the ok approach for now, since it's easier to implement
                    // for either approach, stop the timer / print whenever information is succussfully received 

                    //otherwise, continue as normal
                    //if (iter % 10 == 0) {
                        if (found_solution) {
                            break;
                        }
                    // }
                    float asize = tree_a->size();
                    float bsize = tree_b->size();
                    float ratio = std::abs(asize - bsize) / asize;
                    // std::cout << "Thread " << omp_get_thread_num() << " ratio " << ratio << std::endl;
                    if ((not settings.balance) or ratio < settings.tree_ratio)
                    {
                        std::swap(tree_a, tree_b);
                        tree_a_is_start = not tree_a_is_start;
                    }

                    auto temp = rng.next();
                    Robot::scale_configuration(temp);
                    // std::cout << "Thread " << omp_get_thread_num() << " got random configuration" << std::endl;
                    typename Robot::ConfigurationBuffer temp_array;
                    temp.to_array(temp_array.data());

                    const auto nearest = tree_a->nearest(NNFloatArray<dimension>{temp_array.data()});
                    if (not nearest)
                    {
                        continue;
                    }

                    const auto &[nearest_node, nearest_distance] = *nearest;
                    const auto nearest_radius = radii[nearest_node.index];

                    if (settings.dynamic_domain and nearest_radius < nearest_distance)
                    {
                        continue;
                    }

                    const auto nearest_configuration = nearest_node.as_vector();

                    auto nearest_vector = temp - nearest_configuration;

                    bool reach = nearest_distance < settings.range;
                    auto extension_vector =
                        (reach) ? nearest_vector : nearest_vector * (settings.range / nearest_distance);
                    // std::cout << "Thread " << omp_get_thread_num() << " got extension vector" << std::endl;
                    if (validate_vector<Robot, rake, resolution>(
                            nearest_configuration,
                            extension_vector,
                            (reach) ? nearest_distance : settings.range,
                            environment))
                    {
                        float *new_configuration_index = buffer_index(free_index);
                        auto new_configuration = nearest_configuration + extension_vector;
                        new_configuration.to_array(new_configuration_index);
                        tree_a->insert(NNNode<dimension>{free_index, {new_configuration_index}});

                        parents[free_index] = nearest_node.index;
                        radii[free_index] = std::numeric_limits<float>::max();

                        free_index++;

                        if (settings.dynamic_domain and nearest_radius != std::numeric_limits<float>::max())
                        {
                            radii[nearest_node.index] *= (1 + settings.alpha);
                        }

                        // Extend to goal tree
                        const auto other_nearest =
                            tree_b->nearest(NNFloatArray<dimension>{new_configuration_index});
                        if (not other_nearest)
                        {
                            continue;
                        }

                        const auto &[other_nearest_node, other_nearest_distance] = *other_nearest;
                        const auto other_nearest_configuration = other_nearest_node.as_vector();
                        auto other_nearest_vector = other_nearest_configuration - new_configuration;

                        const std::size_t n_extensions = std::ceil(other_nearest_distance / settings.range);
                        const float increment_length = other_nearest_distance / static_cast<float>(n_extensions);
                        auto increment = other_nearest_vector * (1.0F / static_cast<float>(n_extensions));

                        std::size_t i_extension = 0;
                        auto prior = new_configuration;
                        for (; i_extension < n_extensions and
                            validate_vector<Robot, rake, resolution>(
                                prior, increment, increment_length, environment) and
                            free_index < settings.max_samples;
                            ++i_extension)
                        {
                            auto next = prior + increment;
                            float *next_index = buffer_index(free_index);
                            next.to_array(next_index);
                            tree_a->insert(NNNode<dimension>{free_index, {next_index}});
                            parents[free_index] = free_index - 1;
                            radii[free_index] = std::numeric_limits<float>::max();

                            free_index++;

                            prior = next;
                        }

                        if (i_extension == n_extensions)  // connected
                        {
                            bool somebody_found_solution = false;
                            #pragma omp atomic capture
                            {
                                somebody_found_solution = found_solution;
                                found_solution = true;
                            }

                            if (somebody_found_solution) break;
                            std::cout << "Thread " << omp_get_thread_num() << " found solution" << std::endl;
                            
                            
                            result.cost = other_nearest_distance;
                                                        auto current = free_index - 1;
                            result.path.emplace_back(buffer_index(current));
                            while (parents[current] != current)
                            {
                                auto parent = parents[current];
                                result.path.emplace_back(buffer_index(parent));
                                result.cost += result.path[result.path.size() - 1].distance(
                                    result.path[result.path.size() - 2]);
                                current = parent;
                            }

                            std::reverse(result.path.begin(), result.path.end());
                            current = other_nearest_node.index;

                            while (parents[current] != current)
                            {
                                auto parent = parents[current];
                                result.path.emplace_back(buffer_index(parent));
                                result.cost += result.path[result.path.size() - 1].distance(
                                    result.path[result.path.size() - 2]);
                                current = parent;
                            }

                            if (not tree_a_is_start)
                            {
                                std::reverse(result.path.begin(), result.path.end());
                            }

                            result.size.emplace_back(start_tree.size());
                            result.size.emplace_back(goal_tree.size());
                            result.iterations = iter;
                        }
                        break;
                    }
                    else if (settings.dynamic_domain)
                    {
                        if (nearest_radius == std::numeric_limits<float>::max())
                        {
                            radii[nearest_node.index] = settings.radius;
                        }
                        else
                        {
                            radii[nearest_node.index] =
                                std::max(radii[nearest_node.index] * (1.F - settings.alpha), settings.min_radius);
                        }
                    }
                }
            } // end of parallel region
            result.nanoseconds = vamp::utils::get_elapsed_nanoseconds(start_time);
            return result;
        }
    };
}  // namespace vamp::planning
