import pickle
import time
from tabulate import tabulate
from tqdm import tqdm
from pathlib import Path
import pandas as pd
from typing import Union, List
import threading
# from fire import Fire
import vamp
from vamp import pointcloud as vpc
import sys

#set parallel iff PythonVersion > 3.12
if sys.version_info >= (3, 12):
    parallel = True
else:
    parallel = False

def parallel_solve(vamp_module, planner_func, pset, flags, envs, plan_settings, simp_settings, results, thread_id):
    #results must be pre-initialized to failures

    sampler = vamp_module.halton_offset(thread_id)
    #copy plan_settings
    local_plan_settings = plan_settings.copy()
    local_plan_settings.thread_id = thread_id
    print("Thread ", thread_id, " initialized")
    # print(local_plan_settings)
    for i, data in tqdm(enumerate(pset)):
        # if (thread_id == 0):
            # print("Starting problem ", i)
        #experiments 
        # don't call planner_func if flags[i] is set
        # see performance of resetting sampler

        # sampler.reset() # needs to be private per thread / make sure seeded indivdually 
        # if flags[i].get() >= 0:
        #     # if thread_id == 0:
        #     #     print("Flag is" , flags[i].get())
        #         # print("Skipping problem ", i)
        #     continue

        result = planner_func(data['start'], data['goals'], envs[i], local_plan_settings, sampler, flags[i])
        if(not result.solved):
            continue
        # print("Thread ", thread_id, "solved problem ", i, "with flag ", flags[i].get())
        
        if (flags[i].get() == thread_id):
            simple = vamp_module.simplify(result.path, envs[i], simp_settings, sampler)
            trial_result = vamp.results_to_dict(result, simple)
            results[i] = trial_result

    flag_results = [flag.get() for flag in flags]
    print("Thread ", thread_id, "finished with flag results: ", flag_results)

def par_main(
    robot: str = "panda",                  # Robot to plan for
    planner: str = "rrtc",                 # Planner name to use
    dataset: str = "problems.pkl",         # Pickled dataset to use
    problem: Union[str, List[str]] = [],   # Problem name or list of problems to evaluate
    trials: int = 1,                       # Number of trials to evaluate each instance
    sampler: str = "halton", 
                                # Sampler to use.
    #TODO: here we want to set something like "custom", so as to wait until parallelization to worry 
    # before creating different samplers
    skip_rng_iterations: int = 0,          # Skip a number of RNG iterations
    print_failures: bool = False,          # Print out failures and invalid problems
    pointcloud: bool = False,              # Use pointcloud rather than primitive geometry
    samples_per_object: int = 10000,       # If pointcloud, samples per object to use
    filter_radius: float = 0.02,           # Filter radius for pointcloud filtering
    filter_cull: bool = True,              # Cull pointcloud around robot by maximum distance
    **kwargs,
    ):

    if robot not in vamp.ROBOT_JOINTS:
        raise RuntimeError(f"Robot {robot} does not exist in VAMP!")

    problems_dir = Path(__file__).parent.parent / 'resources' / robot / 'problems'
    with open(problems_dir.parent / dataset, 'rb') as f:
        problems = pickle.load(f)

    problem_names = list(problems['problems'].keys())
    if isinstance(problem, str):
        problem = [problem]

    if not problem:
        problem = problem_names
    else:
        for problem_name in problem:
            if problem_name not in problem_names:
                raise RuntimeError(
                    f"Problem `{problem_name}` not available! Available problems: {problem_names}"
                    )

    (vamp_module, planner_func, plan_settings,
     simp_settings) = vamp.configure_robot_and_planner_with_kwargs(robot, planner, **kwargs)


    total_problems = 0
    valid_problems = 0
    failed_problems = 0
    slow_problems = 0
    
    #initialize results to all have dummy data, preventing threads from having to write
    # anything on a failure
    dummy_data = {
        "planning_time": pd.Timedelta(0),
        "planning_iterations": 0,
        "solved": False,
        "planning_graph_size": 0,
        "initial_path_vertices": 0,
        "initial_path_cost": 0,
        "simplification_time": pd.Timedelta(0),
        "simplified_path_vertices": 0,
        "simplified_path_cost": 0,
        }
    tick = time.perf_counter()
    
    results = []
    num_threads = 4
    per_thread_succeses = []
    for i in range(num_threads):
        per_thread_succeses.append(0)

    for name, pset in problems['problems'].items():
        if name not in problem:
            continue
        solves = []
        failures = []
        invalids = []
        curr_results = [dummy_data for _ in pset]
        flags = [vamp.FinishFlag() for _ in pset]
        envs = [vamp.problem_dict_to_vamp(data) for data in problems['problems'][name]]
        total_problems += len(pset)
        #preprocess data
        for i, data in tqdm(enumerate(pset)):
            if not (data['valid']):
                invalids.append(i)
        valid_problems += len(pset) - len(invalids)
        

        threads = []
        print(f"Evaluating {robot} on {name}: ")
        thread_start_time = time.perf_counter()
        for num in range(num_threads):
            thread = threading.Thread(target=parallel_solve, args=(vamp_module, planner_func, pset, flags, envs, plan_settings, simp_settings, curr_results, num))
            print("starting thread")
            thread.start()
            threads.append(thread)
        thread_end_time = time.perf_counter()
        print(f"Starting threads took {thread_end_time - thread_start_time:.3f} seconds")


        thread_start_time = time.perf_counter()
        for thread in threads:
            print("joining thread")
            thread.join()
        thread_end_time = time.perf_counter()
        print(f"Joining threads took {thread_end_time - thread_start_time:.3f} seconds")

        #do leg work here to calculate failures

        results.extend(curr_results)
    tock = time.perf_counter()


    #clean up results, calculate failures
    print("There are ", len(results), " results")
    i = 0
    while i < len(results):
        if not results[i]["solved"]:
            failed_problems += 1
            failures.append(i)
            results.pop(i)
        else:
            i += 1

    print("There are ", len(failures), " failures")
    print("There are ", len(results), " results")
    df = pd.DataFrame.from_dict(results)

    # Convert to microseconds
    df["planning_time"] = df["planning_time"].dt.microseconds
    df["simplification_time"] = df["simplification_time"].dt.microseconds
    df["avg_time_per_iteration"] = df["planning_iterations"] / df["planning_time"]

    df["total_time"] = df["total_time"].dt.microseconds
    df["overhead_time"] = (tock - tick) * 1000
    df["total_problems"] = total_problems
    df["valid_problems"] = valid_problems
    df["failed_problems"] = failed_problems


    # print(
    #     f"Solved / Valid / Total # Problems: {valid_problems - failed_problems} / {valid_problems} / {total_problems}"
    #     )
    # print(f"Completed all problems in {df['total_time'].sum() / 1000:.3f} milliseconds")
    # print(f"Total time including Python overhead: {(tock - tick) * 1000:.3f} milliseconds")
    return df

def main(
    parallel: bool = False,                 # Run in parallel
    robot: str = "panda",                  # Robot to plan for
    planner: str = "rrtc",                 # Planner name to use
    dataset: str = "problems.pkl",         # Pickled dataset to use
    problem: Union[str, List[str]] = [],   # Problem name or list of problems to evaluate
    trials: int = 1,                       # Number of trials to evaluate each instance
    sampler: str = "halton", 
                                # Sampler to use.
    #TODO: here we want to set something like "custom", so as to wait until parallelization to worry 
    # before creating different samplers
    skip_rng_iterations: int = 0,          # Skip a number of RNG iterations
    print_failures: bool = False,          # Print out failures and invalid problems
    pointcloud: bool = False,              # Use pointcloud rather than primitive geometry
    samples_per_object: int = 10000,       # If pointcloud, samples per object to use
    filter_radius: float = 0.02,           # Filter radius for pointcloud filtering
    filter_cull: bool = True,              # Cull pointcloud around robot by maximum distance
    **kwargs,
    ):

    if robot not in vamp.ROBOT_JOINTS:
        raise RuntimeError(f"Robot {robot} does not exist in VAMP!")

    problems_dir = Path(__file__).parent.parent / 'resources' / robot / 'problems'
    with open(problems_dir.parent / dataset, 'rb') as f:
        problems = pickle.load(f)

    problem_names = list(problems['problems'].keys())
    if isinstance(problem, str):
        problem = [problem]

    if not problem:
        problem = problem_names
    else:
        for problem_name in problem:
            if problem_name not in problem_names:
                raise RuntimeError(
                    f"Problem `{problem_name}` not available! Available problems: {problem_names}"
                    )

    (vamp_module, planner_func, plan_settings,
     simp_settings) = vamp.configure_robot_and_planner_with_kwargs(robot, planner, **kwargs)

    sampler = getattr(vamp_module, sampler)()

    total_problems = 0
    valid_problems = 0
    failed_problems = 0
    slow_problems = 0

    tick = time.perf_counter()
    results = []
    flag = False
    if (parallel):
        flag = vamp.FinishFlag()

    
    for name, pset in problems['problems'].items():
        if name not in problem:
            continue
        solves = []
        failures = []
        invalids = []
        print(f"Evaluating {robot} on {name}: ")
        # for i, data in tqdm(enumerate(pset)):
        for i, data in enumerate(pset):
            total_problems += 1

            if not data['valid']:
                invalids.append(i)
                continue

            valid_problems += 1

            if pointcloud:
                (env, original_pc, filtered_pc, filter_time, build_time) = vpc.problem_dict_to_pointcloud(
                    robot,
                    data,
                    samples_per_object,
                    filter_radius,
                    filter_cull,
                    )

                pointcloud_results = {
                    'original_pointcloud_size': len(original_pc),
                    'filtered_pointcloud_size': len(filtered_pc),
                    'filter_time': pd.Timedelta(nanoseconds = filter_time),
                    'capt_build_time': pd.Timedelta(nanoseconds = build_time)
                    }
            else:
                env = vamp.problem_dict_to_vamp(data)

            sampler.reset()
            sampler.skip(skip_rng_iterations)
            for _ in range(trials):
                if (parallel):
                    # launch multiple threads to solve this problem

                    result = planner_func(data['start'], data['goals'], env, plan_settings, sampler, flag)
                    flag.reset()
                else:
                    result = planner_func(data['start'], data['goals'], env, plan_settings, sampler)
                if not result.solved:
                    failures.append(i)
                    break

                simple = vamp_module.simplify(result.path, env, simp_settings, sampler)

                trial_result = vamp.results_to_dict(result, simple)
                if pointcloud:
                    trial_result.update(pointcloud_results)

                results.append(trial_result)

        failed_problems += len(failures)
        
        if True:
            if invalids:
                print(f"  Invalid problems: {invalids}")

            if failures:
                print(f"  Failed on {failures}")

            if solves:
                print(f" Solved {solves}")
    tock = time.perf_counter()

    df = pd.DataFrame.from_dict(results)

    # Convert to microseconds
    df["planning_time"] = df["planning_time"].dt.microseconds
    df["simplification_time"] = df["simplification_time"].dt.microseconds
    df["avg_time_per_iteration"] = df["planning_iterations"] / df["planning_time"]

    # # Pointcloud data
    if pointcloud:
        df["total_build_and_plan_time"] = df["total_time"] + df["filter_time"] + df["capt_build_time"]
        df["filter_time"] = df["filter_time"].dt.microseconds / 1e3
        df["capt_build_time"] = df["capt_build_time"].dt.microseconds / 1e3
        df["total_build_and_plan_time"] = df["total_build_and_plan_time"].dt.microseconds / 1e3

    df["total_time"] = df["total_time"].dt.microseconds
    df["overhead_time"] = (tock - tick) * 1000
    df["total_problems"] = total_problems
    df["valid_problems"] = valid_problems
    df["failed_problems"] = failed_problems


    if pointcloud:
        print(
            tabulate(
                pointcloud_stats,
                headers = [
                    '  Filter Time (ms)',
                    '    CAPT Build Time (ms)',
                    'Total Time (ms)',
                    ],
                tablefmt = 'github'
                )
            )

    # print(
    #     f"Solved / Valid / Total # Problems: {valid_problems - failed_problems} / {valid_problems} / {total_problems}"
    #     )
    # print(f"Completed all problems in {df['total_time'].sum() / 1000:.3f} milliseconds")
    # print(f"Total time including Python overhead: {(tock - tick) * 1000:.3f} milliseconds")
    return df

bot_dataframes = {bot: [] for bot in ["baxter"]}
# bot_dataframes = {bot: [] for bot in ["panda", "ur5", "baxter", "fetch"]}
if parallel:
    planner = "por_rrtc"
    # for i in range(3):
        # for bot in ["panda", "ur5", "baxter", "fetch"]:
    for bot in ["baxter"]:
        bot_dataframes[bot].append(par_main(planner = planner, robot = bot))
else:
    planner = "rrtc"
# bot_dataframes = {bot: [] for bot in ["baxter"]}
    # for i in range(3):
        # for bot in ["panda", "ur5", "baxter", "fetch"]:
    for bot in ["baxter"]:
        bot_dataframes[bot].append(main(parallel = parallel, planner = planner, robot = bot))

average_dataframes = {}
correctness_stats = {bot: {} for bot in ["panda", "ur5", "baxter", "fetch"]}
# correctness_stats = {bot: {} for bot in ["baxter"]}

print("done running, now calculating results")
for bot, dfs in bot_dataframes.items():
    combined_df = pd.concat(dfs)
    average_df = combined_df.groupby(combined_df.index).mean()
    average_dataframes[bot] = average_df
    stat_names = ["total_problems", "valid_problems", "failed_problems"]
    correctness_stats[bot] = { name: 0 for name in stat_names }
    for df in dfs:
        for name in stat_names:
            correctness_stats[bot][name] += df[name][0]

print("done calculating results, now printing")
# print("average_dataframes is ", average_dataframes)
for bot, df in average_dataframes.items():
    time_stats = df[[
        "planning_time",
        "simplification_time",
        "total_time",
        "planning_iterations",
        "avg_time_per_iteration",
        ]].describe(percentiles = [0.25, 0.5, 0.75, 0.95])
    time_stats.drop(index = ["count"], inplace = True)

    cost_stats = df[[
        "initial_path_cost",
        "simplified_path_cost",
        ]].describe(percentiles = [0.25, 0.5, 0.75, 0.95])
    cost_stats.drop(index = ["count"], inplace = True)

    print()
    print(
        tabulate(
            time_stats,
            headers = [
                'Planning Time (μs)',
                'Simplification Time (μs)',
                'Total Time (μs)',
                'Planning Iters.',
                'Time per Iter. (μs)',
                ],
            tablefmt = 'github'
            )
        )

    print(
        tabulate(
            cost_stats, headers = [
                ' Initial Cost (L2)',
                '    Simplified Cost (L2)',
                ], tablefmt = 'github'
            )
        )
    print()
    print(f"{bot} Average Solved / Valid / Total # Problems: {df['valid_problems'][0] - df['failed_problems'][0]} / {df['valid_problems'][0]} / {df['total_problems'][0]}")
    print(f"{bot} Total   Solved / Valid / Total # Problems: {correctness_stats[bot]['valid_problems'] - correctness_stats[bot]['failed_problems']} / {correctness_stats[bot]['valid_problems']} / {correctness_stats[bot]['total_problems']}")
    print(f"{bot} Completed all problems in {df['total_time'].sum() / 1000:.3f} milliseconds")
    print(f"{bot} Total time including Python overhead: {df['overhead_time'][0]:.3f} milliseconds")
