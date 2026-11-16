# editagent_script.py


from collections import defaultdict
import random
import os
import openai
import re
import yaml
import pandas as pd
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
import json
import concurrent.futures
import threading
import docker

from r2egym.agenthub.runtime.docker import DockerRuntime, DockerRuntimeMap
from r2egym.agenthub.environment.env import EnvArgs, RepoEnv
from r2egym.agenthub.agent.agent import AgentArgs, MultiLanguageAgentArgs, Agent
from r2egym.agenthub.config import SCAFFOLD_NAMES

from r2egym.docker_bash_utils.docker_list_tags import fetch_docker_tags
from r2egym.agenthub.utils.log import get_logger
from r2egym.logging import setup_logging, INFO
from r2egym.agenthub.utils.utils import get_parsed_commit

from fire import Fire
from r2egym.agenthub.utils.utils import match_dockerimage_to_repo
from r2egym.agenthub import SUPPORTED_REPOS
from datasets import load_dataset
from r2egym.agenthub.trajectory import TrajectoryStep, Trajectory
import time

from queue import Queue
from math import ceil

if os.environ.get("LLM_BASE_URL", ""):
    max_workers = int(os.environ["MAX_WORKERS"])
    api_endpoints = os.environ["LLM_BASE_URL"].split(",")
    api_queue = Queue()
    for endpoint in api_endpoints * ceil(max_workers / len(api_endpoints)):
        print("add {}".format(endpoint))
        api_queue.put(endpoint)
else:
    raise ValueError(
        "Please set LLM_BASE_URL environment variable with comma separated values."
    )

##############################################################################
# Initialize Logger
##############################################################################
logger = get_logger(__name__)  # Initialize the logger

##############################################################################
# Initialize File Lock for Thread-Safe Writing
##############################################################################
file_lock = threading.Lock()


##############################################################################
# Utility Function
##############################################################################
def get_docker_images(repo_name) -> List[str]:
    """
    Fetches the list of Docker images available for the base image.

    Returns:
        A list of Docker image tags.
    """
    base_image = f"namanjain12/{repo_name}new"
    tags = fetch_docker_tags(base_image)
    docker_image_list = [f"{base_image}:{x['name']}" for x in tags]
    return docker_image_list


def prepull_docker_image(docker_image: str) -> bool:
    """
    Prepulls a single Docker image.
    
    Args:
        docker_image: The Docker image name to pull
        
    Returns:
        True if successful, False otherwise
    """
    try:
        client = docker.from_env()
        logger.info(f"Pulling Docker image: {docker_image}")
        client.images.pull(docker_image)
        logger.info(f"Successfully pulled Docker image: {docker_image}")
        return True
    except Exception as e:
        logger.error(f"Failed to pull Docker image {docker_image}: {e}")
        return False


def prepull_docker_images(ds_selected: List[Dict], max_workers: Optional[int] = None) -> None:
    """
    Prepulls all Docker images in parallel before starting the main execution.
    
    Args:
        ds_selected: List of dataset entries containing docker_image keys
        max_workers: Maximum number of threads for parallel pulling
    """
    # Extract unique Docker images
    docker_images = list(set([ds_entry["docker_image"] for ds_entry in ds_selected]))
    logger.info(f"Starting parallel prepull of {len(docker_images)} unique Docker images...")
    
    # Use ThreadPoolExecutor for I/O bound operations like Docker pulls
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all pull tasks
        future_to_image = {
            executor.submit(prepull_docker_image, docker_image): docker_image
            for docker_image in docker_images
        }
        
        # Track results
        successful_pulls = []
        failed_pulls = []
        
        for future in concurrent.futures.as_completed(future_to_image):
            docker_image = future_to_image[future]
            try:
                success = future.result()
                if success:
                    successful_pulls.append(docker_image)
                else:
                    failed_pulls.append(docker_image)
            except Exception as e:
                logger.error(f"Exception during prepull of {docker_image}: {e}")
                failed_pulls.append(docker_image)
    
    logger.info(f"Prepull completed. Success: {len(successful_pulls)}, Failed: {len(failed_pulls)}")
    if failed_pulls:
        logger.warning(f"Failed to pull images: {failed_pulls}")


##############################################################################
# editagent Functions
##############################################################################
def run_agent_with_restarts(
    agent,
    env,
    max_steps=40,
    num_restarts=1,
    temperature=0.0,
    top_p=1.0,
    max_steps_absolute=50,
    use_fn_calling: bool = True,
    max_iterations: int = 100,
    scaffold: str = "r2egym",
    max_tokens: int = 65536,
):
    """
    Iterative eval protocol:
    - normally run the agent
    - run for maximum num_iterations = 3 times
    - stop if trajectory.exit_reason == "agent"
    - otherwise continue iteratively till maximum iterations
    - finally choose the trajectory with the lowest number of steps
    - note restarts and iterative_evals are different (so just use one of them | add an assert flag)
    - also if original is at temp = 0, then we do next with 0.1 and 0.2 and so on (max 0.2)
    """
    steps_per_agent = max_steps // num_restarts
    logger.warning(f"running {steps_per_agent} steps per agent")

    # only one of restarts > 1 and iterative_eval can be True
    iterative_eval = max_iterations > 1
    assert not (num_restarts > 1 and iterative_eval), "only one of restarts > 1 and iterative_eval can be True"
    logger.warning(f"Using iterations: {max_iterations}, using iterative protocol: {iterative_eval}")

    # if original is at temp = 0, then we do next with 0.1 and 0.2 and so on (max 0.2)
    # if temperature is 0, create list of increasing temperatures up to 0.2
    if temperature == 0:
        temperatures = [0.0 + 0.1 * i for i in range(max_iterations)]
        temperatures = [min(t, 0.2) for t in temperatures]  # cap at 0.2
    else:
        temperatures = [temperature] * max_iterations
    logger.warning(f"Using temperatures: {temperatures}")

    # run the agent in iterative protocol
    trajectories = []
    for iteration in range(max_iterations):
        for idx in range(num_restarts):
            logger.warning(f"running agent at idx: {idx+1}")
            if scaffold in ["r2egym-parallel"]:
                trajectory, history = agent.run_with_parallel_function_call(
                    env,
                    max_steps=steps_per_agent,
                    temperature=temperatures[iteration],
                    top_p=top_p,
                    max_steps_absolute=max_steps_absolute,
                    use_fn_calling=use_fn_calling,
                    scaffold=scaffold,
                    max_token_limit=max_tokens,
                )
            else:
                trajectory, history = agent.run(
                    env,
                    max_steps=steps_per_agent,
                    temperature=temperatures[iteration],
                    top_p=top_p,
                    max_steps_absolute=max_steps_absolute,
                    use_fn_calling=use_fn_calling,
                    scaffold=scaffold,
                    max_token_limit=max_tokens,
                )
            # remove reproduce.py
            # env.runtime.run('rm reproduce_issue.py')
        if trajectory.exit_reason == "agent":
            logger.warning(f"agent self-finished at iteration: {iteration}")
            # return trajectory
            return trajectory, history
        # otherwise continue iteratively
        # trajectories.append(trajectory)
        trajectories.append((trajectory, history))
        # reset the env
        # env.reset()

    # choose the trajectory with the lowest number of steps
    # trajectory = min(trajectories, key=lambda x: x.num_steps)
    trajectory, history = min(trajectories, key=lambda x: x[0].num_steps)
    # return trajectory
    return trajectory, history

def runagent(
    ds,
    exp_name: Optional[str] = None,
    max_steps=40,
    num_restarts=1,
    max_steps_absolute=50,
    llm_name="gpt-4o",
    temperature=0,
    top_p=1.0,
    use_fn_calling: bool = True,
    backend: str = "kubernetes", # "kubernetes" or "docker"
    max_reward_calc_time: int = 300,
    max_iterations: int = 100,
    scaffold: str = "r2egym",
    max_tokens: int = 65536,
    task_index: int = 0,
) -> Optional[str]:
    """
    Runs the editagent agent on a specified Docker image.

    Args:
        docker_image: The Docker image to use for the environment.
        traj_dir: Directory to save trajectories.
        jsonl_file: Path to the JSONL file to save results. If not provided, generated using traj_dir and exp_name.
        exp_name: Experiment name. Used if jsonl_file is not provided. If not provided, a unique name is generated.
    """
    global api_endpoints, api_queue
    api_url = api_queue.get(timeout=60 * 60 * 3)
    print(f"queue size -> {api_queue.qsize()}")
    logger = setup_logging(
        name=ds["docker_image"].replace("/", "_"),
        log_file=f"run_logs/{exp_name}/{ds['docker_image'].replace('/', '_')}.log",
        console=True,
        level=INFO,
    )
    logger.info(f"using api-url -> {api_url}")
    logger.info(f"Starting editagent on Docker image: {ds['docker_image']}")
    logger.info(f"Using LLM: {llm_name}")
    logger.info(f"Max Steps: {max_steps}")
    logger.info(f"Backend: {backend}")
    logger.info(f"Scaffold: {scaffold}")
    logger.info(f"Temperature: {temperature}")
    logger.info(f"top_p: {top_p}")
    logger.info(f"Use FN Calling: {use_fn_calling}")
    logger.info(f"Max Iterations: {max_iterations}")
    logger.info(f"Max Tokens: {max_tokens}")

    assert scaffold in SCAFFOLD_NAMES, f"Scaffold is {scaffold}, must be one of {SCAFFOLD_NAMES}"
    # Generate a unique experiment name if not provided
    if exp_name is None:
        exp_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Initialize environment arguments
    env_args = EnvArgs(ds=ds)

    # Initialize the RepoEnv
    logger.info("🚀 Initializing RepoEnv (this will create DockerRuntime and start k8s pod)...")
    try:
        env = RepoEnv(env_args, logger=logger, backend=backend)
        logger.info("✅ RepoEnv initialized successfully")
        logger.info(f"✅ Runtime type: {type(env.runtime).__name__}")
        if hasattr(env.runtime, 'container_name'):
            logger.info(f"✅ Container/Pod name: {env.runtime.container_name}")
    except Exception as e:
        logger.error(f"❌ Failed to initialize RepoEnv: {e}")
        import traceback
        logger.error(f"❌ Exception details: {traceback.format_exc()}")
        raise
    # set agent args
    if use_fn_calling:
        assert scaffold != "sweagent", "SWEagent scaffold does not support fn calling"
        # Check if scaffold is multi_swe_bench (uses MultiLanguageAgentArgs)
        if scaffold == "multi_swe_bench":
            try:
                # Use MultiLanguageAgentArgs for multi-language support
                multi_lang_args = MultiLanguageAgentArgs.from_yaml(
                    Path(f"./src/r2egym/agenthub/config/{scaffold}/edit_fn_calling.yaml")
                )
                multi_lang_args.llm_name = llm_name
                
                # Check if the dataset has a 'language' field for multi-language support
                if 'language' in ds and ds['language']:
                    # Convert to AgentArgs with language-specific prompt
                    agent_args = multi_lang_args.to_agent_args(ds['language'])
                    logger.info(f"Using language-specific prompt for: {ds['language']}")
                else:
                    # Use default instance_prompt if no language field
                    agent_args = multi_lang_args.to_agent_args('python')  # default to python
                    logger.info("No language field found, defaulting to python prompt")
            except (ValueError, KeyError) as e:
                logger.error(f"无法为 '{ds.get('language', 'unknown')}' 使用特定语言的 prompt: {e}")
                raise
        else:
            agent_args = AgentArgs.from_yaml(
                Path(f"./src/r2egym/agenthub/config/{scaffold}/edit_fn_calling.yaml")
            )
    else:
        # Check if scaffold is multi_swe_bench (uses MultiLanguageAgentArgs)
        if scaffold == "multi_swe_bench":
            try:
                # Use MultiLanguageAgentArgs for multi-language support
                multi_lang_args = MultiLanguageAgentArgs.from_yaml(
                    Path(f"./src/r2egym/agenthub/config/{scaffold}/edit_non_fn_calling.yaml")
                )
                multi_lang_args.llm_name = llm_name
                
                # Check if the dataset has a 'language' field for multi-language support
                if 'language' in ds and ds['language']:
                    # Convert to AgentArgs with language-specific prompt
                    agent_args = multi_lang_args.to_agent_args(ds['language'])
                    logger.info(f"Using language-specific prompt for: {ds['language']}")
                else:
                    # Use default instance_prompt if no language field
                    agent_args = multi_lang_args.to_agent_args('python')  # default to python
                    logger.info("No language field found, defaulting to python prompt")
            except (ValueError, KeyError) as e:
                logger.error(f"无法为 '{ds.get('language', 'unknown')}' 使用特定语言的 prompt: {e}")
                raise
        else:
            # Use default AgentArgs if no language field
            agent_args = AgentArgs.from_yaml(
                Path(f"./src/r2egym/agenthub/config/{scaffold}/edit_non_fn_calling.yaml")
            )

    agent_args.llm_name = llm_name
    agent_args.llm_base_url = api_url
    # Initialize the agent
    agent = Agent(name="EditAgent", args=agent_args, logger=logger)

    # run agent editagent
    try:
        ret = run_agent_with_restarts(
            agent,
            env,
            max_steps=max_steps,
            num_restarts=num_restarts,
            temperature=temperature,
            top_p=top_p,
            max_steps_absolute=max_steps_absolute,
            use_fn_calling=use_fn_calling,
            max_iterations=max_iterations,
            scaffold=scaffold,
            max_tokens=max_tokens,
        )
        [trajectory, history] = ret
        api_queue.put(api_url)
    except Exception as e:
        logger.error(
            f"Error during agent run for Docker image {ds['docker_image']}: {e}"
        )
        api_queue.put(api_url)
        return None

    # also get the gt outputs
    logger.info("Starting reward calculation...")
    reward_calc_time = time.time()
    reward, test_output = env.runtime._calculate_reward(get_test_output=True, timeout=max_reward_calc_time)
    reward_calc_time = time.time() - reward_calc_time
    aider_step_events = env.runtime.ds.get("aider_step_events", [])
    # Close the environment and runtime
    env.close()

    # update the trajectory object
    trajectory.reward = reward
    trajectory.test_output = test_output
    trajectory.ds = ds
    trajectory.exp_name = exp_name
    trajectory.reward_calc_time = reward_calc_time # time taken to calculate reward
    
    logger.info(f"************* RESULTS *************")
    logger.info(f"Reward: {reward}")
    logger.info(f"Time taken to calculate reward: {reward_calc_time:.2f}s")
    logger.info(f"Trajectory steps: {trajectory.num_steps}")
    logger.info(f"Exit reason: {trajectory.exit_reason}")
    logger.info(f"**********************************")

    logger.info(f"editagent completed for Docker image: {ds['docker_image']}")
    logger.info(f"Closing environment for Docker image: {ds['docker_image']}")
    history = {"exp_name": exp_name, "docker_image": ds['docker_image'], "instance_id": ds["instance_id"], "reward": reward, "messages": history, "aider_step_events": aider_step_events}
    return trajectory.model_dump_json(), history

def runagent_multiple(
    dataset: str,
    split: str,
    k: int = 1,
    traj_dir: str = "./traj",
    exp_name: Optional[str] = None,
    start_idx=0,
    max_steps=40,
    num_restarts=1,
    max_steps_absolute=50,
    max_workers: Optional[int] = None,
    llm_name="gpt-4o",
    use_existing: bool = True,
    skip_existing: bool = False,
    temperature: float = 0,
    top_p=1.0,
    use_fn_calling: bool = True,
    backend: str = "kubernetes", # "kubernetes" or "docker"
    max_reward_calc_time: int = 300,
    max_iterations: int = 100,
    scaffold: str = "r2egym",
    prepull_images: bool = False,
    max_tokens: int = 65536,
):
    """
    Runs the editagent agent on the first k Docker images.

    Args:
        k: The number of Docker images to process.
        traj_dir: Directory to save trajectories.
        exp_name: Experiment name for the JSONL file. If not provided, a unique name is generated.
        start_idx: The starting index in the Docker images list.
        max_steps: Maximum steps for the agent run.
        max_workers: Maximum number of threads to use.
        prepull_images: Whether to prepull Docker images in parallel before starting execution.
    """
    if dataset.endswith(".jsonl"):
        with open(dataset, "r") as f:
            items = []
            for line in f:
                item = json.loads(line)
                items.append(item)
            ds_selected = items
            ds = pd.DataFrame(items)
    else:
        ds = load_dataset(dataset, split=split)
        # shuffle the dataset
        ds = ds.shuffle(seed=42)
        # get selected idxs
        selected_idx = range(start_idx, start_idx + k)
        ds_selected = [ds[i] for i in selected_idx]

    logger.info(
        f"Dataset: {dataset}, Split: {split}, Num_total: {len(ds)}, Start Index: {start_idx}, k: {k}"
    )
    logger.info(f"Starting editagent on {len(ds_selected)} Docker images.")

    # Generate a unique experiment name if not provided
    if exp_name is None:
        exp_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Ensure traj_dir exists
    traj_dir_path = Path(traj_dir)
    traj_dir_path.mkdir(parents=True, exist_ok=True)

    # Generate a filename for the JSONL file
    jsonl_file = traj_dir_path / f"{exp_name}.jsonl"
    history_json_file = traj_dir_path / f"completions_{exp_name}.jsonl"
    if use_existing:
        if jsonl_file.exists():
            existing_dockers_counter = defaultdict(int)
            with open(jsonl_file) as f:
                for line in f.readlines():
                    try:
                        instance_id = Trajectory.load_from_model_dump_json(line).ds[
                            "instance_id"
                        ]
                        existing_dockers_counter[instance_id] += 1
                    except:
                        logger.info("error in jsonl file")

            filtered_ds_selected = []
            for ds_entry in ds_selected:
                instance_id = ds_entry["instance_id"]
                if existing_dockers_counter[instance_id] == 0:
                    filtered_ds_selected.append(ds_entry)
                else:
                    existing_dockers_counter[instance_id] -= 1

            ds_selected = filtered_ds_selected

    if skip_existing:
        old_jsonl_files_glob = f"{exp_name[:-1]}*"
        for old_jsonl_file in traj_dir_path.glob(old_jsonl_files_glob):
            with open(old_jsonl_file) as f:
                existing_dockers = [
                    loadline["ds"]["instance_id"]
                    for line in f
                    for loadline in [json.loads(line)]
                    if loadline["reward"] == 1
                ]

            ds_selected = [
                ds_entry
                for ds_entry in ds_selected
                if ds_entry["instance_id"] not in existing_dockers
            ]

    logger.info(
        f"Starting editagent on {len(ds_selected)} Docker images after filtering."
    )

    # Prepull all Docker images in parallel before starting main execution
    if ds_selected and prepull_images:
        logger.info("Prepulling Docker images before starting main execution...")
        prepull_docker_images(ds_selected, max_workers=max_workers)
        logger.info("Docker image prepull completed.")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_image = {
            executor.submit(
                runagent,
                ds=ds_entry,
                exp_name=exp_name,
                max_steps=max_steps,
                num_restarts=num_restarts,
                max_steps_absolute=max_steps_absolute,
                llm_name=llm_name,
                temperature=temperature,
                top_p=top_p,
                use_fn_calling=use_fn_calling,
                backend=backend,
                max_reward_calc_time=max_reward_calc_time,
                max_iterations=max_iterations,
                scaffold=scaffold,
                max_tokens=max_tokens,
                task_index=index
            ): ds_entry[
                "docker_image"
            ]  # <-- store the docker_image from ds_entry here
            for index, ds_entry in enumerate(ds_selected) 
        }

        with open(jsonl_file, "a+") as f, open(history_json_file, "a+") as hf:
            for future in concurrent.futures.as_completed(future_to_image):
                docker_image = future_to_image[
                    future
                ]  # <-- retrieve that stored docker_image
                try:
                    result, history = future.result()
                    if result is not None:
                        with file_lock:
                            f.write(result + "\n")
                            hf.write(json.dumps(history, ensure_ascii=False)+"\n")
                except Exception as e:
                    # Use docker_image from above when logging
                    logger.error(f"Exception for Docker image {docker_image}: {e}")

    logger.info(f"editagent completed on {len(ds_selected)} Docker images.")

def run_multi_swe_bench_agent_multiple(
    dataset: str,
    split: str,
    k: int = 1,
    traj_dir: str = "./traj",
    exp_name: Optional[str] = None,
    start_idx=0,
    max_steps=40,
    num_restarts=1,
    max_steps_absolute=50,
    max_workers: Optional[int] = None,
    llm_name="gpt-4o",
    use_existing: bool = True,
    skip_existing: bool = False,
    temperature: float = 1.0,
    top_p: float = 0.95,
    use_fn_calling: bool = True,
    backend: str = "kubernetes", # "kubernetes" or "docker"
    max_reward_calc_time: int = 300,
    max_iterations: int = 100,
    scaffold: str = "multi_swe_bench",
    prepull_images: bool = False,
    max_tokens: int = 65536,
):
    """
    Runs the editagent agent on the first k Docker images.

    Args:
        k: The number of Docker images to process.
        traj_dir: Directory to save trajectories.
        exp_name: Experiment name for the JSONL file. If not provided, a unique name is generated.
        start_idx: The starting index in the Docker images list.
        max_steps: Maximum steps for the agent run.
        max_workers: Maximum number of threads to use.
        prepull_images: Whether to prepull Docker images in parallel before starting execution.
    """

    def convert_to_ccr_image_path(row: dict) -> str:
        if row.get("docker_image", ""):
            return row["docker_image"]

        items = row["instance_id"].replace("__", "_m_").split('-')
        image_name = "-".join(items[:-1]) + ":pr-" + items[-1]
        new_image_id = "mswebench/" + image_name
        return new_image_id

    # Load the dataset
    lang_mapping = {
        "js": "javascript",
        "ts": "typescript",
    }
    if dataset.endswith(".jsonl"):
        with open(dataset, "r") as f:
            items = []
            for line in f:
                item = json.loads(line)
                item["docker_image"] = convert_to_ccr_image_path(item)

                item["swe_type"] = "mswebench"

                # init problem_statement
                if "problem_statement" not in item:
                    problem_statement = []
                    if "resolved_issues" in item:
                        for issue in item["resolved_issues"]:
                            problem_statement.append(
                                f"## Issue Title: {issue['title']}\n\n## Issue Description:\n{issue['body']}"
                            )
                        item["problem_statement"] = "\n".join(problem_statement)
                    elif "title" in item and "body" in item:
                        problem_statement = f"## Issue Title: {issue['title']}\n\n## Issue Description:\n{issue['body']}"
                        item["problem_statement"] = problem_statement
                
                if "lang" in item:
                    item["language"] = item.pop("lang")
                
                if item["language"] in lang_mapping:
                    item["language"] = lang_mapping[item["language"]]
                
                if "instance_id" not in item:
                    item["instance_id"] = item["org"] + "/" + item["repo"] + ":pr-" + str(item["number"])

                items.append(item)
            ds_selected = items
            ds = pd.DataFrame(items)
    else:
        ds = load_dataset(dataset, split=split)
        # shuffle the dataset
        ds = ds.shuffle(seed=42)

        # get selected idxs
        selected_idx = range(start_idx, start_idx + k)
        ds_selected = [ds[i] for i in selected_idx]

    logger.info(f"{len(ds)}, {k}, {start_idx}")

    logger.info(
        f"Dataset: {dataset}, Split: {split}, Num_total: {len(ds)}, Start Index: {start_idx}, k: {k}"
    )
    logger.info(f"Starting editagent on {len(ds_selected)} Docker images.")

    # Generate a unique experiment name if not provided
    if exp_name is None:
        exp_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Ensure traj_dir exists
    traj_dir_path = Path(traj_dir)
    traj_dir_path.mkdir(parents=True, exist_ok=True)

    # Generate a filename for the JSONL file
    jsonl_file = traj_dir_path / f"{exp_name}.jsonl"
    history_json_file = traj_dir_path / f"completions_{exp_name}.jsonl"

    if use_existing:
        if jsonl_file.exists():
            with open(jsonl_file) as f:
                existing_dockers = []
                for line in f.readlines():
                    try:
                        existing_dockers.append(
                            Trajectory.load_from_model_dump_json(line).ds[
                                "instance_id"
                            ]
                        )
                    except:
                        logger.info("error in jsonl file")

            ds_selected = [
                ds_entry
                for ds_entry in ds_selected
                if ds_entry["instance_id"] not in existing_dockers
            ]

    if skip_existing:
        old_jsonl_files_glob = f"{exp_name[:-1]}*"
        for old_jsonl_file in traj_dir_path.glob(old_jsonl_files_glob):
            with open(old_jsonl_file) as f:
                existing_dockers = [
                    loadline["ds"]["instance_id"]
                    for line in f
                    for loadline in [json.loads(line)]
                    if loadline["reward"] == 1
                ]

            ds_selected = [
                ds_entry
                for ds_entry in ds_selected
                if ds_entry["instance_id"] not in existing_dockers
            ]

    logger.info(
        f"Starting editagent on {len(ds_selected)} Docker images after filtering."
    )

    # Prepull all Docker images in parallel before starting main execution
    if ds_selected and prepull_images:
        logger.info("Prepulling Docker images before starting main execution...")
        prepull_docker_images(ds_selected, max_workers=max_workers)
        logger.info("Docker image prepull completed.")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks to the executor using keyword arguments
        future_to_image = {
            executor.submit(
                runagent,
                ds=ds_entry,
                exp_name=exp_name,
                max_steps=max_steps,
                num_restarts=num_restarts,
                max_steps_absolute=max_steps_absolute,
                llm_name=llm_name,
                temperature=temperature,
                top_p=top_p,
                use_fn_calling=use_fn_calling,
                backend=backend,
                max_reward_calc_time=max_reward_calc_time,
                max_iterations=max_iterations,
                scaffold=scaffold,
                max_tokens=max_tokens,
                task_index=index
            ): ds_entry[
                "docker_image"
            ]  # <-- store the docker_image from ds_entry here
            for index, ds_entry in enumerate(ds_selected) 
        }

        with open(jsonl_file, "a+") as f, open(history_json_file, "a+") as hf:
            for future in concurrent.futures.as_completed(future_to_image):
                docker_image = future_to_image[
                    future
                ]  # <-- retrieve that stored docker_image
                try:
                    result, history = future.result()
                    if result is not None:
                        with file_lock:
                            f.write(result + "\n")
                            hf.write(json.dumps(history, ensure_ascii=False)+"\n")
                except Exception as e:
                    # Use docker_image from above when logging
                    logger.error(f"Exception for Docker image {docker_image}: {e}")

    logger.info(f"editagent completed on {len(ds_selected)} Docker images.")


def aider_runagent_multiple(
    ds_list: List[Dict],
    traj_dir: str = "./traj",
    exp_name: Optional[str] = None,
    max_steps: int = 5,
    num_restarts: int = 1,
    max_steps_absolute: int = 5,
    max_workers: Optional[int] = None,
    llm_name: str = "gpt-5",
    temperature: float = 0.6,
    top_p: float = 0.95,
    use_fn_calling: bool = True,
    backend: str = "docker",  # "kubernetes" or "docker"
    max_reward_calc_time: int = 900,
    max_iterations: int = 1,
    scaffold: str = "aider_diff",
    prepull_images: bool = False,
    max_tokens: int = 65536,
):
    """
    针对 Aider 的并发运行：使用线程池并发调用 runagent，并把结果写入JSONL（与 runagent_multiple 保持一致）
    """
    if not ds_list:
        logger.warning("aider_runagent_multiple: ds_list 为空，直接返回")
        return

    # 过滤非法/缺字段项
    required_keys = {"docker_image", "instance_id", "repo_name", "repo_path", "file_name", "initial_file_content", "problem_statement"}
    ds_selected = []
    for ds in ds_list:
        if not required_keys.issubset(ds.keys()):
            logger.error(f"aider_runagent_multiple: ds 缺少必要字段，已跳过：{ds.get('instance_id', '<unknown>')}")
            continue
        ds_selected.append(ds)
    logger.info(f"aider_runagent_multiple: 接收 {len(ds_list)} 道题，过滤后 {len(ds_selected)} 道题。")

    # 生成实验名
    if exp_name is None:
        exp_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 输出目录/文件
    traj_dir_path = Path(traj_dir)
    traj_dir_path.mkdir(parents=True, exist_ok=True)
    jsonl_file = traj_dir_path / f"{exp_name}.jsonl"
    history_json_file = traj_dir_path / f"completions_{exp_name}.jsonl"

    logger.info(
        f"[Aider] exp_name={exp_name}, traj_dir={traj_dir_path}, "
        f"max_workers={max_workers}, scaffold={scaffold}, backend={backend}"
    )

    # 预拉取镜像
    if ds_selected and prepull_images:
        logger.info("aider_runagent_multiple: 开始并行预拉取 Docker 镜像...")
        prepull_docker_images(ds_selected, max_workers=max_workers)
        logger.info("aider_runagent_multiple: 预拉取完成")

    # 并发执行
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_instance = {
            executor.submit(
                runagent,
                ds=ds_entry,
                exp_name=exp_name,
                max_steps=max_steps,
                num_restarts=num_restarts,
                max_steps_absolute=max_steps_absolute,
                llm_name=llm_name,
                temperature=temperature,
                top_p=top_p,
                use_fn_calling=use_fn_calling,
                backend=backend,
                max_reward_calc_time=max_reward_calc_time,
                max_iterations=max_iterations,
                scaffold=scaffold,
                max_tokens=max_tokens,
                task_index=index,  # 分流到不同 api_endpoints
            ): ds_entry["instance_id"]
            for index, ds_entry in enumerate(ds_selected)
        }

        # 结果写入（线程安全：依赖全局 file_lock）
        with open(jsonl_file, "a+", encoding="utf-8") as f_out, open(history_json_file, "a+", encoding="utf-8") as h_out:
            for future in concurrent.futures.as_completed(future_to_instance):
                instance_id = future_to_instance[future]
                try:
                    result = future.result()
                    if result is None:
                        logger.error(f"[Aider] {instance_id}: runagent 返回 None（初始化或执行失败）")
                        continue
                    trajectory_json, history = result
                    if trajectory_json is not None:
                        with file_lock:
                            f_out.write(trajectory_json + "\n")
                            h_out.write(json.dumps(history, ensure_ascii=False) + "\n")
                except Exception as e:
                    logger.error(f"[Aider] {instance_id}: 执行异常：{e}")

    logger.info(f"Aider Runagent Multiple: 完成并发运行，共 {len(ds_selected)} 条。输出：{jsonl_file}, {history_json_file}")


if __name__ == "__main__":
    # Expose functions via Fire
    Fire(
        {
            "runagent": runagent,
            "runagent_multiple": runagent_multiple,
            "run_multi_swe_bench_agent_multiple": run_multi_swe_bench_agent_multiple,
            "aider_runagent_multiple": aider_runagent_multiple,
        }
    )
