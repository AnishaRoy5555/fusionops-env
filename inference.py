"""
Inference Script - FusionOps Environment
=========================================
MANDATORY
- Before submitting, ensure the following variables are defined:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    LOCAL_IMAGE_NAME  (optional) Docker image name for from_docker_image()

STDOUT FORMAT
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import asyncio
import os
import sys
import subprocess
import textwrap
from typing import List, Optional

# Ensure required packages are installed (defensive for validators that skip pyproject)
def _ensure_package(module_name: str, pip_name: str):
    try:
        __import__(module_name)
        return
    except ImportError:
        pass
    # Try standard install
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", pip_name],
            stderr=subprocess.DEVNULL,
        )
        return
    except Exception:
        pass
    # Try with --break-system-packages for externally-managed envs
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", "--break-system-packages", pip_name],
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass

_ensure_package("openai", "openai>=1.0.0")
_ensure_package("aiohttp", "aiohttp>=3.9.0")
_ensure_package("pydantic", "pydantic>=2.0.0")

from openai import OpenAI
from fusionops_env import FusionOpsAction, FusionOpsEnv

IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

BENCHMARK = "fusionops"
TASKS = ["task1_linear", "task2_diamond", "task3_matmul", "task4_multistage"]
MAX_STEPS_PER_TASK = {
    "task1_linear": 10,
    "task2_diamond": 12,
    "task3_matmul": 15,
    "task4_multistage": 20,
}
TEMPERATURE = 0.3
MAX_TOKENS = 300
SUCCESS_SCORE_THRESHOLD = 0.05

SYSTEM_PROMPT = textwrap.dedent("""
You are an agent interacting with an ML graph scheduling environment.

Each observation contains:
- LAST ACTION RESULT (only if your previous action was invalid - read this carefully and fix the issue)
- CURRENT STATE (completed ops, ready ops)
- VALID ACTION EXAMPLES (use these as templates)
- CONSTRAINTS and BEST PRACTICES

Your job:
- Pick ops only from the READY OPS list
- Follow the VALID ACTION EXAMPLES format exactly
- When the previous action failed, READ THE FIX HINT and apply it
- Goal: minimize total latency by fusing connected ops and reducing memory transfers

RESPOND WITH EXACTLY ONE LINE IN THIS FORMAT (no explanations, no markdown):
SCHEDULE ops=[op_ids] config=[w,h,k] retain=[tensor_ids]
""").strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    action_clean = action.replace("\n", " ").replace("\r", " ").strip()
    print(
        f"[STEP] step={step} action={action_clean} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def get_model_action(client: OpenAI, observation: str, history: List[str]) -> str:
    history_block = "\n".join(history[-4:]) if history else ""
    user_prompt = observation
    if history_block:
        user_prompt += f"\n\nPrevious actions:\n{history_block}"
    user_prompt += "\n\nChoose your next action:"

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        for line in text.split("\n"):
            line = line.strip()
            if "SCHEDULE" in line.upper() or "ops=" in line.lower():
                return line
        return text if text else "SCHEDULE ops=[0] config=[128,128,1] retain=[]"
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return "SCHEDULE ops=[0] config=[128,128,1] retain=[]"


async def run_task(client: OpenAI, env: FusionOpsEnv, task_name: str) -> None:
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    max_steps = MAX_STEPS_PER_TASK.get(task_name, 15)

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(task=task_name)
        observation = result.observation.text

        for step in range(1, max_steps + 1):
            if result.done:
                break

            action_text = get_model_action(client, observation, history)

            result = await env.step(FusionOpsAction(command=action_text))

            reward = result.reward
            done = result.done
            error = result.observation.error

            rewards.append(reward)
            steps_taken = step
            observation = result.observation.text

            log_step(step=step, action=action_text, reward=reward, done=done, error=error)
            history.append(f"Step {step}: {action_text} -> reward={reward:.2f}")

            if done:
                if result.score is not None:
                    score = result.score
                break

        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Task {task_name} error: {e}", flush=True)

    finally:
        try:
            await env.close()
        except Exception:
            pass
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


async def main() -> None:
    try:
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    except Exception as e:
        print(f"[DEBUG] Failed to create OpenAI client: {e}", flush=True)
        return

    for task_name in TASKS:
        try:
            # Fresh env per task to avoid stale session issues
            env = await FusionOpsEnv.from_docker_image(IMAGE_NAME)
            await run_task(client, env, task_name)
        except Exception as e:
            print(f"[DEBUG] Task {task_name} top-level error: {e}", flush=True)
            log_end(success=False, steps=0, score=0.0, rewards=[])
            continue


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"[DEBUG] Top-level error: {e}", flush=True)
        sys.exit(0)
