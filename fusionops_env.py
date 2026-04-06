"""
FusionOps Client
OpenEnv-compatible client for the FusionOps scheduling environment.
Install: pip install git+https://huggingface.co/spaces/<username>/fusionops
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class FusionOpsObservation:
    text: str
    error: Optional[str] = None


@dataclass
class FusionOpsAction:
    command: str


@dataclass
class FusionOpsStepResult:
    observation: FusionOpsObservation
    reward: float
    done: bool
    score: Optional[float] = None


class FusionOpsEnv:
    """Client for the FusionOps environment."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self._session_id: Optional[str] = None

    @classmethod
    async def from_docker_image(cls, image_name: Optional[str] = None):
        """Create env from a docker image (OpenEnv pattern)."""
        # For local testing, just connect to localhost
        url = "http://localhost:8000"
        if image_name:
            # In production, this would pull and start the container
            pass
        return cls(base_url=url)

    async def reset(self, task: str = "task1_linear") -> FusionOpsStepResult:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/reset",
                json={"task": task},
            ) as resp:
                data = await resp.json()
                self._session_id = data["session_id"]
                return FusionOpsStepResult(
                    observation=FusionOpsObservation(text=data["observation"]),
                    reward=data.get("reward", 0.0),
                    done=data.get("done", False),
                )

    async def step(self, action: FusionOpsAction) -> FusionOpsStepResult:
        import aiohttp
        if self._session_id is None:
            raise RuntimeError("Must call reset() first")

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/step/{self._session_id}",
                json={"command": action.command},
            ) as resp:
                data = await resp.json()
                return FusionOpsStepResult(
                    observation=FusionOpsObservation(
                        text=data["observation"],
                        error=data.get("info", {}).get("error"),
                    ),
                    reward=data.get("reward", 0.0),
                    done=data.get("done", False),
                    score=data.get("score"),
                )

    async def close(self):
        self._session_id = None

    def sync(self):
        """Return a synchronous context manager."""
        return SyncFusionOpsEnv(self)


class SyncFusionOpsEnv:
    def __init__(self, env: FusionOpsEnv):
        self._env = env

    def __enter__(self):
        return self

    def __exit__(self, *args):
        import asyncio
        asyncio.get_event_loop().run_until_complete(self._env.close())

    def reset(self, task: str = "task1_linear"):
        import asyncio
        return asyncio.get_event_loop().run_until_complete(self._env.reset(task))

    def step(self, action: FusionOpsAction):
        import asyncio
        return asyncio.get_event_loop().run_until_complete(self._env.step(action))
