"""
FusionOps Client
OpenEnv-compatible client for the FusionOps scheduling environment.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel


class FusionOpsObservation(BaseModel):
    text: str
    error: Optional[str] = None


class FusionOpsAction(BaseModel):
    command: str


class FusionOpsResult(BaseModel):
    observation: FusionOpsObservation
    reward: float = 0.0
    done: bool = False
    score: Optional[float] = None
    info: dict = {}


class FusionOpsEnv:
    """Client for the FusionOps environment server."""

    def __init__(self, base_url: str = "http://localhost:7860"):
        self.base_url = base_url.rstrip("/")
        self._session_id: Optional[str] = None
        self._last_score: Optional[float] = None

    @classmethod
    async def from_docker_image(cls, image_name: Optional[str] = None):
        if image_name:
            return cls(base_url="http://localhost:7860")
        return cls(base_url="http://localhost:7860")

    async def reset(self, task: str = "task1_linear") -> FusionOpsResult:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/reset",
                json={"task": task},
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                data = await resp.json()
                self._session_id = data.get("session_id")
                return FusionOpsResult(
                    observation=FusionOpsObservation(
                        text=data.get("observation", ""),
                    ),
                    reward=data.get("reward", 0.0),
                    done=data.get("done", False),
                )

    async def step(self, action: FusionOpsAction) -> FusionOpsResult:
        import aiohttp
        if self._session_id is None:
            raise RuntimeError("Must call reset() first")

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/step/{self._session_id}",
                json={"command": action.command},
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                data = await resp.json()
                info = data.get("info", {})
                score = data.get("score")
                if score is not None:
                    self._last_score = score
                return FusionOpsResult(
                    observation=FusionOpsObservation(
                        text=data.get("observation", ""),
                        error=info.get("error"),
                    ),
                    reward=data.get("reward", 0.0),
                    done=data.get("done", False),
                    score=score,
                    info=info,
                )

    async def state(self) -> dict:
        import aiohttp
        if self._session_id is None:
            raise RuntimeError("Must call reset() first")
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.base_url}/state/{self._session_id}",
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                return await resp.json()

    async def close(self):
        self._session_id = None

    def get_score(self) -> float:
        return self._last_score or 0.0
