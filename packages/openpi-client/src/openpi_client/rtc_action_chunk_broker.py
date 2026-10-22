import concurrent.futures
import logging
from typing import Dict

import numpy as np
from typing_extensions import override

from openpi_client import base_policy as _base_policy

logger = logging.getLogger(__name__)


class RTCActionChunkBroker(_base_policy.BasePolicy):
    """Wraps a policy with RTC-aware asynchronous action chunk management.

    Timeline for a single chunk (H=action_horizon, s=exec_steps, d=prefix_len):

        Chunk A: [a0, a1, ..., a_{s-1}, a_s, ..., a_{s+d-1}, ...]
                  |--- execute s ---|--- still executing d ---|
                                    ^                         ^
                            trigger infer()            new chunk B arrives
                            (prefetch point)

    At step s (after executing s actions), the broker fires an async infer request.
    During the next d steps, it continues serving actions from chunk A while waiting
    for chunk B. When chunk B arrives (d steps later), the broker switches to it.

    If d=0, this reduces to synchronous behavior (identical to ActionChunkBroker).
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        action_horizon: int,
        exec_steps: int,
        prefix_len: int = 0,
    ):
        self._policy = policy
        self._action_horizon = action_horizon
        self._exec_steps = exec_steps
        self._prefix_len = prefix_len

        self._cur_step: int = 0
        self._last_results: Dict[str, np.ndarray] | None = None
        self._pending_future: concurrent.futures.Future | None = None
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._last_obs: Dict | None = None

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        self._last_obs = obs

        if self._last_results is None:
            self._last_results = self._policy.infer(obs)
            self._cur_step = 0

        if self._cur_step == self._exec_steps and self._prefix_len > 0 and self._pending_future is None:
            self._pending_future = self._executor.submit(self._policy.infer, obs)

        if self._cur_step >= self._exec_steps + self._prefix_len:
            if self._pending_future is not None:
                self._last_results = self._pending_future.result()
                self._pending_future = None
            else:
                self._last_results = self._policy.infer(obs)
            self._cur_step = 0

        def slicer(x):
            if isinstance(x, np.ndarray):
                return x[self._cur_step, ...]
            else:
                return x

        import tree
        results = tree.map_structure(slicer, self._last_results)
        self._cur_step += 1
        return results

    @override
    def reset(self) -> None:
        if self._pending_future is not None:
            self._pending_future.cancel()
            self._pending_future = None
        self._policy.reset()
        self._last_results = None
        self._cur_step = 0
        self._last_obs = None
