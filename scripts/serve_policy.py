# Modified 2026-10-21: align entry points with the Mobile ALOHA release configurations.
import dataclasses
import logging
import socket

import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config




@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi05_mobile_aloha_rtc").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi05_mobile_aloha_rtc/exp/10000").
    dir: str


@dataclasses.dataclass
class Default:
    """Require an explicit Mobile ALOHA configuration and checkpoint."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False

    # Supply policy:checkpoint with --policy.config and --policy.dir.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)

    # RTC inference overrides. These override the values from the model config.
    rtc_prefix_len: int | None = None
    rtc_exec_steps: int | None = None

    # Number of denoising steps for flow matching inference (default: 10).
    num_steps: int | None = None






def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    sample_kwargs = {}
    if args.num_steps is not None:
        sample_kwargs["num_steps"] = args.num_steps

    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config), args.policy.dir, default_prompt=args.default_prompt,
                sample_kwargs=sample_kwargs or None,
            )
        case Default():
            raise ValueError("Supply policy:checkpoint --policy.config <config-name> --policy.dir <checkpoint-dir>.")


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = policy.metadata

    if args.rtc_prefix_len is not None:
        policy._rtc_prefix_len = args.rtc_prefix_len
    if args.rtc_exec_steps is not None:
        policy._rtc_exec_steps = args.rtc_exec_steps

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
