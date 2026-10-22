# Modified 2026-10-21: remove added adapters for robots outside the ALOHA and Mobile ALOHA families.
# Modified 2026-10-21: disable AlohaOutputs joint conversion by default for 16-dimensional actions.
import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms


def make_aloha_example() -> dict:
    """Creates a random input example for the Aloha policy."""
    return {
        "state": np.ones((14,)),
        "images": {
            "cam_high": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_low": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }


@dataclasses.dataclass(frozen=True)
class AlohaInputs(transforms.DataTransformFn):
    """Inputs for the Aloha policy.

    Expected inputs:
    - images: dict[name, img] where img is [channel, height, width]. name must be in EXPECTED_CAMERAS.
    - state: [14]
    - actions: [action_horizon, 14]
    """

    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model.
    adapt_to_pi: bool = True

    # The expected cameras names. All input cameras must be in this set. Missing cameras will be
    # replaced with black images and the corresponding `image_mask` will be set to False.
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist")

    def __call__(self, data: dict) -> dict:
        data = _decode_aloha(data, adapt_to_pi=self.adapt_to_pi)

        in_images = data["images"]
        if set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")

        # Assume that base image always exists.
        base_image = in_images["cam_high"]

        images = {
            "base_0_rgb": base_image,
        }
        image_masks = {
            "base_0_rgb": np.True_,
        }

        # Add the extra images.
        extra_image_names = {
            "left_wrist_0_rgb": "cam_left_wrist",
            "right_wrist_0_rgb": "cam_right_wrist",
        }
        for dest, source in extra_image_names.items():
            if source in in_images:
                images[dest] = in_images[source]
                image_masks[dest] = np.True_
            else:
                images[dest] = np.zeros_like(base_image)
                image_masks[dest] = np.False_

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": data["state"].reshape(-1), # For yiming's dataset
        }

        # Actions are only available during training.
        if "actions" in data:
            actions = np.asarray(data["actions"])
            actions = _encode_actions_inv(actions, adapt_to_pi=self.adapt_to_pi)
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs



@dataclasses.dataclass(frozen=True)
class MobileAlohaInputs(transforms.DataTransformFn):
    """Inputs for the Aloha policy.

    Expected inputs:
    - images: dict[name, img] where img is [channel, height, width]. name must be in EXPECTED_CAMERAS.
    - state: [14]
    - actions: [action_horizon, 14]
    """

    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model.
    adapt_to_pi: bool = True
    history_length: int = 0

    # The expected cameras names. All input cameras must be in this set. Missing cameras will be
    # replaced with black images and the corresponding `image_mask` will be set to False.
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("cam_high", "cam_top", "cam_left_wrist", "cam_right_wrist")
    
    @staticmethod
    def _decode_aloha(data: dict, *, adapt_to_pi: bool = False) -> dict:
        # state is [left_arm_joint_angles, left_arm_gripper, right_arm_joint_angles, right_arm_gripper]
        # dim sizes: [6, 1, 6, 1]

        def convert_image(img):
            img = np.asarray(img)
            # Convert to uint8 if using float images.
            if np.issubdtype(img.dtype, np.floating):
                img = (255 * img).astype(np.uint8)
            # Convert from [channel, height, width] to [height, width, channel].
            return einops.rearrange(img, "c h w -> h w c")

        images = data["images"]
        images_dict = {name: convert_image(img) for name, img in images.items()}

        data["images"] = images_dict
        return data
    
    def __call__(self, data: dict) -> dict:
        data = self._decode_aloha(data, adapt_to_pi=self.adapt_to_pi)

        in_images = data["images"]
        if set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")

        # Assume that base image always exists.
        base_image = in_images["cam_high"]

        images = {
            "base_0_rgb": base_image,
        }
        image_masks = {
            "base_0_rgb": np.True_,
        }

        # Add the extra images.
        extra_image_names = {
            "left_wrist_0_rgb": "cam_left_wrist",
            "right_wrist_0_rgb": "cam_right_wrist",
        }
        if "cam_top" in in_images:
            extra_image_names["top_0_rgb"] = "cam_top"

        for dest, source in extra_image_names.items():
            if source in in_images:
                images[dest] = in_images[source]
                image_masks[dest] = np.True_
            else:
                images[dest] = np.zeros_like(base_image)
                image_masks[dest] = np.False_

        inputs = {
            "image": images,
            "image_mask": image_masks,
        }

        # Actions are only available during training.

        action_list = []

        if "action_left" in data:
            action_list.append(data["action_left"][self.history_length:])
        if "action_right" in data:
            action_list.append(data["action_right"][self.history_length:])
        if "action_base" in data:
            action_list.append(data["action_base"][self.history_length:,[0, -1]]) # 2D velocity.

        if len(action_list) > 1:
            inputs["actions"] = np.concatenate(action_list, axis=-1)

        states_left = data["observation_states_left"]
        data["observation_states_left"] = states_left

        states_right = data["observation_states_right"]
        data["observation_states_right"] = states_right

        if self.history_length>1:
            inputs['state'] = np.concatenate([data["observation_states_left"][[0, 1, 2, 3, 4, 5, 6]], # joint 1-6
                                              data["observation_states_right"][[0, 1, 2, 3, 4, 5, 6]], # joint 1-6
                                              data["action_base"][0 : self.history_length : 6,[0, -1]].reshape(-1)] # -history_length step
                                            )
        else:
            inputs['state'] = np.concatenate([data["observation_states_left"][[0, 1, 2, 3, 4, 5, 6]],
                                              data["observation_states_right"][[0, 1, 2, 3, 4, 5, 6]],
                                              data["action_base"][0, [0, -1]].reshape(-1)]
                                            )


        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs

@dataclasses.dataclass(frozen=True)
class LingbotMobileAlohaInputs(MobileAlohaInputs):
    include_gripper: bool = False

    def __call__(self, data: dict) -> dict:
        data = self._decode_aloha(data, adapt_to_pi=self.adapt_to_pi)

        in_images = data["images"]
        if set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")

        base_image = in_images["cam_high"]
        images = {
            "base_0_rgb": base_image,
        }
        image_masks = {
            "base_0_rgb": np.True_,
        }

        extra_image_names = {
            "left_wrist_0_rgb": "cam_left_wrist",
            "right_wrist_0_rgb": "cam_right_wrist",
        }
        for dest, source in extra_image_names.items():
            if source in in_images:
                images[dest] = in_images[source]
                image_masks[dest] = np.True_
            else:
                images[dest] = np.zeros_like(base_image)
                image_masks[dest] = np.False_

        action_list = [
            np.asarray(data["action_left"], dtype=np.float32)[self.history_length:],
        ]
        if self.include_gripper:
            action_list.append(np.asarray(data["action_left_gripper"], dtype=np.float32)[self.history_length:])
        action_list.append(np.asarray(data["action_right"], dtype=np.float32)[self.history_length:])
        if self.include_gripper:
            action_list.append(np.asarray(data["action_right_gripper"], dtype=np.float32)[self.history_length:])
        action_list.append(np.asarray(data["action_base"], dtype=np.float32)[self.history_length:, [0, -1]])
        actions = np.concatenate(action_list, axis=-1)

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": actions[0].copy(),
            "actions": actions,
        }

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class LingbotMobileAlohaOutputs(transforms.DataTransformFn):
    include_gripper: bool = False

    def __call__(self, data: dict) -> dict:
        action_dim = 16 if self.include_gripper else 14
        actions = np.asarray(data["actions"][:, :action_dim])
        return {"actions": actions}

def prevent_quaternion_flip(actions: np.ndarray) -> np.ndarray:
    actions = np.asarray(actions).copy()
    quaternions = actions[:, 3:7]
    for i in range(1, len(quaternions)):
        if np.dot(quaternions[i - 1], quaternions[i]) < 0:
            quaternions[i] *= -1
    actions[:, 3:7] = quaternions
    return actions
class MobileAlohaEEFInputs(MobileAlohaInputs):
    """Inputs for the Aloha policy.

    Expected inputs:
    - images: dict[name, img] where img is [channel, height, width]. name must be in EXPECTED_CAMERAS.
    - state: [16]
    - actions: [action_horizon, 16]
    """

    def __call__(self, data: dict) -> dict:
        data = self._decode_aloha(data, adapt_to_pi=self.adapt_to_pi)

        in_images = data["images"]
        if set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")

        # Assume that base image always exists.
        base_image = in_images["cam_high"]

        images = {
            "base_0_rgb": base_image,
        }
        image_masks = {
            "base_0_rgb": np.True_,
        }

        # Add the extra images.
        extra_image_names = {
            "left_wrist_0_rgb": "cam_left_wrist",
            "right_wrist_0_rgb": "cam_right_wrist",
            # "top_0_rgb": "cam_top",
        }
        for dest, source in extra_image_names.items():
            if source in in_images:
                images[dest] = in_images[source]
                image_masks[dest] = np.True_
            else:
                images[dest] = np.zeros_like(base_image)
                image_masks[dest] = np.False_

        inputs = {
            "image": images,
            "image_mask": image_masks,
        }

        # Actions are only available during training.

        action_list = []

        if "action_left" in data:
            action_list.append(prevent_quaternion_flip(data["action_left"][self.history_length:])) # [Horizon, [x,y,z,rx,ry,rz,rw, gripper]]
        if "action_right" in data:
            action_list.append(prevent_quaternion_flip(data["action_right"][self.history_length:]))
        if "action_base" in data:
            action_list.append(data["action_base"][self.history_length:,[0, -1]])

        if len(action_list) > 1:
            inputs["actions"] = np.concatenate(action_list, axis=-1)

        if self.history_length>1:
            inputs['state'] = np.concatenate([data["observation_states_left"],
                                              data["observation_states_right"],
                                              data["action_base"][0 : self.history_length : 6,[0, -1]].reshape(-1)] # -history_length step
                                            )
        else:
            inputs['state'] = np.concatenate([data["observation_states_left"],
                                              data["observation_states_right"],
                                              data["action_base"][0, [0, -1]].reshape(-1)]
                                            )


        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs







@dataclasses.dataclass(frozen=True)
class AlohaOutputs(transforms.DataTransformFn):
    """Outputs for the Aloha policy."""

    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model.
    adapt_to_pi: bool = False

    def __call__(self, data: dict) -> dict:
        # Only return the first 14 dims.
        actions = np.asarray(data["actions"][:, :16])
        return {"actions": _encode_actions(actions, adapt_to_pi=self.adapt_to_pi)}

@dataclasses.dataclass(frozen=True)
class MobileAlohaOutputs(transforms.DataTransformFn):
    """Outputs for the Aloha policy."""

    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model.
    adapt_to_pi: bool = True

    def __call__(self, data: dict) -> dict:
        # Only return the first 16 dims.: base 2, left arm 6 gripper 1, right arm 6 gripper 1
        actions = np.asarray(data["actions"][:, :16])
        return {"actions": actions}

class MobileAlohaEEFOutputs(transforms.DataTransformFn):
    """Outputs for the Aloha policy."""

    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model.
    adapt_to_pi: bool = True

    def __call__(self, data: dict) -> dict:
        # Only return the first 16 dims.: base 2, left arm 6 gripper 1, right arm 6 gripper 1
        actions = np.asarray(data["actions"][:, :18])
        return {"actions": actions}






def _joint_flip_mask() -> np.ndarray:
    """Used to convert between aloha and pi joint angles."""
    return np.array([1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1])


def _normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def _unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


def _gripper_to_angular(value):
    # Aloha transforms the gripper positions into a linear space. The following code
    # reverses this transformation to be consistent with pi0 which is pretrained in
    # angular space.
    #
    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_POSITION_OPEN, PUPPET_GRIPPER_POSITION_CLOSED
    value = _unnormalize(value, min_val=0.01844, max_val=0.05800)

    # This is the inverse of the angular to linear transformation inside the Interbotix code.
    def linear_to_radian(linear_position, arm_length, horn_radius):
        value = (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)
        return np.arcsin(np.clip(value, -1.0, 1.0))

    # The constants are taken from the Interbotix code.
    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)

    # pi0 gripper data is normalized (0, 1) between encoder counts (2405, 3110).
    # There are 4096 total encoder counts and aloha uses a zero of 2048.
    # Converting this to radians means that the normalized inputs are between (0.5476, 1.6296)
    return _normalize(value, min_val=0.5476, max_val=1.6296)


def _gripper_from_angular(value):
    # Convert from the gripper position used by pi0 to the gripper position that is used by Aloha.
    # Note that the units are still angular but the range is different.

    # We do not scale the output since the trossen model predictions are already in radians.
    # See the comment in _gripper_to_angular for a derivation of the constant
    value = value + 0.5476

    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE
    return _normalize(value, min_val=-0.6213, max_val=1.4910)


def _gripper_from_angular_inv(value):
    # Directly inverts the gripper_from_angular function.
    value = _unnormalize(value, min_val=-0.6213, max_val=1.4910)
    return value - 0.5476


def _decode_aloha(data: dict, *, adapt_to_pi: bool = False) -> dict:
    # state is [left_arm_joint_angles, left_arm_gripper, right_arm_joint_angles, right_arm_gripper]
    # dim sizes: [6, 1, 6, 1]
    state = np.asarray(data["state"])

    def convert_image(img):
        img = np.asarray(img)
        # Convert to uint8 if using float images.
        if np.issubdtype(img.dtype, np.floating):
            img = (255 * img).astype(np.uint8)
        # Convert from [channel, height, width] to [height, width, channel].
        return einops.rearrange(img, "c h w -> h w c")

    images = data["images"]
    images_dict = {name: convert_image(img) for name, img in images.items()}

    data["images"] = images_dict
    data["state"] = state
    return data


def _decode_state(state: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        # Flip the joints.
        state = _joint_flip_mask() * state
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        state[[6, 13]] = _gripper_to_angular(state[[6, 13]])
    return state


def _encode_actions(actions: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        # Flip the joints.
        actions = _joint_flip_mask() * actions
        actions[:, [6, 13]] = _gripper_from_angular(actions[:, [6, 13]])
    return actions


def _encode_actions_inv(actions: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        actions = _joint_flip_mask() * actions
        actions[:, [6, 13]] = _gripper_from_angular_inv(actions[:, [6, 13]])
    return actions



