import numpy as np
import gymnasium

import pufferlib
from pufferlib.ocean.drone import binding

class Drone(pufferlib.PufferEnv):
    def __init__(
        self,
        num_envs=16,
        render_mode=None,
        report_interval=1,
        buf=None,
        seed=0,
        n_targets=5,
        moves_left=1000,
        pos_x=0,
        pos_y=0,
        pos_z=0,
        vel_x=0,
        vel_y=0,
        vel_z=0,
        quat_w=0,
        quat_x=0,
        quat_y=0,
        quat_z=0,
        omega_x=0,
        omega_y=0,
        omega_z=0,
        move_target_x=0,
        move_target_y=0,
        move_target_z=0,
        look_target_x=0,
        look_target_y=0,
        look_target_z=0,
        vec_to_target_x=0,
        vec_to_target_y=0,
        vec_to_target_z=0
    ):
        self.single_observation_space = gymnasium.spaces.Box(
            low=-1,
            high=1,
            shape=(16,),
            dtype=np.float32,
        )

        self.single_action_space = gymnasium.spaces.Box(
            low=-1, high=1, shape=(4,), dtype=np.float32
        )

        self.num_agents = num_envs
        self.render_mode = render_mode
        self.report_interval = report_interval
        self.tick = 0

        super().__init__(buf)
        self.actions = self.actions.astype(np.float32)

        c_envs = []
        for env_num in range(num_envs):
            c_envs.append(binding.env_init(
                self.observations[env_num:(env_num+1)],
                self.actions[env_num:(env_num+1)],
                self.rewards[env_num:(env_num+1)],
                self.terminals[env_num:(env_num+1)],
                self.truncations[env_num:(env_num+1)],
                seed,
                report_interval=self.report_interval,
                n_targets=n_targets,
                moves_left=moves_left,
                pos_x=pos_x,
                pos_y=pos_y,
                pos_z=pos_z,
                vel_x=vel_x,
                vel_y=vel_y,
                vel_z=vel_z,
                quat_w=quat_w,
                quat_x=quat_x,
                quat_y=quat_y,
                quat_z=quat_z,
                omega_x=omega_x,
                omega_y=omega_y,
                omega_z=omega_z,
                move_target_x=move_target_x,
                move_target_y=move_target_y,
                move_target_z=move_target_z,
                look_target_x=look_target_x,
                look_target_y=look_target_y,
                look_target_z=look_target_z,
                vec_to_target_x=vec_to_target_x,
                vec_to_target_y=vec_to_target_y,
                vec_to_target_z=vec_to_target_z,
            ))

        self.c_envs = binding.vectorize(*c_envs)

    def reset(self, seed=None):
        self.tick = 0
        binding.vec_reset(self.c_envs, seed)
        return self.observations, []

    def step(self, actions):
        self.actions[:] = actions

        self.tick += 1
        binding.vec_step(self.c_envs)

        info = []
        if self.tick % self.report_interval == 0:
            log_data = binding.vec_log(self.c_envs)
            if log_data:
                info.append(log_data)

        return (self.observations, self.rewards, self.terminals, self.truncations, info)

    def render(self):
        binding.vec_render(self.c_envs, 0)

    def close(self):
        binding.vec_close(self.c_envs)

def test_performance(timeout=10, atn_cache=1024):
    env = Drone(num_envs=1000)
    env.reset()
    tick = 0

    actions = [env.action_space.sample() for _ in range(atn_cache)]

    import time
    start = time.time()
    while time.time() - start < timeout:
        atn = actions[tick % atn_cache]
        env.step(atn)
        tick += 1

    print(f"SPS: {env.num_agents * tick / (time.time() - start)}")

if __name__ == "__main__":
    test_performance()
