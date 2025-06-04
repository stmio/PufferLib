#include "drone.h"

#define Env Drone
#include "../env_binding.h"

static int my_init(Env *env, PyObject *args, PyObject *kwargs) {
  env->n_targets = unpack(kwargs, "n_targets");
  env->moves_left = unpack(kwargs, "moves_left");

  env->pos.x = unpack(kwargs, "pos_x");
  env->pos.y = unpack(kwargs, "pos_y");
  env->pos.z = unpack(kwargs, "pos_z");

  env->vel.x = unpack(kwargs, "vel_x");
  env->vel.y = unpack(kwargs, "vel_y");
  env->vel.z = unpack(kwargs, "vel_z");

  env->quat.w = unpack(kwargs, "quat_w");
  env->quat.x = unpack(kwargs, "quat_x");
  env->quat.y = unpack(kwargs, "quat_y");
  env->quat.z = unpack(kwargs, "quat_z");

  env->omega.x = unpack(kwargs, "omega_x");
  env->omega.y = unpack(kwargs, "omega_y");
  env->omega.z = unpack(kwargs, "omega_z");

  env->move_target.x = unpack(kwargs, "move_target_x");
  env->move_target.y = unpack(kwargs, "move_target_y");
  env->move_target.z = unpack(kwargs, "move_target_z");

  env->look_target.x = unpack(kwargs, "look_target_x");
  env->look_target.y = unpack(kwargs, "look_target_y");
  env->look_target.z = unpack(kwargs, "look_target_z");

  env->vec_to_target.x = unpack(kwargs, "vec_to_target_x");
  env->vec_to_target.y = unpack(kwargs, "vec_to_target_y");
  env->vec_to_target.z = unpack(kwargs, "vec_to_target_z");

  init(env);
  return 0;
}

static int my_log(PyObject *dict, Log *log) {
  assign_to_dict(dict, "perf", log->perf);
  assign_to_dict(dict, "score", log->score);
  assign_to_dict(dict, "episode_return", log->episode_return);
  assign_to_dict(dict, "episode_length", log->episode_length);
  assign_to_dict(dict, "n", log->n);
  return 0;
}
