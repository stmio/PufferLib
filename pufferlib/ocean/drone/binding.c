#include "drone.h"

#define Env Drone
#include "../env_binding.h"

static int my_init(Env *env, PyObject *args, PyObject *kwargs) {
  env->n_targets = unpack(kwargs, "n_targets");
  env->moves_left = unpack(kwargs, "moves_left");
  env->pos = unpack(kwargs, "pos");
  env->vel = unpack(kwargs, "vel");
  env->quat = unpack(kwargs, "quat");
  env->omega = unpack(kwargs, "omega");
  env->move_target = unpack(kwargs, "move_target");
  env->look_target = unpack(kwargs, "look_target");
  env->vec_to_target = unpack(kwargs, "vec_to_target");
  init(env);
  return 0;
}

static int my_log(PyObject *dict, Log *log) {
  assign_to_dict(dict, "perf", log->perf);
  assign_to_dict(dict, "score", log->score);
  assign_to_dict(dict, "episode_return", log->episode_return);
  assign_to_dict(dict, "episode_length", log->episode_length);
  return 0;
}
