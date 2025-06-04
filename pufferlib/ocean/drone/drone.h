// Originally made by Sam Turner and Finlay Sanders, 2025

#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define GRID_SIZE 10.0f
#define DT 0.01

#define MASS 1.0f        // kg
#define IXX 0.005f       // kgm^2
#define IYY 0.005f       // kgm^2
#define IZZ 0.009f       // kgm^2
#define ARM_LEN 0.225f   // m
#define K_THRUST 3e-5f   // thrust coefficient
#define K_ANG_DAMP 0.05f // tune this
#define K_DRAG 1e-7f     // drag (torque) coefficient
#define B_DRAG 0.1f      // linear drag coefficent
#define GRAVITY 9.81f    // m/s^2

#define MAX_RPM 1000.0f // rad/s
#define MAX_VEL 50.0f   // m/s
#define MAX_OMEGA 50.0f // rad/s

typedef struct Log Log;
struct Log {
  float episode_return;
  float episode_length;
  float score;
  float perf;
  float n;
};

// ------------------------------------------------------------
// Helper functions for vector math in ℝ³
// ------------------------------------------------------------

typedef struct {
  float w, x, y, z;
} Quaternion;

typedef struct {
  float x, y, z;
} Vector3;

static inline float clampf(float v, float min, float max) {
  if (v < min)
    return min;
  if (v > max)
    return max;
  return v;
}

static inline float rndf(float a, float b) {
  return a + ((float)rand() / (float)RAND_MAX) * (b - a);
}

static inline int rndi(int a, int b) { return a + rand() % (b - a + 1); }

static inline float dot3(Vector3 a, Vector3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

static inline float norm3(Vector3 a) { return sqrtf(dot3(a, a)); }

// In-place clamp of a vector
static inline void clamp3(Vector3 vec, float min, float max) {
  vec.x = clampf(vec.x, min, max);
  vec.y = clampf(vec.y, min, max);
  vec.z = clampf(vec.z, min, max);
}

// In-place clamp of a vector
static inline void clamp4(float a[4], float min, float max) {
  a[0] = clampf(a[0], min, max);
  a[1] = clampf(a[1], min, max);
  a[2] = clampf(a[2], min, max);
  a[3] = clampf(a[3], min, max);
}

static inline Quaternion quat_mul(Quaternion q1, Quaternion q2) {
  Quaternion out;
  out.w = q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z;
  out.x = q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y;
  out.y = q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x;
  out.z = q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w;
  return out;
}

static inline void quat_normalize(Quaternion q) {
  float n = sqrtf(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
  if (n > 0.0f) {
    q.w /= n;
    q.x /= n;
    q.y /= n;
    q.z /= n;
  }
}

static inline Vector3 quat_rotate(Quaternion q, Vector3 v) {
  Quaternion qv = {0.0f, v.x, v.y, v.z};
  Quaternion tmp = quat_mul(q, qv);
  Quaternion q_conj = {q.w, -q.x, -q.y, -q.z};
  Quaternion res = quat_mul(tmp, q_conj);
  return (Vector3){res.x, res.y, res.z};
}

typedef struct Drone Drone;
struct Drone {
  float *observations;
  float *actions;
  float *rewards;
  unsigned char *terminals;
  Log log;
  int tick;

  int n_targets;
  int moves_left;

  Vector3 *pos;     // global position (x, y, z)
  Vector3 *vel;     // linear velocity (u, v, w)
  Quaternion *quat; // roll/pitch/yaw (phi/theta/psi) as a quaternion
  Vector3 *omega;   // angular velocity (p, q, r)

  Vector3 *move_target;   // move target position
  Vector3 *look_target;   // look target position
  Vector3 *vec_to_target; // vector to target
};

void init(Drone *env) {
  env->log = (Log){0};
  env->tick = 0;

  env->pos = (Vector3 *)calloc(1, sizeof(Vector3));
  env->vel = (Vector3 *)calloc(1, sizeof(Vector3));
  env->quat = (Quaternion *)calloc(1, sizeof(Quaternion));
  env->omega = (Vector3 *)calloc(1, sizeof(Vector3));
  env->move_target = (Vector3 *)calloc(1, sizeof(Vector3));
  env->look_target = (Vector3 *)calloc(1, sizeof(Vector3));
  env->vec_to_target = (Vector3 *)calloc(1, sizeof(Vector3));

  srand(time(NULL));
}

void compute_observations(Drone *env) {
  env->observations[0] = env->move_target->x / GRID_SIZE;
  env->observations[1] = env->move_target->y / GRID_SIZE;
  env->observations[2] = env->move_target->z / GRID_SIZE;

  env->observations[3] = env->pos->x / GRID_SIZE;
  env->observations[4] = env->pos->y / GRID_SIZE;
  env->observations[5] = env->pos->z / GRID_SIZE;

  env->observations[6] = env->quat->w;
  env->observations[7] = env->quat->x;
  env->observations[8] = env->quat->y;
  env->observations[9] = env->quat->z;

  env->observations[10] = env->vel->x / MAX_VEL;
  env->observations[11] = env->vel->y / MAX_VEL;
  env->observations[12] = env->vel->z / MAX_VEL;

  env->observations[13] = env->omega->x / MAX_OMEGA;
  env->observations[14] = env->omega->y / MAX_OMEGA;
  env->observations[15] = env->omega->z / MAX_OMEGA;
}

void c_reset(Drone *env) {
  env->log = (Log){0};
  env->tick = 0;

  // env
  env->n_targets = 5;
  env->moves_left = 1000;

  env->move_target->x = rndf(-9, 9);
  env->move_target->y = rndf(-9, 9);
  env->move_target->z = rndf(-9, 9);

  env->look_target->x = rndf(-9, 9);
  env->look_target->y = rndf(-9, 9);
  env->look_target->z = rndf(-9, 9);

  // state
  env->pos->x = rndf(-9, 9);
  env->pos->y = rndf(-9, 9);
  env->pos->z = rndf(-9, 9);

  env->vel->x = 0.0f;
  env->vel->y = 0.0f;
  env->vel->z = 0.0f;

  env->quat->w = 1.0f;
  env->quat->x = 0.0f;
  env->quat->y = 0.0f;
  env->quat->z = 0.0f;

  env->omega->x = 0.0f;
  env->omega->y = 0.0f;
  env->omega->z = 0.0f;

  compute_observations(env);
}

void c_step(Drone *env) {
  clamp4(env->actions, -1.0f, 1.0f);

  // distance to target pre-step for rew calcs
  Vector3 prev_vec = {env->pos->x - env->move_target->x,
                      env->pos->y - env->move_target->y,
                      env->pos->z - env->move_target->z};

  env->tick += 1;
  env->log.episode_length += 1;
  env->rewards[0] = 0;
  env->terminals[0] = 0;

  // motor thrusts
  float T[4];
  for (int i = 0; i < 4; i++) {
    float rpm = (env->actions[i] + 1.0f) * 0.5f * MAX_RPM;
    T[i] = K_THRUST * rpm * rpm;
  }

  // body frame net force
  Vector3 F_body = {0.0f, 0.0f, T[0] + T[1] + T[2] + T[3]};

  // body frame torques
  Vector3 M = {ARM_LEN * (T[1] - T[3]), ARM_LEN * (T[2] - T[0]),
               K_DRAG * (T[0] - T[1] + T[2] - T[3])};

  // applies angular damping to torques
  M.x -= K_ANG_DAMP * env->omega->x;
  M.y -= K_ANG_DAMP * env->omega->y;
  M.z -= K_ANG_DAMP * env->omega->z;

  // body frame force -> world frame force
  Vector3 F_world = quat_rotate(*env->quat, F_body);

  // world frame linear drag
  F_world.x -= B_DRAG * env->vel->x;
  F_world.y -= B_DRAG * env->vel->y;
  F_world.z -= B_DRAG * env->vel->z;

  // world frame gravity
  Vector3 accel = {F_world.x / MASS, (F_world.y / MASS) - GRAVITY,
                   F_world.z / MASS};

  // integrates quaternion
  Quaternion omega_q = {0.0f, env->omega->x, env->omega->y, env->omega->z};
  Quaternion q_dot = quat_mul(*env->quat, omega_q);

  q_dot.w *= 0.5f;
  q_dot.x *= 0.5f;
  q_dot.y *= 0.5f;
  q_dot.z *= 0.5f;

  env->pos->x += env->vel->x * DT;
  env->pos->y += env->vel->y * DT;
  env->pos->z += env->vel->z * DT;

  env->vel->x += accel.x * DT;
  env->vel->y += accel.y * DT;
  env->vel->z += accel.z * DT;

  env->omega->x += (M.x / IXX) * DT;
  env->omega->y += (M.y / IYY) * DT;
  env->omega->z += (M.z / IZZ) * DT;

  clamp3(*env->vel, -MAX_VEL, MAX_VEL);
  clamp3(*env->omega, -MAX_OMEGA, MAX_OMEGA);

  env->quat->w += q_dot.w * DT;
  env->quat->x += q_dot.x * DT;
  env->quat->y += q_dot.y * DT;
  env->quat->z += q_dot.z * DT;
  quat_normalize(*env->quat);

  // check out of bounds
  bool out_of_bounds = env->pos->x < -10.0f || env->pos->x > 10.0f ||
                       env->pos->y < -10.0f || env->pos->y > 10.0f ||
                       env->pos->z < -10.0f || env->pos->z > 10.0f;

  // give rewards
  if (out_of_bounds) {
    env->rewards[0] -= 1;
    env->log.episode_return -= 1;
    env->terminals[0] = 1;
    c_reset(env);
    compute_observations(env);
    return;
  }

  env->vec_to_target->x = env->pos->x - env->move_target->x;
  env->vec_to_target->y = env->pos->y - env->move_target->y;
  env->vec_to_target->z = env->pos->z - env->move_target->z;

  float dist = norm3(prev_vec) - norm3(*env->vec_to_target);
  env->rewards[0] += dist;
  env->log.episode_return += dist;

  if (norm3(*env->vec_to_target) < 1.5) {
    env->rewards[0] += 1;
    env->log.episode_return += 1;
    env->log.score += 1;
    env->n_targets -= 1;

    env->move_target->x = rndf(-10, 10);
    env->move_target->y = rndf(-10, 10);
    env->move_target->z = rndf(-10, 10);
  }

  env->moves_left -= 1;
  if (env->moves_left == 0 || env->n_targets == 0) {
    env->terminals[0] = 1;
    c_reset(env);
  }

  compute_observations(env);
}
