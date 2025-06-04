// Originally made by Sam Turner and Finlay Sanders, 2025.
// Included in pufferlib under the original project's MIT license.

#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "raylib.h"

#define WIDTH 1080
#define HEIGHT 720

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

typedef struct {
  float w, x, y, z;
} Quat;

typedef struct {
  float x, y, z;
} Vec3;

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

static inline float dot3(Vec3 a, Vec3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

static inline float norm3(Vec3 a) { return sqrtf(dot3(a, a)); }

// In-place clamp of a vector
static inline void clamp3(Vec3 *vec, float min, float max) {
  vec->x = clampf(vec->x, min, max);
  vec->y = clampf(vec->y, min, max);
  vec->z = clampf(vec->z, min, max);
}

// In-place clamp of a vector
static inline void clamp4(float a[4], float min, float max) {
  a[0] = clampf(a[0], min, max);
  a[1] = clampf(a[1], min, max);
  a[2] = clampf(a[2], min, max);
  a[3] = clampf(a[3], min, max);
}

static inline Quat quat_mul(Quat q1, Quat q2) {
  Quat out;
  out.w = q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z;
  out.x = q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y;
  out.y = q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x;
  out.z = q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w;
  return out;
}

static inline void quat_normalize(Quat *q) {
  float n = sqrtf(q->w * q->w + q->x * q->x + q->y * q->y + q->z * q->z);
  if (n > 0.0f) {
    q->w /= n;
    q->x /= n;
    q->y /= n;
    q->z /= n;
  }
}

static inline Vec3 quat_rotate(Quat q, Vec3 v) {
  Quat qv = {0.0f, v.x, v.y, v.z};
  Quat tmp = quat_mul(q, qv);
  Quat q_conj = {q.w, -q.x, -q.y, -q.z};
  Quat res = quat_mul(tmp, q_conj);
  return (Vec3){res.x, res.y, res.z};
}

typedef struct Client Client;
struct Client {
  Camera3D camera;
  float width;
  float height;
};

typedef struct Drone Drone;
struct Drone {
  float *observations;
  float *actions;
  float *rewards;
  unsigned char *terminals;
  Log log;
  unsigned tick;
  unsigned report_interval;
  int episode_return;

  int n_targets;
  int moves_left;

  Vec3 pos;   // global position (x, y, z)
  Vec3 vel;   // linear velocity (u, v, w)
  Quat quat;  // roll/pitch/yaw (phi/theta/psi) as a quaternion
  Vec3 omega; // angular velocity (p, q, r)

  Vec3 move_target;   // move target position
  Vec3 look_target;   // look target position
  Vec3 vec_to_target; // vector to target

  Client *client;
};

void init(Drone *env) {
  env->log = (Log){0};
  env->tick = 0;
  srand(time(NULL));
}

void add_log(Drone *env) {
    env->log.score = env->episode_return;
    env->log.episode_return = env->episode_return;
    env->log.episode_length = env->tick;
    env->log.perf = 0.0f; // make this a 0-1 normalized score
    env->log.n += 1.0f;
}

void compute_observations(Drone *env) {
  env->observations[0] = env->move_target.x / GRID_SIZE;
  env->observations[1] = env->move_target.y / GRID_SIZE;
  env->observations[2] = env->move_target.z / GRID_SIZE;

  env->observations[3] = env->pos.x / GRID_SIZE;
  env->observations[4] = env->pos.y / GRID_SIZE;
  env->observations[5] = env->pos.z / GRID_SIZE;

  env->observations[6] = env->quat.w;
  env->observations[7] = env->quat.x;
  env->observations[8] = env->quat.y;
  env->observations[9] = env->quat.z;

  env->observations[10] = env->vel.x / MAX_VEL;
  env->observations[11] = env->vel.y / MAX_VEL;
  env->observations[12] = env->vel.z / MAX_VEL;

  env->observations[13] = env->omega.x / MAX_OMEGA;
  env->observations[14] = env->omega.y / MAX_OMEGA;
  env->observations[15] = env->omega.z / MAX_OMEGA;
}

void c_reset(Drone *env) {
  env->tick = 0;
  env->episode_return = 0;

  // env
  env->n_targets = 5;
  env->moves_left = 1000;

  env->move_target.x = rndf(-9, 9);
  env->move_target.y = rndf(-9, 9);
  env->move_target.z = rndf(-9, 9);

  env->look_target.x = rndf(-9, 9);
  env->look_target.y = rndf(-9, 9);
  env->look_target.z = rndf(-9, 9);

  // state
  env->pos.x = rndf(-9, 9);
  env->pos.y = rndf(-9, 9);
  env->pos.z = rndf(-9, 9);

  env->vel.x = 0.0f;
  env->vel.y = 0.0f;
  env->vel.z = 0.0f;

  env->quat.w = 1.0f;
  env->quat.x = 0.0f;
  env->quat.y = 0.0f;
  env->quat.z = 0.0f;

  env->omega.x = 0.0f;
  env->omega.y = 0.0f;
  env->omega.z = 0.0f;

  compute_observations(env);
}

void c_step(Drone *env) {
  clamp4(env->actions, -1.0f, 1.0f);

  // distance to target pre-step for rew calcs
  Vec3 prev_vec = {env->pos.x - env->move_target.x,
                   env->pos.y - env->move_target.y,
                   env->pos.z - env->move_target.z};

  env->tick++;
  env->rewards[0] = 0;
  env->terminals[0] = 0;

  // motor thrusts
  float T[4];
  for (int i = 0; i < 4; i++) {
    float rpm = (env->actions[i] + 1.0f) * 0.5f * MAX_RPM;
    T[i] = K_THRUST * rpm * rpm;
  }

  // body frame net force
  Vec3 F_body = {0.0f, 0.0f, T[0] + T[1] + T[2] + T[3]};

  // body frame torques
  Vec3 M = {ARM_LEN * (T[1] - T[3]), ARM_LEN * (T[2] - T[0]),
            K_DRAG * (T[0] - T[1] + T[2] - T[3])};

  // applies angular damping to torques
  M.x -= K_ANG_DAMP * env->omega.x;
  M.y -= K_ANG_DAMP * env->omega.y;
  M.z -= K_ANG_DAMP * env->omega.z;

  // body frame force -> world frame force
  Vec3 F_world = quat_rotate(env->quat, F_body);

  // world frame linear drag
  F_world.x -= B_DRAG * env->vel.x;
  F_world.y -= B_DRAG * env->vel.y;
  F_world.z -= B_DRAG * env->vel.z;

  // world frame gravity
  Vec3 accel = {F_world.x / MASS, (F_world.y / MASS) - GRAVITY,
                F_world.z / MASS};

  // integrates quaternion
  Quat omega_q = {0.0f, env->omega.x, env->omega.y, env->omega.z};
  Quat q_dot = quat_mul(env->quat, omega_q);

  q_dot.w *= 0.5f;
  q_dot.x *= 0.5f;
  q_dot.y *= 0.5f;
  q_dot.z *= 0.5f;

  env->pos.x += env->vel.x * DT;
  env->pos.y += env->vel.y * DT;
  env->pos.z += env->vel.z * DT;

  env->vel.x += accel.x * DT;
  env->vel.y += accel.y * DT;
  env->vel.z += accel.z * DT;

  env->omega.x += (M.x / IXX) * DT;
  env->omega.y += (M.y / IYY) * DT;
  env->omega.z += (M.z / IZZ) * DT;

  clamp3(&env->vel, -MAX_VEL, MAX_VEL);
  clamp3(&env->omega, -MAX_OMEGA, MAX_OMEGA);

  env->quat.w += q_dot.w * DT;
  env->quat.x += q_dot.x * DT;
  env->quat.y += q_dot.y * DT;
  env->quat.z += q_dot.z * DT;
  quat_normalize(&env->quat);

  // check out of bounds
  bool out_of_bounds = env->pos.x < -10.0f || env->pos.x > 10.0f ||
                       env->pos.y < -10.0f || env->pos.y > 10.0f ||
                       env->pos.z < -10.0f || env->pos.z > 10.0f;

  // give rewards
  if (out_of_bounds) {
    env->rewards[0] -= 1;
    env->episode_return -= 1;
    env->terminals[0] = 1;
    add_log(env);
    c_reset(env);
    compute_observations(env);
    return;
  }

  env->vec_to_target.x = env->pos.x - env->move_target.x;
  env->vec_to_target.y = env->pos.y - env->move_target.y;
  env->vec_to_target.z = env->pos.z - env->move_target.z;

  float dist = norm3(prev_vec) - norm3(env->vec_to_target);
  env->rewards[0] += dist;
  env->episode_return += dist;

  if (norm3(env->vec_to_target) < 1.5) {
    env->rewards[0] += 1;
    env->episode_return += 1;
    env->n_targets -= 1;
    env->move_target.x = rndf(-10, 10);
    env->move_target.y = rndf(-10, 10);
    env->move_target.z = rndf(-10, 10);
  }

  env->moves_left -= 1;
  if (env->moves_left == 0 || env->n_targets == 0) {
    env->terminals[0] = 1;
    add_log(env);
    c_reset(env);
  }

  compute_observations(env);
}

void c_close_client(Client *client) {
  CloseWindow();
  free(client);
}

void c_close(Drone *env) {
  if (env->client != NULL) {
    c_close_client(env->client);
  }
}

Client *make_client(Drone *env) {
  Client *client = (Client *)calloc(1, sizeof(Client));

  client->width = WIDTH;
  client->height = HEIGHT;

  InitWindow(WIDTH, HEIGHT, "PufferLib Drone");
  SetTargetFPS(60);

  if (!IsWindowReady()) {
    TraceLog(LOG_ERROR, "Window failed to initialize\n");
    free(client);
    return NULL;
  }

  client->camera.position = (Vector3){
      20.0f, // Same X as target
      20.0f, // 20 units above target
      20.0f  // 20 units behind target
  };
  ;
  client->camera.target = (Vector3){0.0f, 0.0f, 0.0f};
  client->camera.up = (Vector3){0.0f, -1.0f, 0.0f}; // Y is up
  client->camera.fovy = 45.0f;
  client->camera.projection = CAMERA_PERSPECTIVE;

  return client;
}

void c_render(Drone *env) {
  if (env->client == NULL) {
    env->client = make_client(env);
    if (env->client == NULL) {
      TraceLog(LOG_ERROR, "Failed to initialize client for rendering\n");
      return;
    }
  }

  if (!WindowShouldClose() && IsWindowReady()) {
    if (IsKeyDown(KEY_ESCAPE)) {
      exit(0);
    }

    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});

    BeginMode3D(env->client->camera);
    DrawGrid(20, 1.0f);

    DrawSphere(
        (Vector3){env->move_target.x, env->move_target.y, env->move_target.z},
        0.2f, BLUE);

    DrawSphere(
        (Vector3){env->look_target.x, env->look_target.y, env->look_target.z},
        0.15f, GREEN);

    DrawSphere((Vector3){env->pos.x, env->pos.y, env->pos.z}, 0.25f, RED);

    DrawLine3D(
        (Vector3){env->pos.x, env->pos.y, env->pos.z},
        (Vector3){env->move_target.x, env->move_target.y, env->move_target.z},
        DARKGRAY);
    EndMode3D();

    DrawText(TextFormat("Targets left: %d", env->n_targets), 10, 10, 20,
             DARKGRAY);
    DrawText(TextFormat("Moves left: %d", env->moves_left), 10, 40, 20,
             DARKGRAY);
    DrawText(TextFormat("Episode Return: %.2f", env->log.episode_return), 10,
             70, 20, DARKGRAY);

    EndDrawing();
  } else {
    TraceLog(LOG_WARNING, "Window is not ready or should close");
  }
}
