// Standalone C demo for Drone environment
// Compile using: ./scripts/build_ocean.sh drone [local|fast]
// Run with: ./drone

#include "drone.h"
#include <time.h>

void generate_dummy_actions(Drone *env) {
  // Generate random floats in [-1, 1] range
  env->actions[0] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
  env->actions[1] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
  env->actions[2] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
  env->actions[3] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
}

void demo() {
  Drone env = {0};

  size_t obs_size = 16;
  size_t act_size = 4;
  env.observations = (float *)calloc(obs_size, sizeof(float));
  env.actions = (float *)calloc(act_size, sizeof(float));
  env.rewards = (float *)calloc(1, sizeof(float));
  env.terminals = (float *)calloc(1, sizeof(float));

  if (!env.observations || !env.actions || !env.rewards) {
    fprintf(stderr, "ERROR: Failed to allocate memory for demo buffers.\n");
    free(env.observations);
    free(env.actions);
    free(env.rewards);
    return;
  }

  init(&env);
  Client *client = make_client(&env);

  if (client == NULL) {
    fprintf(stderr,
            "ERROR: Failed to create rendering client during initial setup.\n");
    c_close(&env);
    free(env.observations);
    free(env.actions);
    free(env.rewards);
    return;
  }
  env.client = client;

  // Initial reset
  c_reset(&env);
  int total_steps = 0;

  printf("Starting Drone demo. Press ESC to exit.\n");

  while (!WindowShouldClose()) {
    generate_dummy_actions(&env);
    c_step(&env);
    c_render(&env);
    total_steps++;
  }

  c_close(&env);
  free(env.observations);
  free(env.actions);
  free(env.rewards);
  // ----------------------------------------
}

int main() {
  srand(time(NULL)); // Seed random number generator
  demo();
  return 0;
}
