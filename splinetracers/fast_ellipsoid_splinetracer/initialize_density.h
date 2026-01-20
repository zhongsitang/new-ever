// Copyright 2024 Google LLC
// Licensed under the Apache License, Version 2.0

#pragma once
#include "Forward.h"

void initialize_density(Params* params, OptixAabb* aabbs, int* d_touch_count = nullptr, int* d_touch_inds = nullptr);
void initialize_density_so(Params* params);
void initialize_density_zero(Params* params);
