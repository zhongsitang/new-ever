// Copyright 2024 Google LLC
// ...
#pragma once

#include "types.h"
#include <cstddef>
#include <stdexcept>
#include <string>

// =============================================================================
// Runtime Layout Validator
// =============================================================================
//
// Validates C++ Params struct layout against Slang reflection at runtime.
// Call once at initialization to catch layout mismatches early.
//
// Usage:
//   ParamsValidator::validate(reflection_json);
//
// =============================================================================

class ParamsValidator {
public:
    struct FieldLayout {
        const char* name;
        size_t offset;
        size_t size;
    };

    /// Validate C++ Params against Slang reflection JSON
    static void validate(const char* reflection_json) {
        // Expected C++ layout (generated from offsetof/sizeof)
        static const FieldLayout cpp_layout[] = {
            {"image",             offsetof(Params, image),             sizeof(Params::image)},
            {"depth_out",         offsetof(Params, depth_out),         sizeof(Params::depth_out)},
            {"iters",             offsetof(Params, iters),             sizeof(Params::iters)},
            {"last_prim",         offsetof(Params, last_prim),         sizeof(Params::last_prim)},
            {"prim_hits",         offsetof(Params, prim_hits),         sizeof(Params::prim_hits)},
            {"last_delta_contrib",offsetof(Params, last_delta_contrib),sizeof(Params::last_delta_contrib)},
            {"last_state",        offsetof(Params, last_state),        sizeof(Params::last_state)},
            {"hit_collection",    offsetof(Params, hit_collection),    sizeof(Params::hit_collection)},
            {"ray_origins",       offsetof(Params, ray_origins),       sizeof(Params::ray_origins)},
            {"ray_directions",    offsetof(Params, ray_directions),    sizeof(Params::ray_directions)},
            {"camera",            offsetof(Params, camera),            sizeof(Params::camera)},
            {"means",             offsetof(Params, means),             sizeof(Params::means)},
            {"scales",            offsetof(Params, scales),            sizeof(Params::scales)},
            {"quats",             offsetof(Params, quats),             sizeof(Params::quats)},
            {"densities",         offsetof(Params, densities),         sizeof(Params::densities)},
            {"features",          offsetof(Params, features),          sizeof(Params::features)},
            {"sh_degree",         offsetof(Params, sh_degree),         sizeof(Params::sh_degree)},
            {"max_iters",         offsetof(Params, max_iters),         sizeof(Params::max_iters)},
            {"tmin",              offsetof(Params, tmin),              sizeof(Params::tmin)},
            {"tmax",              offsetof(Params, tmax),              sizeof(Params::tmax)},
            {"initial_contrib",   offsetof(Params, initial_contrib),   sizeof(Params::initial_contrib)},
            {"max_prim_size",     offsetof(Params, max_prim_size),     sizeof(Params::max_prim_size)},
            {"handle",            offsetof(Params, handle),            sizeof(Params::handle)},
        };

        // Parse Slang reflection and compare
        // (In production, use a proper JSON parser)
        validate_against_json(cpp_layout, sizeof(cpp_layout)/sizeof(cpp_layout[0]),
                              reflection_json);
    }

    /// Generate C++ layout table for debugging
    static void print_cpp_layout() {
        printf("C++ Params layout (sizeof=%zu, alignof=%zu):\n",
               sizeof(Params), alignof(Params));
        printf("  %-20s  %6s  %6s\n", "Field", "Offset", "Size");
        printf("  %-20s  %6zu  %6zu\n", "image",             offsetof(Params, image),             sizeof(Params::image));
        printf("  %-20s  %6zu  %6zu\n", "depth_out",         offsetof(Params, depth_out),         sizeof(Params::depth_out));
        printf("  %-20s  %6zu  %6zu\n", "iters",             offsetof(Params, iters),             sizeof(Params::iters));
        printf("  %-20s  %6zu  %6zu\n", "last_prim",         offsetof(Params, last_prim),         sizeof(Params::last_prim));
        printf("  %-20s  %6zu  %6zu\n", "prim_hits",         offsetof(Params, prim_hits),         sizeof(Params::prim_hits));
        printf("  %-20s  %6zu  %6zu\n", "last_delta_contrib",offsetof(Params, last_delta_contrib),sizeof(Params::last_delta_contrib));
        printf("  %-20s  %6zu  %6zu\n", "last_state",        offsetof(Params, last_state),        sizeof(Params::last_state));
        printf("  %-20s  %6zu  %6zu\n", "hit_collection",    offsetof(Params, hit_collection),    sizeof(Params::hit_collection));
        printf("  %-20s  %6zu  %6zu\n", "ray_origins",       offsetof(Params, ray_origins),       sizeof(Params::ray_origins));
        printf("  %-20s  %6zu  %6zu\n", "ray_directions",    offsetof(Params, ray_directions),    sizeof(Params::ray_directions));
        printf("  %-20s  %6zu  %6zu\n", "camera",            offsetof(Params, camera),            sizeof(Params::camera));
        printf("  %-20s  %6zu  %6zu\n", "means",             offsetof(Params, means),             sizeof(Params::means));
        printf("  %-20s  %6zu  %6zu\n", "scales",            offsetof(Params, scales),            sizeof(Params::scales));
        printf("  %-20s  %6zu  %6zu\n", "quats",             offsetof(Params, quats),             sizeof(Params::quats));
        printf("  %-20s  %6zu  %6zu\n", "densities",         offsetof(Params, densities),         sizeof(Params::densities));
        printf("  %-20s  %6zu  %6zu\n", "features",          offsetof(Params, features),          sizeof(Params::features));
        printf("  %-20s  %6zu  %6zu\n", "sh_degree",         offsetof(Params, sh_degree),         sizeof(Params::sh_degree));
        printf("  %-20s  %6zu  %6zu\n", "max_iters",         offsetof(Params, max_iters),         sizeof(Params::max_iters));
        printf("  %-20s  %6zu  %6zu\n", "tmin",              offsetof(Params, tmin),              sizeof(Params::tmin));
        printf("  %-20s  %6zu  %6zu\n", "_pad_tmin",         offsetof(Params, _pad_tmin),         sizeof(Params::_pad_tmin));
        printf("  %-20s  %6zu  %6zu\n", "tmax",              offsetof(Params, tmax),              sizeof(Params::tmax));
        printf("  %-20s  %6zu  %6zu\n", "initial_contrib",   offsetof(Params, initial_contrib),   sizeof(Params::initial_contrib));
        printf("  %-20s  %6zu  %6zu\n", "max_prim_size",     offsetof(Params, max_prim_size),     sizeof(Params::max_prim_size));
        printf("  %-20s  %6zu  %6zu\n", "_pad_handle",       offsetof(Params, _pad_handle),       sizeof(Params::_pad_handle));
        printf("  %-20s  %6zu  %6zu\n", "handle",            offsetof(Params, handle),            sizeof(Params::handle));
    }

private:
    static void validate_against_json(const FieldLayout* cpp, size_t count,
                                       const char* json);
};
