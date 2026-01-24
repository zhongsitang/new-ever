// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// ...

#include "launch_params.h"

#include <algorithm>
#include <cctype>
#include <sstream>

// =============================================================================
// Minimal JSON Parser (for reflection data)
// =============================================================================
//
// Only parses the subset of JSON needed for Slang reflection output.
// For production, consider using nlohmann/json or rapidjson.
//
// =============================================================================

namespace {

struct JsonParser {
    const char* ptr;
    const char* end;

    void skip_ws() {
        while (ptr < end && std::isspace(*ptr)) ++ptr;
    }

    bool match(char c) {
        skip_ws();
        if (ptr < end && *ptr == c) { ++ptr; return true; }
        return false;
    }

    std::string parse_string() {
        skip_ws();
        if (*ptr != '"') throw std::runtime_error("Expected string");
        ++ptr;
        std::string result;
        while (ptr < end && *ptr != '"') {
            if (*ptr == '\\' && ptr + 1 < end) {
                ++ptr;
                switch (*ptr) {
                    case 'n': result += '\n'; break;
                    case 't': result += '\t'; break;
                    case '"': result += '"'; break;
                    case '\\': result += '\\'; break;
                    default: result += *ptr;
                }
            } else {
                result += *ptr;
            }
            ++ptr;
        }
        if (ptr >= end) throw std::runtime_error("Unterminated string");
        ++ptr; // skip closing "
        return result;
    }

    int64_t parse_int() {
        skip_ws();
        int64_t result = 0;
        bool neg = false;
        if (*ptr == '-') { neg = true; ++ptr; }
        while (ptr < end && std::isdigit(*ptr)) {
            result = result * 10 + (*ptr - '0');
            ++ptr;
        }
        return neg ? -result : result;
    }

    void skip_value() {
        skip_ws();
        if (*ptr == '"') { parse_string(); return; }
        if (*ptr == '{') { skip_object(); return; }
        if (*ptr == '[') { skip_array(); return; }
        // number or literal
        while (ptr < end && !std::strchr(",}]", *ptr) && !std::isspace(*ptr)) ++ptr;
    }

    void skip_object() {
        if (!match('{')) throw std::runtime_error("Expected {");
        if (match('}')) return;
        do {
            parse_string(); // key
            if (!match(':')) throw std::runtime_error("Expected :");
            skip_value();
        } while (match(','));
        if (!match('}')) throw std::runtime_error("Expected }");
    }

    void skip_array() {
        if (!match('[')) throw std::runtime_error("Expected [");
        if (match(']')) return;
        do { skip_value(); } while (match(','));
        if (!match(']')) throw std::runtime_error("Expected ]");
    }
};

} // namespace

// =============================================================================
// LaunchParams Implementation
// =============================================================================

LaunchParams::LaunchParams(const char* reflection_json) {
    parse_reflection(reflection_json);
}

void LaunchParams::parse_reflection(const char* json) {
    JsonParser p{json, json + std::strlen(json)};

    // Find globalParams object
    if (!p.match('{')) throw std::runtime_error("Invalid JSON");

    size_t total_size = 0;
    size_t alignment = 1;

    while (!p.match('}')) {
        std::string key = p.parse_string();
        if (!p.match(':')) throw std::runtime_error("Expected :");

        if (key == "size") {
            total_size = p.parse_int();
        } else if (key == "alignment") {
            alignment = p.parse_int();
        } else if (key == "fields") {
            // Parse fields array
            if (!p.match('[')) throw std::runtime_error("Expected [");

            while (!p.match(']')) {
                FieldInfo field;
                if (!p.match('{')) throw std::runtime_error("Expected {");

                while (!p.match('}')) {
                    std::string field_key = p.parse_string();
                    if (!p.match(':')) throw std::runtime_error("Expected :");

                    if (field_key == "name") {
                        field.name = p.parse_string();
                    } else if (field_key == "offset") {
                        field.offset = p.parse_int();
                    } else if (field_key == "size") {
                        field.size = p.parse_int();
                    } else if (field_key == "alignment") {
                        field.alignment = p.parse_int();
                    } else if (field_key == "type") {
                        field.type = p.parse_string();
                    } else {
                        p.skip_value();
                    }
                    p.match(',');
                }

                fields_[field.name] = field;
                p.match(',');
            }
        } else {
            p.skip_value();
        }
        p.match(',');
    }

    // Allocate buffer with proper alignment
    buffer_.resize(total_size, 0);
}
