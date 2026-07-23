/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#pragma once
#include <cstddef>

#include "VecSim/memory/vecsim_base.h"
#include "VecSim/spaces/spaces.h"
#include "VecSim/types/sq8.h"
#include "VecSim/utils/alignment.h"

// We need this "wrapper" class to hold the DistanceCalculatorInterface in the index, that is not
// templated according to the distance function signature.
template <typename DistType>
class IndexCalculatorInterface : public VecsimBaseObject {
public:
    explicit IndexCalculatorInterface(std::shared_ptr<VecSimAllocator> allocator)
        : VecsimBaseObject(allocator) {}

    virtual ~IndexCalculatorInterface() = default;

    virtual DistType calcDistance(const void *v1, const void *v2, size_t dim) const = 0;

    virtual DistType calcDistanceForQuery(const void *candidate_vector, const void *query_vector,
                                          size_t dim) const = 0;

    // Raw distance function; cached by the index to skip the vtable on the hot path.
    virtual spaces::dist_func_t<DistType> getDistFunc() const = 0;

    virtual spaces::dist_func_t<DistType> getDistFuncForQuery() const = 0;
};

/**
 * This object purpose is to calculate the distance between two vectors.
 * It extends the IndexCalculatorInterface class' type to hold the distance function.
 * Every specific implementation of the distance calculator should hold by reference or by value the
 * parameters required for the calculation. The distance calculation API of all DistanceCalculator
 * classes is: calc_dist(v1,v2,dim). Internally it calls the distance function according the
 * template signature, allowing flexibility in the distance function arguments.
 */
template <typename DistType, typename DistFuncType>
class DistanceCalculatorInterface : public IndexCalculatorInterface<DistType> {
public:
    DistanceCalculatorInterface(std::shared_ptr<VecSimAllocator> allocator, DistFuncType dist_func,
                                DistFuncType query_dist_func = nullptr)
        : IndexCalculatorInterface<DistType>(allocator), dist_func(dist_func),
          query_dist_func(query_dist_func ? query_dist_func : dist_func) {}
    virtual DistType calcDistance(const void *v1, const void *v2, size_t dim) const = 0;
    virtual DistType calcDistanceForQuery(const void *candidate_vector, const void *query_vector,
                                          size_t dim) const = 0;

protected:
    DistFuncType dist_func;
    DistFuncType query_dist_func;
};

template <typename DistType>
class DistanceCalculatorCommon
    : public DistanceCalculatorInterface<DistType, spaces::dist_func_t<DistType>> {
public:
    DistanceCalculatorCommon(std::shared_ptr<VecSimAllocator> allocator,
                             spaces::dist_func_t<DistType> dist_func,
                             spaces::dist_func_t<DistType> query_dist_func = nullptr)
        : DistanceCalculatorInterface<DistType, spaces::dist_func_t<DistType>>(allocator, dist_func,
                                                                               query_dist_func) {}

    DistType calcDistance(const void *v1, const void *v2, size_t dim) const override {
        return this->dist_func(v1, v2, dim);
    }

    DistType calcDistanceForQuery(const void *candidate_vector, const void *query_vector,
                                  size_t dim) const override {
        return this->query_dist_func(candidate_vector, query_vector, dim);
    }

    spaces::dist_func_t<DistType> getDistFunc() const override { return this->dist_func; }

    spaces::dist_func_t<DistType> getDistFuncForQuery() const override {
        return this->query_dist_func;
    }
};

/**
 * Distance calculator for mean-normalized SQ8 indices.
 *
 * Stored vectors are SQ8-quantized from x' = x - mean. This calculator applies analytical
 * correction terms derived from the per-vector x_mean_ip / y_mean_ip metadata stored in
 * the blobs by QuantPreprocessor WithNorm=True to recover correct distances between the
 * original (un-shifted) vectors x and y.
 *
 * Correction formulas (expressed in terms of distance semantics):
 *
 * Asymmetric (query y as FP32/FP16, stored vector x as SQ8 of x'):
 *   IP distance:  dist(x,y) = asym_func(x_blob, y_blob) - y_mean_ip
 *   L2 distance:  dist(x,y) = asym_func(x_blob, y_blob) + 2*(x_mean_ip - y_mean_ip)
 *                              - mean_sum_squares
 *
 * Symmetric (both x and y as SQ8 blobs):
 *   IP distance:  dist(x,y) = sym_func(x_blob, y_blob) - x_mean_ip - y_mean_ip
 *                              + mean_sum_squares
 *   L2 distance:  dist(x,y) = sym_func(x_blob, y_blob)   (mean terms cancel exactly)
 *
 * Note: IP dist functions return 1 - IP(x',y'), not raw inner product. L2 dist functions
 * return ||x'-y'||² directly.
 *
 * DataType is float or float16
 * DistType is float
 */
template <typename DataType, typename DistType, VecSimMetric Metric>
class DistanceCalculatorWithNorm
    : public DistanceCalculatorInterface<DistType, spaces::dist_func_t<DistType>> {
    static_assert(Metric == VecSimMetric_L2 || Metric == VecSimMetric_IP,
                  "DistanceCalculatorWithNorm only supports L2 and IP metrics");

private:
    using sq8 = vecsim_types::sq8;

    float mean_sum_squares_;

public:
    DistanceCalculatorWithNorm(std::shared_ptr<VecSimAllocator> allocator,
                               spaces::dist_func_t<DistType> asym_func,
                               spaces::dist_func_t<DistType> sym_func, float mean_sum_squares)
        : DistanceCalculatorInterface<DistType, spaces::dist_func_t<DistType>>(allocator, sym_func,
                                                                               asym_func),
          mean_sum_squares_(mean_sum_squares) {}

    // Symmetric: both v1 and v2 are stored SQ8-of-x' blobs.
    DistType calcDistance(const void *v1, const void *v2, size_t dim) const override {
        DistType base = this->dist_func(v1, v2, dim);
        if constexpr (Metric == VecSimMetric_IP) {
            // base = 1 - IP(x', y'). We want 1 - IP(x, y).
            // IP(x, y) = IP(x', y') + x_mean_ip + y_mean_ip - mean_sum_squares
            // => 1 - IP(x, y) = base - x_mean_ip - y_mean_ip + mean_sum_squares
            float x_mean_ip =
                load_unaligned<float>(static_cast<const uint8_t *>(v1) + dim +
                                      sq8::template mean_ip_index<Metric>() * sizeof(float));
            float y_mean_ip =
                load_unaligned<float>(static_cast<const uint8_t *>(v2) + dim +
                                      sq8::template mean_ip_index<Metric>() * sizeof(float));
            return base - x_mean_ip - y_mean_ip + mean_sum_squares_;
        } else { // L2: ||x - y||² = ||x' - y'||² (mean terms cancel exactly)
            return base;
        }
    }

    // Asymmetric: query is raw FP32/FP16 blob, candidate is stored SQ8-of-x' blob.
    // Note: query_dist_func expects (storage, query) argument order.
    DistType calcDistanceForQuery(const void *candidate, const void *query,
                                  size_t dim) const override {
        DistType base = this->query_dist_func(candidate, query, dim);
        float y_mean_ip =
            load_unaligned<float>(static_cast<const uint8_t *>(query) + dim * sizeof(DataType) +
                                  sq8::template query_mean_ip_index<Metric>() * sizeof(float));
        if constexpr (Metric == VecSimMetric_IP) {
            // base = 1 - IP(x', y). We want 1 - IP(x, y).
            // IP(x, y) = IP(x', y) + y_mean_ip
            // => 1 - IP(x, y) = base - y_mean_ip
            return base - y_mean_ip;
        } else { // L2
            // base = ||x' - y||². We want ||x - y||².
            // ||x - y||² = ||x' - y||² + 2*(x_mean_ip - y_mean_ip) - mean_sum_squares
            float x_mean_ip =
                load_unaligned<float>(static_cast<const uint8_t *>(candidate) + dim +
                                      sq8::template mean_ip_index<Metric>() * sizeof(float));
            return base + 2.0f * (x_mean_ip - y_mean_ip) - mean_sum_squares_;
        }
    }

    spaces::dist_func_t<DistType> getDistFunc() const override {
        if constexpr (Metric == VecSimMetric_IP) {
            return nullptr;
        } else {
            // For L2, no correction term, we can cache distance fuction.
            return this->dist_func;
        }
    }

    spaces::dist_func_t<DistType> getDistFuncForQuery() const override { return nullptr; }
};
