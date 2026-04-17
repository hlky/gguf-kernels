#include "ggml-quants.h"
#include "ggml-impl.h"

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#if defined(_WIN32) && !defined(__MINGW32__)
#define GGUF_GGML_REF_API __declspec(dllexport)
#else
#define GGUF_GGML_REF_API __attribute__((visibility("default")))
#endif

static ggml_abort_callback_t gguf_abort_callback = NULL;

GGUF_GGML_REF_API ggml_abort_callback_t ggml_set_abort_callback(ggml_abort_callback_t callback) {
    ggml_abort_callback_t previous = gguf_abort_callback;
    gguf_abort_callback = callback;
    return previous;
}

GGUF_GGML_REF_API void ggml_abort(const char * file, int line, const char * fmt, ...) {
    char message[1024];
    va_list args;
    va_start(args, fmt);
    vsnprintf(message, sizeof(message), fmt, args);
    va_end(args);

    if (gguf_abort_callback != NULL) {
        gguf_abort_callback(message);
    } else {
        fprintf(stderr, "%s:%d: %s\n", file, line, message);
    }
    abort();
}

GGUF_GGML_REF_API float ggml_fp16_to_fp32(ggml_fp16_t x) {
    return GGML_FP16_TO_FP32(x);
}

GGUF_GGML_REF_API ggml_fp16_t ggml_fp32_to_fp16(float x) {
    return GGML_FP32_TO_FP16(x);
}

GGUF_GGML_REF_API void ggml_fp16_to_fp32_row(const ggml_fp16_t * x, float * y, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        y[i] = GGML_FP16_TO_FP32(x[i]);
    }
}

GGUF_GGML_REF_API void ggml_fp32_to_fp16_row(const float * x, ggml_fp16_t * y, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        y[i] = GGML_FP32_TO_FP16(x[i]);
    }
}

static void gguf_ggml_ref_quantize_init(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
            iq2xs_init_impl(type);
            break;
        case GGML_TYPE_IQ3_XXS:
            iq3xs_init_impl(256);
            break;
        case GGML_TYPE_IQ3_S:
            iq3xs_init_impl(512);
            break;
        default:
            break;
    }
}

GGUF_GGML_REF_API void gguf_ggml_ref_quantize_free(void) {
    iq2xs_free_impl(GGML_TYPE_IQ2_XXS);
    iq2xs_free_impl(GGML_TYPE_IQ2_XS);
    iq2xs_free_impl(GGML_TYPE_IQ2_S);
    iq2xs_free_impl(GGML_TYPE_IQ1_S);
    iq2xs_free_impl(GGML_TYPE_IQ1_M);
    iq3xs_free_impl(256);
    iq3xs_free_impl(512);
}

GGUF_GGML_REF_API bool gguf_ggml_ref_quantize_requires_imatrix(enum ggml_type type) {
    return type == GGML_TYPE_IQ2_XXS ||
           type == GGML_TYPE_IQ2_XS  ||
           type == GGML_TYPE_IQ1_S;
}

GGUF_GGML_REF_API size_t gguf_ggml_ref_row_size(enum ggml_type type, int64_t n_per_row) {
    switch (type) {
        case GGML_TYPE_Q1_0:    return (size_t) n_per_row * sizeof(block_q1_0)    / QK1_0;
        case GGML_TYPE_Q4_0:    return (size_t) n_per_row * sizeof(block_q4_0)    / QK4_0;
        case GGML_TYPE_Q4_1:    return (size_t) n_per_row * sizeof(block_q4_1)    / QK4_1;
        case GGML_TYPE_Q5_0:    return (size_t) n_per_row * sizeof(block_q5_0)    / QK5_0;
        case GGML_TYPE_Q5_1:    return (size_t) n_per_row * sizeof(block_q5_1)    / QK5_1;
        case GGML_TYPE_Q8_0:    return (size_t) n_per_row * sizeof(block_q8_0)    / QK8_0;
        case GGML_TYPE_Q2_K:    return (size_t) n_per_row * sizeof(block_q2_K)    / QK_K;
        case GGML_TYPE_Q3_K:    return (size_t) n_per_row * sizeof(block_q3_K)    / QK_K;
        case GGML_TYPE_Q4_K:    return (size_t) n_per_row * sizeof(block_q4_K)    / QK_K;
        case GGML_TYPE_Q5_K:    return (size_t) n_per_row * sizeof(block_q5_K)    / QK_K;
        case GGML_TYPE_Q6_K:    return (size_t) n_per_row * sizeof(block_q6_K)    / QK_K;
        case GGML_TYPE_IQ2_XXS: return (size_t) n_per_row * sizeof(block_iq2_xxs) / QK_K;
        case GGML_TYPE_IQ2_XS:  return (size_t) n_per_row * sizeof(block_iq2_xs)  / QK_K;
        case GGML_TYPE_IQ2_S:   return (size_t) n_per_row * sizeof(block_iq2_s)   / QK_K;
        case GGML_TYPE_IQ3_XXS: return (size_t) n_per_row * sizeof(block_iq3_xxs) / QK_K;
        case GGML_TYPE_IQ3_S:   return (size_t) n_per_row * sizeof(block_iq3_s)   / QK_K;
        case GGML_TYPE_IQ1_S:   return (size_t) n_per_row * sizeof(block_iq1_s)   / QK_K;
        case GGML_TYPE_IQ1_M:   return (size_t) n_per_row * sizeof(block_iq1_m)   / QK_K;
        case GGML_TYPE_IQ4_NL:  return (size_t) n_per_row * sizeof(block_iq4_nl)  / QK4_NL;
        case GGML_TYPE_IQ4_XS:  return (size_t) n_per_row * sizeof(block_iq4_xs)  / QK_K;
        case GGML_TYPE_TQ1_0:   return (size_t) n_per_row * sizeof(block_tq1_0)   / QK_K;
        case GGML_TYPE_TQ2_0:   return (size_t) n_per_row * sizeof(block_tq2_0)   / QK_K;
        case GGML_TYPE_MXFP4:   return (size_t) n_per_row * sizeof(block_mxfp4)   / QK_MXFP4;
        case GGML_TYPE_NVFP4:   return (size_t) n_per_row * sizeof(block_nvfp4)   / QK_NVFP4;
        default:
            return 0;
    }
}

GGUF_GGML_REF_API size_t ggml_type_size(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:    return sizeof(block_q1_0);
        case GGML_TYPE_Q4_0:    return sizeof(block_q4_0);
        case GGML_TYPE_Q4_1:    return sizeof(block_q4_1);
        case GGML_TYPE_Q5_0:    return sizeof(block_q5_0);
        case GGML_TYPE_Q5_1:    return sizeof(block_q5_1);
        case GGML_TYPE_Q8_0:    return sizeof(block_q8_0);
        case GGML_TYPE_Q8_1:    return sizeof(block_q8_1);
        case GGML_TYPE_Q2_K:    return sizeof(block_q2_K);
        case GGML_TYPE_Q3_K:    return sizeof(block_q3_K);
        case GGML_TYPE_Q4_K:    return sizeof(block_q4_K);
        case GGML_TYPE_Q5_K:    return sizeof(block_q5_K);
        case GGML_TYPE_Q6_K:    return sizeof(block_q6_K);
        case GGML_TYPE_Q8_K:    return sizeof(block_q8_K);
        case GGML_TYPE_IQ2_XXS: return sizeof(block_iq2_xxs);
        case GGML_TYPE_IQ2_XS:  return sizeof(block_iq2_xs);
        case GGML_TYPE_IQ3_XXS: return sizeof(block_iq3_xxs);
        case GGML_TYPE_IQ1_S:   return sizeof(block_iq1_s);
        case GGML_TYPE_IQ4_NL:  return sizeof(block_iq4_nl);
        case GGML_TYPE_IQ3_S:   return sizeof(block_iq3_s);
        case GGML_TYPE_IQ2_S:   return sizeof(block_iq2_s);
        case GGML_TYPE_IQ4_XS:  return sizeof(block_iq4_xs);
        case GGML_TYPE_IQ1_M:   return sizeof(block_iq1_m);
        case GGML_TYPE_TQ1_0:   return sizeof(block_tq1_0);
        case GGML_TYPE_TQ2_0:   return sizeof(block_tq2_0);
        case GGML_TYPE_MXFP4:   return sizeof(block_mxfp4);
        case GGML_TYPE_NVFP4:   return sizeof(block_nvfp4);
        case GGML_TYPE_F16:     return sizeof(ggml_fp16_t);
        case GGML_TYPE_BF16:    return sizeof(ggml_bf16_t);
        case GGML_TYPE_F32:     return sizeof(float);
        case GGML_TYPE_F64:     return sizeof(double);
        case GGML_TYPE_I8:      return sizeof(int8_t);
        case GGML_TYPE_I16:     return sizeof(int16_t);
        case GGML_TYPE_I32:     return sizeof(int32_t);
        case GGML_TYPE_I64:     return sizeof(int64_t);
        default:
            return 0;
    }
}

GGUF_GGML_REF_API const char * ggml_type_name(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:     return "f32";
        case GGML_TYPE_F16:     return "f16";
        case GGML_TYPE_Q4_0:    return "q4_0";
        case GGML_TYPE_Q4_1:    return "q4_1";
        case GGML_TYPE_Q5_0:    return "q5_0";
        case GGML_TYPE_Q5_1:    return "q5_1";
        case GGML_TYPE_Q8_0:    return "q8_0";
        case GGML_TYPE_Q8_1:    return "q8_1";
        case GGML_TYPE_Q2_K:    return "q2_K";
        case GGML_TYPE_Q3_K:    return "q3_K";
        case GGML_TYPE_Q4_K:    return "q4_K";
        case GGML_TYPE_Q5_K:    return "q5_K";
        case GGML_TYPE_Q6_K:    return "q6_K";
        case GGML_TYPE_Q8_K:    return "q8_K";
        case GGML_TYPE_IQ2_XXS: return "iq2_xxs";
        case GGML_TYPE_IQ2_XS:  return "iq2_xs";
        case GGML_TYPE_IQ3_XXS: return "iq3_xxs";
        case GGML_TYPE_IQ1_S:   return "iq1_s";
        case GGML_TYPE_IQ4_NL:  return "iq4_nl";
        case GGML_TYPE_IQ3_S:   return "iq3_s";
        case GGML_TYPE_IQ2_S:   return "iq2_s";
        case GGML_TYPE_IQ4_XS:  return "iq4_xs";
        case GGML_TYPE_I8:      return "i8";
        case GGML_TYPE_I16:     return "i16";
        case GGML_TYPE_I32:     return "i32";
        case GGML_TYPE_I64:     return "i64";
        case GGML_TYPE_F64:     return "f64";
        case GGML_TYPE_IQ1_M:   return "iq1_m";
        case GGML_TYPE_BF16:    return "bf16";
        case GGML_TYPE_TQ1_0:   return "tq1_0";
        case GGML_TYPE_TQ2_0:   return "tq2_0";
        case GGML_TYPE_MXFP4:   return "mxfp4";
        case GGML_TYPE_NVFP4:   return "nvfp4";
        case GGML_TYPE_Q1_0:    return "q1_0";
        default:
            return "unknown";
    }
}

GGUF_GGML_REF_API size_t ggml_row_size(enum ggml_type type, int64_t n_per_row) {
    return gguf_ggml_ref_row_size(type, n_per_row);
}

GGUF_GGML_REF_API size_t gguf_ggml_ref_quantize_chunk(
        enum ggml_type   type,
        const float *    src,
        void *           dst,
        int64_t          start,
        int64_t          nrows,
        int64_t          n_per_row,
        const float *    imatrix) {
    const size_t row_size = gguf_ggml_ref_row_size(type, n_per_row);
    const size_t start_row = (size_t) (start / n_per_row);

    GGML_ASSERT(row_size != 0);
    GGML_ASSERT(start % n_per_row == 0);

    if (gguf_ggml_ref_quantize_requires_imatrix(type)) {
        GGML_ASSERT(imatrix != NULL);
    }

    gguf_ggml_ref_quantize_init(type);

    switch (type) {
        case GGML_TYPE_Q1_0:
            return quantize_q1_0(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix);
        case GGML_TYPE_Q4_0:
            return quantize_q4_0(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix);
        case GGML_TYPE_Q4_1:
            return quantize_q4_1(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix);
        case GGML_TYPE_Q5_0:
            return quantize_q5_0(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix);
        case GGML_TYPE_Q5_1:
            return quantize_q5_1(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix);
        case GGML_TYPE_Q8_0:
            return quantize_q8_0(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix);
        case GGML_TYPE_Q2_K:
            return quantize_q2_K(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix);
        case GGML_TYPE_Q3_K:
            return quantize_q3_K(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix);
        case GGML_TYPE_Q4_K:
            return quantize_q4_K(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix);
        case GGML_TYPE_Q5_K:
            return quantize_q5_K(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix);
        case GGML_TYPE_Q6_K:
            return quantize_q6_K(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix);
        case GGML_TYPE_IQ2_XXS:
            return quantize_iq2_xxs(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix);
        case GGML_TYPE_IQ2_XS:
            return quantize_iq2_xs(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix);
        case GGML_TYPE_IQ2_S:
            return quantize_iq2_s(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix);
        case GGML_TYPE_IQ3_XXS:
            return quantize_iq3_xxs(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix);
        case GGML_TYPE_IQ3_S:
            return quantize_iq3_s(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix);
        case GGML_TYPE_IQ1_S:
            return quantize_iq1_s(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix);
        case GGML_TYPE_IQ1_M:
            return quantize_iq1_m(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix);
        case GGML_TYPE_IQ4_NL:
            return quantize_iq4_nl(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix);
        case GGML_TYPE_IQ4_XS:
            return quantize_iq4_xs(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix);
        case GGML_TYPE_TQ1_0:
            return quantize_tq1_0(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix);
        case GGML_TYPE_TQ2_0:
            return quantize_tq2_0(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix);
        case GGML_TYPE_MXFP4:
            return quantize_mxfp4(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix);
        case GGML_TYPE_NVFP4:
            return quantize_nvfp4(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix);
        default:
            GGML_ASSERT(false && "unsupported quantization type");
            return 0;
    }
}
