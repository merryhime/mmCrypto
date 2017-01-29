/* This file is part of the mmCrypto project.
 *
 * Written in 2017 by MerryMage
 *
 * To the extent possible under law, the author(s) have dedicated all
 * copyright and related and neighboring rights to this software to
 * the public domain worldwide. This software is distributed without
 * any warranty.
 *
 * You should have received a copy of the CC0 Public Domain Dedication
 * along with this software.
 * If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
 */

/* Literature References:
 *
 * Daniel J. Bernstein, "ChaCha, a variant of Salsa20",
 *   <https://cr.yp.to/chacha/chacha-20080128.pdf>.
 *
 * M. Goll, S. Gueron, "Vectorization on ChaCha Stream Cipher", IEEE
 *   Proceedings of 11th International Conference on Information
 *   Technology: New Generations (ITNG 2014), 612-615 (2014).
 */

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <x86intrin.h>

#include "chacha20-avx2.h"

#ifdef _MSC_VER
#define forceinline __forceinline
#else
#define forceinline __attribute__((always_inline)) inline
#endif

forceinline static __m256i add_pi32(const __m256i a, const __m256i b) {
    return _mm256_add_epi32(a, b);
}

forceinline static __m256i xor_pi32(const __m256i a, const __m256i b) {
    return _mm256_xor_si256(a, b);
}

/* We special-case the rotation of 8 and 16 because pshufb is faster. */
forceinline static __m256i rotl_pi32(const __m256i value, size_t count) {
    count %= 32;

    if (count == 0) {
        return value;
    }

    if (count == 8) {
        const __m256i shuf = _mm256_set_epi8(14, 13, 12, 15,
                                             10,  9,  8, 11,
                                              6,  5,  4,  7,
                                              2,  1,  0,  3,
                                             14, 13, 12, 15,
                                             10,  9,  8, 11,
                                              6,  5,  4,  7,
                                              2,  1,  0,  3);
        return _mm256_shuffle_epi8(value, shuf);
    }

    if (count == 16) {
        const __m256i shuf = _mm256_set_epi8(13, 12, 15, 14,
                                              9,  8, 11, 10,
                                              5,  4,  7,  6,
                                              1,  0,  3,  2,
                                             13, 12, 15, 14,
                                              9,  8, 11, 10,
                                              5,  4,  7,  6,
                                              1,  0,  3,  2);
        return _mm256_shuffle_epi8(value, shuf);
    }

    __m256i hi = _mm256_slli_epi32(value, count);
    __m256i lo = _mm256_srli_epi32(value, 32-count);
    return _mm256_or_si256(hi, lo);
}

forceinline static __m256i rotl_vector(const __m256i vector, size_t count) {
    if (count % 4 == 0)
        return vector;
    if (count % 4 == 1)
        return _mm256_shuffle_epi32(vector, _MM_SHUFFLE(0,3,2,1));
    if (count % 4 == 2)
        return _mm256_shuffle_epi32(vector, _MM_SHUFFLE(1,0,3,2));
    if (count % 4 == 3)
        return _mm256_shuffle_epi32(vector, _MM_SHUFFLE(2,1,0,3));
    abort();
}

/* This is the core function. It performs 8 quarter-rounds on two matices in parallel.
 * {a, b, c, d} make up 2 matrices of 16x32-bit state words, like so:
 * a = {{ x0, x1, x2, x3},{ x0, x1, x2, x3}}
 * b = {{ x4, x5, x6, x7},{ x4, x5, x6, x7}}
 * c = {{ x8, x9,x10,x11},{ x8, x9,x10,x11}}
 * d = {{x12,x13,x14,x15},{x12,x13,x14,x15}}
 * i.e.: the upper and lower 128-bits of each vector belong to two different matrices.
 */
forceinline static void chacha20_double_round(__m256i *pa, __m256i *pb, __m256i *pc, __m256i *pd) {
    __m256i a = *pa, b = *pb, c = *pc, d = *pd;

    /* Column Rounds */
    a = add_pi32(a, b); d = rotl_pi32(xor_pi32(d, a), 16);
    c = add_pi32(c, d); b = rotl_pi32(xor_pi32(b, c), 12);
    a = add_pi32(a, b); d = rotl_pi32(xor_pi32(d, a),  8);
    c = add_pi32(c, d); b = rotl_pi32(xor_pi32(b, c),  7);

    /* Diagonal Rounds */
    b = rotl_vector(b, 1); c = rotl_vector(c, 2); d = rotl_vector(d, 3);
    a = add_pi32(a, b); d = rotl_pi32(xor_pi32(d, a), 16);
    c = add_pi32(c, d); b = rotl_pi32(xor_pi32(b, c), 12);
    a = add_pi32(a, b); d = rotl_pi32(xor_pi32(d, a),  8);
    c = add_pi32(c, d); b = rotl_pi32(xor_pi32(b, c),  7);
    b = rotl_vector(b, 3); c = rotl_vector(c, 2); d = rotl_vector(d, 1);

    *pa = a; *pb = b; *pc = c; *pd = d;
}

static __m128i load_i128_from_bytes(const void* bytes) {
    return _mm_loadu_si128((__m128i*)bytes);
}

static __m256i load_i256_from_bytes(const void* bytes) {
    return _mm256_loadu_si256((__m256i*)bytes);
}

static void store_i256_to_bytes(const void* bytes, __m256i value) {
    _mm256_storeu_si256((__m256i*)bytes, value);
}

static __m256i load_i256_with_nonce_and_counter(
        const uint8_t nonce[MMCRYPTO_CHACHA20_NONCE_BYTES],
        const uint64_t counter
    ) {
    union {
        __m256i vector;
        struct {
            uint64_t counter0;
            uint8_t nonce0[MMCRYPTO_CHACHA20_NONCE_BYTES];
            uint64_t counter1;
            uint8_t nonce1[MMCRYPTO_CHACHA20_NONCE_BYTES];
        } s;
    } u;
    u.s.counter0 = counter;
    u.s.counter1 = counter + 1;
    memcpy(u.s.nonce0, nonce, MMCRYPTO_CHACHA20_NONCE_BYTES);
    memcpy(u.s.nonce1, nonce, MMCRYPTO_CHACHA20_NONCE_BYTES);
    return u.vector;
}

void mmcrypto_chacha20(
        uint8_t *out,
        const uint8_t *in,
        size_t in_length,
        const uint8_t key[MMCRYPTO_CHACHA20_KEY_BYTES],
        const uint8_t nonce[MMCRYPTO_CHACHA20_NONCE_BYTES],
        const uint64_t counter
    ) {
    const __m256i im0 = _mm256_broadcastsi128_si256(load_i128_from_bytes("expand 32-byte k"));
    const __m256i im1 = _mm256_broadcastsi128_si256(load_i128_from_bytes(key));
    const __m256i im2 = _mm256_broadcastsi128_si256(load_i128_from_bytes(key+16));
    __m256i im3 = load_i256_with_nonce_and_counter(nonce, counter);

    for (; in_length >= 128; in_length -= 128, in += 128, out += 128) {
        __m256i v0 = im0, v1 = im1, v2 = im2, v3 = im3;

        /* 20 rounds = 10 doublerounds */
        for (unsigned i = 0; i < 10; i++) {
            chacha20_double_round(&v0, &v1, &v2, &v3);
        }
        v0 = add_pi32(v0, im0);
        v1 = add_pi32(v1, im1);
        v2 = add_pi32(v2, im2);
        v3 = add_pi32(v3, im3);

        /* Permute keystream into byte-order */
        __m256i f0 = _mm256_permute2x128_si256(v0, v1, _MM_SHUFFLE(0,2,0,0));
        __m256i f1 = _mm256_permute2x128_si256(v2, v3, _MM_SHUFFLE(0,2,0,0));
        __m256i f2 = _mm256_permute2x128_si256(v0, v1, _MM_SHUFFLE(0,3,0,1));
        __m256i f3 = _mm256_permute2x128_si256(v2, v3, _MM_SHUFFLE(0,3,0,1));

        /* xor the keystream with plaintext */
        f0 = xor_pi32(f0, load_i256_from_bytes(in));
        f1 = xor_pi32(f1, load_i256_from_bytes(in + 32));
        f2 = xor_pi32(f2, load_i256_from_bytes(in + 64));
        f3 = xor_pi32(f3, load_i256_from_bytes(in + 96));
        store_i256_to_bytes(out,      f0);
        store_i256_to_bytes(out + 32, f1);
        store_i256_to_bytes(out + 64, f2);
        store_i256_to_bytes(out + 96, f3);

        /* Increment counter */
        im3 = _mm256_add_epi64(im3, _mm256_set_epi64x(0,2,0,2));
    }

    if (in_length > 0) {
        __m256i v0 = im0, v1 = im1, v2 = im2, v3 = im3;

        /* 20 rounds = 10 doublerounds */
        for (unsigned i = 0; i < 10; i++) {
            chacha20_double_round(&v0, &v1, &v2, &v3);
        }
        v0 = add_pi32(v0, im0);
        v1 = add_pi32(v1, im1);
        v2 = add_pi32(v2, im2);
        v3 = add_pi32(v3, im3);

        /* Permute keystream into byte-order */
        __m256i f0 = _mm256_permute2x128_si256(v0, v1, _MM_SHUFFLE(0,2,0,0));
        __m256i f1 = _mm256_permute2x128_si256(v2, v3, _MM_SHUFFLE(0,2,0,0));
        __m256i f2 = _mm256_permute2x128_si256(v0, v1, _MM_SHUFFLE(0,3,0,1));
        __m256i f3 = _mm256_permute2x128_si256(v2, v3, _MM_SHUFFLE(0,3,0,1));
        uint8_t f[128] = {};
        store_i256_to_bytes(f,      f0);
        store_i256_to_bytes(f + 32, f1);
        store_i256_to_bytes(f + 64, f2);
        store_i256_to_bytes(f + 96, f3);

        for (size_t i = 0; i < in_length; i++) {
            out[i] = f[i] ^ in[i];
        }
    }
}
