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

#ifndef _MMCRYPTO_CHACHA20_AVX_H_
#define _MMCRYPTO_CHACHA20_AVX_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MMCRYPTO_CHACHA20_KEY_BYTES 32
#define MMCRYPTO_CHACHA20_NONCE_BYTES 8

/* This function performs unauthenticated encryption with the chacha20 stream cipher.
 * out must be a buffer of size in_length.
 * It is the caller's responsibility to ensure the uniqueness of nonces and counters.
 */
void mmcrypto_chacha20(
    uint8_t *out,
    const uint8_t *in,
    size_t in_length,
    const uint8_t key[MMCRYPTO_CHACHA20_KEY_BYTES],
    const uint8_t nonce[MMCRYPTO_CHACHA20_NONCE_BYTES],
    const uint64_t counter
);

#ifdef __cplusplus
}
#endif

#endif // _MMCRYPTO_CHACHA20_AVX_H_
