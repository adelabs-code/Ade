#ifndef ADE_NATIVE_H
#define ADE_NATIVE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Hash arbitrary data using SHA256
 * 
 * @param data Input data to hash
 * @param len Length of input data
 * @param out Output buffer (must be at least 32 bytes)
 * @return 0 on success, -1 on error
 */
int32_t ade_hash_data(const uint8_t* data, size_t len, uint8_t* out);

/**
 * Verify an Ed25519 signature
 * 
 * @param message Message that was signed
 * @param message_len Length of message
 * @param signature 64-byte signature
 * @param public_key 32-byte public key
 * @return 0 if signature is valid, -1 on input error, -2 if signature is invalid
 */
int32_t ade_verify_signature(
    const uint8_t* message,
    size_t message_len,
    const uint8_t* signature,
    const uint8_t* public_key
);

/**
 * Compute Merkle root from array of hashes
 * 
 * @param hashes Array of pointers to 32-byte hashes
 * @param count Number of hashes
 * @param out Output buffer (must be at least 32 bytes)
 * @return 0 on success, -1 on error
 */
int32_t ade_compute_merkle_root(
    const uint8_t** hashes,
    size_t count,
    uint8_t* out
);

#ifdef __cplusplus
}
#endif

#endif // ADE_NATIVE_H








