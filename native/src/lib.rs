pub mod crypto;
pub mod hash;

pub use crypto::{verify_signature_batch, sign_message};
pub use hash::{compute_merkle_root, hash_transaction_batch};

use std::os::raw::c_char;
use std::ffi::{CStr, CString};

#[no_mangle]
pub extern "C" fn ade_hash_data(data: *const u8, len: usize, out: *mut u8) -> i32 {
    if data.is_null() || out.is_null() {
        return -1;
    }

    unsafe {
        let slice = std::slice::from_raw_parts(data, len);
        let hash = hash::hash_data(slice);
        std::ptr::copy_nonoverlapping(hash.as_ptr(), out, hash.len());
    }

    0
}

#[no_mangle]
pub extern "C" fn ade_verify_signature(
    message: *const u8,
    message_len: usize,
    signature: *const u8,
    public_key: *const u8,
) -> i32 {
    if message.is_null() || signature.is_null() || public_key.is_null() {
        return -1;
    }

    unsafe {
        let msg = std::slice::from_raw_parts(message, message_len);
        let sig = std::slice::from_raw_parts(signature, 64);
        let pubkey = std::slice::from_raw_parts(public_key, 32);

        match crypto::verify_signature_c(msg, sig, pubkey) {
            true => 0,
            false => -2,
        }
    }
}

#[no_mangle]
pub extern "C" fn ade_compute_merkle_root(
    hashes: *const *const u8,
    count: usize,
    out: *mut u8,
) -> i32 {
    if hashes.is_null() || out.is_null() {
        return -1;
    }

    unsafe {
        let hash_ptrs = std::slice::from_raw_parts(hashes, count);
        let mut hash_vec = Vec::new();
        
        for ptr in hash_ptrs {
            let hash = std::slice::from_raw_parts(*ptr, 32);
            hash_vec.push(hash.to_vec());
        }

        let root = hash::compute_merkle_root(&hash_vec);
        std::ptr::copy_nonoverlapping(root.as_ptr(), out, root.len());
    }

    0
}


