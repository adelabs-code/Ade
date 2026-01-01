use ed25519_dalek::{PublicKey, Signature, Verifier};

pub fn verify_signature_c(message: &[u8], signature: &[u8], public_key: &[u8]) -> bool {
    let pubkey = match PublicKey::from_bytes(public_key) {
        Ok(pk) => pk,
        Err(_) => return false,
    };

    let sig = match Signature::from_bytes(signature) {
        Ok(s) => s,
        Err(_) => return false,
    };

    pubkey.verify(message, &sig).is_ok()
}

pub fn verify_signature_batch(
    messages: &[&[u8]],
    signatures: &[&[u8]],
    public_keys: &[&[u8]],
) -> Result<bool, &'static str> {
    if messages.len() != signatures.len() || messages.len() != public_keys.len() {
        return Err("Input arrays must have the same length");
    }

    for i in 0..messages.len() {
        if !verify_signature_c(messages[i], signatures[i], public_keys[i]) {
            return Ok(false);
        }
    }

    Ok(true)
}

pub fn sign_message(message: &[u8], secret_key: &[u8]) -> Result<Vec<u8>, &'static str> {
    use ed25519_dalek::Keypair;

    if secret_key.len() != 64 {
        return Err("Invalid secret key length");
    }

    let keypair = Keypair::from_bytes(secret_key)
        .map_err(|_| "Invalid keypair")?;

    use ed25519_dalek::Signer;
    let signature = keypair.sign(message);
    Ok(signature.to_bytes().to_vec())
}


