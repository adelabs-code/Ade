use anyhow::Result;
use tracing::debug;

/// Compression algorithms for network optimization
pub enum CompressionAlgorithm {
    None,
    Gzip(flate2::Compression),
    Zstd(i32), // Compression level
    Lz4,
}

pub struct Compressor {
    algorithm: CompressionAlgorithm,
}

impl Compressor {
    pub fn new(algorithm: CompressionAlgorithm) -> Self {
        Self { algorithm }
    }

    /// Compress data
    pub fn compress(&self, data: &[u8]) -> Result<Vec<u8>> {
        match &self.algorithm {
            CompressionAlgorithm::None => Ok(data.to_vec()),
            
            CompressionAlgorithm::Gzip(level) => {
                use flate2::write::GzEncoder;
                use std::io::Write;
                
                let mut encoder = GzEncoder::new(Vec::new(), *level);
                encoder.write_all(data)?;
                let compressed = encoder.finish()?;
                
                debug!("Gzip compressed: {} -> {} bytes ({:.2}% ratio)",
                    data.len(), compressed.len(),
                    (compressed.len() as f64 / data.len() as f64) * 100.0
                );
                
                Ok(compressed)
            }
            
            CompressionAlgorithm::Zstd(level) => {
                let compressed = zstd::encode_all(data, *level)?;
                
                debug!("Zstd compressed: {} -> {} bytes ({:.2}% ratio)",
                    data.len(), compressed.len(),
                    (compressed.len() as f64 / data.len() as f64) * 100.0
                );
                
                Ok(compressed)
            }
            
            CompressionAlgorithm::Lz4 => {
                let compressed = lz4_flex::compress_prepend_size(data);
                
                debug!("LZ4 compressed: {} -> {} bytes ({:.2}% ratio)",
                    data.len(), compressed.len(),
                    (compressed.len() as f64 / data.len() as f64) * 100.0
                );
                
                Ok(compressed)
            }
        }
    }

    /// Decompress data
    pub fn decompress(&self, data: &[u8]) -> Result<Vec<u8>> {
        match &self.algorithm {
            CompressionAlgorithm::None => Ok(data.to_vec()),
            
            CompressionAlgorithm::Gzip(_) => {
                use flate2::read::GzDecoder;
                use std::io::Read;
                
                let mut decoder = GzDecoder::new(data);
                let mut decompressed = Vec::new();
                decoder.read_to_end(&mut decompressed)?;
                
                Ok(decompressed)
            }
            
            CompressionAlgorithm::Zstd(_) => {
                Ok(zstd::decode_all(data)?)
            }
            
            CompressionAlgorithm::Lz4 => {
                Ok(lz4_flex::decompress_size_prepended(data)?)
            }
        }
    }

    /// Get compression ratio
    pub fn compression_ratio(&self, original: usize, compressed: usize) -> f64 {
        if original == 0 {
            return 0.0;
        }
        (compressed as f64 / original as f64) * 100.0
    }

    /// Check if data is compressible
    pub fn is_worth_compressing(&self, data: &[u8]) -> bool {
        // Small data might not benefit from compression
        data.len() > 1024
    }
}

/// Adaptive compressor that selects best algorithm
pub struct AdaptiveCompressor {
    threshold_size: usize,
}

impl AdaptiveCompressor {
    pub fn new(threshold_size: usize) -> Self {
        Self { threshold_size }
    }

    /// Compress with best algorithm for data
    pub fn compress(&self, data: &[u8]) -> Result<(Vec<u8>, CompressionAlgorithm)> {
        if data.len() < self.threshold_size {
            return Ok((data.to_vec(), CompressionAlgorithm::None));
        }

        // Try different algorithms and pick best
        let algorithms = vec![
            CompressionAlgorithm::Lz4,  // Fast
            CompressionAlgorithm::Zstd(3),  // Balanced
        ];

        let mut best_compressed = data.to_vec();
        let mut best_algorithm = CompressionAlgorithm::None;

        for algo in algorithms {
            let compressor = Compressor::new(algo);
            if let Ok(compressed) = compressor.compress(data) {
                if compressed.len() < best_compressed.len() {
                    best_compressed = compressed;
                    best_algorithm = compressor.algorithm;
                }
            }
        }

        Ok((best_compressed, best_algorithm))
    }
}

impl Default for AdaptiveCompressor {
    fn default() -> Self {
        Self::new(1024)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gzip_compression() {
        let compressor = Compressor::new(CompressionAlgorithm::Gzip(flate2::Compression::default()));
        
        let data = vec![1u8; 1000];
        let compressed = compressor.compress(&data).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();
        
        assert_eq!(data, decompressed);
        assert!(compressed.len() < data.len());
    }

    #[test]
    fn test_zstd_compression() {
        let compressor = Compressor::new(CompressionAlgorithm::Zstd(3));
        
        let data = b"Hello, world! ".repeat(100);
        let compressed = compressor.compress(&data).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();
        
        assert_eq!(data, decompressed.as_slice());
    }

    #[test]
    fn test_lz4_compression() {
        let compressor = Compressor::new(CompressionAlgorithm::Lz4);
        
        let data = vec![42u8; 500];
        let compressed = compressor.compress(&data).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();
        
        assert_eq!(data, decompressed);
    }

    #[test]
    fn test_adaptive_compressor() {
        let compressor = AdaptiveCompressor::new(100);
        
        let data = vec![1u8; 2000];
        let (compressed, _algo) = compressor.compress(&data).unwrap();
        
        assert!(compressed.len() <= data.len());
    }

    #[test]
    fn test_small_data_no_compression() {
        let compressor = AdaptiveCompressor::new(1000);
        
        let data = vec![1u8; 500];
        let (compressed, algo) = compressor.compress(&data).unwrap();
        
        assert!(matches!(algo, CompressionAlgorithm::None));
        assert_eq!(compressed, data);
    }
}



