use anyhow::{Result, Context};
use tokio::net::{TcpListener, TcpStream, UdpSocket};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use std::sync::Arc;
use std::net::SocketAddr;
use tracing::{info, debug, warn, error};

use crate::network::GossipMessage;

/// Network transport layer
pub struct Transport {
    tcp_listener: Option<Arc<TcpListener>>,
    udp_socket: Option<Arc<UdpSocket>>,
    tcp_port: u16,
    udp_port: u16,
}

impl Transport {
    pub async fn new(tcp_port: u16, udp_port: u16) -> Result<Self> {
        Ok(Self {
            tcp_listener: None,
            udp_socket: None,
            tcp_port,
            udp_port,
        })
    }

    /// Start TCP listener
    pub async fn start_tcp(&mut self) -> Result<()> {
        let addr = format!("0.0.0.0:{}", self.tcp_port);
        info!("Starting TCP listener on {}", addr);
        
        let listener = TcpListener::bind(&addr).await
            .context("Failed to bind TCP listener")?;
        
        self.tcp_listener = Some(Arc::new(listener));
        Ok(())
    }

    /// Start UDP socket
    pub async fn start_udp(&mut self) -> Result<()> {
        let addr = format!("0.0.0.0:{}", self.udp_port);
        info!("Starting UDP socket on {}", addr);
        
        let socket = UdpSocket::bind(&addr).await
            .context("Failed to bind UDP socket")?;
        
        self.udp_socket = Some(Arc::new(socket));
        Ok(())
    }

    /// Send message via TCP
    pub async fn send_tcp(&self, peer_addr: &str, data: &[u8]) -> Result<()> {
        debug!("Sending {} bytes via TCP to {}", data.len(), peer_addr);
        
        let mut stream = TcpStream::connect(peer_addr).await
            .context("Failed to connect to peer")?;
        
        // Write length prefix (4 bytes)
        stream.write_u32(data.len() as u32).await?;
        
        // Write data
        stream.write_all(data).await?;
        stream.flush().await?;
        
        Ok(())
    }

    /// Send message via UDP
    pub async fn send_udp(&self, peer_addr: &str, data: &[u8]) -> Result<()> {
        if let Some(ref socket) = self.udp_socket {
            debug!("Sending {} bytes via UDP to {}", data.len(), peer_addr);
            
            socket.send_to(data, peer_addr).await?;
            Ok(())
        } else {
            Err(anyhow::anyhow!("UDP socket not initialized"))
        }
    }

    /// Receive message via TCP
    pub async fn receive_tcp(&self) -> Result<(Vec<u8>, SocketAddr)> {
        if let Some(ref listener) = self.tcp_listener {
            let (mut stream, addr) = listener.accept().await?;
            
            // Read length prefix
            let len = stream.read_u32().await? as usize;
            
            if len > 10_000_000 {
                return Err(anyhow::anyhow!("Message too large: {}", len));
            }
            
            // Read data
            let mut data = vec![0u8; len];
            stream.read_exact(&mut data).await?;
            
            debug!("Received {} bytes via TCP from {}", len, addr);
            Ok((data, addr))
        } else {
            Err(anyhow::anyhow!("TCP listener not initialized"))
        }
    }

    /// Receive message via UDP
    pub async fn receive_udp(&self) -> Result<(Vec<u8>, SocketAddr)> {
        if let Some(ref socket) = self.udp_socket {
            let mut buf = vec![0u8; 65536]; // Max UDP packet size
            let (len, addr) = socket.recv_from(&mut buf).await?;
            
            buf.truncate(len);
            debug!("Received {} bytes via UDP from {}", len, addr);
            
            Ok((buf, addr))
        } else {
            Err(anyhow::anyhow!("UDP socket not initialized"))
        }
    }

    /// Broadcast message to multiple peers via UDP
    pub async fn broadcast_udp(&self, peer_addrs: &[String], data: &[u8]) -> Result<usize> {
        if let Some(ref socket) = self.udp_socket {
            let mut sent = 0;
            
            for addr in peer_addrs {
                match socket.send_to(data, addr).await {
                    Ok(_) => sent += 1,
                    Err(e) => warn!("Failed to send to {}: {}", addr, e),
                }
            }
            
            debug!("Broadcasted to {}/{} peers", sent, peer_addrs.len());
            Ok(sent)
        } else {
            Err(anyhow::anyhow!("UDP socket not initialized"))
        }
    }

    /// Spawn TCP receiver task
    pub fn spawn_tcp_receiver<F>(&self, handler: F)
    where
        F: Fn(Vec<u8>, SocketAddr) + Send + 'static,
    {
        if let Some(listener) = self.tcp_listener.clone() {
            tokio::spawn(async move {
                loop {
                    match listener.accept().await {
                        Ok((mut stream, addr)) => {
                            let handler = &handler;
                            
                            tokio::spawn(async move {
                                match read_tcp_message(&mut stream).await {
                                    Ok(data) => {
                                        handler(data, addr);
                                    }
                                    Err(e) => {
                                        warn!("Error reading TCP message: {}", e);
                                    }
                                }
                            });
                        }
                        Err(e) => {
                            error!("TCP accept error: {}", e);
                        }
                    }
                }
            });
        }
    }

    /// Spawn UDP receiver task
    pub fn spawn_udp_receiver<F>(&self, handler: F)
    where
        F: Fn(Vec<u8>, SocketAddr) + Send + 'static,
    {
        if let Some(socket) = self.udp_socket.clone() {
            tokio::spawn(async move {
                let mut buf = vec![0u8; 65536];
                
                loop {
                    match socket.recv_from(&mut buf).await {
                        Ok((len, addr)) => {
                            let data = buf[..len].to_vec();
                            handler(data, addr);
                        }
                        Err(e) => {
                            error!("UDP receive error: {}", e);
                        }
                    }
                }
            });
        }
    }
}

/// Read TCP message with length prefix
async fn read_tcp_message(stream: &mut TcpStream) -> Result<Vec<u8>> {
    let len = stream.read_u32().await? as usize;
    
    if len > 10_000_000 {
        return Err(anyhow::anyhow!("Message too large: {}", len));
    }
    
    let mut data = vec![0u8; len];
    stream.read_exact(&mut data).await?;
    
    Ok(data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_transport_creation() {
        let transport = Transport::new(9000, 9001).await.unwrap();
        assert_eq!(transport.tcp_port, 9000);
        assert_eq!(transport.udp_port, 9001);
    }

    #[tokio::test]
    async fn test_tcp_listener_start() {
        let mut transport = Transport::new(0, 0).await.unwrap(); // OS-assigned port
        let result = transport.start_tcp().await;
        assert!(result.is_ok());
        assert!(transport.tcp_listener.is_some());
    }

    #[tokio::test]
    async fn test_udp_socket_start() {
        let mut transport = Transport::new(0, 0).await.unwrap();
        let result = transport.start_udp().await;
        assert!(result.is_ok());
        assert!(transport.udp_socket.is_some());
    }
}




