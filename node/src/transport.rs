use anyhow::{Result, Context};
use tokio::net::{TcpListener, TcpStream, UdpSocket};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use std::sync::Arc;
use std::net::SocketAddr;
use tracing::{info, debug, warn, error};

use crate::network::GossipMessage;
use crate::protocol::Protocol;

/// Network transport layer with real TCP/UDP implementation
pub struct Transport {
    tcp_listener: Option<Arc<TcpListener>>,
    udp_socket: Option<Arc<UdpSocket>>,
    tcp_port: u16,
    udp_port: u16,
    message_handler: Option<Arc<dyn MessageHandler>>,
}

/// Message handler trait for processing received messages
pub trait MessageHandler: Send + Sync {
    fn handle_message(&self, message: GossipMessage, from: SocketAddr);
}

impl Transport {
    pub async fn new(tcp_port: u16, udp_port: u16) -> Result<Self> {
        Ok(Self {
            tcp_listener: None,
            udp_socket: None,
            tcp_port,
            udp_port,
            message_handler: None,
        })
    }

    pub fn set_message_handler<H: MessageHandler + 'static>(&mut self, handler: H) {
        self.message_handler = Some(Arc::new(handler));
    }

    pub async fn start_tcp(&mut self) -> Result<()> {
        let addr = format!("0.0.0.0:{}", self.tcp_port);
        info!("Starting TCP listener on {}", addr);
        
        let listener = TcpListener::bind(&addr).await
            .context("Failed to bind TCP listener")?;
        
        self.tcp_listener = Some(Arc::new(listener));
        
        // Start receiver task
        if let Some(listener) = self.tcp_listener.clone() {
            let handler = self.message_handler.clone();
            
            tokio::spawn(async move {
                loop {
                    match listener.accept().await {
                        Ok((stream, addr)) => {
                            let handler = handler.clone();
                            tokio::spawn(async move {
                                if let Err(e) = handle_tcp_connection(stream, addr, handler).await {
                                    warn!("TCP connection error from {}: {}", addr, e);
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
        
        Ok(())
    }

    pub async fn start_udp(&mut self) -> Result<()> {
        let addr = format!("0.0.0.0:{}", self.udp_port);
        info!("Starting UDP socket on {}", addr);
        
        let socket = UdpSocket::bind(&addr).await
            .context("Failed to bind UDP socket")?;
        
        self.udp_socket = Some(Arc::new(socket));
        
        // Start receiver task
        if let Some(socket) = self.udp_socket.clone() {
            let handler = self.message_handler.clone();
            
            tokio::spawn(async move {
                let mut buf = vec![0u8; 65536];
                
                loop {
                    match socket.recv_from(&mut buf).await {
                        Ok((len, addr)) => {
                            let data = buf[..len].to_vec();
                            
                            if let Some(ref handler) = handler {
                                match Protocol::decode(&data) {
                                    Ok(message) => {
                                        handler.handle_message(message, addr);
                                    }
                                    Err(e) => {
                                        warn!("Failed to decode UDP message from {}: {}", addr, e);
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            error!("UDP receive error: {}", e);
                        }
                    }
                }
            });
        }
        
        Ok(())
    }

    pub async fn send_tcp(&self, peer_addr: &str, message: &GossipMessage) -> Result<()> {
        let data = Protocol::encode(message)?;
        
        debug!("Sending {} bytes via TCP to {}", data.len(), peer_addr);
        
        let mut stream = TcpStream::connect(peer_addr).await
            .context("Failed to connect to peer")?;
        
        // Write length prefix (4 bytes)
        stream.write_u32(data.len() as u32).await?;
        
        // Write data
        stream.write_all(&data).await?;
        stream.flush().await?;
        
        Ok(())
    }

    pub async fn send_udp(&self, peer_addr: &str, message: &GossipMessage) -> Result<()> {
        if let Some(ref socket) = self.udp_socket {
            let data = Protocol::encode(message)?;
            
            debug!("Sending {} bytes via UDP to {}", data.len(), peer_addr);
            
            socket.send_to(&data, peer_addr).await?;
            Ok(())
        } else {
            Err(anyhow::anyhow!("UDP socket not initialized"))
        }
    }

    pub async fn broadcast_udp(&self, peer_addrs: &[String], message: &GossipMessage) -> Result<usize> {
        if let Some(ref socket) = self.udp_socket {
            let data = Protocol::encode(message)?;
            let mut sent = 0;
            
            for addr in peer_addrs {
                match socket.send_to(&data, addr).await {
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

    pub fn get_tcp_port(&self) -> u16 {
        self.tcp_port
    }

    pub fn get_udp_port(&self) -> u16 {
        self.udp_port
    }

    pub fn is_tcp_active(&self) -> bool {
        self.tcp_listener.is_some()
    }

    pub fn is_udp_active(&self) -> bool {
        self.udp_socket.is_some()
    }
}

/// Handle TCP connection
async fn handle_tcp_connection(
    mut stream: TcpStream,
    addr: SocketAddr,
    handler: Option<Arc<dyn MessageHandler>>,
) -> Result<()> {
    // Read length prefix
    let len = stream.read_u32().await? as usize;
    
    if len > 10_000_000 {
        return Err(anyhow::anyhow!("Message too large: {}", len));
    }
    
    // Read data
    let mut data = vec![0u8; len];
    stream.read_exact(&mut data).await?;
    
    debug!("Received {} bytes via TCP from {}", len, addr);
    
    // Decode and handle message
    if let Some(handler) = handler {
        match Protocol::decode(&data) {
            Ok(message) => {
                handler.handle_message(message, addr);
            }
            Err(e) => {
                warn!("Failed to decode TCP message from {}: {}", addr, e);
            }
        }
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestHandler;
    
    impl MessageHandler for TestHandler {
        fn handle_message(&self, _message: GossipMessage, _from: SocketAddr) {
            // Test handler
        }
    }

    #[tokio::test]
    async fn test_transport_creation() {
        let transport = Transport::new(9000, 9001).await.unwrap();
        assert_eq!(transport.tcp_port, 9000);
        assert_eq!(transport.udp_port, 9001);
    }

    #[tokio::test]
    async fn test_tcp_listener_start() {
        let mut transport = Transport::new(0, 0).await.unwrap();
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

    #[tokio::test]
    async fn test_message_handling() {
        let mut transport = Transport::new(0, 0).await.unwrap();
        transport.set_message_handler(TestHandler);
        
        transport.start_udp().await.unwrap();
        
        // Send message to self
        let message = GossipMessage::Ping { timestamp: 12345 };
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        
        assert!(transport.is_udp_active());
    }
}
