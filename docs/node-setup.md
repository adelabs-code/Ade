# Node Setup Guide

## System Requirements

### Minimum Specifications

- **CPU**: 12 cores / 24 threads
- **RAM**: 32 GB
- **Storage**: 500 GB NVMe SSD
- **Network**: 1 Gbps bandwidth
- **OS**: Linux (Ubuntu 20.04+ recommended)

### Recommended Specifications

- **CPU**: 16+ cores / 32+ threads
- **RAM**: 128 GB
- **Storage**: 2 TB NVMe SSD
- **Network**: 10 Gbps bandwidth

## Installation

### From Source

```bash
# Install dependencies
sudo apt update
sudo apt install -y build-essential pkg-config libssl-dev

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# Clone repository
git clone https://github.com/ade-sidechain/ade.git
cd ade

# Build
make build

# Verify installation
./target/release/ade-node --version
```

### Using Docker

```bash
# Pull image
docker pull adechain/node:latest

# Run node
docker run -d \
  --name ade-node \
  -p 8899:8899 \
  -p 9900:9900 \
  -v /data/ade:/data \
  adechain/node:latest
```

## Configuration

### Generate Validator Keypair

```bash
# Generate new keypair
ade-node generate-keypair --output validator-keypair.json

# View public key
ade-node show-pubkey --keypair validator-keypair.json
```

### Node Configuration File

Create `node-config.toml`:

```toml
[node]
rpc_port = 8899
gossip_port = 9900
data_dir = "./data"
validator_mode = true

[network]
bootstrap_nodes = [
    "bootstrap1.example.com:9900",
    "bootstrap2.example.com:9900"
]

[consensus]
min_stake = 100000
slot_duration_ms = 400

[storage]
prune_old_blocks = true
retention_days = 30
snapshot_interval_slots = 10000
```

## Running the Node

### Validator Mode

```bash
./target/release/ade-node \
  --config node-config.toml \
  --validator-keypair ./validator-keypair.json \
  --validator-mode
```

### Full Node Mode

```bash
./target/release/ade-node \
  --config node-config.toml
```

### RPC-Only Node

```bash
./target/release/ade-node \
  --rpc-port 8899 \
  --data-dir ./data \
  --bootstrap-nodes bootstrap1.example.com:9900
```

## Systemd Service

Create `/etc/systemd/system/ade-node.service`:

```ini
[Unit]
Description=Ade Sidechain Node
After=network.target

[Service]
Type=simple
User=ade
WorkingDirectory=/opt/ade
ExecStart=/opt/ade/target/release/ade-node \
  --config /opt/ade/node-config.toml \
  --validator-keypair /opt/ade/validator-keypair.json \
  --validator-mode
Restart=on-failure
RestartSec=10
LimitNOFILE=1000000

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable ade-node
sudo systemctl start ade-node
sudo systemctl status ade-node
```

## Monitoring

### Logs

```bash
# Follow logs
journalctl -u ade-node -f

# View recent logs
journalctl -u ade-node -n 100

# Filter by priority
journalctl -u ade-node -p err
```

### Metrics

Access metrics endpoint:

```bash
curl http://localhost:8899/metrics
```

Response:
```json
{
  "slot": 12345678,
  "transaction_count": 9876543,
  "validator_count": 42,
  "tps": 8543
}
```

### Health Check

```bash
curl http://localhost:8899/health
```

## Staking (Validator Mode)

### Register as Validator

```bash
# Stake tokens
ade-cli stake \
  --validator-keypair ./validator-keypair.json \
  --amount 100000 \
  --rpc-url http://localhost:8899
```

### Check Stake Status

```bash
ade-cli show-stake \
  --validator-pubkey $(ade-node show-pubkey --keypair validator-keypair.json) \
  --rpc-url http://localhost:8899
```

### Commission Settings

```bash
ade-cli set-commission \
  --validator-keypair ./validator-keypair.json \
  --commission 5 \
  --rpc-url http://localhost:8899
```

## Maintenance

### Update Node

```bash
# Pull latest code
git pull origin main

# Rebuild
make build

# Restart service
sudo systemctl restart ade-node
```

### Backup

```bash
# Backup keypair
cp validator-keypair.json validator-keypair.backup.json

# Backup data
tar -czf ade-data-backup-$(date +%Y%m%d).tar.gz data/
```

### Restore from Snapshot

```bash
# Download snapshot
wget https://snapshots.example.com/latest.tar.gz

# Extract
tar -xzf latest.tar.gz -C data/

# Start node
sudo systemctl start ade-node
```

## Troubleshooting

### Node Won't Start

Check logs:
```bash
journalctl -u ade-node -n 50
```

Common issues:
- Port already in use
- Insufficient permissions
- Missing keypair file
- Network connectivity

### Slow Sync

Enable snapshot:
```bash
./target/release/ade-node \
  --fast-sync \
  --snapshot-url https://snapshots.example.com/latest.tar.gz
```

### High Memory Usage

Adjust cache settings in `node-config.toml`:
```toml
[storage]
cache_size_mb = 2048
write_buffer_size_mb = 512
```

### Network Issues

Test connectivity:
```bash
# Test bootstrap nodes
nc -zv bootstrap1.example.com 9900

# Check firewall
sudo ufw status

# Open ports
sudo ufw allow 8899/tcp
sudo ufw allow 9900/tcp
```

## Security

### Firewall Configuration

```bash
# Allow RPC (if public)
sudo ufw allow 8899/tcp

# Allow gossip
sudo ufw allow 9900/tcp

# Enable firewall
sudo ufw enable
```

### Secure Keypair

```bash
# Set restrictive permissions
chmod 600 validator-keypair.json

# Encrypt keypair
gpg -c validator-keypair.json

# Store in secure location
mv validator-keypair.json.gpg /secure/location/
```

### SSL/TLS for RPC

Use a reverse proxy (nginx):

```nginx
server {
    listen 443 ssl;
    server_name rpc.example.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://localhost:8899;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Performance Tuning

### System Optimization

```bash
# Increase file descriptors
echo "* soft nofile 1000000" | sudo tee -a /etc/security/limits.conf
echo "* hard nofile 1000000" | sudo tee -a /etc/security/limits.conf

# TCP tuning
sudo sysctl -w net.core.rmem_max=134217728
sudo sysctl -w net.core.wmem_max=134217728
sudo sysctl -w net.ipv4.tcp_rmem="4096 87380 67108864"
sudo sysctl -w net.ipv4.tcp_wmem="4096 65536 67108864"
```

### RocksDB Tuning

In `node-config.toml`:
```toml
[storage]
max_open_files = 10000
write_buffer_size_mb = 512
max_background_jobs = 8
```

## Advanced Configuration

### Custom Genesis

```bash
ade-node init-genesis \
  --genesis-file genesis.json \
  --validators validators.json
```

### Network Fork

```bash
# Start from specific slot
ade-node \
  --fork-slot 12345678 \
  --expected-fork-hash abc123...
```

