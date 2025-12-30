.PHONY: build build-node build-rpc build-native build-ts install-deps test clean help

help:
	@echo "Ade Sidechain Build System"
	@echo ""
	@echo "Available targets:"
	@echo "  build         - Build all components"
	@echo "  build-node    - Build node binary"
	@echo "  build-rpc     - Build RPC server"
	@echo "  build-native  - Build native C library"
	@echo "  build-ts      - Build TypeScript SDK"
	@echo "  install-deps  - Install all dependencies"
	@echo "  test          - Run all tests"
	@echo "  clean         - Clean build artifacts"

build: build-node build-rpc build-native build-ts

build-node:
	@echo "Building node..."
	cargo build --release -p ade-node

build-rpc:
	@echo "Building RPC server..."
	cargo build --release -p ade-rpc

build-native:
	@echo "Building native library..."
	cd native && cargo build --release

build-ts:
	@echo "Building TypeScript SDK..."
	npm install
	npm run build

install-deps:
	@echo "Installing Rust dependencies..."
	cargo fetch
	@echo "Installing Node.js dependencies..."
	npm install
	@echo "Installing Python dependencies..."
	cd python-sdk && pip install -e .

test:
	@echo "Running Rust tests..."
	cargo test --all
	@echo "Running TypeScript tests..."
	npm test

clean:
	@echo "Cleaning build artifacts..."
	cargo clean
	rm -rf dist/
	rm -rf node_modules/
	rm -rf python-sdk/build/
	rm -rf python-sdk/dist/
	rm -rf python-sdk/*.egg-info

