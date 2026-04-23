default: check

check:
    cargo fmt --all -- --check
    cargo clippy --all-targets --all-features -- -D warnings

test:
    cargo test --all-targets --all-features

audit:
    cargo audit

install:
    cargo install --path .
