FROM ghcr.io/tenstorrent/tt-xla/tt-xla-slim:979080cd242eb7eb1e31afee345d72aca4c389d2

RUN apt-get update && \
    apt-get install -y curl build-essential libblis-dev libblis-openmp-dev && \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
    rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app

COPY . .

ARG FEATURES=""

RUN if [ -n "$FEATURES" ]; then \
        cargo build --release --features "$FEATURES"; \
    else \
        cargo build --release; \
    fi

ENTRYPOINT ["/bin/sh", "-c"]
CMD ["cargo run --release -- --help"]
