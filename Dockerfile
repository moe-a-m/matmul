FROM rust:1.88-bookworm

RUN apt-get update && \
    apt-get install -y --no-install-recommends libblis-dev libblis-openmp-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# Optional backends: avx2 (x86-64 only) and blis. Note: target-cpu=native tunes the
# binary for the machine that builds the image.
ARG FEATURES="avx2,blis"
ENV FEATURES=${FEATURES}
ENV RUSTFLAGS="-C target-cpu=native"

RUN cargo build --release ${FEATURES:+--features "$FEATURES"}

ENTRYPOINT ["/bin/sh", "-c"]
CMD ["cargo run --release ${FEATURES:+--features \"$FEATURES\"}"]
