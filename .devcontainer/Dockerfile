FROM nvcr.io/nvidia/pytorch:25.12-py3

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    g++ \
    gdb \
    clang-format \
    clang-tidy \
    cmake \
    make \
    git \
    pre-commit \
    nlohmann-json3-dev \
    && rm -rf /var/lib/apt/lists/*

# Get rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:$PATH"

# get uv 
RUN cargo install --locked uv 

# Install Python dependencies with uv
COPY requirements.txt /tmp/requirements.txt
RUN uv venv /opt/venv && \
    uv pip install --python /opt/venv/bin/python -r /tmp/requirements.txt
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /luminal

RUN echo "umask 000" >> /etc/bash.bashrc

CMD ["bash"]