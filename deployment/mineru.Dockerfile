FROM vllm/vllm-openai:v0.21.0

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        fonts-noto-core \
        fonts-noto-cjk \
        fontconfig \
        libgl1 && \
    fc-cache -fv && \
    rm -rf /var/lib/apt/lists/*

ENV HF_HOME=/opt/mineru/huggingface \
    MINERU_TOOLS_CONFIG_JSON=/opt/mineru/mineru.json \
    MINERU_MODEL_SOURCE=local \
    HOME=/opt/mineru/runtime \
    XDG_CACHE_HOME=/opt/mineru/runtime/cache

RUN python3 -m pip install -U "mineru[core]==3.2.1" --break-system-packages && \
    python3 -m pip cache purge && \
    mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" && \
    mineru-models-download -s huggingface -m all && \
    chgrp -R 0 /opt/mineru && \
    chmod -R g=u /opt/mineru

ENTRYPOINT ["/bin/bash", "-c", "exec \"$@\"", "--"]
