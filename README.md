# Liquid AI Space Hackathon — FireEdge

Satellite onboard AI wildfire detection using LFM 2.5-VL-450M × Sentinel-2 SWIR imagery.

## Quick Start

```bash
git clone <this repo>
git submodule update --init   # SimSat (DPhi Space satellite simulator) を取得

# SimSat を起動
cd simsat && docker compose up -d && cd ..

# ヘルスチェック
curl http://localhost:9005/   # → {"message":"Simulation API is online"}
```

## Apps

- **FireEdge** (`apps/fireedge/`) — 煙透過型野火検知パイプライン

## License

This project is licensed under the [MIT License](LICENSE).

> **Note**: The `simsat/` submodule ([DPhi-Space/SimSat](https://github.com/DPhi-Space/SimSat)) is licensed under **AGPL-3.0** by DPhi Space.
