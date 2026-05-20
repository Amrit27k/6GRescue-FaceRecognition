# TCP Headers Dataset – Visualization & Analysis
## Project Description

 This dataset contains TCP header captures from a Jetson-based face recognition application and deployment, collected as part of the **6G-Rescue** project under **6G-PATH** to benchmark network behaviour. This dataset contains TCP header captures from a Jetson-based face recognition device, collected as part of the 6GRescue project to benchmark network behaviour across cloud (50 ms delay) and edge (7 ms delay) deployments. 6G-PATH project has received funding from the Smart Networks and Services Joint Undertaking (SNS JU) under the European Union's Horizon Europe research and innovation programme.


### Dataset Structure
This dataset contains network packet captures (`.pcap`) collected from a **Jetson-based edge AI device** running a face recognition inference workload, as part of the **6GRescue** project — a 6G-enabled emergency response system. The captures characterise TCP-layer behaviour across two deployment strategies and two network conditions, providing a controlled benchmark for evaluating the impact of network latency and workload type on real-time AI inference delivery.

| | **Type 1 — Container image pull** | **Type 2 — Build script + on-device build** |
|---|---|---|
| **Cloud deployment (50 ms delay)** | `tcp_headers_baseline_cloud_delay50ms_type1.pcap` | `tcp_headers_baseline_cloud_delay50ms_type2.pcap` |
| **Edge deployment (7 ms delay)** | `tcp_headers_baseline_edge_delay7ms.pcap` | `tcp_headers_baseline_edge_delay7ms_type2.pcap` |

- **Type 1**: The device downloads a complete pre-built container image (~135.5 MB) from the server — a sustained, high-volume bulk TCP transfer.
- **Type 2**: Only a lightweight build script (~15 kB) is sent to the device, which then compiles/builds locally — a brief initial TCP transfer followed by local processing silence.
- **Cloud (50 ms)**: Server at `10.70.0.71`, 2 routing hops from the client, with an artificially imposed 50 ms one-way delay simulating WAN/cloud latency.
- **Edge (7 ms)**: Server at `10.70.0.64`, 1 routing hop from the client, with a 7 ms one-way delay simulating edge deployment proximity.

These are evaluated under two **network conditions**:
- **Cloud deployment (50 ms one-way delay)**: inference served from a remote cloud server (`10.70.0.71`), simulating WAN latency typical of cloud-offloaded computation.
- **Edge deployment (7 ms one-way delay)**: inference served from a nearby edge server (`10.70.0.64`), simulating low-latency edge computing proximity.

All four captures use a **96-byte snapshot length (snaplen)**, capturing full TCP/IP/Ethernet headers but truncating payload data. Packet size values throughout this document reflect original wire-length frames, not captured bytes. The dataset is intended for research into TCP performance modelling, edge vs. cloud offloading trade-offs, and network-aware deployment optimisation for AI workloads.

---

## Dataset 1: `tcp_headers_baseline_cloud_delay50ms_type1.pcap`

### Overview

| Property | Value |
|---|---|
| **File size** | 1056 kB |
| **Format** | Wireshark/tcpdump pcap |
| **Encapsulation** | Ethernet |
| **Snapshot length (snaplen)** | 96 bytes |
| **Capture start** | 2026-04-01 17:08:20 |
| **Capture end** | 2026-04-01 17:09:07 |
| **Elapsed time** | 46.849 seconds |
| **Total packets** | 10,644 |
| **Total bytes** | 142,623,017 (~143 MB) |
| **Average packet size** | 13,399 bytes |
| **Average packet rate** | 227.2 packets/sec |
| **Average throughput** | ~24 Mbit/s (3,044 kB/s) |

> **Note:** The snapshot length of 96 bytes means only the first 96 bytes of each packet were captured. This is sufficient to capture full TCP/IP headers but truncates payload data, which explains the very large reported average packet size — the size values reflect the *original* packets on the wire, not the captured bytes.

---

### Endpoints

Two IPv4 endpoints are involved in this capture, communicating exclusively with each other:

| Address | Role | Tx Packets | Tx Bytes | Rx Packets | Rx Bytes |
|---|---|---|---|---|---|
| `10.70.0.71` | Cloud/Server | 6,186 | ~142 MB | 4,458 | ~305 kB |
| `192.168.2.100` | Local/Client | 4,458 | ~305 kB | 6,186 | ~142 MB |

The highly asymmetric traffic pattern confirms a **download-dominant flow**: the server (`10.70.0.71`) transmits the bulk of the data (~142 MB), while the client (`192.168.2.100`) sends only ~305 kB (primarily ACKs and control packets).

---

### Source TTLs (IPv4)

| Source IP | TTL | Destination | Packet Count | Share |
|---|---|---|---|---|
| `10.70.0.71` | 62 | `192.168.2.100` | 6,186 | 58.12% |
| `192.168.2.100` | 64 | `10.70.0.71` | 4,458 | 41.88% |

The TTL of **64** from `192.168.2.100` indicates a typical Linux/macOS origin (default TTL = 64, no hops consumed). The TTL of **62** from `10.70.0.71` suggests 2 routing hops between the server and the capture point, consistent with a cloud endpoint traversing a small number of network segments.

---

### Packet Length Distribution

| Size Bucket | Count | Share | Avg Size |
|---|---|---|---|
| 40–79 bytes | 4,540 | 42.65% | 66.1 B |
| 5120+ bytes | 4,942 | 46.43% | 28,155 B |
| 2560–5119 bytes | 820 | 7.70% | 3,303 B |
| 1280–2559 bytes | 267 | 2.51% | 1,586 B |
| 640–1279 bytes | 33 | 0.31% | 947 B |
| 320–639 bytes | 30 | 0.28% | 433 B |
| 160–319 bytes | 12 | 0.11% | 253 B |

The distribution is strongly **bimodal**:

- **Small packets (40–79 B, 42.65%)** — These are almost exclusively TCP ACKs and control packets (SYN, FIN, RST) from the client side. The 66-byte minimum aligns with a standard TCP ACK on Ethernet (14 B Ethernet + 20 B IP + 20 B TCP + no payload = 54 B, with 12 B of padding reaching 66 B minimum frame size).
- **Large packets (5120+ B, 46.43%)** — These represent large-segment data frames from the server. The average of ~28 kB and a maximum of 65,226 bytes indicate **TCP segmentation offload (TSO)** or **Generic Segmentation Offload (GSO)** is active, allowing the kernel to hand oversized logical segments to the NIC, which splits them before transmission.

Together these two buckets account for ~89% of all packets, with mid-range sizes representing a small minority.

---

### I/O Graph

The I/O graph plots packet rate (packets per 1-second interval) across the 46-second capture window.

- Packet rates reach up to approximately **2,500–3,000 packets/sec** at peak intervals.
- Activity is largely **continuous** throughout the capture, with no obvious long idle gaps.
- The average rate of 227.2 pps is consistent with sustained data transfer punctuated by ACK traffic.
- The burst rate of **6.8 packets/ms** (6,800 pps) recorded starting at t=1.165 s corresponds to the initial burst at connection ramp-up.

---

### Frame Number vs. Time (Sequence Plot)

The frame-vs-time and sequence-number plots reveal TCP flow behavior:

- **Frame numbers** increase linearly with time, confirming a near-constant capture rate with no major gaps or retransmission storms.
- **Sequence numbers** show a steady, near-linear climb, consistent with a single long-running bulk TCP data stream with no significant reordering or large retransmission events.
- The imposed **50 ms artificial delay** (as indicated by the filename `delay50ms`) is reflected in the RTT characteristics of the flow.

---

### Throughput

The throughput graph for the stream `10.70.0.71:5000 → 192.168.2.100:53736` shows:

- **Segment lengths** in the range of ~20–100 bytes at the granularity plotted — note this reflects the *captured* (truncated) segment length due to the 96-byte snaplen, not the full wire-length payload.
- **Average throughput** in the range of ~200–800 bits/sec at the plotted time window — again, this reflects captured bytes only and should not be interpreted as the true wire throughput (which is ~24 Mbit/s as shown in capture properties).
- The flow corresponds to TCP source port **5000** (server application) to ephemeral client port **53736**.

---

### Key Observations

1. **Header-only capture**: With snaplen=96, this dataset captures TCP/IP/Ethernet headers and the first few bytes of payload. It is well-suited for studying TCP behavior (flags, sequence numbers, ACK patterns, window sizes, options) but does not contain full application payloads.

2. **Artificial 50 ms delay**: The filename explicitly labels this as a `delay50ms` scenario. This simulates WAN or cloud latency and will affect TCP congestion window growth, ACK timing, and overall goodput relative to a zero-delay baseline.

3. **Asymmetric bulk transfer (container image pull)**: The capture represents a large one-directional data transfer (~143 MB server→client), consistent with a full container image pull (~135.5 MB payload). The client generates only acknowledgment traffic in return.

4. **TSO/GSO active**: Large frames exceeding standard MTU (1500 B) suggest hardware/software offloading on the sending side, a common configuration on Linux servers.

---

---

## Dataset 2: `tcp_headers_baseline_cloud_delay50ms_type2.pcap`

### Overview

| Property | Value |
|---|---|
| **File size** | 777 kB |
| **Format** | Wireshark/tcpdump pcap |
| **Encapsulation** | Ethernet |
| **Snapshot length (snaplen)** | 96 bytes |
| **Capture start** | 2026-04-01 16:59:12 |
| **Capture end** | 2026-04-01 17:00:35 |
| **Elapsed time** | 83.357 seconds |
| **Total packets** | 7,773 |
| **Total bytes** | 105,440,670 (~105 MB) |
| **Average packet size** | 13,565 bytes |
| **Average packet rate** | 93.2 packets/sec |
| **Average throughput** | ~10 Mbit/s (1,264 kB/s) |

> **Note:** As with Dataset 1, the 96-byte snaplen means only headers are captured. Packet size values reflect original wire-length frames, not the truncated captured bytes.

---

### Endpoints

| Address | Role | Tx Packets | Tx Bytes | Rx Packets | Rx Bytes |
|---|---|---|---|---|---|
| `10.70.0.71` | Cloud/Server | 4,727 | ~105 MB | 3,046 | ~208 kB |
| `192.168.2.100` | Local/Client | 3,046 | ~208 kB | 4,727 | ~105 MB |

The same two endpoints as Dataset 1 are present, again with a strongly download-dominant flow. The server transmits ~105 MB while the client returns only ~208 kB of acknowledgment traffic.

---

### Source TTLs (IPv4)

| Source IP | TTL | Destination | Packet Count | Share |
|---|---|---|---|---|
| `10.70.0.71` | 62 | `192.168.2.100` | 4,726 | 60.79% |
| `10.70.0.71` | 63 | `192.168.2.100` | 1 | 0.01% |
| `192.168.2.100` | 64 | `10.70.0.71` | 3,046 | 39.19% |

TTL values are consistent with Dataset 1: the client originates at TTL 64 (Linux/macOS default), and the server arrives at TTL 62 (2 hops consumed). One anomalous packet from `10.70.0.71` arrived with TTL 63 at t=83.357 s — this single packet likely took a slightly different routing path at the very end of the session, or represents a brief routing change.

---

### Packet Length Distribution

| Size Bucket | Count | Share | Avg Size |
|---|---|---|---|
| 40–79 bytes | 3,098 | 39.86% | 66.1 B |
| 5120+ bytes | 3,708 | 47.70% | 27,620 B |
| 2560–5119 bytes | 754 | 9.70% | 3,343 B |
| 1280–2559 bytes | 172 | 2.21% | 1,588 B |
| 640–1279 bytes | 17 | 0.22% | 1,016 B |
| 320–639 bytes | 18 | 0.23% | 429 B |
| 160–319 bytes | 6 | 0.08% | 249 B |

The bimodal pattern from Dataset 1 is preserved:

- **Small packets (40–79 B, 39.86%)** — TCP ACKs and control frames from the client, essentially identical in character to Dataset 1.
- **Large packets (5120+ B, 47.70%)** — TSO/GSO data frames from the server, slightly higher share than Dataset 1 (47.70% vs 46.43%), with a similar average size (~27.6 kB vs ~28.2 kB).

The proportions are very close to Dataset 1, indicating the same underlying TCP/application stack behaviour. The lower total packet count (7,773 vs 10,644) reflects the lower overall throughput of this capture.

---

### I/O Graph

The I/O graph plots packet rate over the full 83-second capture window.

- Peak packet rates reach approximately **2,000–2,500 packets/sec**, somewhat lower than Dataset 1's peaks of ~3,000 pps.
- Activity appears **continuous** throughout with no significant idle periods, consistent with a sustained bulk transfer.
- The burst rate of **5.47 packets/ms** (5,470 pps) at t=1.190 s reflects the same connection ramp-up pattern seen in Dataset 1.
- The longer capture duration with lower sustained rate accounts for the reduced total throughput (~10 Mbit/s vs ~24 Mbit/s).

---

### Frame Number vs. Time (Sequence Plot)

- **Frame numbers** increase linearly with time across the ~83-second window, indicating steady continuous capture.
- **Sequence numbers** show a near-linear increase up to approximately frame 8,000, consistent with a single bulk TCP stream with no major retransmissions.
- The time axis extends to ~100 s (with data from roughly -25 s to 100 s relative to an internal reference point), slightly wider than Dataset 1's 46-second window.
- The 50 ms artificial delay is the same as Dataset 1, so RTT-driven behavior (ACK pacing, congestion window dynamics) should be comparable.

---

### Throughput

The throughput graph for the stream `10.70.0.71:5000 → 192.168.2.100:53576` shows:

- **Segment lengths** in the same ~20–100 byte range as Dataset 1, reflecting the 96-byte snaplen truncation.
- **Average throughput** similarly in the ~200–800 bits/sec range at the plotted granularity — interpreted as captured bytes only, not true wire throughput.
- Client ephemeral port is **53576**, differing from Dataset 1's port 53736, confirming these are two separate TCP sessions.

---

### Key Observations & Comparison with Dataset 1

| Metric | Dataset 1 (type1) | Dataset 2 (type2) |
|---|---|---|
| Duration | 46.8 s | 83.4 s |
| Total packets | 10,644 | 7,773 |
| Total bytes (wire) | ~143 MB | ~105 MB |
| Avg throughput | ~24 Mbit/s | ~10 Mbit/s |
| Avg packet rate | 227.2 pps | 93.2 pps |
| Large packet share | 46.43% | 47.70% |
| ACK packet share | 42.65% | 39.86% |
| Server TTL | 62 | 62 (one anomalous 63) |
| Client port | 53736 | 53576 |

1. **Lower throughput, longer duration**: Type2 runs nearly twice as long as Type1 but transfers less total data, resulting in roughly 2.4× lower average throughput. This is the most significant difference between the two captures.

2. **Same network path**: Identical TTL values and endpoints confirm the same physical/logical network path was used for both captures. The conditions differ in workload, not topology.

3. **Consistent protocol behaviour**: Packet size distributions, TSO/GSO signatures, and ACK ratios are nearly identical, suggesting the same application and OS stack for both captures.

4. **Separate TCP sessions**: Different client ephemeral ports (53736 vs 53576) confirm these are independent TCP connections, not a continuation of the same session.

> **On "type1" vs "type2"**: The two types represent fundamentally different **deployment strategies** for the face recognition workload:
> - **Type 1 — Pull full container images**: The device downloads complete pre-built container images (~135.5 MB) from the server. This produces a sustained, high-volume bulk TCP transfer, which is exactly what the ~143 MB, ~24 Mbit/s, 47-second capture reflects.
> - **Type 2 — Send build script, build on-device**: Only a lightweight build script (~15 kB) is transferred to the device, which then compiles/builds locally. The TCP transfer is minimal (only ~105 MB captured, but the bulk of that is likely unrelated data or prior state); the defining characteristic is a **very short initial transfer** followed by local processing silence. The longer capture duration (83 s) at lower throughput (~10 Mbit/s) is consistent with a brief data transfer plus lingering connection overhead.

---

---

## Dataset 3: `tcp_headers_baseline_edge_delay7ms.pcap`

### Overview

| Property | Value |
|---|---|
| **File size** | 818 kB |
| **Format** | Wireshark/tcpdump pcap |
| **Encapsulation** | Ethernet |
| **Snapshot length (snaplen)** | 96 bytes |
| **Capture start** | 2026-04-01 13:30:37 |
| **Capture end** | 2026-04-01 13:31:16 |
| **Elapsed time** | 39.072 seconds |
| **Total packets** | 8,322 |
| **Total bytes** | 142,450,730 (~142 MB) |
| **Average packet size** | 17,117 bytes |
| **Average packet rate** | 213.0 packets/sec |
| **Average throughput** | ~29 Mbit/s (3,645 kB/s) |

This capture differs from Datasets 1 and 2 in two important ways: it uses an **edge server** (`10.70.0.64`) rather than a cloud server (`10.70.0.71`), and the imposed artificial delay is only **7 ms** instead of 50 ms. The combination of a closer edge node and lower RTT produces noticeably higher throughput despite a similar packet count and transfer size.

> **Note:** The 96-byte snaplen is consistent across all captures. Packet size values reflect original wire-length frames.

---

### Endpoints

| Address | Role | Tx Packets | Tx Bytes | Rx Packets | Rx Bytes |
|---|---|---|---|---|---|
| `10.70.0.64` | Edge/Server | 4,574 | ~142 MB | 3,748 | ~257 kB |
| `192.168.2.100` | Local/Client | 3,748 | ~257 kB | 4,574 | ~142 MB |

The same client (`192.168.2.100`) is present but the server IP has changed from `10.70.0.71` (cloud) to `10.70.0.64` (edge). The traffic pattern remains strongly download-dominant: ~142 MB server→client vs ~257 kB client→server. The TCP endpoint count dropped to **25 connections** (vs 30 for type1 cloud), which may reflect a smaller or more efficiently served workload at the edge.

---

### Source TTLs (IPv4)

| Source IP | TTL | Destination | Packet Count | Share |
|---|---|---|---|---|
| `10.70.0.64` | 63 | `192.168.2.100` | 4,574 | 54.96% |
| `192.168.2.100` | 64 | `10.70.0.64` | 3,748 | 45.04% |

The client TTL of **64** is unchanged (Linux/macOS default, 0 hops consumed at capture point). The edge server arrives with TTL **63**, indicating only **1 routing hop** between the edge node and the capture point — compared to TTL 62 (2 hops) for the cloud server. This is consistent with an edge server being topologically closer to the client, as expected in an edge computing deployment.

---

### Packet Length Distribution

| Size Bucket | Count | Share | Avg Size |
|---|---|---|---|
| 40–79 bytes | 3,815 | 45.84% | 66.1 B |
| 5120+ bytes | 3,995 | 48.01% | 35,230 B |
| 2560–5119 bytes | 348 | 4.18% | 3,581 B |
| 1280–2559 bytes | 110 | 1.32% | 1,566 B |
| 640–1279 bytes | 25 | 0.30% | 981 B |
| 320–639 bytes | 23 | 0.28% | 421 B |
| 160–319 bytes | 6 | 0.07% | 249 B |

The bimodal distribution is again preserved, but with a notable difference in the large-packet bucket:

- **Small packets (40–79 B, 45.84%)** — TCP ACKs and control frames from the client. The share is slightly higher than in both cloud captures (42.65% and 39.86%), consistent with the lower RTT of 7 ms allowing ACKs to return faster and be captured more densely.
- **Large packets (5120+ B, 48.01%)** — TSO/GSO data frames from the edge server. The **average size here is ~35.2 kB**, markedly larger than the cloud captures (~28.2 kB for type1, ~27.6 kB for type2). With a much lower RTT, TCP's congestion window can grow larger before being constrained, allowing the sender to push bigger logical segments before needing to pause for ACKs.

---

### I/O Graph

The I/O graph covers the ~39-second capture window.

- Peak packet rates reach approximately **3,000–5,000 packets/sec**, noticeably higher than the cloud captures (which peaked at ~2,500–3,000 pps).
- The burst rate of **8.78 packets/ms** (8,780 pps) at t=0.285 s is the highest initial burst across all three datasets, reflecting the faster RTT allowing TCP slow-start to ramp up more aggressively.
- Activity appears continuous with no idle gaps, consistent with a sustained bulk transfer completing in ~39 seconds.

---

### Frame Number vs. Time (Sequence Plot)

- **Frame numbers** increase linearly, confirming steady capture throughout the 39-second window.
- **Sequence numbers** climb steeply and linearly, but the plot notably shows a **negative-time region** (frames appearing from roughly -2,500 to 0 relative to the first data point). This suggests Wireshark's reference point was set mid-capture, or that the sequence number axis captured a reset/retransmit near the start. The main bulk of the transfer progresses cleanly.
- The steeper slope of sequence numbers relative to time (compared to the cloud captures) is consistent with higher throughput: more bytes transferred per unit time.

---

### Throughput

The throughput graph for the stream `10.70.0.64:5000 → 192.168.2.100:46446` shows:

- **Segment lengths** in the same ~20–100 byte snaplen-limited range as the cloud captures.
- **Average throughput** similarly in the ~200–800 bits/sec plotted range (captured bytes only — true wire throughput is ~29 Mbit/s).
- Client ephemeral port is **46446**, a new independent TCP session distinct from all cloud captures.
- The time axis extends only to ~22.5 ms in the throughput plot window, reflecting the much shorter per-ACK cycle time at 7 ms RTT compared to 50 ms RTT.

---

### Key Observations & Cross-Dataset Comparison

| Metric | Cloud type1 (50ms) | Cloud type2 (50ms) | Edge (7ms) |
|---|---|---|---|
| Server IP | `10.70.0.71` | `10.70.0.71` | `10.70.0.64` |
| Delay | 50 ms | 50 ms | 7 ms |
| Duration | 46.8 s | 83.4 s | 39.1 s |
| Total packets | 10,644 | 7,773 | 8,322 |
| Total bytes (wire) | ~143 MB | ~105 MB | ~142 MB |
| Avg throughput | ~24 Mbit/s | ~10 Mbit/s | **~29 Mbit/s** |
| Avg packet rate | 227.2 pps | 93.2 pps | 213.0 pps |
| Large packet avg size | ~28.2 kB | ~27.6 kB | **~35.2 kB** |
| Large packet share | 46.43% | 47.70% | 48.01% |
| Server TTL (hops) | 62 (2 hops) | 62 (2 hops) | **63 (1 hop)** |
| TCP connections | 30 | 21 | 25 |
| Client port | 53736 | 53576 | 46446 |

1. **RTT reduction improves throughput significantly**: Reducing artificial delay from 50 ms to 7 ms increases average throughput from ~24 Mbit/s to ~29 Mbit/s for a comparable transfer size (~142–143 MB), a ~21% improvement. The improvement would likely be larger still for a zero-delay baseline.

2. **Lower RTT enables larger TCP segments**: The average large-packet size of ~35.2 kB at 7 ms RTT is ~25% larger than the ~28 kB seen at 50 ms RTT. With less time spent waiting for ACKs, TCP's congestion window grows higher before stabilising, allowing the sender to pipeline more data per burst.

3. **Edge server is one hop closer**: TTL 63 vs 62 from the server confirms the edge node is one routing hop nearer to the client than the cloud node, directly contributing to the lower RTT.

4. **Workload is type1 equivalent**: The ~142 MB transfer size closely matches Dataset 1 (type1 cloud), confirming this is also a full container image pull scenario, just served from the edge instead of the cloud.

---

---

## Dataset 4: `tcp_headers_baseline_edge_delay7ms_type2.pcap`

### Overview

| Property | Value |
|---|---|
| **File size** | 568 kB |
| **Format** | Wireshark/tcpdump pcap |
| **Encapsulation** | Ethernet |
| **Snapshot length (snaplen)** | 96 bytes |
| **Capture start** | 2026-04-01 14:57:43 |
| **Capture end** | 2026-04-01 14:58:06 |
| **Elapsed time** | 22.729 seconds |
| **Total packets** | 5,806 |
| **Total bytes** | 105,310,257 (~105 MB) |
| **Average packet size** | 18,138 bytes |
| **Average packet rate** | 255.4 packets/sec |
| **Average throughput** | ~37 Mbit/s (4,633 kB/s) |

This is the **edge + type2** capture: the build script deployment strategy (~15 kB initial transfer, then on-device build) served from the edge node at 7 ms delay. It is the shortest and fastest capture in the dataset, and achieves the **highest average throughput** of all four captures.

---

### Endpoints

| Address | Role | Tx Packets | Tx Bytes | Rx Packets | Rx Bytes |
|---|---|---|---|---|---|
| `10.70.0.64` | Edge/Server | 3,103 | ~105 MB | 2,703 | ~185 kB |
| `192.168.2.100` | Local/Client | 2,703 | ~185 kB | 3,103 | ~105 MB |

The same edge server (`10.70.0.64`) and client (`192.168.2.100`) as Dataset 3. The download-dominant pattern holds, with the client sending only ~185 kB (ACKs and control frames). TCP connection count dropped to **19**, the lowest across all four datasets, consistent with the type2 workload's minimal initial transfer.

---

### Source TTLs (IPv4)

| Source IP | TTL | Destination | Packet Count | Share |
|---|---|---|---|---|
| `10.70.0.64` | 63 | `192.168.2.100` | 3,103 | 53.44% |
| `192.168.2.100` | 64 | `10.70.0.64` | 2,703 | 46.56% |

TTL values are identical to Dataset 3 (edge type1): server TTL 63 (1 hop), client TTL 64 (Linux/macOS default, 0 hops). No anomalous TTL values, confirming a stable routing path throughout the capture.

---

### Packet Length Distribution

| Size Bucket | Count | Share | Avg Size |
|---|---|---|---|
| 40–79 bytes | 2,752 | 47.40% | 66.1 B |
| 5120+ bytes | 2,708 | 46.64% | 38,456 B |
| 2560–5119 bytes | 226 | 3.89% | 3,714 B |
| 1280–2559 bytes | 83 | 1.43% | 1,532 B |
| 640–1279 bytes | 13 | 0.22% | 1,010 B |
| 320–639 bytes | 17 | 0.29% | 424 B |
| 160–319 bytes | 7 | 0.12% | 241 B |

The bimodal distribution is again present, with a striking characteristic: the **average large-packet size of ~38.5 kB** is the highest across all four datasets. This is the edge + low-RTT effect taken further — with only 7 ms RTT and fewer total connections, TCP can sustain an even larger congestion window, allowing the server to send longer logical segments per burst.

- **Small packets (40–79 B, 47.40%)** — the highest ACK share of all four datasets. With fewer total packets (5,806) and a higher proportion of the transfer being acknowledgment-driven, the ACK fraction is naturally elevated.
- **Large packets (5120+ B, 46.64%)** — slightly lower share by count than other captures, but the dramatically higher average segment size means these packets carry substantially more data per frame.

---

### I/O Graph

The I/O graph covers the ~22-second capture window — the shortest window in the dataset.

- Peak packet rates reach approximately **2,500–3,000 packets/sec**, similar in magnitude to the cloud type1 capture.
- The burst rate of **7.32 packets/ms** (7,320 pps) at t=0.280 s is high, consistent with the edge deployment's fast ramp-up.
- Despite the shorter window, the transfer completes ~105 MB, reflecting the ~37 Mbit/s average throughput.
- Activity is fully continuous with no idle periods — the entire capture is active transfer.

---

### Frame Number vs. Time (Sequence Plot)

- **Frame numbers** increase linearly across the ~22-second window with no gaps.
- **Sequence numbers** show a steady near-linear climb, reaching approximately 6,000 frames, consistent with a clean bulk transfer and no significant retransmissions.
- The time axis extends to ~30 s in the plot (slightly beyond the actual capture end), with all data concentrated in the 0–22 s window.
- The slope of sequence numbers per unit time is the steepest of all four captures, directly reflecting the highest throughput in the dataset.

---

### Throughput

The throughput graph for the stream `10.70.0.64:5000 → 192.168.2.100:47270` shows:

- **Segment lengths** in the ~20–100 byte snaplen-limited range, consistent with all other captures.
- **Average throughput** in the ~200–800 bits/sec plotted range (captured bytes only — true wire throughput is ~37 Mbit/s).
- Client ephemeral port **47270** confirms this is a new independent TCP session.
- The throughput time axis again extends only to ~22.5 ms per window, reflecting the 7 ms RTT ACK cycle.

---

### Key Observations & Full Cross-Dataset Summary

| Metric | Cloud T1 (50ms) | Cloud T2 (50ms) | Edge T1 (7ms) | Edge T2 (7ms) |
|---|---|---|---|---|
| File | `cloud_50ms_type1` | `cloud_50ms_type2` | `edge_7ms` | `edge_7ms_type2` |
| Duration | 46.8 s | 83.4 s | 39.1 s | **22.7 s** |
| Total packets | 10,644 | 7,773 | 8,322 | **5,806** |
| Total bytes (wire) | ~143 MB | ~105 MB | ~142 MB | ~105 MB |
| Avg throughput | ~24 Mbit/s | ~10 Mbit/s | ~29 Mbit/s | **~37 Mbit/s** |
| Avg packet rate | 227.2 pps | 93.2 pps | 213.0 pps | **255.4 pps** |
| Large pkt avg size | ~28.2 kB | ~27.6 kB | ~35.2 kB | **~38.5 kB** |
| ACK share | 42.65% | 39.86% | 45.84% | **47.40%** |
| Server TTL (hops) | 62 (2 hops) | 62 (2 hops) | 63 (1 hop) | 63 (1 hop) |
| TCP connections | 30 | 21 | 25 | **19** |
| Client port | 53736 | 53576 | 46446 | 47270 |

**Effect of deployment location (Cloud vs Edge, same workload type):**

- Type1: Edge achieves ~29 Mbit/s vs Cloud's ~24 Mbit/s — a **~21% throughput gain** from reducing RTT from 50 ms to 7 ms.
- Type2: Edge achieves ~37 Mbit/s vs Cloud's ~10 Mbit/s — a **~270% throughput gain**. The type2 workload is far more sensitive to RTT because its smaller transfer size means TCP never fully exits slow-start at 50 ms delay; at 7 ms, it ramps up fast enough to saturate the link before the transfer ends.

**Effect of workload type (Type1 vs Type2, same deployment):**

- Cloud: Type2 is ~2.4× slower than Type1 despite transferring ~26% less data. The 50 ms RTT severely limits TCP ramp-up for a shorter transfer.
- Edge: Type2 is ~28% *faster* than Type1. At 7 ms RTT, TCP reaches high throughput quickly enough that the shorter type2 transfer actually benefits — it captures only the high-throughput phase without the tail-off seen in longer type1 captures.

**Largest TCP segments at edge + low RTT:** The progression of average large-packet size (~28 kB → ~28 kB → ~35 kB → ~38.5 kB) across cloud-T1, cloud-T2, edge-T1, edge-T2 directly tracks the combination of lower RTT and fewer concurrent connections, both of which allow the TCP congestion window to grow larger.

---
*README created for Zenodo dataset. Last updated: 2026-05-20.*
