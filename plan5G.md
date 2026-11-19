# 5G Network KPI Forecasting & Capacity-Risk Monitoring
## Master Plan v6.1 — 8 Weeks · High-Impact Portfolio Edition · Solo

| Hạng mục | Quyết định |
|---|---|
| Hình thức | Solo project, trình độ intermediate |
| Kế hoạch chính thức | **8 tuần × 12 giờ/tuần = 96 giờ** |
| Trọng tâm nâng cấp | Tuần 5–8: Model Quality → Final Evaluation → Product Engineering → Portfolio Release |
| Dataset | NETOP GOTS-F (`netop/gotsf-ds`) |
| Target | DLPRB |
| Input KPI | DLPRB, DLThpVol, DLThpTime, MR_number |
| Lookback | 168 giờ |
| Horizons | +1h, +6h, +24h |
| Models | 3 baselines, Global LightGBM, GRU-A, GRU-B |
| Training & evaluation | **100% trên Google Colab/Kaggle**; local chỉ coding, unit tests tổng hợp và demo CPU inference |
| Product | Inference pipeline, FastAPI, Streamlit, Docker Compose |
| Tổng giờ | **96 giờ / 8 tuần**, đã gồm buffer |
| Nguyên tắc | Không làm quá 12 giờ/tuần; không thêm model ngoài scope |

> **Scope freeze:** Giữ nguyên mục tiêu và ngân sách từng task của tuần 1–4 từ v5.0 (48 giờ), **đồng bộ sửa lỗi static/scaler và GPU trước lúc train ở W3–W4**, không chờ đến W5 để retrain. Nâng cấp riêng tuần 5–8 (48 giờ). Không vượt 12 giờ/tuần. **Điều kiện:** Các gate G1–G4 đã pass; nếu chưa, dùng thời gian tuần 5 để sửa, cắt nâng cấp optional. Không thể cam kết deadline nếu data/GPU liên tục không khả dụng.

---

# 0. Mục tiêu, phạm vi và tiêu chí thành công

## 0.1. Mục tiêu dự án

Xây dựng prototype dự báo KPI mạng 5G ở cấp beam, tập trung vào DLPRB, sau đó chuyển dự báo thành các chỉ báo theo dõi khả năng vượt ngưỡng thống kê lịch sử.

Dự án phải chứng minh được ba nhóm năng lực:

1. **Applied ML:** panel time-series forecasting, feature engineering, baselines, LightGBM, GRU, multi-horizon forecasting.
2. **Experimental rigor:** chronological split, leakage prevention, rolling-origin evaluation, model selection bằng validation, final holdout độc lập, bootstrap CI và error analysis.
3. **ML Engineering:** pipeline có thể tái lập, model artifacts, inference độc lập notebook, API, dashboard, Docker và tests.

## 0.2. Deliverables của bản 8 tuần

- Data download, manifest, panel builder và data audit.
- Data dictionary và temporal split decision.
- Ba forecasting baselines.
- Global LightGBM cho ba horizons.
- GRU-A (univariate) và GRU-B (multivariate); **mỗi model 1 seed là P0**, seed thứ hai là P1 nếu còn giờ/cloud GPU.
- Validation benchmark, ablation có giới hạn và **common evaluation mask** cho các model.
- Champion selection chỉ bằng validation.
- Final refit và holdout evaluation.
- Per-beam P90/P95 threshold-exceedance engine.
- Risk evaluation so với baselines.
- Independent inference pipeline.
- FastAPI với các endpoint cốt lõi.
- Streamlit dashboard một trang, có overview và forecast detail.
- Docker Compose.
- Unit tests, integration tests và leakage tests.
- **Error-case gallery**, model comparison, experiment report, model card, architecture diagram, README, screenshots và **demo video ngắn**.
- **Quality assurance**: clean clone, demo assets manifest, inference latency đo trên máy cụ thể, model/data provenance.
- **P1 có điều kiện:** LightGBM P10/P90 quantile forecasting để minh họa prediction bands và kiểm tra empirical coverage; không claim calibrated coverage nếu chưa chứng minh.

## 0.3. Claims được phép

Có thể tuyên bố:
- Global beam-level multivariate time-series forecasting.
- Multi-horizon DLPRB forecasting.
- Historical threshold-exceedance monitoring.
- Offline historical replay.
- So sánh model trên holdout theo protocol đã công bố.

Không được tuyên bố:
- Spatial-temporal forecasting hoặc GNN.
- Real-time network monitoring.
- Fault detection hoặc incident prediction.
- Congestion prediction đã được xác thực.
- Ngưỡng P90/P95 là ngưỡng vận hành chính thức của nhà mạng.
- DLPRB utilization (%) nếu đơn vị chưa được xác minh.

## 0.4. Definition of Done chung

Một task chỉ được đánh dấu Done khi:
- Code nằm trong đúng module, không viết toàn bộ logic trong notebook.
- Có type hints và docstring ngắn nêu input/output.
- Config được đọc từ `configs/`, không hard-code đường dẫn máy cá nhân.
- Có test trên fixture nhỏ hoặc dữ liệu thật phù hợp.
- Output được lưu đúng đường dẫn quy định.
- Có log/config/seed/dataset revision/git commit khi task tạo kết quả thực nghiệm.
- `pytest` và `ruff check` chạy được.
- Đã commit Git.

**Quy tắc workload:** 12 giờ là trần làm việc mỗi tuần. Nếu task trễ, dùng thứ tự cắt scope ở **Mục 10**; không bù bằng cách làm 15–18 giờ tuần kế tiếp.

---

# 1. Dataset và data policy

## 1.1. Dataset dự kiến

Dataset: Hugging Face `netop/gotsf-ds`, repository code `netop-team/gotsf`.

Theo dataset card cần xác minh lại trên file thực tế:
- 2.880 beams = 30 BS × 3 cell × 32 beam.
- Sampling hourly.
- Train file có 840 giờ.
- Test 5w–6w có 168 giờ.
- Test 10w–11w có 168 giờ.
- Bốn KPI: `DLPRB`, `DLThpVol`, `DLThpTime`, `MR_number`.

Không coi các con số trên là đúng cho đến khi audit xong.

## 1.2. Panel format

Nếu dữ liệu khớp dataset card:

```python
values.shape == (1008, 2880, 4)
```

Thứ tự KPI:

```python
KPIS = ["DLPRB", "DLThpVol", "DLThpTime", "MR_number"]
```

Beam ID phải được sắp xếp theo thứ tự số `(bs, cell, beam)`, không sort lexical. Lưu `week11` riêng, không nối vào panel chính.

## 1.3. Missing và zero policy

- Giữ NaN trong raw panel; không tự đổi NaN thành zero.
- Giữ zero như observation hợp lệ cho đến khi có bằng chứng ngược lại.
- LightGBM dùng native missing handling.
- GRU dùng **global per-KPI median imputer và global per-KPI scaler** (mỗi KPI một bộ tham số chung cho mọi beam) fit **chỉ trên fit segment**, đi cùng missingness mask. Không dùng per-beam scaler fit toàn bộ chuỗi khi chưa có origin-causal protocol và test riêng.
- Nếu target DLPRB là NaN, loại sample đó khỏi training/evaluation và ghi số lượng bị loại.
- Không xóa observation âm thầm.
- Báo zero ratio, active ratio, active-only MAE và zero-observation false-positive rate.

## 1.4. Unit policy

Nếu không xác minh được đơn vị DLPRB, dùng thuật ngữ `DLPRB usage`, không ghi `DLPRB utilization (%)`.

## 1.5. Temporal boundary policy

Week 6 chỉ được dùng làm external holdout nếu có căn cứ về thứ tự thời gian liên tục từ tài liệu dataset hoặc metadata có thẩm quyền, đồng thời file thật khớp sampling interval và thứ tự index.

Không dùng profile correlation đơn thuần để chứng minh continuity.

Nếu không xác minh được continuity, dùng internal chronological holdout trong 840 giờ train. Khi đó week 6 chỉ là exploratory data, không phải primary holdout.

Quyết định phải được ghi trong `reports/split_decision.md` trước khi train model.

---

# 2. Temporal split và leakage-free evaluation

Đây là phần bắt buộc phải đúng trước khi train bất kỳ model nào.

## 2.1. Primary split — chỉ khi week 6 hợp lệ

| Segment | Index | Vai trò |
|---|---:|---|
| Fit | 0–671 | Train model và fit preprocessing |
| Validation | 672–839 | Early stopping, model selection |
| Holdout | 840–1007 | Final evaluation, dùng sau khi freeze |
| Week 11 | Riêng | Không thuộc P0 |

Model validation được fit đến index 671. Vì vậy validation origin đầu tiên là 671, target đầu tiên là 672.

Sau khi chọn champion, final model được refit trên 0–839. Vì vậy holdout origin đầu tiên là 839, target đầu tiên là 840.

## 2.2. Fallback split — nếu week 6 không đủ bằng chứng continuity

| Segment | Index | Vai trò |
|---|---:|---|
| Fit | 0–503 | Train model |
| Validation | 504–671 | Model selection |
| Internal holdout | 672–839 | Final test |

Sau khi chọn champion bằng validation, refit trên 0–671 rồi đánh giá trên internal holdout.

Không được đổi validation thành final test sau khi đã dùng nó để tune/select model.

## 2.3. Forecast-origin rule

Tại origin `t`, model chỉ được dùng dữ liệu có index `<= t`.

Một origin hợp lệ phải:
1. Có đủ 168 giờ history.
2. Không sớm hơn `fit_end`.
3. Có target `t+h` nằm trong evaluation segment.
4. Không dùng future values trong feature, scaler, static statistics hoặc imputer.

```python
def evaluation_origins(fit_end, segment_end, horizon, lookback=168):
    start = max(lookback - 1, fit_end)
    end = segment_end - horizon
    return np.arange(start, end + 1)
```

Với primary split:

| Segment | Horizon | Origins | Số lượng |
|---|---:|---|---:|
| Train | +1h | 167–670 | 504 |
| Train | +6h | 167–665 | 499 |
| Train | +24h | 167–647 | 481 |
| Validation | +1h | 671–838 | 168 |
| Validation | +6h | 671–833 | 163 |
| Validation | +24h | 671–815 | 145 |
| Holdout | +1h | 839–1006 | 168 |
| Holdout | +6h | 839–1001 | 163 |
| Holdout | +24h | 839–983 | 145 |

GRU joint training chỉ dùng origins có đủ cả ba target, tức `167–647` trong primary split.

Với fallback split, áp dụng cùng công thức nhưng thay các mốc fit/validation/holdout tương ứng.

## 2.4. Rolling-origin evaluation

- Model được fit cố định ở đầu evaluation segment.
- Tại các origin tiếp theo, history được cập nhật bằng actual đã quan sát đến origin đó.
- Không retrain model ở mỗi origin.
- Evaluation stride = 1.
- Training stride có thể là 2 hoặc 3 để giảm compute.
- Tất cả model dùng cùng origin set cho từng horizon.

Đây là fixed-model rolling-origin evaluation, không phải retraining-at-every-step evaluation.

## 2.5. Refit rule

Sau model selection:
- LightGBM: refit với `n_estimators = best_iteration` từ validation.
- GRU: refit với số epoch bằng `best_epoch` từ validation.
- Fit lại **global per-KPI scaler và imputer** trên toàn bộ final development fit segment. Không fit per-beam scaler; các static features đã bỏ khỏi model input không được tái thêm khi refit.
- Tính lại P90/P95 thresholds trên final fit segment.
- Không đọc holdout trong bất kỳ bước selection hoặc fitting nào.

## 2.6. Leakage tests bắt buộc

- Target không chồng giữa train/validation/holdout.
- Rolling features chỉ kết thúc tại origin.
- Future-perturbation canary: thay dữ liệu sau origin bằng nhiễu, features/predictions tại origin không được đổi.
- Scaler/imputer không đổi khi perturb dữ liệu ngoài fit segment; với GRU phải chứng minh tham số normalization có shape `(n_kpis,)` (không có trục beam), được lưu cùng model, và train/inference dùng chung. Với feature động, canary theo origin vẫn phải pass.
- Holdout runner từ chối chạy nếu chưa có `pre-holdout` tag hoặc hash config không khớp.

---

# 3. Models và metrics

## 3.1. Baselines — P0

Giữ ba baselines từ tuần 1–4:
1. Persistence: `pred(t+h) = actual(t)`.
2. Seasonal Naive 24h: `actual(t+h-24)`.
3. Seasonal Naive 168h: `actual(t+h-168)`.

Seasonal Mean 7d là optional; chỉ thêm nếu hoàn thành sớm và không làm trễ milestone.

Với horizon +24h, Seasonal Naive 24h trùng Persistence. Ghi rõ điều này, không coi là hai baseline độc lập.

## 3.2. Global LightGBM — P0

Ba direct models độc lập cho +1h, +6h, +24h.

Feature groups:

**Origin lags (9):**
```python
lags = [0, 1, 2, 3, 6, 12, 24, 48, 167]
```

**Target-aligned seasonal features (5):**
```python
seasonal_lags = [1, 2, 3, 7]
```
Dùng `DLPRB[t + h - 24*k]` và seasonal mean 7 ngày. Với `h <= 24`, các giá trị này không vượt quá origin.

**Rolling (9):** mean/std/max cho cửa sổ 6, 24, 168 giờ, tất cả kết thúc tại `t`.

**Phase (2):** `(t+h) % 24`, `(t+h) % 168`. Đây là phase index, không gán thành giờ lịch.

**KPI phụ (15):** mỗi KPI phụ có lag 0/1/24 và rolling mean 6h/24h.

**Static beam statistics:** **Không đưa** mean/std/P90/active-ratio của cả fit segment vào **model input** trong P0. Các thống kê tổng hợp theo beam chỉ dùng cho **stratified sampling, load-tier analysis và risk thresholds**, tách biệt khỏi predictor. Nếu về sau thử biến thể causal-as-of-origin, đó là experiment P1 độc lập, cần cùng protocol ở train/inference và test riêng.

**Chốt P0: 40 features (9 lag + 5 seasonal + 9 rolling + 2 phase + 15 secondary KPI), không phải 44.** `feature_manifest.json`, artifacts, evaluator và API phải khớp chính xác 40 feature names. Chốt trước khi training tuần 3 để tránh retrain tuần 5.

Chỉ chạy hai configs:

```yaml
# configs/lightgbm.yaml
configs:
  c1:
    objective: regression_l1
    num_leaves: 31
    learning_rate: 0.05
    n_estimators: 500
    early_stopping_rounds: 50
  c2:
    objective: regression
    num_leaves: 31
    learning_rate: 0.05
    n_estimators: 500
    early_stopping_rounds: 50
```

Không grid search lớn. Training stride mặc định 2; evaluation stride 1.

## 3.3. GRU-A và GRU-B — P0 có giới hạn

**GRU-A:** DLPRB + missingness mask.

**GRU-B:** bốn KPI + missingness masks.

Lookback 168. **P0 chỉ giữ phase vector 4 chiều** `[sin(2πt/24), cos(2πt/24), sin(2πt/168), cos(2πt/168)]`. Không ghép beam mean/std từ toàn fit segment vào GRU input. `static_dim=4`, GRU head dùng `Linear(hidden_size + static_dim, 64)` → với `hidden_size=64`, input dimension là **68**, không hard-code 70.

**Chuẩn hóa GRU (khóa từ T4.1, audit lại tại T5.1):** Một **global per-KPI scaler** trên tất cả beam, fit duy nhất trên training fit segment, với một center/scale cho mỗi KPI; GRU-A dùng tham số DLPRB, GRU-B dùng cùng bộ tham số DLPRB và ba KPI còn lại. **Global per-KPI imputer** cũng chỉ fit trên fit segment. Missingness mask được giữ nguyên sau imputation, target NaN bị loại theo quy tắc công bố. Lưu `scaler.json`/`imputer.json` và hash cùng checkpoint; inverse transform DLPRB trước khi đánh giá MAE.

**Phân biệt bản chất:** Scaler fit trên cả training segment là quy trình offline hợp lệ cho model được huấn luyện tại `fit_end` rồi dùng ở validation/holdout; nó **không phải leakage của validation/holdout**. Nhưng nếu tính theo từng beam rồi giả vờ mỗi training sample tại origin sớm là dự báo được deploy ngay lúc đó, tham số per-beam sẽ mang thông tin từ cuối tập train. Vì project yêu cầu pipeline đơn giản và nhất quán, chọn global per-KPI normalization, không per-beam normalization. Chỉ một thiết kế causal expanding scaler đầy đủ mới được phép trở thành biến thể nghiên cứu sau này.

Architecture:

```text
Sequence (168 × channels)
       |
       v
GRU(hidden=64, layers=2, dropout=0.1)
       |
       v
Last hidden state
       |
       +---- Phase vector (4)
       |
       v
Concatenate
       |
       v
Linear(hidden_size + static_dim=68 -> 64) -> ReLU -> Linear(64 -> 3)
       |
       v
Forecast: +1h, +6h, +24h
```

Training config:

```yaml
# configs/gru.yaml
hidden_size: 64
num_layers: 2
dropout: 0.1
optimizer: AdamW
learning_rate: 0.001
loss: L1
batch_size: 256
max_epochs: 25
early_stopping_patience: 5
gradient_clip: 1.0
seeds: [17]   # seed 42 là P1 sau khi pass G4
```

Giới hạn compute:
- Stratified sample khoảng 600–800 beams để train GRU.
- Phân tầng theo mean, std và active ratio; dùng cùng beam sample cho GRU-A/B.
- Đánh giá trên toàn bộ beams hợp lệ.
- Training stride mặc định 3.
- Tổng GPU budget mục tiêu tối đa 4 giờ.
- Checkpoint mỗi epoch.
- **Kaggle GPU là chính; Google Colab GPU là backup** cho GRU-A/B và GRU champion refit. Colab CPU dùng cho baselines, LightGBM và evaluation. Không training model trên máy cá nhân.
- Không train full GRU bằng CPU.

Một seed cho mỗi GRU là P0, chỉ đủ kết luận mô tả cho một lần chạy. Nếu chạy hai seeds, báo từng seed và mean ± std mang tính mô tả; không tuyên bố statistical stability mạnh.

Nếu GPU quota không đủ: ưu tiên ít nhất GRU-B một seed, có thể bỏ GRU-A và ablation; giữ LightGBM và toàn bộ evaluation protocol. Ghi rõ thay đổi scope.

## 3.4. Metrics

Primary:
- MAE theo từng horizon.

Secondary:
- RMSE.
- WAPE.
- Macro-WAPE.
- Per-beam macro MAE.
- Active-only MAE.
- Zero-observation false-positive rate.
- MAE theo load tier.
- Skill score so với baseline tốt nhất.

Không dùng MAPE.

Không mặc định pooled MAE bằng macro MAE. Báo số observation hợp lệ và số beam bị loại do thiếu dữ liệu.

Load tiers Low/Mid/High được tạo từ tertiles của mean DLPRB từng beam trên fit segment.

## 3.5. Confidence intervals

- Paired cluster bootstrap theo BS nếu mapping đã xác minh.
- 500–1.000 repetitions, seed cố định.
- CI 95% cho chênh lệch MAE.
- Nếu mapping không đáng tin cậy, bootstrap theo beam và ghi rõ hạn chế phụ thuộc giữa các beam.
- Với 30 BS, diễn giải CI thận trọng; không coi đó là bằng chứng generalization sang mọi mạng.

---

# 4. Champion selection

Chọn champion chỉ bằng validation.

Ứng viên:
- **Best baseline theo validation cho mỗi horizon** (được phép trở thành champion).
- LightGBM.
- GRU-A.
- GRU-B.

Quy tắc:
1. Tính MAE theo từng horizon.
2. Chuẩn hóa MAE theo best baseline của từng horizon.
3. Tính trung bình normalized MAE trên ba horizons.
4. Chọn họ model có điểm thấp nhất.
5. Nếu chênh lệch không rõ theo paired CI, chọn model đơn giản/rẻ hơn.
6. Chỉ cho phép chọn model khác nhau theo horizon nếu cải thiện ít nhất 5% và CI hỗ trợ.
7. Nếu ML không tốt hơn baseline, **deploy baseline thắng validation** thay vì buộc dùng ML kém hơn; báo cáo kết quả trung thực.
8. **Common evaluation mask:** chênh lệch MAE và CI phải được tính trên cùng cặp `(origin, beam)` có actual hợp lệ và dự báo hữu hạn của các model đem so. Báo thêm coverage của từng model.
9. Risk score không được can thiệp vào tiêu chí point-forecast champion nếu chưa được chốt từ trước; risk trade-off được báo cáo riêng.

Risk metrics được báo cáo riêng, không dùng holdout để điều chỉnh champion hoặc threshold.

Trước holdout:
- Tạo `configs/champion.yaml`.
- Lưu config hash.
- Lưu validation selection report.
- Commit.
- Tạo Git tag `pre-holdout`.

---

# 5. Capacity-Risk Engine

## 5.1. Thresholds

Primary: per-beam P90/P95.
- Warning: P90.
- Critical: P95.

Threshold được tính từ fit segment, không dùng validation/holdout để fit.

Global P90/P95 chỉ là optional; không ưu tiên. **Risk score là scenario threshold exceedance, không phải xác suất congestion.**

## 5.2. Risk state

```text
NORMAL   : prediction < P90
WARNING  : P90 <= prediction < P95
CRITICAL : prediction >= P95
```

Nếu `P90 == P95`:
- Không tạo Warning band giả.
- `prediction < P90` → NORMAL.
- `prediction >= P95` → CRITICAL.
- Ghi `warning_band_degenerate = true`.

## 5.3. Guard

Loại beam khỏi risk evaluation nếu:
- P90 không vượt epsilon đã định trước; hoặc
- Active ratio thấp hơn ngưỡng cấu hình được chọn từ training data.

Báo số beam bị loại và lý do.

## 5.4. Evaluation

Actual exceedance: `actual >= threshold`.

Predicted exceedance: `prediction >= threshold`.

Báo:
- Precision, Recall, F1, FPR.
- TP, FP, FN, TN.
- Base rate.
- False alarm count.
- Missed exceedance count.
- Số beam hợp lệ/bị loại.

So sánh champion với Persistence và Seasonal Naive 168h.

Không positive samples thì metric tương ứng trả NaN kèm cờ, không chia cho zero.

**Cách diễn giải:** đây là historical threshold-exceedance monitoring trên ngưỡng thống kê, không phải congestion/incident detection.

---

# 6. Repository structure

```text
5g-network-intelligence/
├── configs/
│   ├── data.yaml
│   ├── evaluation.yaml
│   ├── lightgbm.yaml
│   ├── gru.yaml
│   ├── risk.yaml
│   └── champion.yaml
├── data/
│   ├── raw/                  # gitignored
│   └── processed/            # gitignored
├── notebooks/
│   ├── 01_data_audit_eda.ipynb
│   ├── 02_baselines.ipynb
│   ├── 03_lightgbm.ipynb
│   ├── 04_gru.ipynb
│   └── 05_final_evaluation.ipynb
├── src/
│   ├── utils/
│   │   ├── config.py
│   │   ├── seed.py
│   │   └── metadata.py
│   ├── data/
│   │   ├── io.py
│   │   ├── schema.py
│   │   ├── panel.py
│   │   ├── audit.py
│   │   └── splits.py
│   ├── features/
│   │   ├── static_stats.py
│   │   ├── lightgbm_features.py
│   │   ├── beam_sampling.py
│   │   └── gru_inputs.py
│   ├── models/
│   │   ├── baselines.py
│   │   ├── lgbm.py
│   │   ├── gru.py
│   │   └── gru_train.py
│   ├── evaluation/
│   │   ├── origins.py
│   │   ├── metrics.py
│   │   ├── evaluator.py
│   │   ├── bootstrap.py
│   │   ├── common_mask.py
│   │   ├── diagnostics.py
│   │   └── reports.py
│   ├── risk/
│   │   ├── thresholds.py
│   │   ├── scorer.py
│   │   ├── evaluator.py
│   │   └── aggregator.py
│   ├── inference/
│   │   ├── replay_store.py
│   │   └── predictor.py
│   └── api/
│       ├── schemas.py
│       └── main.py
├── app/
│   ├── backends.py
│   └── dashboard.py
├── scripts/
│   ├── prepare_data.py
│   ├── run_baselines.py
│   ├── train_lgbm.py
│   ├── train_gru.py
│   ├── evaluate_models.py
│   ├── run_risk.py
│   ├── select_champion.py
│   ├── refit_champion.py
│   ├── run_holdout.py
│   ├── quality_audit.py
│   ├── fetch_demo_assets.py
│   ├── bench_inference.py
│   └── build_report.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── utils/
├── demo_assets/              # bundle nhỏ, chỉ chứa dữ liệu được phép phân phối
├── artifacts/                # gitignored; champion bundle được quản lý riêng
├── reports/
├── docs/
├── requirements-ml.txt
├── requirements-app.txt
├── pyproject.toml
├── Dockerfile.api
├── Dockerfile.dashboard
├── docker-compose.yml
├── .gitignore
├── .dockerignore
└── README.md
```

Tạo file theo milestone, không cần tạo file rỗng ngay từ ngày đầu.

---

# 7. Notebook, Colab và Kaggle

Notebook chỉ điều phối. Logic tái sử dụng phải nằm trong `src/` và `scripts/`.

| Notebook | Nội dung | Nơi chạy chính |
|---|---|---|
| `01_data_audit_eda.ipynb` | Download, panel, audit, EDA, split decision | **Colab CPU** |
| `02_baselines.ipynb` | Origins, metrics, baselines, benchmark | **Colab CPU** |
| `03_lightgbm.ipynb` | Feature build, LightGBM, validation | **Colab CPU** |
| `04_gru.ipynb` | GRU training, checkpoint, ablation | **Kaggle GPU**, Colab GPU fallback |
| `05_final_evaluation.ipynb` | Champion, refit, holdout, risk, plots | **Colab CPU / Kaggle GPU nếu GRU refit** |

## 7.1. Colab workflow

1. Clone GitHub repo.
2. Mount Google Drive.
3. Cài `requirements-ml.txt` hoặc cài dependency còn thiếu.
4. Copy dữ liệu cần dùng vào `/content` để giảm I/O.
5. Chạy script từ notebook.
6. Lưu checkpoint/artifacts quan trọng về Drive sau mỗi epoch.
7. Ghi GPU type, runtime, duration vào `run_meta.json`.
8. Không dựa vào `/content` để lưu file duy nhất.

## 7.2. Kaggle fallback

Dùng cùng code và config với Colab.

- Upload/attach dataset hoặc lấy dữ liệu từ nguồn công khai.
- Lưu checkpoint trong Kaggle working/output và tải về Drive khi cần.
- Không giả định quota cố định; ghi quota thực tế thấy trong tài khoản.
- Không train lại từ đầu nếu đã có checkpoint hợp lệ.

## 7.3. Run metadata

Mỗi run lưu:

```text
run_id
git_commit
config_sha256
dataset_revision
seed
python_version
torch_version
lightgbm_version
device
gpu_name
started_at
duration_seconds
```

---


## 7.4. Quy tắc compute và assets cho bản 8 tuần

- **100% full-data processing, baselines, LightGBM train, GRU train, final refit, bootstrap và holdout** trên Colab/Kaggle. Máy cá nhân chỉ code, Git, synthetic unit tests, CPU inference/API/dashboard/Docker, latency benchmark.
- **Kaggle GPU** làm môi trường train GRU chính, **Colab GPU** backup; Colab CPU làm baseline/LightGBM/evaluation chính. Không giả định GPU được cấp, phải đo quota thực tế.
- Code chính ở GitHub, checkpoints và full dataset ở Drive/Kaggle Output. Pin `git_commit`, `data_revision`, `config_hash`, `feature_schema`, versions.
- **GRU preprocessing default:** global per-KPI scaler + median imputer fit trên fit segment; **không per-beam**. Checkpoint, run metadata và API đều dùng cùng artifact preprocessing. Kaggle GPU primary, Colab GPU backup (kể cả refit).
- **Demo assets**: repo có `demo_assets/README.md` và `scripts/fetch_demo_assets.py` tải model + replay subset từ đường dẫn chính thức do chính dự án phát hành sau khi release; kiểm tra SHA256, báo lỗi nếu thiếu; **không tự bịa URL**. Một fixture tổng hợp nhỏ có thể được commit để smoke test nhưng phải ghi **SYNTHETIC**, không dùng nó làm kết quả ML.
- Dữ liệu thật chỉ được phân phối trong bundle nếu license tại revision đã kiểm tra cho phép và đã ghi attribution. Dashboard phải ghi rõ số beam trong **demo subset**, không dùng số 2.880 như số beam đang được nạp nếu chỉ có 12.
- API và notebook dùng chung feature builder/scaler/thresholds. Không triển khai một phiên bản inference khác logic training.

# 8. Lịch triển khai chi tiết 8 tuần (giữ phạm vi W1–W4)

**Tổng: 96 giờ. W1–W4 giữ nguyên các task từ v5.0, W5–W8 được thiết kế lại. Mỗi tuần tối đa 12 giờ, kể cả buffer.**

> **Ghi chú cloud:** Các task W1–W4 giữ nguyên scope, nhưng mọi thao tác data/ML full scale thực hiện trên Colab/Kaggle; local chỉ viết code và chạy test trên synthetic fixtures. W3 LightGBM chạy Colab CPU, W4 GRU chạy Kaggle GPU/Colab GPU fallback.

## WEEK 1 — Data engineering và audit

**Ngân sách: 12 giờ**

**Mục tiêu:** tạo panel đáng tin cậy, xác minh dataset và freeze temporal split.

### T1.1 — Repo và môi trường (1h)

Làm:
- Tạo GitHub repo `5g-network-intelligence`.
- Tạo virtual environment.
- Tạo `pyproject.toml`, `.gitignore`.
- Tạo `requirements-ml.txt`, `requirements-app.txt`.
- Setup Colab, Drive và Kaggle.

Code:
- `src/utils/config.py::load_config(path)`
- `src/utils/seed.py::set_seed(seed)`
- `src/utils/metadata.py::collect_run_metadata(config)`

Output:
- Repo có thể cài dependencies.
- `pytest -q` và `ruff check` chạy được.
- `docs/infra.md` ghi môi trường và tình trạng GPU thực tế.

### T1.2 — Download dataset và manifest (1h)

Code:
- `src/data/io.py::download_dataset(repo_id, revision, dest)`
- `src/data/io.py::build_manifest(raw_dir)`
- `scripts/prepare_data.py --step download`

Output:
- Các CSV cần thiết nằm trong `data/raw/`.
- `reports/data_manifest.json` có revision, license, checksum, size, rows, columns.
- Chạy lại không tải lại file có checksum khớp.

### T1.3 — Panel builder (2h)

Code:
- `src/data/schema.py`: KPI constants và `parse_beam_id`.
- `src/data/panel.py`: `Panel`, `build_panel`, `save_panel`, `load_panel`.

Làm:
- Align Time và beam giữa bốn KPI.
- Numeric-sort beam IDs.
- Giữ NaN/zero.
- Ghép train + week 6 chỉ khi split decision cho phép.
- Lưu week 11 riêng.

Output:
- `data/processed/panel.npz`
- `data/processed/panel_meta.json`
- `data/processed/week11.npz`
- Save/load parity test pass.

### T1.4 — Data audit và EDA (2.5h)

Code:
- `src/data/audit.py::audit_missing`
- `audit_zeros`
- `audit_range`
- `audit_duplicates`
- `audit_beams`

Notebook:
- `notebooks/01_data_audit_eda.ipynb`

Output:
- `reports/data_audit.md`
- `docs/data_dictionary.md`
- Hình phân phối DLPRB, zero ratio, time series mẫu, phase 24h/168h.
- Có quyết định NaN policy, zero policy và unit policy.

### T1.5 — Temporal boundary và split decision (2h)

Code:
- `src/data/audit.py::check_mapping`
- `src/data/audit.py::check_temporal_boundary`
- `src/data/splits.py::build_split_config`

Output:
- `reports/split_decision.md`
- `configs/data.yaml`
- `configs/evaluation.yaml`
- `data/processed/split.json`

Phải chốt một trong hai:
- Primary: external week 6 holdout.
- Fallback: internal chronological holdout.

Không train model trước khi task này xong.

### T1.6 — Tests và buffer (1.5h)

Files:
- `tests/unit/test_panel.py`
- `tests/unit/test_audit.py`

Output:
- Test sorting, alignment, NaN, zero, save/load, split metadata pass.
- Commit `week1-data-audit`.

### Buffer tuần 1 (2h)

Dùng cho lỗi download/schema hoặc sửa data pipeline. Nếu không dùng hết, không tự thêm scope.

**Gate G1:** panel, audit, data policy và split decision hoàn tất.

---

## WEEK 2 — Evaluation framework và baselines

**Ngân sách: 12 giờ**

**Mục tiêu:** có evaluator đúng index, leakage tests và baseline benchmark.

### T2.1 — Origin generator (2h)

Code:
- `src/evaluation/origins.py::train_origins`
- `evaluation_origins`
- `train_origins_joint`
- `assert_no_target_overlap`

Output:
- `tests/unit/test_origins.py`
- `data/processed/split.json` có origins theo từng horizon.
- Test các mốc origin ở mục 2 pass.

### T2.2 — Metrics (1.5h)

Code:
- `src/evaluation/metrics.py`

Functions:
- `mae`
- `rmse`
- `wape`
- `macro_wape`
- `macro_mae`
- `active_only_mae`
- `zero_fp_rate`
- `assign_load_tiers`

Output:
- `tests/unit/test_metrics.py`
- Mảng tính tay cho từng metric.
- WAPE mẫu số zero trả NaN có cờ.
- NaN được loại và đếm rõ.

### T2.3 — Evaluator (2h)

Code:
- `src/evaluation/evaluator.py`
- `Forecaster` protocol.
- `evaluate(forecaster, panel, origins, horizon)`.
- `save_forecasts`.
- `to_benchmark_rows`.

Output:
- `tests/unit/test_evaluator.py`
- Oracle predictor cho MAE = 0.
- Constant predictor cho MAE đúng giá trị tính tay.
- Forecast record lưu được.

### T2.4 — Baselines (1.5h)

Code:
- `src/models/baselines.py`
- `Persistence`
- `SeasonalNaive(period=24)`
- `SeasonalNaive(period=168)`

Output:
- `tests/unit/test_baselines.py`
- Mọi tham chiếu đều `<= origin`.
- Horizon 24 seasonal naive 24h trùng Persistence.
- Không đọc index âm.

### T2.5 — Leakage canary (1.5h)

Code:
- `tests/utils/perturb.py`
- `tests/unit/test_leakage.py`

Làm:
- Perturb toàn bộ dữ liệu sau origin.
- Kiểm tra feature/prediction tại origin không đổi.
- Tạo một forecaster cố tình đọc future để chứng minh canary phát hiện được lỗi.

Output:
- Canary pass cho baselines.
- Test-of-test chứng minh canary fail với forecaster cố tình leakage.

### T2.6 — Baseline benchmark và report (1.5h)

Code:
- `scripts/run_baselines.py --segment val`
- `notebooks/02_baselines.ipynb`

Output:
- `reports/baseline_benchmark_val.csv`
- `reports/forecasts/baselines_val.npz`
- `reports/baseline_analysis.md`

Báo best baseline theo horizon, load tiers, active/zero và top beam errors.

### Buffer tuần 2 (2h)

Dùng cho index bug, evaluator mismatch hoặc sửa leakage canary.

**Gate G2:** evaluator, origins, baselines và leakage tests pass.

---

## WEEK 3 — LightGBM forecasting

**Ngân sách: 12 giờ**

**Mục tiêu:** hoàn thành global LightGBM cho ba horizons và có benchmark validation.

### T3.1 — Static stats và feature builder (3h)

Code:
- `src/features/static_stats.py::compute_static`
- `src/features/lightgbm_features.py::build_features`

Làm:
- Tạo **40 features** theo mục 3.2; loại bốn static beam statistics khỏi model input **trước khi train**.
- Vectorize theo origin/beam.
- Không materialize toàn bộ sliding windows nếu không cần.
- Các thống kê beam cho sampling/diagnostics/risk đặt ngoài feature builder dự báo.
- Training stride 2; validation stride 1.

Output:
- `tests/unit/test_lightgbm_features.py`
- `reports/feature_manifest.json`
- Feature names đúng **40**, schema có hash và API dùng đúng thứ tự.
- Test tính tay ít nhất 10 feature.
- Future-perturbation canary pass.

### T3.2 — LightGBM train/evaluate pipeline (2.5h)

Code:
- `src/models/lgbm.py::train_lgbm`
- `LGBMForecaster`
- `save_lgbm`
- `load_lgbm`
- `scripts/train_lgbm.py`

Làm:
- Train từng horizon.
- Early stopping trên validation.
- Lưu best iteration.
- Lưu feature names, config, seed và metadata.

Output:
- `artifacts/lgbm/{config}/h{h}/model.txt`
- `artifacts/lgbm/{config}/h{h}/meta.json`
- Save/load prediction parity pass.

### T3.3 — So sánh hai config (1.5h)

Chạy C1 và C2.

Output:
- `reports/lightgbm_benchmark_val.csv`
- `configs/lightgbm.yaml` có config được chọn.
- Không chạy grid search ngoài hai configs.

### T3.4 — Error analysis (2h)

Code:
- `src/evaluation/reports.py`
- `scripts/evaluate_models.py --model lightgbm --segment val`
- `notebooks/03_lightgbm.ipynb`

Output:
- `reports/lightgbm_analysis.md`
- MAE theo horizon/load tier.
- Active-only và zero FP.
- Feature importance.
- Ít nhất ba failure examples.
- Skill score so với best baseline.

### T3.5 — Sanity checks và buffer (3h)

Kiểm tra:
- LightGBM h=1 không bất thường so với Persistence.
- Feature canary pass.
- Model save/load pass.
- Metrics tái lập được.
- Chạy lại benchmark từ config.

Nếu LightGBM thua baseline, debug index/features trước; không tune vô hạn.

**Gate G3:** LightGBM hoàn thành, validation benchmark và error analysis có thể tái lập.

---

## WEEK 4 — GRU và model comparison

**Ngân sách: 12 giờ**

**Mục tiêu:** có GRU baseline đủ tốt để so sánh, không biến tuần này thành cuộc chạy GPU vô hạn.

### T4.1 — GRU input pipeline (2h)

Code:
- `src/features/beam_sampling.py`
- `src/features/gru_inputs.py`

Làm:
- Stratified beam sample 600–800 beams.
- Fit **global per-KPI** median imputer và scaler trên fit segment (`center/scale` mỗi KPI, **không** theo beam).
- Lưu `scaler.json`/`imputer.json` và test transform + inverse-transform.
- Tạo missingness masks; xử lý KPI có toàn NaN theo policy, không dùng thống kê validation để chữa.
- Tạo **phase vector 4 chiều**, bỏ beam mean/std khỏi model input.
- Tạo input tensors và targets cho ba horizons.

Output:
- `tests/unit/test_gru_inputs.py`
- `reports/gru_sample_manifest.json`
- Test shape (`GRU-A C=2`, `GRU-B C=8` với mask), target alignment, `static_dim=4`, scaler/imputer chỉ fit trong segment, **same global DLPRB transform** ở A/B, inverse target đúng và future canary pass.

### T4.2 — GRU architecture và training loop (2.5h)

Code:
- `src/models/gru.py::GRUNet`
- `src/models/gru_train.py::train_gru`
- `scripts/train_gru.py`

Làm:
- Implement GRU-A/B.
- AdamW + L1.
- Gradient clipping.
- Early stopping.
- Checkpoint/resume.

Output:
- `tests/unit/test_gru.py`
- `tests/unit/test_gru_train.py`
- Overfit test trên synthetic mini-batch.
- Resume test chạy tiếp được sau khi dừng.

### T4.3 — GPU pilot (1h)

Làm:
- Chạy một epoch trên **Kaggle GPU**.
- Nếu Kaggle không cấp GPU hoặc quota hết, chuyển **Colab GPU**.
- Đo seconds/epoch, VRAM, sample count.

Output:
- `reports/gru_compute.md`
- Chốt batch size, stride, sample beams và số seeds trong `configs/gru.yaml`.

### T4.4 — Training runs (2.5h active time)

Ưu tiên P0:
- GRU-A seed 17.
- GRU-B seed 17.

P1 nếu GPU và giờ còn đủ:
- GRU-A seed 42.
- GRU-B seed 42.

Checkpoint mỗi epoch vào Drive hoặc Kaggle Output.

Output:
- `artifacts/gru/{A|B}/seed{seed}/best.pt`
- `history.jsonl`
- `meta.json`

Nếu compute không đủ:
- Bỏ GRU-A trước.
- Giữ GRU-B một seed.
- Ghi rõ đây là exploratory neural benchmark.
- Không chạy CPU training dài.

### T4.5 — Validation comparison (1.5h)

Code:
- `scripts/evaluate_models.py --model gru --segment val`
- `GRUForecaster`

Output:
- `reports/gru_benchmark_val.csv`
- `reports/ablation_study.csv`
- `reports/gru_error_analysis.md`

So sánh:
- GRU-A vs GRU-B.
- GRU vs LightGBM.
- GRU vs best baseline.

Nếu chỉ một seed/model, báo metric từng run và ghi **single-run exploratory ablation**; chỉ báo mean ± std khi thực sự có hai seeds.

### Buffer tuần 4 (2.5h)

Dành cho GPU disconnect, checkpoint resume, lỗi tensor alignment và đánh giá.

**Gate G4:** ít nhất GRU-B chạy được và được đánh giá; GRU-A/B ablation hoàn thành nếu compute cho phép.

---


## WEEK 5 — Model Quality & Research Evidence

**12 giờ tối đa (9.5 giờ P0 + 1.5 giờ P1 + 1 giờ buffer).**  
**Môi trường:** Colab CPU cho benchmark/rerun LightGBM, Kaggle GPU hoặc Colab GPU cho GRU rerun. Không training local.

**Mục tiêu:** biến kết quả tuần 1–4 thành benchmark thật sự công bằng, phát hiện lỗi và có câu chuyện nghiên cứu thuyết phục. **Không mở holdout tuần này.**

### T5.1 — Quality gate cho artifacts W1–W4 (2h, P0)

**Kiểm tra:**
- `feature_manifest.json`, `run_meta.json`, dataset revision, config và seed của mọi experiment.
- Training/validation origins và số sample thực tế theo từng horizon.
- LightGBM/GRU features sau origin không phụ thuộc vào future trong **feature construction**.
- Xác nhận **không có** beam mean/std/P90/active ratio tổng hợp trên fit segment trong model input: LightGBM đúng **40 features**; GRU đúng **4 phase features**, head input `hidden_size + 4`. Nếu artifact W3/W4 sai thì **FAIL và retrain cloud**; không chỉ sửa inference để che lỗi.
- **Scaler/imputer audit bắt buộc:** kiểm tra `src/features/gru_inputs.py::fit_scaler` và `fit_imputer` dùng **global per-KPI**, không `per-beam`; fit chỉ trên `fit_segment`; các tham số có kích thước số KPI (GRU-A: 1; GRU-B: 4), không chứa beam dimension; target DLPRB inverse transform về đơn vị gốc. Đảm bảo GRU-A và GRU-B có chung thống kê DLPRB từ training data.
- Test đổi dữ liệu **ngoài fit segment** không làm scaler/imputer thay đổi; model save/load + offline/inference dùng đúng cùng phiên bản preprocessing. Ghi rõ normalization được fit ở cutoff, không diễn giải training origins sớm như các phiên deployed độc lập.
- Pin feature schema và preprocessing artifacts. Nếu phải retrain, bỏ **toàn bộ P1** của W5.

**Code/file:** `scripts/quality_audit.py`, `src/features/lightgbm_features.py`, `src/features/gru_inputs.py`, `src/models/gru.py`, `tests/unit/test_leakage.py`, `tests/unit/test_gru_inputs.py`, `tests/unit/test_preprocessing_contract.py`.

**Đầu ra:** `reports/model_quality_gate.md` với PASS/FAIL, ảnh hưởng đến artifact nào, hành động sửa và hash phiên bản. **Gate:** Không chọn champion nếu FAIL.

### T5.2 — Common evaluation mask + paired comparison (1.5h, P0)

**Code:** `src/evaluation/common_mask.py::comparable_mask(y_true, predictions)`; `src/evaluation/bootstrap.py::paired_cluster_ci(...)`; `tests/unit/test_common_mask.py`.

- Khóa cùng cặp `(origin, beam)` cho các candidate được so sánh tại **cùng horizon**.
- Mask yêu cầu actual hữu hạn và forecasts hữu hạn cho các candidate đang so sánh; không yêu cầu toàn bộ chuỗi 168 giờ hoàn toàn không có NaN nếu model xử lý thiếu hợp lệ.
- Báo `n_origins`, `n_pairs`, `n_beams`, `coverage_pct`, `% samples excluded` cho từng candidate.
- Chênh lệch MAE và paired BS-cluster bootstrap trên cùng mask; chỉ dùng mapping BS khi đã xác minh; nếu thiếu mapping, nêu limitation.

**Đầu ra:** `reports/model_comparison_val.csv` và test oracle/missingness. Không tuyên bố A thắng B chỉ nhờ MAE khác vài chữ số.

### T5.3 — Error atlas theo beam/tải/horizon (2h, P0)

**Code:** `src/evaluation/diagnostics.py`, `scripts/evaluate_models.py --segment val --diagnostics`.

Chọn **có quy tắc, không cherry-pick:**
- Top 5 beams có MAE cao nhất.
- 5 beams đại diện cho tải thấp/trung bình/cao.
- 3 ví dụ miss peak hoặc zero-to-active transition.
- So sánh LightGBM, GRU và best baseline trên cùng đoạn.

**Đầu ra:** `reports/error_atlas.md`, `reports/figures/actual_vs_forecast.png`, `reports/figures/mae_by_horizon.png`, `reports/figures/mae_by_load_tier.png`. Ghi cases selection rule và failure modes.

### T5.4 — GRU A/B & model story (1.5h, P0)

- Xác nhận cùng beam sample, cùng origins, cùng seed, cùng target mask khi đối chiếu A/B.
- So sánh: GRU-A vs GRU-B, GRU vs LightGBM, mô hình ML vs best baseline.
- Nếu chỉ có một seed, báo cáo `single-run exploratory ablation`, không kết luận seed robustness.
- Không bắt buộc GRU thắng LightGBM. Nếu GRU thua, giải thích dữ liệu ngắn, bias/variance, sampling, train resources ở mức giả thuyết, không biến giả thuyết thành kết luận đo được.

**Đầu ra:** `reports/ablation_study.csv`, `reports/model_analysis.md`.

### T5.5 — Champion-ready artifacts & baseline fallback (2.5h, P0)

**Code:** `src/inference/registry.py` hoặc `scripts/select_champion.py` (chưa freeze cuối); `tests/unit/test_champion_selection.py`.

- Cho **baseline** làm champion nếu thắng hoặc không kém đáng kể so với ML.
- Chọn dựa trên validation MAE/normalized MAE theo quy tắc công bố trước, ưu tiên mô hình rẻ hơn nếu CI không phân biệt được.
- Test trường hợp baseline tốt nhất và ML không thắng; test phân biệt model family/3 horizons.
- Tạo `reports/champion_candidate_table.csv`; không đọc holdout.

### T5.6 — Optional: prediction bands (1.5h, P1 — chỉ khi P0 pass)

**Lựa chọn duy nhất:** chạy thử **LightGBM quantile** mức 0.1/0.9 trên **một horizon ưu tiên h=24** bằng code chung. Nếu chạy đủ nhanh mới mở rộng 3 horizons ở thời gian P1 còn dư.

**Code:** `scripts/train_lgbm_quantiles.py`, `configs/quantile.yaml`.

Báo `pinball loss`, empirical interval coverage và width trên validation. Với **conditional quantiles `q10/q90`**, nominal central interval là **80%**, nhưng không được gọi là **calibrated** hoặc "80% guaranteed coverage". **Đây là quantile dự báo** `forecast_q10/forecast_q90`, **khác hoàn toàn** với per-beam historical risk thresholds `historical_warning_p90/historical_critical_p95`. Không dùng quantile output để thay đổi point-forecast champion nếu quy tắc chưa được chốt trước.

**Naming/UX contract:** `Forecast Band (q10–q90)` cho dải dự báo và `Historical Warning Threshold (P90)` / `Historical Critical Threshold (P95)` cho hai đường ngang. Chỉ vẽ band ở horizon thực sự có quantile models và validation report; nếu champion không phải LightGBM, phải ghi rõ đây là **auxiliary quantile LightGBM**, không gán nhầm uncertainty cho GRU champion; an toàn nhất là ẩn band khi không đồng nhất model. Dashboard phải có legend/tooltip phân biệt hai P90. Nếu không đủ thời gian, bỏ band (P1).

### Buffer W5 (1h)

Nếu T5.1 phát hiện lỗi nghiêm trọng, **cắt T5.6 và ưu tiên retrain cloud**. Mục tiêu W5 là data/model correctness, không phải thêm công nghệ cho đẹp.

**Gate G5 — Research quality:** benchmark chung mask, provenance, ablation và error atlas pass; mọi artifact đủ điều kiện selection, chưa mở holdout.

---

## WEEK 6 — Champion, Risk & Final Holdout

**12 giờ tối đa (10.5 giờ P0 + 1.5 giờ buffer).**  
**Môi trường:** Colab CPU cho selection/risk/holdout và LightGBM refit; Kaggle GPU hoặc Colab GPU nếu GRU là champion. Local không train.

**Mục tiêu:** tạo **final result** đáng tin, có model lựa chọn hợp lý, baseline fallback, threshold-risk evaluation, test độc lập và phân tích tình huống bỏ lỡ cảnh báo.

### T6.1 — Risk engine + deterministic policy (2.5h)

**Code:**
- `src/risk/thresholds.py::fit_thresholds(fit_panel, cfg)`
- `src/risk/scorer.py::risk_status(pred, threshold)`
- `src/risk/evaluator.py::evaluate_exceedance(y, p, thresholds, mask)`
- `configs/risk.yaml`, `tests/unit/test_risk.py`

**P0:** per-beam P90/P95; Normal/Warning/Critical; `P90 == P95` → no warning band; guard near-zero beams/low activity; threshold `epsilon` và `min_active_ratio` được chốt bằng **fit segment trước holdout**. Báo `n_excluded`, `base_rate`, `TP/FP/FN/TN`, Precision, Recall, F1, FPR, false alarms/misses theo horizon và threshold.

**Guard không có positive:** metric tương ứng NaN kèm flag, không gán 100% chính xác.

### T6.2 — Validation risk & trade-off report (1h)

**Code:** `scripts/run_risk.py --segment val`.

- Champion candidates và best baselines trên **common evaluation mask**; không tối ưu rules theo holdout.
- P90/P95 là statistical thresholds; đánh giá được **exceedance**, không có ground truth congestion.
- Báo false-alarm vs missed-exceedance, base rate, và cảnh báo rõ nếu số positive quá ít.

**Đầu ra:** `reports/risk_validation.csv`, `reports/risk_tradeoff_val.md`.

### T6.3 — Freeze selection, model registry & configs (1h)

**Code:** `scripts/select_champion.py`, `configs/champion.yaml`, `tests/unit/test_champion_selection.py`.

- Áp dụng lựa chọn từ W5, baseline được phép là champion.
- Chốt model family, 3 horizon routes, seed deploy (nếu GRU), số epoch/iterations refit, feature schemas, imputer, scaler, thresholds policy, artifact paths.
- `git tag pre-holdout` **trước khi đọc holdout target hoặc xem holdout metrics**.
- `run_holdout.py` kiểm tra commit/tag/config hash và refuse nếu không khớp.

**Đầu ra:** `reports/champion_selection.md` và `configs/champion.yaml`.

### T6.4 — Full-development refit (2h)

**Code:** `scripts/refit_champion.py`.

- Nếu primary split: fit 0–839; nếu fallback: 0–671.
- LightGBM: dùng số cây đã chọn ở validation, **không early stop bằng holdout**.
- GRU: dùng số epoch đã freeze ở validation, **không tune bằng holdout**.
- Baseline: chỉ lưu definition/parameters, không huấn luyện giả.
- Recompute scaler/imputer/thresholds chỉ từ final development fit segment; đảm bảo causal feature protocol như W5.

**Đầu ra:** `artifacts/champion/{model...}`, `champion_meta.json`, `feature_schema.json`, `thresholds.npz`, `artifacts_manifest.json` (SHA256). Verify save/load.

### T6.5 — Final holdout once + paired CI (2h)

**Code:** `scripts/run_holdout.py --frozen configs/champion.yaml`.

- Dùng origins với fit cutoff hợp lệ; chung mask để so với **best validation baseline**.
- Không điều chỉnh threshold, feature hoặc model sau khi xem kết quả.
- Tính MAE/RMSE/WAPE/macro MAE, errors theo load tier, active/zero, paired BS-cluster CI (500–1,000 reps).
- Optional quantile: đánh giá coverage/width sau freeze, không retrofit calibration bằng holdout.
- Nếu phát hiện bug nghiêm trọng, log số lần rerun, lý do, diff, không chọn cấu hình mới theo holdout.

**Đầu ra:** `reports/holdout_benchmark.csv`, `reports/holdout_uncertainty.csv` (nếu quantile có), `reports/forecasts/champion_holdout.npz`, `reports/holdout_run.log`.

### T6.6 — Risk holdout + missed-event case study (2h)

**Code:** `scripts/run_risk.py --segment holdout`, `src/evaluation/diagnostics.py`.

- Có bảng risk metrics **đặt cạnh baseline**, ít nhất 2–3 false alarms và missed exceedances (nếu tồn tại), kèm actual, forecast, threshold, horizon.
- Nêu rõ số positive events, số beam bị loại và sai số theo từng nhóm.
- Không tự suy luận các event là sự cố thật.

**Đầu ra:** `reports/risk_holdout.csv`, `reports/risk_evaluation.md`, `reports/holdout_analysis.md`.

### Buffer W6 (1.5h)

Ưu tiên fix reproducibility/final test; không bắt đầu UI khi model registry và holdout còn lỗi.

**Gate G6 — Final science frozen:** model selection chỉ bằng validation, freeze trước holdout, tất cả holdout metrics/risk có provenance, baseline comparator và limitations.

---

## WEEK 7 — Polished Product: Inference, API, Dashboard & Docker

**12 giờ tối đa (11 giờ P0 + 1 giờ buffer).**  
**Môi trường:** VS Code local cho code + synthetic tests, **local CPU inference** và Docker demo. **Không training trên máy cá nhân**. Nếu cần re-evaluation full dataset, dùng Colab CPU.

**Mục tiêu:** biến mô hình thành sản phẩm người tuyển dụng có thể thao tác và quan sát logic dự báo/rủi ro trong dưới hai phút.

### T7.1 — Shared inference + artifact registry (3h)

**Code:**
- `src/inference/replay_store.py::ReplayStore`
- `src/inference/predictor.py::Predictor`
- `src/inference/registry.py::load_champion` (nếu chưa tạo W5)
- `tests/unit/test_predictor.py`

**Contract:**
- `predict(beam_ids, origin)` trả `history`, `predictions` cho 3 horizons, `target_indices`, `actual` nếu có, `model_version`, `is_replay`.
- `predict_risk(...)` trả `historical_warning_p90`, `historical_critical_p95`, status từng horizon, valid_mask và reason nếu excluded; không gọi threshold P90 là predictive q90.
- Nạp đúng `global_per_kpi` scaler/imputer từ champion artifacts; kiểm tra feature schema và GRU `static_dim=4` trước inference.
- Chỉ load artifacts một lần; cache nguyên tắc data/read-only; không train trong request.
- Dùng shared feature builder của notebook; baseline champion cũng chạy được.
- **Parity:** 20 cặp `(beam, origin)` so với offline forecasts, `atol` được chốt theo model type.

**Đầu ra:** `reports/inference_parity.csv` và test pass.

### T7.2 — FastAPI 5 endpoints (2h)

**Code:** `src/api/schemas.py`, `src/api/main.py`, `tests/integration/test_api.py`.

**P0 endpoints:**
- `GET /health`: model/data readiness.
- `GET /model-info`: champion, horizons, limitations, revision.
- `GET /replay/meta`: beam ids, origin range, dataset scope, beam subset count.
- `POST /forecast`: input `{origin, beam_ids}`; response includes **last 168 historical points** for plotting, point forecasts, target indices, known actuals and metadata.
- `POST /risk`: input `{origin, beam_ids}`; response includes per-horizon status, **`historical_warning_p90` và `historical_critical_p95`**, guards, error flags; nếu có quantiles thì `forecast_q10/forecast_q90` thuộc schema dự báo riêng, không trộn vào threshold.

Bắt buộc validate unknown beam (404), origin ngoài phạm vi (422), batch quá lớn (422), artifact missing (503); không lộ stack trace nội bộ. Swagger `/docs` có example request.

### T7.3 — Streamlit 1-page professional dashboard (3h)

**Code:** `app/backends.py::ApiBackend`, `app/dashboard.py`.

Bố cục:
1. Header: **5G Network Intelligence**, badge **HISTORICAL / OFFLINE REPLAY**, model/version/data revision.
2. Sidebar: BS/cell/beam selector (chỉ khi mapping được xác minh), origin selector, horizon selector.
3. Four concise KPI tiles: beams loaded, chosen model, selected horizon, risk status; không fake live KPIs.
4. Main Plotly/Altair chart: **168-step history**, 3 forecast markers; hai đường ngang có label **`Historical Warning Threshold (P90)`** và **`Historical Critical Threshold (P95)`**. Nếu T5.6 pass và model/horizon tương thích, dải dự báo dùng label **`Forecast Band (q10–q90)`** và legend riêng. Không vẽ dải q10–q90 cho h=1/h=6 nếu mới train h=24.
5. Risk detail: Normal/Warning/Critical, threshold values, actual if known; tooltip giải thích **historical risk P90/P95** là phân vị theo beam trên fit segment, **không phải forecast conditional q90** và không phải congestion probability.
6. Comparison panel: champion MAE by horizon vs best baseline, link/description report.
7. Error-state UI: invalid origin, beam excluded, missing data, no known target, unavailable model.

**UX acceptance:** demo beam A → beam B → origin mới chạy ổn; các con số khớp API; không dùng ví dụ giả như kết quả thật.

### T7.4 — Docker Compose + assets strategy (2h)

**Files:** `Dockerfile.api`, `Dockerfile.dashboard`, `docker-compose.yml`, `.dockerignore`, `scripts/fetch_demo_assets.py`, `demo_assets/README.md`.

- Services `api` và `dashboard`, health checks, read-only artifacts/demo store, env config, **CPU-only inference**.
- **CPU-only GRU deployment:** Trong `requirements-app.txt` chỉ để các dependencies ứng dụng, **không** để `torch` GPU/CUDA, `torchvision`, `torchaudio` hoặc `nvidia-*`. Trong `Dockerfile.api`, nếu champion là GRU, cài **PyTorch Linux CPU wheel từ index chính thức**: `pip install --no-cache-dir "torch==${TORCH_VERSION}" --index-url https://download.pytorch.org/whl/cpu` (pin phiên bản đã thử nghiệm trong `docs/infra.md`); nếu champion là LightGBM/baseline thì **không cài torch**. Dùng `ARG MODEL_FAMILY`/build target để chọn dependency profile trước khi build; không dùng `--extra-index-url` cho torch nếu có nguy cơ chọn CUDA wheels. Không cài torch trong Dockerfile dashboard.
- GRU artifact dùng `state_dict`/config và `torch.load(..., map_location="cpu")`; smoke test không yêu cầu GPU, xác nhận **`torch.version.cuda is None`**, kiểm tra `pip freeze` không có CUDA-only packages và đo image size thật. Chỉ cân nhắc ONNX Runtime ở P1 sau khi kiểm tra numerical parity; không bắt buộc chuyển ONNX.
- **Cold start path:** `python scripts/fetch_demo_assets.py` có kiểm tra SHA256 nếu asset cần tải; sau đó `docker compose up --build`.
- Không commit raw dataset lớn, passwords/keys. Model/demo assets từ **chính release của repo khi đã phát hành**, không ghi URL chưa tồn tại.
- Nếu license không cho phép publish subset, chỉ lưu metadata + synthetic fixture cho tests và hướng dẫn người dùng tạo demo data từ nguồn gốc; demo phải nói rõ type dữ liệu.
- Từ clean clone, startup được với documented assets; dashboard phải hiển thị đúng số beam **đang loaded**, không mặc định 2.880.

### T7.5 — Smoke + API/dashboard parity (1h)

**Files:** `tests/integration/test_smoke.py`, `tests/integration/test_api.py`.

- `GET /health` 200, metadata đúng, forecast/risk sample đúng.
- Negative cases 404/422/503, invalid horizon, NaN guard.
- 20 sample pairs offline/API within declared tolerance, không bỏ qua parity nếu model GRU float khác nhẹ trên CPU/GPU.

### Buffer W7 (1h)

Nếu Docker hoặc API vướng dependency, bỏ visual polish và optional quantile band trước; **giữ functional API, one-beam plot và risk status**.

**Gate G7 — Working product:** API + dashboard + CPU inference + Compose hoạt động, parity pass, có lịch sử và điểm dự báo thực sự.

---

## WEEK 8 — Quality Assurance, Reproducibility & HR-Ready Release

**12 giờ tối đa (11 giờ P0 + 1 giờ buffer).**  
**Môi trường:** VS Code local cho tests, Docker, video; Colab/Kaggle chỉ nếu cần full-data evaluation rerun do bug có ghi log. Không local training.

**Mục tiêu:** người xem hiểu tác động của project, kiểm chứng được evidence và dùng thử nhanh; không chỉ thấy danh sách công nghệ.

### T8.1 — End-to-end QA & reproducibility (3h)

**Files:** `tests/integration/test_end_to_end.py`, `tests/unit/test_artifact_io.py`, `scripts/fetch_demo_assets.py`, `requirements-ml.txt`, `requirements-app.txt`.

- Test clean clone trên thư mục mới (code + optional demo assets) và `docker compose up --build`.
- Test baseline champion / LightGBM champion / GRU champion **bằng fixture synthetic** (không cần train lại) để routing đúng.
- Kiểm tra NaN, missing beam, origin outside range, `P90==P95`, no-positive risk, missing artifact, file checksum, runtime mismatch.
- Pin package versions được thử nghiệm; `pytest -q`, `ruff check`, API smoke đều pass. Nếu champion GRU, kiểm tra CPU-only `torch` bằng `torch.version.cuda is None`, không có `nvidia-*` wheel và `map_location="cpu"`; nếu champion LightGBM/baseline, Docker image không kéo PyTorch.
- Giữ checklist `reports/release_qa.md` có pass/fail cụ thể.

### T8.2 — CPU inference latency & demo footprint (1.5h)

**Code:** `scripts/bench_inference.py`.

Đo trên máy local (inference thôi, không train):
- P50/P95 cho 1 beam và batch 10 beams, warm-up trước khi đo.
- Model load latency, RAM, **Docker image size theo champion family** (baseline/LightGBM/GRU). Nếu GRU image bất thường nhiều GB, audit dependency tree và CPU wheel trước khi release; không gán một ngưỡng GB cố định khi chưa đo.
- Ghi cấu hình CPU và số lần chạy, không suy rộng thành production SLA.

**Đầu ra:** `reports/inference_benchmark.md`.

### T8.3 — Experiment report với evidence mạnh (2h)

**Code:** `scripts/build_report.py`; report figures từ CSV, không điền số tay.

**Đầu ra:** `reports/experiment_report.md` gồm:
1. Research questions và data audit.
2. Protocol, masks, horizons, sample counts và leakage canary.
3. Comparison table baseline/LightGBM/GRU, MAE/RMSE/WAPE, paired CI.
4. GRU-A/B ablation (ghi rõ số seed và limitations).
5. Holdout benchmark, skill vs baseline, failure atlas.
6. Risk Precision/Recall/F1/FPR, base rate, false alarms/misses.
7. Optional quantile coverage nếu hoàn tất; không claim guaranteed confidence.
8. Threats to validity, honest negative results và next steps.

### T8.4 — README, model card, architecture, limitations (1.5h)

**Files:** `README.md`, `docs/model_card.md`, `docs/architecture.md`, `docs/limitations.md`, `docs/data_dictionary.md`.

README trên cùng màn hình đầu tiên phải trả lời được:
- **Problem** (1–2 câu).
- **What it does** (horizons, risks, replay nature).
- **Key results** (bảng có số thực + baseline + holdout).
- **Try demo** (2–3 lệnh hoặc dưới 2 phút với assets đã tải).
- **How evaluated** (time-aware, leak-free, limitations).
- **Architecture** (một sơ đồ).

Model card nêu dataset, use cases, do-not-use cases, training environment, intended user, output semantics, limitations.

### T8.5 — HR demo assets & CV-ready evidence (2h)

**Deliverables:**
- `docs/screenshots/dashboard.png`, `docs/screenshots/api_docs.png` (screenshots thật từ sản phẩm).
- Video 90–150 giây: problem → select beam/origin → forecast plot → risk status → model comparison → limitations. Link/asset trong README nếu hoàn thành.
- `docs/portfolio_summary.md`: 150–250 từ tiếng Anh, 3–4 CV bullets **chỉ dựa trên kết quả thật**, mô tả contribution cá nhân và kỹ thuật đã tự kiểm chứng.
- `docs/interview_questions.md`: tối thiểu 8 câu và câu trả lời ngắn: baselines, leakage, rolling origin, global model, GRU ablation, risk thresholds, false alarms, inference parity, why no real-time.

Không giả số liệu, không tuyên bố production, không viết "SOTA" nếu không có benchmark tương thích.

### T8.6 — Release v1.0.0 (1h)

- Tạo `v1.0.0` Git tag và GitHub Release.
- Đính kèm champion model artifact manifest và demo assets nếu được phép; không commit raw dataset lớn hoặc secrets.
- Kiểm tra link trong README, công khai license/attribution, và xác nhận ít nhất một người khác có thể theo hướng dẫn chạy demo (nếu có người thử) hoặc tự clean-clone.
- `reports/release_qa.md` mọi P0 đạt hoặc limitation minh bạch.

### Buffer W8 (1h)

Không thêm mô hình mới tuần cuối. Chỉ sửa blocker của release, parity, docs/asset path.

**Gate G8 — Portfolio ready:** recruiter có thể xem problem, bảng kết quả thật, demo và limits; reviewer có thể tái lập core evaluation; local app chạy bằng CPU.

---

# 9. Timeline tổng hợp, ngân sách và milestone gates

| Week | Goal | P0/P1 hours + buffer | Compute chính | Gate |
|---|---|---:|---|---|
| W1 | Data audit, panel, split | 12 | Colab CPU | G1 |
| W2 | Evaluator, 3 baselines, leakage tests | 12 | Colab CPU | G2 |
| W3 | LightGBM 3 horizons | 12 | Colab CPU | G3 |
| W4 | GRU experiments, ablation | 12 | Kaggle GPU / Colab GPU | G4 |
| **W5** | **Model quality, common mask, error atlas, optional quantile pilot** | **12** | **Colab CPU + GPU if rerun** | **G5** |
| **W6** | **Risk engine, champion, refit, final holdout** | **12** | **Colab CPU / Kaggle GPU** | **G6** |
| **W7** | **Predictor, FastAPI, Streamlit, Docker** | **12** | **Local CPU inference only** | **G7** |
| **W8** | **QA, reports, benchmarks, demo, release** | **12** | **Local CPU inference only** | **G8** |
| **Total** | | **96** | | |

**Cách chia P0/P1:** W5 có 9.5h P0 + 1.5h P1 + 1h buffer. W6 có 10.5h P0 + 1.5h buffer. W7 có 11h P0 + 1h buffer. W8 có 11h P0 + 1h buffer. Có blocker thì lấy giờ P1 trước, không bù bằng giờ làm thêm ngoài lịch.

## 9.1. Định nghĩa ba mức hoàn thành

**Gold (ưu tiên):** G1–G8 đạt, GRU-A/B ít nhất 1 seed mỗi, quantile pilot được báo cáo nếu làm xong, sản phẩm có demo + clean clone + measured holdout numbers.

**Silver (vẫn rất tốt):** G1–G3, G5–G8 đạt; GRU-B 1 seed, GRU-A/quantile optional bị cắt; có baseline, LightGBM, risk, holdout, API/dashboard/Docker parity, report.

**Bronze (scientifically valid fallback):** data/baselines/LightGBM/holdout/risk/inference/API/README đúng; dashboard hoặc Docker cần giản lược và phải ghi giới hạn; không claim hoàn thành full Gold.

## 9.2. Dấu hiệu được phép chuyển sang tuần tiếp theo

- Không chuyển W1→W2 nếu split chưa chốt.
- Không chuyển W2→W3 nếu oracle/canary chưa pass.
- Không chuyển W4→W5 nếu GRU pipeline chưa có checkpoint hợp lệ hoặc đã ghi rõ fallback.
- Không chuyển W5→W6 nếu feature leakage/common mask chưa sửa.
- Không mở final holdout trước khi đã freeze selection.
- Không demo kết quả từ model artifact không khớp evaluation artifact metadata.

---

# 10. Scope & cut policy — tăng cạnh tranh nhưng không quá sức

**P0 không được cắt:** audit/split đúng, baselines, LightGBM, 3 horizons, common mask, leakage canary, champion+baseline fallback, frozen final holdout, risk evaluation, predictor/API parity, dashboard forecast detail, tests, report/limitations.

**P1 theo thứ tự chỉ làm khi P0 xanh:** quantile LightGBM h=24; thêm h=1/h=6; GRU seed thứ hai; sampled SHAP/permutation feature analysis; dashboard advanced overview; public demo hosting; CI workflow.

**Nếu trễ, cắt theo thứ tự:** public hosting → CI → SHAP → quantile → extra GRU seed → dashboard top-risk comparison → GRU-A (giữ GRU-B exploratory) → advanced visualization. Không bỏ holdout hay common mask.

**Rủi ro:**
- Dataset week 6 continuity không có evidence → dùng fallback internal chronological holdout từ W1, không tùy tiện đổi sau khi xem kết quả.
- GPU free không cấp → Kaggle/Colab fallback, giảm seed/beam sample/epochs; không train GRU full local.
- Quantile model chậm → cắt h=24 pilot hoặc toàn bộ P1.
- Refit GRU quá lâu → quyết định fallback/selection bằng validation; refit theo epochs frozen; nếu không thể tái lập phải ghi rõ và triển khai baseline/LightGBM nếu validation policy cho phép (không đổi chỉ vì đã xem holdout).
- Docker asset unavailable → sửa fetch/checksum/release packaging; không fake "works from clean clone".

---

# 11. Artifact contracts & reproducibility

## 11.1. Forecast records

`reports/forecasts/{run_id}_{segment}.npz` chứa từng horizon:
- `origins_h1/h6/h24`.
- `beam_ids`.
- `y_true_h*`, `y_pred_h*` và `valid_mask_h*`.
- Mảng có hình dạng nhất quán và thứ tự beam cố định.

## 11.2. Benchmark CSV fields

`run_id, model, seed, segment, horizon, n_origins, n_beams, n_valid_pairs, coverage_pct, mae, rmse, wape, macro_mae, macro_wape, active_mae, zero_fp_rate, mae_low, mae_mid, mae_high, skill_vs_best_baseline`.

## 11.3. Risk CSV fields

`run_id, model, segment, horizon, level, threshold_mode, n_valid_pairs, n_beams_eval, n_beams_excluded, base_rate, tp, fp, fn, tn, precision, recall, f1, fpr`.

## 11.4. Champion manifest

`family, artifact_paths, model_per_horizon, seed, feature_names, feature_schema_hash, scaler_path, scaler_scope=global_per_kpi, imputer_path, imputer_scope=global_per_kpi, gru_static_dim=4, threshold_path, historical_warning_p90, historical_critical_p95, fit_end, lookback, horizons, config_sha256, dataset_revision, git_commit, validation_selection_metrics, demo_asset_version`.

Nếu có P1 quantile: lưu riêng `forecast_q10`, `forecast_q90`, `quantile_model_family`, `quantile_horizons`, `quantile_validation_coverage`; **không thay thế** risk threshold fields.

## 11.5. Demo assets

- Release asset URLs chỉ được ghi sau khi đã upload và test chúng.
- `assets_manifest.json` cần SHA256 và dataset license/attribution.
- Full dataset không commit trong repo.
- `demo_assets/` có thể chứa small real replay subset **nếu redistribution hợp lệ**; nếu không, generate từ download script với nguồn chính thức, hoặc synthetic fixtures chỉ dùng smoke test.

---

# 12. Final product acceptance & HR reviewer experience

## 12.1. Scientific acceptance

- [ ] Split provenance và cutoff hợp lệ, target không leakage.
- [ ] Baselines đúng, same horizon/common evaluation mask.
- [ ] LightGBM/GRU benchmark và ablation có số thực.
- [ ] Diagnostic failure cases được chọn có quy tắc.
- [ ] Champion có baseline fallback; freeze trước holdout.
- [ ] Final holdout + risk evaluation có CI/base rate và số valid samples.
- [ ] Limits: một tuần holdout, same beams, scenario thresholds, no congestion labels, offline replay.

## 12.2. Product acceptance

- [ ] `fetch_demo_assets.py` nếu cần chạy và kiểm tra checksum.
- [ ] `docker compose up --build` chạy với các assets đã khai báo.
- [ ] API 5 endpoints hoạt động, validation/error responses rõ.
- [ ] Dashboard chọn beam + origin, có history/forecast/risk thật.
- [ ] Offline/API parity pass trên 20 cặp beam/origin.
- [ ] Không train trong API request; CPU inference độc lập Colab/Kaggle.
- [ ] `pytest` + `ruff` + integration smoke pass.
- [ ] GRU champion (nếu có) dùng PyTorch CPU-only trong Docker, không CUDA runtime; image size và CPU benchmark có số đo.
- [ ] Dashboard tách nhãn `Forecast Band (q10–q90)` khỏi `Historical Warning Threshold (P90)` và `Historical Critical Threshold (P95)` (nếu P1 band được triển khai).
- [ ] README trong 2 phút chỉ cho người xem problem, result, demo và giới hạn.

## 12.3. Portfolio acceptance

- [ ] README có measured holdout metrics và baseline.
- [ ] Architecture diagram, experiment report, model card, limitations.
- [ ] Demo screenshots thật và video 90–150s nếu xong.
- [ ] CV bullets có dẫn chứng số đo thực tế.
- [ ] GitHub Release `v1.0.0` có artifact provenance.

## 12.4. Không được đưa ra các claims sau

- "5G congestion detection accuracy" khi không có congestion labels.
- "Realtime monitoring" khi chỉ là offline replay.
- "Uncertainty-calibrated prediction" khi mới có raw quantile models và empirical coverage.
- "GRU statistically outperforms" chỉ từ một seed.
- "Production-ready" khi chỉ là local Docker prototype.
- "SOTA" khi chưa có benchmark đồng nhất.

---

# 13. Interview-ready story

**Problem:** Dự báo biến động sử dụng tài nguyên DLPRB của các beams trong hệ thống 5G và nhận biết sớm khả năng vượt các ngưỡng lịch sử.

**Approach:** Audit và chuẩn hóa panel, causal feature engineering, baselines, LightGBM, GRU multivariate, same-origin evaluation, validation selection và frozen holdout.

**Evidence:** Real dataset revision/checksum, comparison with baselines, error-by-load analysis, paired confidence interval, reproducible model artifacts, offline/API parity, latency đo thực.

**Product:** Local Docker application cho phép chọn beam và origin, xem history, forecast +1/+6/+24 và scenario-risk status, có report và model card.

**Limitation:** Statistical threshold exceedance ≠ confirmed network congestion; dataset ngắn, không test BS mới, offline replay.

**Competitive value:** Không phải số lượng framework; là khả năng giải thích và tái lập toàn bộ quyết định kỹ thuật từ dataset đến sản phẩm demo.

---

# 14. Changelog v6.1 — Consistency, preprocessing & deployment audit

Phiên bản này chỉ sửa **tính nhất quán và tính đúng đắn**, không thêm model architecture hoặc tăng workload 96 giờ.

1. **Mục 0.4:** sửa tham chiếu chính sách cắt scope từ mục 12 sang **Mục 10**.
2. **Mục 3.3 + W4:** thống nhất **Kaggle GPU primary / Colab GPU backup**, phase vector **4 chiều** và head input `hidden_size + 4`; LightGBM P0 **40 features**. Những điều kiện này được khóa ở **T3.1 và T4.1**, T5.1 chỉ audit lại để tránh retrain muộn.
3. **Scaler/imputer:** `global_per_kpi`, fit trên training segment, lưu và version cùng model; không per-beam. Training prefix được dùng để fit preprocessing của **model fixed at cutoff**; đây không phải cách mô phỏng deployment ở từng training origin. T5.1 audit cả scope, shape, inverse transform và train/inference parity.
4. **T5.6/T7.2/T7.3:** `Forecast Band (q10–q90)` là quantile dự báo có điều kiện, còn `Historical Warning Threshold (P90)` / `Historical Critical Threshold (P95)` là phân vị lịch sử của beam. Không vẽ/claim band chưa có model/horizon đã validate.
5. **T7.4/T8.1/T8.2:** PyTorch CPU-only từ `https://download.pytorch.org/whl/cpu` chỉ khi GRU là champion, không kéo CUDA runtime vào image; test CPU load và đo image size thật. ONNX là tùy chọn P1 nếu có parity.

**Nguồn kiểm tra cho CPU wheel:** [PyTorch Get Started Locally](https://pytorch.org/get-started/locally/) và [Previous PyTorch Versions](https://pytorch.org/get-started/previous-versions/). Chọn Python/PyTorch version tương thích tại thời điểm build rồi pin phiên bản đã thử nghiệm; không mặc định rằng Colab/Kaggle GPU wheel là dependency phù hợp cho Docker CPU.
