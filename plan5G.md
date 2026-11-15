# 5G Network KPI Forecasting & Capacity-Risk Monitoring

## Master Execution Plan v5.0 --- Solo, 5--6 tuần, workload có kiểm soát

  -----------------------------------------------------------------------
  Hạng mục                            Quyết định
  ----------------------------------- -----------------------------------
  Hình thức                           Solo project, trình độ intermediate

  Kế hoạch khuyến nghị                **6 tuần × 12 giờ/tuần = 72 giờ**

  Phương án rút gọn                   5 tuần × 12 giờ/tuần = 60 giờ, có
                                      cắt scope

  Dataset                             NETOP GOTS-F (`netop/gotsf-ds`)

  Target                              DLPRB

  Input KPI                           DLPRB, DLThpVol, DLThpTime,
                                      MR_number

  Lookback                            168 giờ

  Horizons                            +1h, +6h, +24h

  Models                              3 baselines, Global LightGBM,
                                      GRU-A, GRU-B

  Training                            CPU cho preprocessing/LightGBM;
                                      Colab GPU chính, Kaggle GPU dự
                                      phòng cho GRU

  Product                             Inference pipeline, FastAPI,
                                      Streamlit, Docker Compose

  Tổng giờ                            72 giờ trong phương án 6 tuần, đã
                                      gồm buffer

  Nguyên tắc                          Không làm quá 12 giờ/tuần; không
                                      thêm model ngoài scope
  -----------------------------------------------------------------------

> **Khuyến nghị:** Chọn kế hoạch 6 tuần. Bản 5 tuần là phương án rút gọn
> có chủ đích, không phải ép toàn bộ 72 giờ vào 60 giờ.

------------------------------------------------------------------------

# 0. Mục tiêu, phạm vi và tiêu chí thành công

## 0.1. Mục tiêu dự án

Xây dựng prototype dự báo KPI mạng 5G ở cấp beam, tập trung vào DLPRB,
sau đó chuyển dự báo thành các chỉ báo theo dõi khả năng vượt ngưỡng
thống kê lịch sử.

Dự án phải chứng minh được ba nhóm năng lực:

1.  **Applied ML:** panel time-series forecasting, feature engineering,
    baselines, LightGBM, GRU, multi-horizon forecasting.
2.  **Experimental rigor:** chronological split, leakage prevention,
    rolling-origin evaluation, model selection bằng validation, final
    holdout độc lập, bootstrap CI và error analysis.
3.  **ML Engineering:** pipeline có thể tái lập, model artifacts,
    inference độc lập notebook, API, dashboard, Docker và tests.

## 0.2. Deliverables của bản 6 tuần

-   Data download, manifest, panel builder và data audit.
-   Data dictionary và temporal split decision.
-   Ba forecasting baselines.
-   Global LightGBM cho ba horizons.
-   GRU-A (univariate) và GRU-B (multivariate), tối đa hai seeds mỗi
    experiment.
-   Validation benchmark và ablation.
-   Champion selection chỉ bằng validation.
-   Final refit và holdout evaluation.
-   Per-beam P90/P95 threshold-exceedance engine.
-   Risk evaluation so với baselines.
-   Independent inference pipeline.
-   FastAPI với các endpoint cốt lõi.
-   Streamlit dashboard một trang, có overview và forecast detail.
-   Docker Compose.
-   Unit tests, integration tests và leakage tests.
-   Experiment report, model card, architecture diagram và README.

## 0.3. Claims được phép

Có thể tuyên bố: - Global beam-level multivariate time-series
forecasting. - Multi-horizon DLPRB forecasting. - Historical
threshold-exceedance monitoring. - Offline historical replay. - So sánh
model trên holdout theo protocol đã công bố.

Không được tuyên bố: - Spatial-temporal forecasting hoặc GNN. -
Real-time network monitoring. - Fault detection hoặc incident
prediction. - Congestion prediction đã được xác thực. - Ngưỡng P90/P95
là ngưỡng vận hành chính thức của nhà mạng. - DLPRB utilization (%) nếu
đơn vị chưa được xác minh.

## 0.4. Definition of Done chung

Một task chỉ được đánh dấu Done khi: - Code nằm trong đúng module, không
viết toàn bộ logic trong notebook. - Có type hints và docstring ngắn nêu
input/output. - Config được đọc từ `configs/`, không hard-code đường dẫn
máy cá nhân. - Có test trên fixture nhỏ hoặc dữ liệu thật phù hợp. -
Output được lưu đúng đường dẫn quy định. - Có log/config/seed/dataset
revision/git commit khi task tạo kết quả thực nghiệm. - `pytest` và
`ruff check` chạy được. - Đã commit Git.

**Quy tắc workload:** 12 giờ là trần làm việc mỗi tuần. Nếu task trễ,
dùng thứ tự cắt scope ở mục 12; không bù bằng cách làm 15--18 giờ tuần
kế tiếp.

------------------------------------------------------------------------

# 1. Dataset và data policy

## 1.1. Dataset dự kiến

Dataset: Hugging Face `netop/gotsf-ds`, repository code
`netop-team/gotsf`.

Theo dataset card cần xác minh lại trên file thực tế: - 2.880 beams = 30
BS × 3 cell × 32 beam. - Sampling hourly. - Train file có 840 giờ. -
Test 5w--6w có 168 giờ. - Test 10w--11w có 168 giờ. - Bốn KPI: `DLPRB`,
`DLThpVol`, `DLThpTime`, `MR_number`.

Không coi các con số trên là đúng cho đến khi audit xong.

## 1.2. Panel format

Nếu dữ liệu khớp dataset card:

``` python
values.shape == (1008, 2880, 4)
```

Thứ tự KPI:

``` python
KPIS = ["DLPRB", "DLThpVol", "DLThpTime", "MR_number"]
```

Beam ID phải được sắp xếp theo thứ tự số `(bs, cell, beam)`, không sort
lexical. Lưu `week11` riêng, không nối vào panel chính.

## 1.3. Missing và zero policy

-   Giữ NaN trong raw panel; không tự đổi NaN thành zero.
-   Giữ zero như observation hợp lệ cho đến khi có bằng chứng ngược lại.
-   LightGBM dùng native missing handling.
-   GRU dùng median imputer fit trên training segment và missingness
    mask.
-   Nếu target DLPRB là NaN, loại sample đó khỏi training/evaluation và
    ghi số lượng bị loại.
-   Không xóa observation âm thầm.
-   Báo zero ratio, active ratio, active-only MAE và zero-observation
    false-positive rate.

## 1.4. Unit policy

Nếu không xác minh được đơn vị DLPRB, dùng thuật ngữ `DLPRB usage`,
không ghi `DLPRB utilization (%)`.

## 1.5. Temporal boundary policy

Week 6 chỉ được dùng làm external holdout nếu có căn cứ về thứ tự thời
gian liên tục từ tài liệu dataset hoặc metadata có thẩm quyền, đồng thời
file thật khớp sampling interval và thứ tự index.

Không dùng profile correlation đơn thuần để chứng minh continuity.

Nếu không xác minh được continuity, dùng internal chronological holdout
trong 840 giờ train. Khi đó week 6 chỉ là exploratory data, không phải
primary holdout.

Quyết định phải được ghi trong `reports/split_decision.md` trước khi
train model.

------------------------------------------------------------------------

# 2. Temporal split và leakage-free evaluation

Đây là phần bắt buộc phải đúng trước khi train bất kỳ model nào.

## 2.1. Primary split --- chỉ khi week 6 hợp lệ

  Segment            Index Vai trò
  ------------ ----------- ---------------------------------------
  Fit               0--671 Train model và fit preprocessing
  Validation      672--839 Early stopping, model selection
  Holdout        840--1007 Final evaluation, dùng sau khi freeze
  Week 11            Riêng Không thuộc P0

Model validation được fit đến index 671. Vì vậy validation origin đầu
tiên là 671, target đầu tiên là 672.

Sau khi chọn champion, final model được refit trên 0--839. Vì vậy
holdout origin đầu tiên là 839, target đầu tiên là 840.

## 2.2. Fallback split --- nếu week 6 không đủ bằng chứng continuity

  Segment                 Index Vai trò
  ------------------ ---------- -----------------
  Fit                    0--503 Train model
  Validation           504--671 Model selection
  Internal holdout     672--839 Final test

Sau khi chọn champion bằng validation, refit trên 0--671 rồi đánh giá
trên internal holdout.

Không được đổi validation thành final test sau khi đã dùng nó để
tune/select model.

## 2.3. Forecast-origin rule

Tại origin `t`, model chỉ được dùng dữ liệu có index `<= t`.

Một origin hợp lệ phải: 1. Có đủ 168 giờ history. 2. Không sớm hơn
`fit_end`. 3. Có target `t+h` nằm trong evaluation segment. 4. Không
dùng future values trong feature, scaler, static statistics hoặc
imputer.

``` python
def evaluation_origins(fit_end, segment_end, horizon, lookback=168):
    start = max(lookback - 1, fit_end)
    end = segment_end - horizon
    return np.arange(start, end + 1)
```

Với primary split:

  Segment        Horizon Origins       Số lượng
  ------------ --------- ----------- ----------
  Train              +1h 167--670           504
  Train              +6h 167--665           499
  Train             +24h 167--647           481
  Validation         +1h 671--838           168
  Validation         +6h 671--833           163
  Validation        +24h 671--815           145
  Holdout            +1h 839--1006          168
  Holdout            +6h 839--1001          163
  Holdout           +24h 839--983           145

GRU joint training chỉ dùng origins có đủ cả ba target, tức `167–647`
trong primary split.

Với fallback split, áp dụng cùng công thức nhưng thay các mốc
fit/validation/holdout tương ứng.

## 2.4. Rolling-origin evaluation

-   Model được fit cố định ở đầu evaluation segment.
-   Tại các origin tiếp theo, history được cập nhật bằng actual đã quan
    sát đến origin đó.
-   Không retrain model ở mỗi origin.
-   Evaluation stride = 1.
-   Training stride có thể là 2 hoặc 3 để giảm compute.
-   Tất cả model dùng cùng origin set cho từng horizon.

Đây là fixed-model rolling-origin evaluation, không phải
retraining-at-every-step evaluation.

## 2.5. Refit rule

Sau model selection: - LightGBM: refit với
`n_estimators = best_iteration` từ validation. - GRU: refit với số epoch
bằng `best_epoch` từ validation. - Fit lại scaler, imputer, static stats
trên toàn bộ final fit segment. - Tính lại P90/P95 thresholds trên final
fit segment. - Không đọc holdout trong bất kỳ bước selection hoặc
fitting nào.

## 2.6. Leakage tests bắt buộc

-   Target không chồng giữa train/validation/holdout.
-   Rolling features chỉ kết thúc tại origin.
-   Future-perturbation canary: thay dữ liệu sau origin bằng nhiễu,
    features/predictions tại origin không được đổi.
-   Scaler/imputer/static stats không đổi khi perturb dữ liệu ngoài fit
    segment.
-   Holdout runner từ chối chạy nếu chưa có `pre-holdout` tag hoặc hash
    config không khớp.

------------------------------------------------------------------------

# 3. Models và metrics

## 3.1. Baselines --- P0

Giữ ba baselines để workload hợp lý: 1. Persistence:
`pred(t+h) = actual(t)`. 2. Seasonal Naive 24h: `actual(t+h-24)`. 3.
Seasonal Naive 168h: `actual(t+h-168)`.

Seasonal Mean 7d là optional; chỉ thêm nếu hoàn thành sớm và không làm
trễ milestone.

Với horizon +24h, Seasonal Naive 24h trùng Persistence. Ghi rõ điều này,
không coi là hai baseline độc lập.

## 3.2. Global LightGBM --- P0

Ba direct models độc lập cho +1h, +6h, +24h.

Feature groups:

**Origin lags (9):**

``` python
lags = [0, 1, 2, 3, 6, 12, 24, 48, 167]
```

**Target-aligned seasonal features (5):**

``` python
seasonal_lags = [1, 2, 3, 7]
```

Dùng `DLPRB[t + h - 24*k]` và seasonal mean 7 ngày. Với `h <= 24`, các
giá trị này không vượt quá origin.

**Rolling (9):** mean/std/max cho cửa sổ 6, 24, 168 giờ, tất cả kết thúc
tại `t`.

**Phase (2):** `(t+h) % 24`, `(t+h) % 168`. Đây là phase index, không
gán thành giờ lịch.

**KPI phụ (15):** mỗi KPI phụ có lag 0/1/24 và rolling mean 6h/24h.

**Static (4):** mean, std, P90, active ratio của DLPRB trên fit segment.

Tổng 44 features.

Chỉ chạy hai configs:

``` yaml
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

## 3.3. GRU-A và GRU-B --- P0 có giới hạn

**GRU-A:** DLPRB + missingness mask.

**GRU-B:** bốn KPI + missingness masks.

Lookback 168. Static vector gồm beam mean/std đã scale và sin/cos phase
24h, 168h.

Architecture:

``` text
Sequence (168 × channels)
       |
       v
GRU(hidden=64, layers=2, dropout=0.1)
       |
       v
Last hidden state
       |
       +---- Static vector (6)
       |
       v
Concatenate
       |
       v
Linear(70 -> 64) -> ReLU -> Linear(64 -> 3)
       |
       v
Forecast: +1h, +6h, +24h
```

Training config:

``` yaml
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
seeds: [17, 42]
```

Giới hạn compute: - Stratified sample khoảng 600--800 beams để train
GRU. - Phân tầng theo mean, std và active ratio; dùng cùng beam sample
cho GRU-A/B. - Đánh giá trên toàn bộ beams hợp lệ. - Training stride mặc
định 3. - Tổng GPU budget mục tiêu tối đa 4 giờ. - Checkpoint mỗi
epoch. - Colab là GPU chính; Kaggle là fallback. - Không train full GRU
bằng CPU.

Hai seeds mỗi experiment chỉ dùng để báo cáo mean ± standard deviation
mang tính mô tả. Không tuyên bố statistical stability mạnh.

Nếu GPU quota không đủ: ưu tiên GRU-B một seed, bỏ GRU-A; giữ LightGBM
và toàn bộ evaluation protocol.

## 3.4. Metrics

Primary: - MAE theo từng horizon.

Secondary: - RMSE. - WAPE. - Macro-WAPE. - Per-beam macro MAE. -
Active-only MAE. - Zero-observation false-positive rate. - MAE theo load
tier. - Skill score so với baseline tốt nhất.

Không dùng MAPE.

Không mặc định pooled MAE bằng macro MAE. Báo số observation hợp lệ và
số beam bị loại do thiếu dữ liệu.

Load tiers Low/Mid/High được tạo từ tertiles của mean DLPRB từng beam
trên fit segment.

## 3.5. Confidence intervals

-   Paired cluster bootstrap theo BS nếu mapping đã xác minh.
-   500--1.000 repetitions, seed cố định.
-   CI 95% cho chênh lệch MAE.
-   Nếu mapping không đáng tin cậy, bootstrap theo beam và ghi rõ hạn
    chế phụ thuộc giữa các beam.
-   Với 30 BS, diễn giải CI thận trọng; không coi đó là bằng chứng
    generalization sang mọi mạng.

------------------------------------------------------------------------

# 4. Champion selection

Chọn champion chỉ bằng validation.

Ứng viên: - LightGBM. - GRU-A. - GRU-B.

Quy tắc: 1. Tính MAE theo từng horizon. 2. Chuẩn hóa MAE theo best
baseline của từng horizon. 3. Tính trung bình normalized MAE trên ba
horizons. 4. Chọn họ model có điểm thấp nhất. 5. Nếu chênh lệch không rõ
theo paired CI, chọn model đơn giản/rẻ hơn. 6. Chỉ cho phép chọn model
khác nhau theo horizon nếu cải thiện ít nhất 5% và CI hỗ trợ. 7. Nếu ML
model không thắng baseline, ghi kết quả trung thực.

Risk metrics được báo cáo riêng, không dùng holdout để điều chỉnh
champion hoặc threshold.

Trước holdout: - Tạo `configs/champion.yaml`. - Lưu config hash. - Lưu
validation selection report. - Commit. - Tạo Git tag `pre-holdout`.

------------------------------------------------------------------------

# 5. Capacity-Risk Engine

## 5.1. Thresholds

Primary: per-beam P90/P95. - Warning: P90. - Critical: P95.

Threshold được tính từ fit segment, không dùng validation/holdout để
fit.

Global P90/P95 chỉ là optional; nếu thiếu thời gian, bỏ.

## 5.2. Risk state

``` text
NORMAL   : prediction < P90
WARNING  : P90 <= prediction < P95
CRITICAL : prediction >= P95
```

Nếu `P90 == P95`: - Không tạo Warning band giả. - `prediction < P90` →
NORMAL. - `prediction >= P95` → CRITICAL. - Ghi
`warning_band_degenerate = true`.

## 5.3. Guard

Loại beam khỏi risk evaluation nếu: - P90 không vượt epsilon đã định
trước; hoặc - Active ratio thấp hơn ngưỡng cấu hình được chọn từ
training data.

Báo số beam bị loại và lý do.

## 5.4. Evaluation

Actual exceedance: `actual >= threshold`.

Predicted exceedance: `prediction >= threshold`.

Báo: - Precision, Recall, F1, FPR. - TP, FP, FN, TN. - Base rate. -
False alarm count. - Missed exceedance count. - Số beam hợp lệ/bị loại.

So sánh champion với Persistence và Seasonal Naive 168h.

Không positive samples thì metric tương ứng trả NaN kèm cờ, không chia
cho zero.

**Cách diễn giải:** đây là historical threshold-exceedance monitoring
trên ngưỡng thống kê, không phải congestion/incident detection.

------------------------------------------------------------------------

# 6. Repository structure

``` text
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
│   └── build_report.py
├── tests/
│   ├── unit/
│   └── integration/
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

------------------------------------------------------------------------

# 7. Notebook, Colab và Kaggle

Notebook chỉ điều phối. Logic tái sử dụng phải nằm trong `src/` và
`scripts/`.

  -----------------------------------------------------------------------------
  Notebook                      Nội dung                Nơi chạy chính
  ----------------------------- ----------------------- -----------------------
  `01_data_audit_eda.ipynb`     Download, panel, audit, Local hoặc Colab CPU
                                EDA, split decision     

  `02_baselines.ipynb`          Origins, metrics,       Local hoặc Colab CPU
                                baselines, benchmark    

  `03_lightgbm.ipynb`           Feature build,          Colab CPU / local
                                LightGBM, validation    

  `04_gru.ipynb`                GRU training,           **Colab GPU**, Kaggle
                                checkpoint, ablation    fallback

  `05_final_evaluation.ipynb`   Champion, refit,        Colab CPU/GPU hoặc
                                holdout, risk, plots    local
  -----------------------------------------------------------------------------

## 7.1. Colab workflow

1.  Clone GitHub repo.
2.  Mount Google Drive.
3.  Cài `requirements-ml.txt` hoặc cài dependency còn thiếu.
4.  Copy dữ liệu cần dùng vào `/content` để giảm I/O.
5.  Chạy script từ notebook.
6.  Lưu checkpoint/artifacts quan trọng về Drive sau mỗi epoch.
7.  Ghi GPU type, runtime, duration vào `run_meta.json`.
8.  Không dựa vào `/content` để lưu file duy nhất.

## 7.2. Kaggle fallback

Dùng cùng code và config với Colab.

-   Upload/attach dataset hoặc lấy dữ liệu từ nguồn công khai.
-   Lưu checkpoint trong Kaggle working/output và tải về Drive khi cần.
-   Không giả định quota cố định; ghi quota thực tế thấy trong tài
    khoản.
-   Không train lại từ đầu nếu đã có checkpoint hợp lệ.

## 7.3. Run metadata

Mỗi run lưu:

``` text
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

------------------------------------------------------------------------

# 8. Kế hoạch triển khai chi tiết 6 tuần

**Tổng: 72 giờ. Mỗi tuần tối đa 12 giờ, đã bao gồm buffer.**

## WEEK 1 --- Data engineering và audit

**Ngân sách: 12 giờ**

**Mục tiêu:** tạo panel đáng tin cậy, xác minh dataset và freeze
temporal split.

### T1.1 --- Repo và môi trường (1h)

Làm: - Tạo GitHub repo `5g-network-intelligence`. - Tạo virtual
environment. - Tạo `pyproject.toml`, `.gitignore`. - Tạo
`requirements-ml.txt`, `requirements-app.txt`. - Setup Colab, Drive và
Kaggle.

Code: - `src/utils/config.py::load_config(path)` -
`src/utils/seed.py::set_seed(seed)` -
`src/utils/metadata.py::collect_run_metadata(config)`

Output: - Repo có thể cài dependencies. - `pytest -q` và `ruff check`
chạy được. - `docs/infra.md` ghi môi trường và tình trạng GPU thực tế.

### T1.2 --- Download dataset và manifest (1h)

Code: - `src/data/io.py::download_dataset(repo_id, revision, dest)` -
`src/data/io.py::build_manifest(raw_dir)` -
`scripts/prepare_data.py --step download`

Output: - Các CSV cần thiết nằm trong `data/raw/`. -
`reports/data_manifest.json` có revision, license, checksum, size, rows,
columns. - Chạy lại không tải lại file có checksum khớp.

### T1.3 --- Panel builder (2h)

Code: - `src/data/schema.py`: KPI constants và `parse_beam_id`. -
`src/data/panel.py`: `Panel`, `build_panel`, `save_panel`, `load_panel`.

Làm: - Align Time và beam giữa bốn KPI. - Numeric-sort beam IDs. - Giữ
NaN/zero. - Ghép train + week 6 chỉ khi split decision cho phép. - Lưu
week 11 riêng.

Output: - `data/processed/panel.npz` -
`data/processed/panel_meta.json` - `data/processed/week11.npz` -
Save/load parity test pass.

### T1.4 --- Data audit và EDA (2.5h)

Code: - `src/data/audit.py::audit_missing` - `audit_zeros` -
`audit_range` - `audit_duplicates` - `audit_beams`

Notebook: - `notebooks/01_data_audit_eda.ipynb`

Output: - `reports/data_audit.md` - `docs/data_dictionary.md` - Hình
phân phối DLPRB, zero ratio, time series mẫu, phase 24h/168h. - Có quyết
định NaN policy, zero policy và unit policy.

### T1.5 --- Temporal boundary và split decision (2h)

Code: - `src/data/audit.py::check_mapping` -
`src/data/audit.py::check_temporal_boundary` -
`src/data/splits.py::build_split_config`

Output: - `reports/split_decision.md` - `configs/data.yaml` -
`configs/evaluation.yaml` - `data/processed/split.json`

Phải chốt một trong hai: - Primary: external week 6 holdout. - Fallback:
internal chronological holdout.

Không train model trước khi task này xong.

### T1.6 --- Tests và buffer (1.5h)

Files: - `tests/unit/test_panel.py` - `tests/unit/test_audit.py`

Output: - Test sorting, alignment, NaN, zero, save/load, split metadata
pass. - Commit `week1-data-audit`.

### Buffer tuần 1 (2h)

Dùng cho lỗi download/schema hoặc sửa data pipeline. Nếu không dùng hết,
không tự thêm scope.

**Gate G1:** panel, audit, data policy và split decision hoàn tất.

------------------------------------------------------------------------

## WEEK 2 --- Evaluation framework và baselines

**Ngân sách: 12 giờ**

**Mục tiêu:** có evaluator đúng index, leakage tests và baseline
benchmark.

### T2.1 --- Origin generator (2h)

Code: - `src/evaluation/origins.py::train_origins` -
`evaluation_origins` - `train_origins_joint` -
`assert_no_target_overlap`

Output: - `tests/unit/test_origins.py` - `data/processed/split.json` có
origins theo từng horizon. - Test các mốc origin ở mục 2 pass.

### T2.2 --- Metrics (1.5h)

Code: - `src/evaluation/metrics.py`

Functions: - `mae` - `rmse` - `wape` - `macro_wape` - `macro_mae` -
`active_only_mae` - `zero_fp_rate` - `assign_load_tiers`

Output: - `tests/unit/test_metrics.py` - Mảng tính tay cho từng
metric. - WAPE mẫu số zero trả NaN có cờ. - NaN được loại và đếm rõ.

### T2.3 --- Evaluator (2h)

Code: - `src/evaluation/evaluator.py` - `Forecaster` protocol. -
`evaluate(forecaster, panel, origins, horizon)`. - `save_forecasts`. -
`to_benchmark_rows`.

Output: - `tests/unit/test_evaluator.py` - Oracle predictor cho MAE =
0. - Constant predictor cho MAE đúng giá trị tính tay. - Forecast record
lưu được.

### T2.4 --- Baselines (1.5h)

Code: - `src/models/baselines.py` - `Persistence` -
`SeasonalNaive(period=24)` - `SeasonalNaive(period=168)`

Output: - `tests/unit/test_baselines.py` - Mọi tham chiếu đều
`<= origin`. - Horizon 24 seasonal naive 24h trùng Persistence. - Không
đọc index âm.

### T2.5 --- Leakage canary (1.5h)

Code: - `tests/utils/perturb.py` - `tests/unit/test_leakage.py`

Làm: - Perturb toàn bộ dữ liệu sau origin. - Kiểm tra feature/prediction
tại origin không đổi. - Tạo một forecaster cố tình đọc future để chứng
minh canary phát hiện được lỗi.

Output: - Canary pass cho baselines. - Test-of-test chứng minh canary
fail với forecaster cố tình leakage.

### T2.6 --- Baseline benchmark và report (1.5h)

Code: - `scripts/run_baselines.py --segment val` -
`notebooks/02_baselines.ipynb`

Output: - `reports/baseline_benchmark_val.csv` -
`reports/forecasts/baselines_val.npz` - `reports/baseline_analysis.md`

Báo best baseline theo horizon, load tiers, active/zero và top beam
errors.

### Buffer tuần 2 (2h)

Dùng cho index bug, evaluator mismatch hoặc sửa leakage canary.

**Gate G2:** evaluator, origins, baselines và leakage tests pass.

------------------------------------------------------------------------

## WEEK 3 --- LightGBM forecasting

**Ngân sách: 12 giờ**

**Mục tiêu:** hoàn thành global LightGBM cho ba horizons và có benchmark
validation.

### T3.1 --- Static stats và feature builder (3h)

Code: - `src/features/static_stats.py::compute_static` -
`src/features/lightgbm_features.py::build_features`

Làm: - Tạo 44 features theo mục 3.2. - Vectorize theo origin/beam. -
Không materialize toàn bộ sliding windows nếu không cần. - Static stats
chỉ fit trên fit segment. - Training stride 2; validation stride 1.

Output: - `tests/unit/test_lightgbm_features.py` -
`reports/feature_manifest.json` - Feature names đúng 44. - Test tính tay
ít nhất 10 feature. - Future-perturbation canary pass.

### T3.2 --- LightGBM train/evaluate pipeline (2.5h)

Code: - `src/models/lgbm.py::train_lgbm` - `LGBMForecaster` -
`save_lgbm` - `load_lgbm` - `scripts/train_lgbm.py`

Làm: - Train từng horizon. - Early stopping trên validation. - Lưu best
iteration. - Lưu feature names, config, seed và metadata.

Output: - `artifacts/lgbm/{config}/h{h}/model.txt` -
`artifacts/lgbm/{config}/h{h}/meta.json` - Save/load prediction parity
pass.

### T3.3 --- So sánh hai config (1.5h)

Chạy C1 và C2.

Output: - `reports/lightgbm_benchmark_val.csv` - `configs/lightgbm.yaml`
có config được chọn. - Không chạy grid search ngoài hai configs.

### T3.4 --- Error analysis (2h)

Code: - `src/evaluation/reports.py` -
`scripts/evaluate_models.py --model lightgbm --segment val` -
`notebooks/03_lightgbm.ipynb`

Output: - `reports/lightgbm_analysis.md` - MAE theo horizon/load tier. -
Active-only và zero FP. - Feature importance. - Ít nhất ba failure
examples. - Skill score so với best baseline.

### T3.5 --- Sanity checks và buffer (3h)

Kiểm tra: - LightGBM h=1 không bất thường so với Persistence. - Feature
canary pass. - Model save/load pass. - Metrics tái lập được. - Chạy lại
benchmark từ config.

Nếu LightGBM thua baseline, debug index/features trước; không tune vô
hạn.

**Gate G3:** LightGBM hoàn thành, validation benchmark và error analysis
có thể tái lập.

------------------------------------------------------------------------

## WEEK 4 --- GRU và model comparison

**Ngân sách: 12 giờ**

**Mục tiêu:** có GRU baseline đủ tốt để so sánh, không biến tuần này
thành cuộc chạy GPU vô hạn.

### T4.1 --- GRU input pipeline (2h)

Code: - `src/features/beam_sampling.py` - `src/features/gru_inputs.py`

Làm: - Stratified beam sample 600--800 beams. - Fit imputer/scaler trên
fit segment. - Tạo missingness masks. - Tạo static vector 6 chiều. - Tạo
input tensors và targets cho ba horizons.

Output: - `tests/unit/test_gru_inputs.py` -
`reports/gru_sample_manifest.json` - Test shape, target alignment,
scaler boundary và future canary pass.

### T4.2 --- GRU architecture và training loop (2.5h)

Code: - `src/models/gru.py::GRUNet` -
`src/models/gru_train.py::train_gru` - `scripts/train_gru.py`

Làm: - Implement GRU-A/B. - AdamW + L1. - Gradient clipping. - Early
stopping. - Checkpoint/resume.

Output: - `tests/unit/test_gru.py` - `tests/unit/test_gru_train.py` -
Overfit test trên synthetic mini-batch. - Resume test chạy tiếp được sau
khi dừng.

### T4.3 --- GPU pilot (1h)

Làm: - Chạy một epoch trên Colab GPU. - Nếu Colab không có GPU phù hợp,
chuyển Kaggle. - Đo seconds/epoch, VRAM, sample count.

Output: - `reports/gru_compute.md` - Chốt batch size, stride, sample
beams và số seeds trong `configs/gru.yaml`.

### T4.4 --- Training runs (2.5h active time)

Ưu tiên: - GRU-A seed 17. - GRU-A seed 42. - GRU-B seed 17. - GRU-B seed
42.

Checkpoint mỗi epoch vào Drive hoặc Kaggle Output.

Output: - `artifacts/gru/{A|B}/seed{seed}/best.pt` - `history.jsonl` -
`meta.json`

Nếu compute không đủ: - Bỏ GRU-A trước. - Giữ GRU-B một seed. - Ghi rõ
đây là exploratory neural benchmark. - Không chạy CPU training dài.

### T4.5 --- Validation comparison (1.5h)

Code: - `scripts/evaluate_models.py --model gru --segment val` -
`GRUForecaster`

Output: - `reports/gru_benchmark_val.csv` -
`reports/ablation_study.csv` - `reports/gru_error_analysis.md`

So sánh: - GRU-A vs GRU-B. - GRU vs LightGBM. - GRU vs best baseline.

Báo mean ± sd theo seeds, nhưng không khẳng định mạnh khi chỉ có hai
seeds.

### Buffer tuần 4 (2.5h)

Dành cho GPU disconnect, checkpoint resume, lỗi tensor alignment và đánh
giá.

**Gate G4:** ít nhất GRU-B chạy được và được đánh giá; GRU-A/B ablation
hoàn thành nếu compute cho phép.

------------------------------------------------------------------------

## WEEK 5 --- Champion, holdout và risk engine

**Ngân sách: 12 giờ**

**Mục tiêu:** đóng băng model, đánh giá holdout một cách độc lập và hoàn
thành risk analysis.

### T5.1 --- Risk engine (2.5h)

Code: - `src/risk/thresholds.py::compute_thresholds` -
`src/risk/scorer.py::score` -
`src/risk/evaluator.py::evaluate_exceedance` -
`src/risk/aggregator.py::aggregate` - `configs/risk.yaml`

Output: - `tests/unit/test_risk.py` - Test P90/P95 boundaries. - Test
P90 = P95. - Test guard. - Test precision/recall/FPR bằng ví dụ tính
tay. - No-positive case trả NaN có cờ.

### T5.2 --- Validation risk (1h)

Code: - `scripts/run_risk.py --segment val`

Output: - `reports/risk_validation.csv` - So sánh champion candidates và
baselines. - Threshold chỉ fit từ validation fit segment.

### T5.3 --- Champion selection và freeze (1h)

Code: - `scripts/select_champion.py` - `configs/champion.yaml`

Output: - `reports/champion_selection.md` - Champion selection dựa trên
validation. - Ghi normalized MAE, CI, config hash, dataset revision. -
Git tag `pre-holdout`.

### T5.4 --- Final refit (1.5h)

Code: - `scripts/refit_champion.py`

Làm: - Refit champion trên final fit segment. - Lưu
scaler/imputer/static stats. - Fit thresholds trên final fit segment. -
Không sử dụng holdout.

Output: - `artifacts/champion/` - `champion_meta.json` -
`thresholds.npz` - Load lại artifact thành công.

### T5.5 --- Holdout một lần (1.5h)

Code: - `scripts/run_holdout.py`

Làm: - Guard `pre-holdout` tag và config hash. - Chạy champion và
baselines. - Tính metrics. - Bootstrap CI. - Không tune model sau khi
xem kết quả.

Output: - `reports/holdout_benchmark.csv` -
`reports/forecasts/champion_holdout.npz` - `reports/holdout_run.log`

Nếu phát hiện bug thực sự phải chạy lại, ghi lại lý do và lịch sử lần
chạy.

### T5.6 --- Risk holdout và error analysis (2h)

Output: - `reports/risk_holdout.csv` - `reports/risk_evaluation.md` -
`reports/holdout_analysis.md`

Báo: - Precision/Recall/F1/FPR. - False alarms và missed exceedances. -
Base rate. - Per-horizon results. - Load-tier analysis. - Top 10 beam
errors. - Risk distribution.

### T5.7 --- Buffer và model card draft (2.5h)

Files: - `docs/model_card.md` - `docs/limitations.md` -
`notebooks/05_final_evaluation.ipynb`

Ghi rõ: - Holdout chỉ có một tuần hoặc một internal window. - Cùng beam
đã xuất hiện trong training. - Chưa đánh giá generalization sang BS
mới. - Threshold là statistical scenario thresholds. - Không có
ground-truth congestion/incident labels. - Không phải realtime
telemetry.

**Gate G5:** champion được freeze trước holdout; holdout benchmark và
risk evaluation hoàn thành.

------------------------------------------------------------------------

## WEEK 6 --- Inference, API, dashboard, Docker và release

**Ngân sách: 12 giờ**

**Mục tiêu:** người khác có thể chạy thử inference mà không cần mở
notebook.

### T6.1 --- Replay store và Predictor (2h)

Code: - `src/inference/replay_store.py` - `src/inference/predictor.py`

Functions:

``` python
class ReplayStore:
    def history(self, origin, beam_id, lookback=168):
        ...

class Predictor:
    @classmethod
    def from_artifacts(cls, artifact_dir):
        ...

    def predict(self, origin, beam_ids):
        ...

    def predict_risk(self, origin, beam_ids, threshold_mode="per_beam"):
        ...
```

Output: - `tests/unit/test_predictor.py` - Parity với offline evaluator
trên tối thiểu 20 cặp beam/origin. - Invalid origin trả lỗi rõ. -
Artifact chỉ load một lần.

### T6.2 --- FastAPI (2h)

Code: - `src/api/schemas.py` - `src/api/main.py`

P0 endpoints:

``` text
GET  /health
GET  /model-info
POST /forecast
POST /risk
```

Optional nếu còn thời gian:

``` text
GET /replay/meta
GET /replay/series
```

Output: - `tests/integration/test_api.py` - Pydantic validation. - 404
beam không tồn tại. - 422 origin không hợp lệ. - Swagger `/docs`. - API
response khớp offline predictor.

### T6.3 --- Streamlit dashboard (2h)

Code: - `app/backends.py` - `app/dashboard.py`

Một trang gồm:

**Overview** - Số beam/cell/BS nếu mapping hợp lệ. - Risk
distribution. - Top 10 beams có forecast cao.

**Forecast detail** - Chọn beam. - Chọn origin. - Chọn horizon. - Actual
history. - Forecast. - P90/P95. - Risk state.

Phải hiển thị:

`Historical/offline replay — not live network telemetry.`

Output: - Dashboard chạy được. - Các giá trị chính khớp API. - Không cần
nhiều trang hoặc animation phức tạp.

### T6.4 --- Docker Compose (1.5h)

Files: - `Dockerfile.api` - `Dockerfile.dashboard` -
`docker-compose.yml` - `.dockerignore`

Output: - `docker compose up --build` chạy được. - `/health` trả 200. -
Dashboard mở được. - Artifacts và replay store mount read-only khi phù
hợp. - Ghi cấu hình máy và image size vào README.

### T6.5 --- Integration, parity và smoke tests (1.5h)

Files: - `tests/integration/test_smoke.py` -
`tests/unit/test_artifact_io.py`

Kiểm tra: - Load artifact. - Predict. - Score risk. - API response. -
Dashboard backend. - Offline/API parity. - Toàn bộ `pytest`. -
`ruff check`.

### T6.6 --- Documentation và release (2h)

Files: - `README.md` - `docs/architecture.md` - `docs/model_card.md` -
`docs/limitations.md` - `scripts/build_report.py`

README phải có: - Problem statement. - Dataset và license. -
Architecture diagram. - Model comparison table. - Holdout results. -
Risk results. - Demo instructions. - Reproducibility. - Limitations. -
Screenshots.

Output: - `reports/experiment_report.md` - `docs/architecture.md` -
`README.md` - Git tag `v1.0.0`.

### Buffer tuần 6 (1h)

Dùng cho lỗi Docker, clean-clone và chỉnh README.

**Gate G6:** inference/API/dashboard chạy được; Docker Compose hoạt
động; tests pass; repo có tài liệu hoàn chỉnh.

------------------------------------------------------------------------

# 9. Lịch tổng hợp 6 tuần

  Tuần       Trọng tâm                                             Giờ
  ---------- ------------------------------------------------ --------
  1          Data engineering, audit, split decision                12
  2          Evaluation framework, baselines, leakage tests         12
  3          LightGBM và validation benchmark                       12
  4          GRU, GPU training, ablation                            12
  5          Champion, holdout, risk engine                         12
  6          Predictor, API, dashboard, Docker, release             12
  **Tổng**                                                      **72**

Mỗi tuần có buffer bên trong ngân sách. Không chuyển buffer thành yêu
cầu phải làm thêm feature.

------------------------------------------------------------------------

# 10. Phương án rút xuống 5 tuần

Chỉ dùng nếu deadline bắt buộc là 5 tuần. Tổng tối đa 60 giờ.

Giữ nguyên những thứ bảo vệ scientific validity: - Data audit. -
Split. - Origin alignment. - Leakage tests. - Baselines. - LightGBM. -
Final holdout. - Risk evaluation. - Inference tối thiểu.

Cắt theo thứ tự:

  Hạng mục            Bản 6 tuần          Bản 5 tuần
  ------------------- ------------------- --------------------------
  GRU-A               Có                  Bỏ
  GRU-B               2 seeds             1 seed
  GRU ablation        Có                  Không
  LightGBM configs    2                   1
  Bootstrap           1.000               500
  Global thresholds   Optional            Bỏ
  Dashboard           Overview + detail   Forecast detail tối giản
  Docker Compose      Có                  Optional
  CI                  Optional            Bỏ
  Latency benchmark   Optional            Bỏ
  Demo public         Optional            Bỏ

## Lịch 5 tuần

  Tuần   Task                                                        Giờ
  ------ --------------------------------------------------------- -----
  1      Data audit, panel, split                                     12
  2      Origins, evaluator, baselines, leakage tests                 12
  3      LightGBM, GRU-B một seed, validation                         12
  4      Champion, refit, holdout, risk                               12
  5      Predictor, FastAPI, dashboard tối giản, README và tests      12

Nếu 5 tuần: - Không cố làm GRU-A/B ablation. - Không cố thêm Docker nếu
API/dashboard chưa ổn. - Không bỏ holdout hoặc leakage checks. - Không
tuyên bố kết quả GRU ổn định từ một seed.

------------------------------------------------------------------------

# 11. Final acceptance checklist

## Data & methodology

-   [ ] Dataset revision và checksum được lưu.
-   [ ] Panel đúng schema.
-   [ ] NaN/zero policy được ghi rõ.
-   [ ] Temporal boundary được xác minh hoặc fallback split được dùng.
-   [ ] Forecast origins đúng.
-   [ ] Không có target overlap.
-   [ ] Future-perturbation canary pass.
-   [ ] Scaler/imputer/static stats không leakage.

## Modeling

-   [ ] Ba baselines.
-   [ ] LightGBM ba horizons.
-   [ ] GRU-B ít nhất một run.
-   [ ] GRU-A/B ablation nếu đủ compute.
-   [ ] Validation benchmark.
-   [ ] Metrics theo load tier.
-   [ ] Active/zero analysis.
-   [ ] Bootstrap CI.

## Holdout & risk

-   [ ] Champion chọn bằng validation.
-   [ ] Config freeze trước holdout.
-   [ ] Refit đúng fit segment.
-   [ ] Holdout chỉ dùng sau freeze.
-   [ ] Risk threshold fit trên training data.
-   [ ] P90=P95 được xử lý.
-   [ ] Risk so với baseline.
-   [ ] Không claim congestion/incident detection.

## Engineering

-   [ ] Independent Predictor.
-   [ ] Offline/API parity.
-   [ ] FastAPI.
-   [ ] Streamlit.
-   [ ] Docker Compose trong bản 6 tuần.
-   [ ] Unit tests.
-   [ ] Integration tests.
-   [ ] Clean-clone test.

## Portfolio

-   [ ] README.
-   [ ] Architecture diagram.
-   [ ] Experiment report.
-   [ ] Model card.
-   [ ] Limitations.
-   [ ] Screenshots.
-   [ ] Git tag release.
-   [ ] CV bullets chỉ dùng số liệu đã đo.

------------------------------------------------------------------------

# 12. Scope control và thứ tự cắt khi trễ

Nếu trễ tiến độ, cắt theo thứ tự:

1.  Demo public.
2.  CI automation.
3.  Latency benchmark.
4.  Global threshold mode.
5.  Dashboard overview, giữ forecast detail.
6.  GRU-A ablation.
7.  GRU seeds từ hai xuống một.
8.  GRU training sample giảm còn 400--600 beams.

Không cắt: - Data audit. - Split và origin alignment. - Leakage tests. -
Baselines. - LightGBM. - Final holdout. - Risk engine cơ bản. -
Independent inference. - Critical tests. - README và limitations.

**Nguyên tắc:** giảm độ rộng sản phẩm và số thí nghiệm, không giảm tính
hợp lệ của evaluation.

------------------------------------------------------------------------

# 13. Coding workflow và quản lý thời gian

## 13.1. Mỗi task chỉ làm theo một vòng

1.  Đọc yêu cầu task.
2.  Xác định file được phép sửa.
3.  Viết code.
4.  Review assumptions.
5.  Chạy unit test.
6.  Chạy trên fixture.
7.  Chạy trên dữ liệu thật.
8.  Kiểm tra output.
9.  Commit.

Không yêu cầu coding agent viết lại cả repository.

## 13.2. Các module phải tự review kỹ

-   `src/evaluation/origins.py`
-   `src/evaluation/metrics.py`
-   `src/features/lightgbm_features.py`
-   `src/features/gru_inputs.py`
-   `src/risk/thresholds.py`
-   `scripts/select_champion.py`
-   `scripts/run_holdout.py`

## 13.3. Quy tắc cuối mỗi tuần

Ghi vào `reports/weekly_log.md`:

``` text
Week:
Hours used:
Tasks completed:
Tasks unfinished:
Blocking issue:
Decision:
Next week's priority:
```

Nếu một task vượt estimate: - Không cố làm hết trong ngày. - Ghi nguyên
nhân. - Dùng buffer. - Nếu hết buffer, áp dụng scope cut.

## 13.4. Những điều tuyệt đối không làm

1.  Không thêm Transformer/GNN/TCN.
2.  Không random split.
3.  Không tune bằng holdout.
4.  Không coi zero là missing.
5.  Không gọi threshold exceedance là fault detection.
6.  Không claim realtime khi chỉ replay dữ liệu lịch sử.
7.  Không đưa metric chưa đo vào CV.
8.  Không làm dashboard trước khi predictor parity pass.
9.  Không train GRU trên CPU hàng giờ.
10. Không vượt 12 giờ làm việc mỗi tuần để cứu timeline.

------------------------------------------------------------------------

# 14. Kết quả dự án sau 6 tuần

Nếu hoàn thành đúng P0, sản phẩm cuối cùng sẽ có:

**Research layer** - Data audit. - Leakage-aware evaluation. - Baseline
benchmark. - LightGBM/GRU comparison. - Multi-horizon forecasting. -
Bootstrap confidence intervals. - Holdout analysis. - Risk evaluation.

**Engineering layer** - Config-driven training. - Reproducible
artifacts. - Independent inference. - FastAPI. - Streamlit dashboard. -
Docker Compose. - Automated tests.

**Portfolio layer** - GitHub repository sạch. - README có kết quả thực
nghiệm. - Architecture diagram. - Experiment report. - Model card. -
Screenshots/demo.

Dự án được định vị là một **Applied ML + ML Engineering prototype cho 5G
KPI forecasting**, không phải hệ thống vận hành mạng viễn thông
production.

## Nguyên tắc cuối cùng

Mục tiêu không phải là có nhiều model hoặc nhiều công nghệ nhất.

Mục tiêu là hoàn thành một hệ thống: - Có bài toán rõ. - Có data
pipeline đáng tin cậy. - Có benchmark công bằng. - Có holdout hợp lệ. -
Có kết quả được giải thích. - Có inference chạy được ngoài notebook. -
Có tài liệu để người khác tái lập.

**Ưu tiên hoàn thành bản 6 tuần với chất lượng tốt hơn là cố nhồi bản 5
tuần và tạo ra một dự án có nhiều thành phần nhưng thiếu độ tin cậy.**
