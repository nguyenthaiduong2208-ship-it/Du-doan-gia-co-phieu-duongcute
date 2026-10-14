
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# =========================
# CAU HINH UNG DUNG
# =========================
st.set_page_config(
    page_title="Du doan gia co phieu",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Dự đoán giá đóng cửa cổ phiếu Việt Nam")
st.write(
    "Mô hình LSTM sử dụng dữ liệu lịch sử để dự đoán "
    "giá Close của các phiên giao dịch tiếp theo."
)

# =========================
# NHAP THONG TIN
# =========================
with st.sidebar:
    st.header("Cấu hình dự báo")

    ticker_input = st.text_input(
        "Mã cổ phiếu",
        value="VNM",
        help="Ví dụ: VNM, FPT, HPG, MWG, VCB"
    ).strip().upper()

    forecast_days = st.number_input(
        "Số phiên muốn dự đoán",
        min_value=1,
        max_value=30,
        value=10,
        step=1
    )

    start_date = st.date_input(
        "Ngày bắt đầu lấy dữ liệu",
        value=pd.Timestamp("2018-01-01").date()
    )

    run_button = st.button(
        "Bắt đầu dự đoán",
        type="primary",
        use_container_width=True
    )

st.caption(
    "Lưu ý: Đây là dự báo tham khảo, không phải cam kết "
    "về giá cổ phiếu trong tương lai."
)

# =========================
# HAM CHUAN BI DU LIEU
# =========================
def get_close_data(ticker, start_date):
    symbol = ticker + ".VN"

    df = yf.download(
        symbol,
        start=str(start_date),
        end=(pd.Timestamp.today() + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        auto_adjust=False,
        progress=False
    )

    if df.empty:
        raise ValueError(
            f"Không tải được dữ liệu {symbol}. "
            "Hãy kiểm tra mã cổ phiếu hoặc thử lại sau."
        )

    # Xử lý trường hợp yfinance trả về MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
    else:
        close = df["Close"]

    close = pd.to_numeric(close, errors="coerce").dropna()
    close = close[~close.index.duplicated(keep="last")]

    if len(close) < 100:
        raise ValueError(
            "Dữ liệu lịch sử quá ít. Hãy chọn ngày bắt đầu sớm hơn "
            "hoặc kiểm tra lại mã cổ phiếu."
        )

    return close, symbol


# =========================
# HAM TAO CHUOI DU LIEU
# =========================
def create_sequences(data, window_size=60):
    X, y = [], []

    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i])
        y.append(data[i])

    return np.array(X), np.array(y)


# =========================
# CHAY MO HINH
# =========================
if run_button:
    if not ticker_input:
        st.error("Vui lòng nhập mã cổ phiếu.")
        st.stop()

    # Giữ kết quả ổn định hơn giữa các lần chạy
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

    try:
        with st.spinner("Đang tải dữ liệu cổ phiếu..."):
            close, symbol = get_close_data(
                ticker_input,
                start_date
            )

        st.success(f"Đã tải dữ liệu cho {symbol}.")

        data = close.to_numpy(dtype=float).reshape(-1, 1)

        # Chia tập train/test theo thứ tự thời gian
        train_size = int(0.8 * len(data))
        train_data = data[:train_size]
        test_data = data[train_size:]

        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(train_data)
        test_scaled = scaler.transform(test_data)

        window_size = 60

        X_train, y_train = create_sequences(
            train_scaled,
            window_size
        )

        test_combined = np.vstack(
            (train_scaled[-window_size:], test_scaled)
        )

        X_test, y_test = create_sequences(
            test_combined,
            window_size
        )

        if len(X_train) < 50 or len(X_test) < 1:
            st.error("Không đủ dữ liệu để huấn luyện và kiểm tra mô hình.")
            st.stop()

        # Tách validation từ phần cuối tập train
        val_size = max(1, int(0.1 * len(X_train)))

        X_train_final = X_train[:-val_size]
        y_train_final = y_train[:-val_size]

        X_val = X_train[-val_size:]
        y_val = y_train[-val_size:]

        if len(X_train_final) == 0:
            st.error("Không đủ dữ liệu để huấn luyện mô hình.")
            st.stop()

        # Xây dựng mô hình LSTM
        model = Sequential([
            tf.keras.Input(shape=(window_size, 1)),
            LSTM(50, return_sequences=True),
            LSTM(50),
            Dense(1)
        ])

        model.compile(
            optimizer="adam",
            loss="mean_squared_error"
        )

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        )

        progress = st.progress(0)
        status = st.empty()

        status.info("Đang huấn luyện mô hình LSTM...")

        history = model.fit(
            X_train_final,
            y_train_final,
            validation_data=(X_val, y_val),
            epochs=10,
            batch_size=32,
            shuffle=False,
            callbacks=[early_stop],
            verbose=0
        )

        progress.progress(60)
        status.info("Đang đánh giá mô hình...")

        # Dự đoán tập test
        test_pred_scaled = model.predict(X_test, verbose=0)

        test_pred = scaler.inverse_transform(test_pred_scaled)
        y_test_real = scaler.inverse_transform(y_test)

        rmse = np.sqrt(
            mean_squared_error(y_test_real, test_pred)
        )
        mae = mean_absolute_error(y_test_real, test_pred)

        if len(y_test_real) > 1:
            r2 = r2_score(y_test_real, test_pred)
        else:
            r2 = np.nan

        # Dự báo các phiên tương lai
        last_60_days = scaler.transform(data)[-window_size:]
        last_60_days = last_60_days.reshape(1, window_size, 1)

        future_predictions_scaled = []

        for _ in range(int(forecast_days)):
            next_pred = model.predict(
                last_60_days,
                verbose=0
            )

            future_predictions_scaled.append(next_pred[0, 0])

            last_60_days = np.concatenate(
                [
                    last_60_days[:, 1:, :],
                    next_pred.reshape(1, 1, 1)
                ],
                axis=1
            )

        future_predictions = scaler.inverse_transform(
            np.array(future_predictions_scaled).reshape(-1, 1)
        ).flatten()

        progress.progress(100)
        status.success("Hoàn tất dự báo!")

        # =========================
        # HIEN THI KET QUA
        # =========================
        st.subheader(f"Kết quả dự báo: {symbol}")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Giá Close gần nhất",
            f"{data[-1, 0]:,.2f}"
        )

        col2.metric(
            f"Dự báo sau {int(forecast_days)} phiên",
            f"{future_predictions[-1]:,.2f}"
        )

        col3.metric(
            "RMSE trên tập kiểm tra",
            f"{rmse:,.2f}"
        )

        st.caption(
            "Đơn vị giá theo dữ liệu Yahoo Finance; "
            "cần kiểm tra quy ước giá và điều chỉnh của nguồn dữ liệu."
        )

        # Bảng dự báo
        # Ngày giao dịch tương lai chưa biết trước chính xác.
        # Dùng số thứ tự phiên để tránh gán ngày lịch sai.
        future_table = pd.DataFrame({
            "Phiên dự báo": [
                f"Phiên +{i}"
                for i in range(1, int(forecast_days) + 1)
            ],
            "Giá Close dự báo": future_predictions
        })

        st.subheader("Bảng giá Close dự báo")
        st.dataframe(
            future_table.style.format({
                "Giá Close dự báo": "{:,.2f}"
            }),
            use_container_width=True,
            hide_index=True
        )

        csv = future_table.to_csv(
            index=False
        ).encode("utf-8-sig")

        st.download_button(
            "Tải bảng dự báo CSV",
            data=csv,
            file_name=f"du_bao_{ticker_input}.csv",
            mime="text/csv"
        )

        # Biểu đồ dự báo tương lai
        st.subheader("Biểu đồ dự báo các phiên tiếp theo")

        fig, ax = plt.subplots(figsize=(12, 5))

        ax.plot(
            range(1, len(data) + 1),
            data.flatten(),
            label="Giá Close lịch sử",
            linewidth=1
        )

        future_x = range(
            len(data),
            len(data) + int(forecast_days)
        )

        ax.plot(
            future_x,
            future_predictions,
            marker="o",
            label="Giá Close dự báo",
            color="orange"
        )

        ax.axvline(
            len(data),
            linestyle="--",
            alpha=0.6,
            label="Bắt đầu dự báo"
        )

        ax.set_xlabel("Thứ tự phiên")
        ax.set_ylabel("Giá Close")
        ax.set_title(f"Dự báo giá đóng cửa {symbol}")
        ax.legend()
        ax.grid(alpha=0.25)

        st.pyplot(fig)
        plt.close(fig)

        # Biểu đồ loss
        st.subheader("Quá trình huấn luyện")

        loss_fig, loss_ax = plt.subplots(figsize=(10, 4))

        loss_ax.plot(
            history.history["loss"],
            label="Train Loss"
        )
        loss_ax.plot(
            history.history["val_loss"],
            label="Validation Loss"
        )

        loss_ax.set_xlabel("Epoch")
        loss_ax.set_ylabel("MSE Loss")
        loss_ax.legend()
        loss_ax.grid(alpha=0.25)

        st.pyplot(loss_fig)
        plt.close(loss_fig)

        # Biểu đồ giá thật và dự báo trên tập test
        st.subheader("Đánh giá trên dữ liệu lịch sử")

        eval_fig, eval_ax = plt.subplots(figsize=(12, 5))

        eval_ax.plot(
            y_test_real.flatten(),
            label="Giá Close thật"
        )
        eval_ax.plot(
            test_pred.flatten(),
            label="Giá Close dự đoán"
        )

        eval_ax.set_xlabel("Phiên trong tập kiểm tra")
        eval_ax.set_ylabel("Giá Close")
        eval_ax.legend()
        eval_ax.grid(alpha=0.25)

        st.pyplot(eval_fig)
        plt.close(eval_fig)

        st.write("**Chỉ số đánh giá trên tập kiểm tra**")
        st.write(f"- RMSE: {rmse:.4f}")
        st.write(f"- MAE: {mae:.4f}")
        st.write(f"- R²: {r2:.4f}" if np.isfinite(r2) else "- R²: không đủ dữ liệu")

        st.warning(
            "Giá tương lai được dự báo lặp từng phiên bằng chính dự báo "
            "trước đó. Sai số có thể tích lũy; kết quả chỉ mang tính "
            "tham khảo, không phải lời khuyên đầu tư."
        )

    except Exception as e:
        st.error(f"Không thể chạy dự báo: {e}")
        st.info(
            "Nếu lỗi liên quan đến tải dữ liệu, hãy kiểm tra mã cổ phiếu "
            "hoặc thử lại sau. Nếu lỗi liên quan đến TensorFlow, cần kiểm "
            "tra phiên bản Python và thư viện trên Streamlit Cloud."
        )
else:
    st.info(
        "Nhập mã cổ phiếu ở thanh bên trái rồi bấm "
        "**Bắt đầu dự đoán**."
    )
