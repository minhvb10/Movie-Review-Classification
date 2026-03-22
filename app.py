import os
import pickle
import streamlit as st

MODEL_FILE = os.path.join("saved_models", "svm.pkl")


def load_model(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    with open(model_path, "rb") as f:
        model_obj = pickle.load(f)

    if isinstance(model_obj, dict):
        model = model_obj.get("model") or model_obj.get("pipeline") or model_obj.get("classifier") or model_obj
        vectorizer = model_obj.get("vectorizer")
    elif isinstance(model_obj, (tuple, list)) and len(model_obj) >= 2:
        model, vectorizer = model_obj[0], model_obj[1]
    else:
        model, vectorizer = model_obj, None

    if vectorizer is None:
        vectorizer_path = os.path.join("saved_models", "vectorizer.pkl")
        if os.path.exists(vectorizer_path):
            try:
                with open(vectorizer_path, "rb") as vf:
                    vectorizer = pickle.load(vf)
            except Exception:
                vectorizer = None

    return model, vectorizer


def normalize_label(pred):
    if isinstance(pred, (list, tuple, set)):
        pred = next(iter(pred), None)

    if pred is None:
        return "unknown"

    if isinstance(pred, (int, float)):
        if pred in (0, "0"):
            return "negative"
        if pred in (1, "1"):
            return "positive"
        return "positive" if float(pred) >= 0.5 else "negative"

    s = str(pred).strip().lower()
    if s in {"pos", "positive", "1"}:
        return "positive"
    if s in {"neg", "negative", "0"}:
        return "negative"

    return s


def predict_text(text: str, model, vectorizer=None):
    if vectorizer is not None:
        X = vectorizer.transform([text])
        preds = model.predict(X)
        probs = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)
    else:
        try:
            preds = model.predict([text])
        except Exception as e:
            if "could not convert string to float" in str(e):
                raise RuntimeError("Model chưa có vectorizer, cần save kèm vectorizer hoặc pipeline (vd. {'model': model, 'vectorizer': vectorizer}).") from e
            preds = model.predict(text)

        probs = None
        if hasattr(model, "predict_proba"):
            try:
                probs = model.predict_proba([text])
            except Exception:
                probs = None

    raw_pred = preds[0] if hasattr(preds, "__iter__") else preds
    label = normalize_label(raw_pred)

    score = None
    if probs is not None:
        try:
            probs_ = probs[0]
            if len(probs_) >= 2:
                score = max(probs_)
            else:
                score = float(probs_[0])
        except Exception:
            score = None

    return label, score, raw_pred


@st.cache_resource
def load_model_cached():
    """Load model once and cache it"""
    return load_model(MODEL_FILE)


def main():
    st.set_page_config(page_title="Movie Review Sentiment", page_icon="🎬")
    st.title("🎬 Movie Review Sentiment Classification")

    st.markdown(
        """
Nhập một đoạn review phim để phân loại sentiment thành **positive** hoặc **negative**.
"""
    )

    try:
        with st.spinner("Đang load model ..."):
            model, vectorizer = load_model_cached()
    except FileNotFoundError:
        st.error(f"❌ Lỗi: Không tìm thấy file model. Vui lòng đảm bảo file `svm.pkl` nằm trong thư mục `saved_models/`")
        return
    except Exception as e:
        st.error(f"❌ Lỗi khi load model: {e}")
        return

    st.success("✅ Model đã load thành công!")

    review = st.text_area("📝 Nhập đoạn review phim", height=160, placeholder="Ví dụ: This movie is absolutely amazing!")

    if st.button("🔍 Dự đoán"):
        if not review or review.strip() == "":
            st.warning("⚠️ Vui lòng nhập câu review.")
        else:
            try:
                label, score, raw = predict_text(review, model, vectorizer)
                
                st.markdown("---")
                st.subheader("📊 Kết quả dự đoán")
                
                if label == "positive":
                    st.success(f"😊 Cảm xúc: **POSITIVE**")
                else:
                    st.error(f"😞 Cảm xúc: **NEGATIVE**")
                
                if score is not None:
                    st.metric("Xác suất", f"{score:.2%}")
                
            except Exception as e:
                st.error(f"❌ Lỗi khi dự đoán: {e}")


if __name__ == "__main__":
    main()
